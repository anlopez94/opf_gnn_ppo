import tensorflow as tf
import tensorflow_probability as tfp
from environment.environment import Environment
from lib.actor import Actor
from lib.critic import Critic

import copy
import numpy as np
import random
import os
import logging
import time
import csv
import gin.tf

import utils.tf_logs as tf_logs
import utils.tf_logs as tf_logs
from utils.results import Results
import timeit






@gin.configurable
class PPOAgent(object):
    '''An implementation of a GNN-based PPO Agent'''

    def __init__(self,
                 env,
                 eval_env_type='case9,case30',
                 num_eval_samples=10,
                 max_simultaneous_actions=1,
                 clip_param=0.2,
                 critic_loss_factor=0.5,
                 entropy_loss_factor=0.001,
                 normalize_advantages=True,
                 max_grad_norm=1.0,
                 gamma=0.99,
                 gae_lambda=0.95,
                 horizon=None,
                 default_horizon=60,
                 batch_size=30,
                 epochs=3,
                 last_training_sample=99,
                 eval_period=5,
                 max_evals=100,
                 change_load=False,
                 change_load_period=1,
                 select_max_action=False,
                 optimizer=tf.keras.optimizers.Adam(
                     learning_rate=0.0003,
                     beta_1=0.9,
                     epsilon=0.00001),
                 base_dir='logs',
                 checkpoint_base_dir='checkpoints',
                 save_checkpoints=True):

        self.env = env
        eval_env_type = [env for env in eval_env_type.split('+')]
        self.eval_env_type = eval_env_type
        self.num_eval_samples = num_eval_samples
        self.max_simultaneous_actions = max_simultaneous_actions
        self.clip_param = clip_param

        self.eval_logs = False

        self.generate_training_actor_critic_functions()
        self.add_training_actor_critic()
        self.num_actions = 1

        self.optimizer = optimizer
        self.critic_loss_factor = critic_loss_factor
        self.entropy_loss_factor = entropy_loss_factor
        self.normalize_advantages = normalize_advantages
        self.max_grad_norm = max_grad_norm

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.given_horizon = horizon
        self.default_horizon = default_horizon

        self.define_horizon()
        self.epochs = epochs
        self.batch_size = batch_size
        self.last_training_sample = last_training_sample
        self.eval_period = eval_period
        self.max_evals = max_evals
        self.select_max_action = select_max_action

        self.eval_step = 0
        self.eval_episode = 0
        self.base_dir = base_dir
        self.checkpoint_base_dir = checkpoint_base_dir
        self.save_checkpoints = save_checkpoints
        self.reload_model = False
        self.change_sample = False
        self.change_load = change_load
        self.change_load_period = change_load_period

    def generate_training_actor_critic_functions(self):
        self.training_actor = {}
        self.training_critic = {}
        self.add_training_actor_critic()

    def add_training_actor_critic(self):
        self.training_actor[self.env.network] = Actor(self.env.graph_info)
        self.training_actor[self.env.network].build()
        self.training_critic[self.env.network] = Critic(self.env.graph_info)
        self.training_critic[self.env.network].build()

    def define_horizon(self):

        self.horizon = int(len(self.env.generators_value) * self.env.generator_slices * self.env.horizon_percentage)

    def reset_env(self):
        previous_env_type = self.env.network
        self.env.reset(change_sample=self.change_sample)
        if self.change_sample and len(self.env.env_type) > 1:
            if previous_env_type != self.env.network:
                self.add_training_actor_critic()
            self.load_latest_model(previous_env_type)
            self.define_horizon()
        self.change_sample = False

    def gae_estimation(self, rewards, values, last_value):
        last_gae_lambda = 0
        advantages = np.zeros_like(values, dtype=np.float32)
        for i in reversed(range(self.horizon)):
            if i == self.horizon - 1:
                next_value = last_value
            else:
                next_value = values[i + 1]
            delta = rewards[i] + self.gamma * next_value - values[i]
            advantages[i] = last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
        returns = values + advantages
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return returns, advantages

    def run_episode(self):
        # reset state at the beginning of each iteration
        self.reset_env()
        state = self.env.get_state()
        states = np.zeros((self.horizon, self.env.n_nodes * self.env.num_features),
                          dtype=np.float32)
        actions = []  # np.zeros(self.horizon, dtype=np.float32)
        rewards = np.zeros(self.horizon, dtype=np.float32)
        log_probs = []  # np.zeros(self.horizon, dtype=np.float32)
        values = np.zeros(self.horizon, dtype=np.float32)

        for t in range(self.horizon):
            current_actions, current_log_probs = self.act(state)
            value = self.run_critic(state)
            total_reward = 0
            for action in current_actions:
                next_state, reward = self.env.step(action)
                total_reward += reward
            states[t] = state
            actions.append(current_actions)
            rewards[t] = total_reward
            log_probs.append(current_log_probs)
            values[t] = value.numpy()[0]
            state = next_state
            #state will be false when the power flow dit not converge
            if state is False:
                break
        if state is False:
            last_value = 0
        else:
            value = self.run_critic(state)
            last_value = value.numpy()[0]

        return states, actions, rewards, log_probs, values, last_value, t

    def run_update(self, training_episode, states, actions, returns, advantages, log_probs):
        actor_losses, critic_losses, losses = [], [], []
        inds = np.arange(self.horizon)
        inds = np.arange(len(actions))
        for epoch in range(self.epochs):
            print(f'epoch {epoch}')
            np.random.shuffle(inds)
            #i change start going from 0 to horizon, becouse episode could have stopped before horizon
            for start in range(0, len(actions), self.batch_size):
                end = start + self.batch_size
                if end > len(actions):
                    end = len(actions)
                minibatch_ind = inds[start:end]
                minibatch_actions = tf.ragged.constant([actions[i] for i in list(minibatch_ind)])
                minibatch_log_probs = tf.ragged.constant([log_probs[i] for i in list(minibatch_ind)])
                actor_loss, critic_loss, loss, grads = self.compute_losses_and_grads(states[minibatch_ind],
                                                                                     minibatch_actions,
                                                                                     returns[minibatch_ind],
                                                                                     advantages[minibatch_ind],
                                                                                     minibatch_log_probs)
                # print('Antes apply gradients')
                self.apply_grads(grads)
                # print('Despues apply gradients')
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                losses.append(loss.numpy())

        return actor_losses, critic_losses, losses

    def train_and_evaluate(self):
        training_episode = -1
        results = Results(self.set_experiment_identifier(), 'logs_csv')

        while not (self.env.num_sample == self.last_training_sample and self.change_sample == True):
            training_episode += 1
            print(f'Episode {training_episode} starting {self.env.num_sample}')
            states, actions, rewards, log_probs, values, last_value, t = self.run_episode()

            returns, advantages = self.gae_estimation(rewards, values, last_value)


            actor_losses, critic_losses, losses = self.run_update(training_episode, states, actions, returns,
                                                                  advantages, log_probs)

            tf_logs.training_episode_logs(self.writer, self.env, training_episode, states, rewards, losses,
                                          actor_losses, critic_losses, t)
            results.save_episode(self.env, training_episode, states, rewards, losses, actor_losses, critic_losses, t)
            results.save_results()
            print(f'Episode {training_episode} finished con sample {self.env.num_sample}')

            if (training_episode + 1) % self.eval_period == 0:
                print('entra a checkpoint')
                print(f' num sample {self.env.num_sample}')
                #pongo esta linea xk comente la otra
                self.eval_episode += 1
                # self.training_eval()
                if self.save_checkpoints:
                    print('Guardar modelo')
                    self.training_actor[self.env.network]._set_inputs(states[0])
                    self.training_critic[self.env.network]._set_inputs(states[0])
                    self.save_model(os.path.join(self.checkpoint_dir, 'episode' + str(self.eval_episode)))
                if self.change_load and self.eval_episode % self.change_load_period == 0:
                    self.change_sample = True
                elif not self.change_load:
                    self.change_sample = True

    def only_evaluate(self):
        self.env.initialize_environment(num_sample=self.last_training_sample + 1)
        results_eval = Results(self.set_experiment_identifier(only_eval=True), 'results')
        for _ in range(self.max_evals):
            results_eval = self.evaluation(results_eval)
            self.change_sample = True
            results_eval.save_results()


    def generate_eval_env(self):
        self.eval_envs = {}
        for eval_env_type in self.eval_env_type:
            self.eval_envs[eval_env_type] = Environment()

    def generate_eval_actor_critic_functions(self):
        self.eval_actor = {}
        self.eval_critic = {}
        for eval_env_type in self.eval_env_type:
            self.eval_actor[eval_env_type] = Actor(self.eval_envs[eval_env_type].graph_info)
            self.eval_actor[eval_env_type].build()
            self.eval_critic[eval_env_type] = Critic(self.eval_envs[eval_env_type].graph_info)
            self.eval_critic[eval_env_type].build()

    def update_eval_actor_critic_functions(self):
        for eval_env_type in self.eval_env_type:
            for w_model, w_eval_actor in zip(self.training_actor[self.env.network].trainable_variables,
                                             self.eval_actor[eval_env_type].trainable_variables):
                w_eval_actor.assign(w_model)
            for w_model, w_eval_critic in zip(self.training_critic[self.env.network].trainable_variables,
                                              self.eval_critic[eval_env_type].trainable_variables):
                w_eval_critic.assign(w_model)

    def training_eval(self):
        # Evaluation phase
        print('\n\tEvaluation ' + str(self.eval_episode) + '...\n')

        if self.eval_episode == 0:
            self.generate_eval_env()
            self.generate_eval_actor_critic_functions()

        self.update_eval_actor_critic_functions()

        for eval_env_type in self.eval_env_type:
            self.eval_envs[eval_env_type].define_num_sample(100)

            total_min_max = []
            mini_eval_episode = self.eval_episode * self.num_eval_samples
            min_costs = []
            for _ in range(self.num_eval_samples):
                self.eval_envs[eval_env_type].reset(change_sample=True)
                state = self.eval_envs[eval_env_type].get_state()
                tf_logs.eval_step_logs(self.writer, self.eval_envs[eval_env_type], self.eval_step, state)
                probs, values = [], []

                for i in range(self.horizon):
                    self.eval_step += 1
                    actions, log_probs = self.eval_act(self.eval_actor[eval_env_type], state,
                                                       select_max=self.select_max_action)
                    value = self.eval_critic[eval_env_type](state)
                    for action in actions:
                        next_state, reward = self.eval_envs[eval_env_type].step(action)
                    # probs.append(np.exp(log_prob))
                    values.append(value.numpy()[0])
                    state = next_state
                    if state is False:
                        break
                    # tf_logs.eval_step_logs(self.writer, self.eval_envs[eval_env_type], self.eval_step, state,
                    #                        actions=actions)

                min_costs.append(np.max(self.eval_envs[eval_env_type].cost))
                # tf_logs.eval_final_log(self.writer, mini_eval_episode, max_link_utilization, eval_env_type)
                mini_eval_episode += 1

            tf_logs.eval_top_log(self.writer, self.eval_episode, min_costs, eval_env_type)
        self.eval_episode += 1

    def evaluation(self, results):

        # Evaluation phase
        print('\n\tEvaluation ' + str(self.eval_episode) + '...\n')
        self.reset_env()
        state = self.env.get_state()
        # if self.eval_logs: tf_logs.eval_step_logs(self.writer, self.env, self.eval_step, state)
        # if self.env.link_traffic_to_states:
        #     max_link_utilization = [np.max(state[:self.env.n_links])]
        #     mean_link_utilization = [np.mean(state[:self.env.n_links])]
        probs, values = [], []
        start = timeit.default_timer()
        for i in range(self.horizon):
            self.eval_step += 1
            actions, log_probs = self.act(state, select_max=self.select_max_action)
            #value = self.run_critic(state)
            for action in actions:
                next_state, reward = self.env.step(action)
            # probs.append(np.exp(log_prob))
            values.append(0)
            state = next_state
            # if self.env.link_traffic_to_states:
            #     max_link_utilization.append(np.max(state[:self.env.n_links]))
            #     mean_link_utilization.append(np.mean(state[:self.env.n_links]))

            # if self.eval_logs: tf_logs.eval_step_logs(self.writer, self.env, self.eval_step, state, actions, reward,
            #                                           None, values[i])
            if state is False:
                break


        stop = timeit.default_timer()
        time = round((stop - start) * 1000, 1)
        results.save_eval(self.env, i, time)

        # if self.env.link_traffic_to_states:
        #     if self.eval_logs: tf_logs.eval_final_log(self.writer, self.eval_episode, max_link_utilization,
        #                                               ('+').join(self.env.env_type))
        #     if self.only_eval: self.write_eval_results(self.eval_episode, np.min(max_link_utilization))
        #
        # # self.eval_step += 10
        self.eval_episode += 1
        return results

    # @tf.function
    def compute_actor_loss(self, new_log_probs, old_log_probs, advantages):
        ratio = tf.exp(new_log_probs - old_log_probs)
        pg_loss_1 = tf.map_fn(lambda x: -x[0] * x[1], (advantages, ratio),
                              fn_output_signature=tf.RaggedTensorSpec(shape=[None],
                                                                      dtype=tf.float32))  # - advantages * ratio
        pg_loss_2 = tf.map_fn(lambda x: -x[0] * tf.clip_by_value(x[1], 1.0 - self.clip_param, 1.0 + self.clip_param),
                              (advantages, ratio), fn_output_signature=tf.RaggedTensorSpec(shape=[None],
                                                                                           dtype=tf.float32))  # - advantages * tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        pg_loss = tf.maximum(pg_loss_1, pg_loss_2)
        actor_loss = tf.reduce_mean(pg_loss)
        return actor_loss

    # @tf.function
    def get_new_log_prob_and_entropy(self, state, actions):
        logits = self.training_actor[self.env.network](state, training=True)
        probs = tfp.distributions.Categorical(logits=logits)
        log_probs = tf.map_fn(lambda x: probs.log_prob(x), actions,
                              fn_output_signature=tf.float32)  # [probs.log_prob(action) for action in actions]
        return (log_probs, probs.entropy())

    # @tf.function
    def compute_losses_and_grads(self, states, actions, returns, advantages, old_log_probs):
        with tf.GradientTape(persistent=True) as tape:
            new_log_probs, entropy = tf.map_fn(lambda x: self.get_new_log_prob_and_entropy(x[0], x[1]),
                                               (states, actions), fn_output_signature=(
                tf.RaggedTensorSpec(shape=[None], dtype=tf.float32), tf.float32))

            values = tf.map_fn(lambda x: self.training_critic[self.env.network](x, training=True), states,
                               fn_output_signature=tf.float32)
            values = tf.reshape(values, [-1])

            critic_loss = tf.reduce_mean(tf.square(returns - values))
            entropy_loss = tf.reduce_mean(entropy)
            actor_loss = self.compute_actor_loss(new_log_probs, old_log_probs, advantages)
            loss = actor_loss - self.entropy_loss_factor * entropy_loss + self.critic_loss_factor * critic_loss

        grads = tape.gradient(loss, self.training_actor[self.env.network].trainable_variables + self.training_critic[
            self.env.network].trainable_variables)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        return actor_loss, critic_loss, loss, grads

    def apply_grads(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.training_actor[self.env.network].trainable_variables +
                                           self.training_critic[self.env.network].trainable_variables))

    # @tf.function
    def act(self, state, select_max=False):
        logits = self.training_actor[self.env.network](state)
        # print(logits)
        probs = tfp.distributions.Categorical(logits=logits)
        if select_max:
            actions = [tf.argmax(logits)]
        else:
            actions = list(set([probs.sample().numpy() for _ in range(self.max_simultaneous_actions)]))
        log_probs = [probs.log_prob(action).numpy() for action in actions]
        return actions, log_probs

    # @tf.function
    def eval_act(self, actor, state, select_max=False):
        logits = actor(state)
        probs = tfp.distributions.Categorical(logits=logits)
        if select_max:
            actions = [tf.argmax(logits)]
        else:
            actions = list(set([probs.sample().numpy() for _ in range(self.max_simultaneous_actions)]))
        log_probs = [probs.log_prob(action).numpy() for action in actions]
        return actions, log_probs

    # @tf.function
    def run_critic(self, state):
        return self.training_critic[self.env.network](state)

    def save_model(self, checkpoint_dir):
        self.training_actor[self.env.network].save(checkpoint_dir + '/actor')
        self.training_critic[self.env.network].save(checkpoint_dir + '/critic')

    def load_latest_model(self, previous_env_type):
        for w_model, w_actor in zip(self.training_actor[previous_env_type].trainable_variables,
                                    self.training_actor[self.env.network].trainable_variables):
            w_actor.assign(w_model)
        for w_model, w_critic in zip(self.training_critic[previous_env_type].trainable_variables,
                                     self.training_critic[self.env.network].trainable_variables):
            w_critic.assign(w_model)

    def load_saved_model(self, model_dir, only_eval):
        model = tf.keras.models.load_model(model_dir + '/actor')
        for w_model, w_actor in zip(model.trainable_variables,
                                    self.training_actor[self.env.network].trainable_variables):
            w_actor.assign(w_model)
        if not only_eval:
            model = tf.keras.models.load_model(model_dir + '/critic')
            for w_model, w_critic in zip(model.trainable_variables,
                                         self.training_critic[self.env.network].trainable_variables):
                w_critic.assign(w_model)
        self.model_dir = model_dir
        self.reload_model = True

    def write_eval_results(self, step, value):
        csv_dir = os.path.join('./notebooks/logs', self.experiment_identifier)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        with open(csv_dir + '/results.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([step, value])

    def set_experiment_identifier(self, only_eval=False):
        self.only_eval = only_eval
        mode = 'eval' if only_eval else 'training'

        if mode == 'training':
            # ENVIRONMENT

            num_actions = 'actions' + str(self.max_simultaneous_actions)
            generator_slices = 'generator_slices' + str(self.env.generator_slices)

            env_folder = self.env.network + '_' + str(self.env.env_number)

            # MULTI-AGENT
            # pre = 'MESS'
            # num_actions = 'actions' + str(self.max_simultaneous_actions)
            create_message = 'create_message' + str(self.training_actor[self.env.network].create_message_nn)
            generator_slices = 'generator_slices' + str(self.env.generator_slices)
            horizons = 'horizon' + str(self.horizon)
            reward = 'reward' + str(self.env.reward_magnitude)
            reward_computation = 'reward_comp' + str(self.env.reward_computation)
            action_folder = ('-').join([create_message, generator_slices, horizons, reward, reward_computation])

            # PPOAGENT
            batch = 'batch' + str(self.batch_size)
            gae_lambda = 'gae' + str(self.gae_lambda)
            lr = 'lr' + str(self.optimizer.get_config()['learning_rate'])
            epsilon = 'epsilon' + str(self.optimizer.epsilon)
            clip = 'clip' + str(self.clip_param)
            gamma = 'gamma' + str(self.gamma)
            period = 'period' + str(self.eval_period)
            epoch = 'epoch' + str(self.epochs)
            node_state_size = 'node_state_size' + str(self.training_actor[self.env.network].node_state_size)
            message_iterations = 'message_iterations' + str(self.training_actor[self.env.network].message_iterations)
            agent_folder = ('-').join([batch, lr, epsilon, gae_lambda, clip, gamma, period, epoch, node_state_size, message_iterations])

            # ACTOR-CRITIC
            state_size = 'size' + str(self.training_actor[self.env.network].node_state_size)
            iters = 'iters' + str(self.training_actor[self.env.network].message_iterations)
            # nn_size = 'nnsize' + ('-').join([str(x) for x in self.training_actor[self.env.network].hidden_units_readout])
            dropout = 'drop' + str(self.training_actor[self.env.network].dropout_rate)
            activation = self.training_actor[self.env.network].activation_fn
            # activation  ='tanh' if activation == tf.nn.tanh else 'relu'
            function_folder = ('-').join([state_size, iters, dropout, activation])

            self.experiment_identifier = os.path.join(mode, env_folder, action_folder, agent_folder, function_folder)

        else:
            model_dir = self.model_dir

            network = '+'.join([str(elem) for elem in self.env.env_type])

            eval_env_folder = ('_').join([network, str(self.env.env_number)])
            # eval_num_actions = 'iters'+str(self.training_actor[self.env.network].message_iterations)
            model_name_parts = model_dir.split('/')
            # RELOADED MODEL
            env_folder = model_dir.split('/')[8]
            agent_folder = model_dir.split('/')[9]
            agent_folder2 = model_dir.split('/')[10]
            function_folder = model_dir.split('/')[11]
            episode = model_dir.split('/')[12]

            self.experiment_identifier = os.path.join(mode, eval_env_folder, env_folder, agent_folder, agent_folder2
                                                      ,
                                                      function_folder, episode)

        return self.experiment_identifier

    def set_writer_and_checkpoint_dir(self, writer_dir, checkpoint_dir):
        self.writer_dir = writer_dir
        self.checkpoint_dir = checkpoint_dir
        self.writer = tf.summary.create_file_writer(self.writer_dir)