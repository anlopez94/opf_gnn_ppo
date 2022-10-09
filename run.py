

import os
import sys

import numpy as np
import tensorflow as tf
from agents.ppo_agent import PPOAgent
from lib.run_experiment import Runner
from utils.functions import load_gin_configs

from absl import app
from absl import flags


flags.DEFINE_string('base_dir', 'logs',
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string('network', 'logs',
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', ["configs/general.gin", "configs/ppo.gin"], 'List of paths to gin configuration files')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files ')

FLAGS = flags.FLAGS



# def main():
#     """Main method.
#     Args:
#       unused_argv: Arguments (unused).
#     """
#
#
#     runner = Runner()
#
#
#     runner.run_experiment()
def set_experiment_identifier(network,env_number ,create_message_nn,generator_slices,horizon,reward_magnitude,reward_computation,batch_size,gae_lambda,learning_rate,epsilon,clip_param,gamma,eval_period,epochs,node_state_size,message_iterations,dropout_rate,activation_fn,mode):
    env_folder = network + '_' + str(env_number)

    # MULTI-AGENT
    # pre = 'MESS'
    # num_actions = 'actions' + str(self.max_simultaneous_actions)
    create_message = 'create_message' + str(create_message_nn)
    generator_slices = 'generator_slices' + str(generator_slices)
    horizons = 'horizon' + str(horizon)
    reward = 'reward' + str(reward_magnitude)
    reward_computation = 'reward_comp' + str(reward_computation)
    action_folder = ('-').join([create_message, generator_slices, horizons, reward, reward_computation])

    # PPOAGENT
    batch = 'batch' + str(batch_size)
    gae_lambda = 'gae' + str(gae_lambda)
    lr = 'lr' + str(learning_rate)
    epsilon = 'epsilon' + str(epsilon)
    clip = 'clip' + str(clip_param)
    gamma = 'gamma' + str(gamma)
    period = 'period' + str(eval_period)
    epoch = 'epoch' + str(epochs)
    node_state_size_path = 'node_state_size' + str(node_state_size)
    message_iterations_path = 'message_iterations' + str(message_iterations)
    agent_folder = ('-').join([batch, lr, epsilon, gae_lambda, clip, gamma, period, epoch, node_state_size_path, message_iterations_path])

    # ACTOR-CRITIC
    state_size = 'size' + str(node_state_size)
    iters = 'iters' + str(message_iterations)
    # nn_size = 'nnsize' + ('-').join([str(x) for x in self.training_actor[self.env.network].hidden_units_readout])
    dropout = 'drop' + str(dropout_rate)
    activation = activation_fn
    # activation  ='tanh' if activation == tf.nn.tanh else 'relu'
    function_folder = ('-').join([state_size, iters, dropout, activation])

    experiment_identifier = os.path.join(mode, env_folder, action_folder, agent_folder, function_folder)
    return experiment_identifier
network = 'case30'
env_number = 'v3'
create_message_nn = True
generator_slices = 50
horizon = 125
reward_magnitude = 'cost'
reward_computation = 'change'
batch_size = 25
gae_lambda = 0.9
learning_rate = 0.003
epsilon = 0.1
clip_param = 0.3
gamma = 0.95
eval_period = 1
epochs = 3
node_state_size = 16
message_iterations = 4
dropout_rate = 0.2
activation_fn='tanh'
mode ='training'
episode = 224
experiment_identifier = set_experiment_identifier(network,env_number ,create_message_nn,generator_slices,horizon,reward_magnitude,reward_computation,batch_size,gae_lambda,learning_rate,epsilon,clip_param,gamma,eval_period,epochs,node_state_size,message_iterations,dropout_rate,activation_fn,mode)
dir_check = 'checkpoints_historicos/' + experiment_identifier + '/episode' + str(episode)

def main(argv=None):
    """Main method.
    Args:
      unused_argv: Arguments (unused).
    """


    FLAGS = flags.FLAGS
    load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

    # orig_stdout = sys.stdout
    # orig_stderr = sys.stderr

    #runner = Runner(model_dir=sys.path[0] + '/' + dir_check, only_eval=True)


    runner = Runner()

    # f = open(os.path.join(runner.agent.writer_dir, 'out.txt'), 'w+')
    # sys.stdout = f
    # sys.stderr = f

    runner.run_experiment()
    #
    # sys.stdout = orig_stdout
    # sys.stderr = orig_stderr
    # f.close()

if __name__ == '__main__':
    app.run(main)
    # main()
