#!/usr/bin/python3

import os

# ENVIRONMENT
env_type_list = ["case30"]
env_number_list = [3]
reward_magnitude_list = ["cost"]
reward_computation_list = ["change"]
horizon_percentage_list = [0.50]
generator_slices_list = [50]


# MULTI-AGENT
num_action_list = [1]
num_action = 1

# PPOAGENT
batch_list = [25]
gae_lambda_list = [0.9]
gae_lambda = 0.9
eval_period_list = [1]
last_training_sample_list = [800]
gamma_list = [0.95]
clip_list = [0.2]
epoch_list = [3]
epoch = 3
# sacar dropout de NN menos readout
lr_list = [0.003]
epsilon_list = [0.1]
create_message_nn_list = [True]

# ACTOR-CRITIC
# node_state_size_list = [32]
# message_iteration_list = [4]
# hidden_units_update_list = ["256-128-32"]
# critic_hidden_units_readout_list = ["128-32"]
# critic_hidden_units_readout = "128-32"
# actor_hidden_units_readout_list = ["64-32"]
# actor_hidden_units_readout = "64-32"
# hidden_units_message_list = ["128-32"]
# hidden_units_readout = "64-32"

node_state_size_list = [16]
message_iteration_list = [4]
hidden_units_update_list = ["128-64-16"]
critic_hidden_units_readout_list = ["64-16"]
critic_hidden_units_readout = "64-16"
actor_hidden_units_readout_list = ["32-16"]
actor_hidden_units_readout = "32-16"
hidden_units_message_list = ["64-16"]
hidden_units_readout = "32-16"

dropout_list = [0.2]
dropout = 0.2
activation_fn_list = ["tanh"]


# horizon_percentage_list = [0.4, 0.5, 0.6]

horizon_percentage_list = [0.5]
generator_slices_list = [50]


for activation_fn in activation_fn_list:
    # for dropout in dropout_list:
    for create_message_nn in create_message_nn_list:
        for generator_slices in generator_slices_list:
            for horizon_percentage in horizon_percentage_list:
                for message_iteration in message_iteration_list:
                    for node_state_size in node_state_size_list:
                        for hidden_units_update in hidden_units_update_list:
                            for hidden_units_message in hidden_units_message_list:
                                for env_type in env_type_list:
                                    for reward_magnitude in reward_magnitude_list:
                                        for (
                                            reward_computation
                                        ) in reward_computation_list:
                                            for eval_period in eval_period_list:
                                                for (
                                                    last_training_sample
                                                ) in last_training_sample_list:
                                                    for gamma in gamma_list:
                                                        # for epoch in epoch_list:
                                                        for (
                                                            env_number
                                                        ) in env_number_list:
                                                            for batch in batch_list:
                                                                for lr in lr_list:
                                                                    for (
                                                                        epsilon
                                                                    ) in epsilon_list:
                                                                        # for num_action in num_action_list:
                                                                        for (
                                                                            clip
                                                                        ) in clip_list:
                                                                            for gae_lambda in gae_lambda_list:
                                                                                cmd = (
                                                                                    "python ./run.py"
                                                                                    " --gin_bindings='Environment.env_type = \""
                                                                                    + env_type
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Environment.env_number = "
                                                                                    + str(
                                                                                        env_number
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Environment.reward_magnitude = \""
                                                                                    + reward_magnitude
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Environment.reward_computation = \""
                                                                                    + reward_computation
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Environment.horizon_percentage = "
                                                                                    + str(
                                                                                        horizon_percentage
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Environment.generator_slices = "
                                                                                    + str(
                                                                                        generator_slices
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.max_simultaneous_actions ="
                                                                                    + str(
                                                                                        num_action
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.eval_period = "
                                                                                    + str(
                                                                                        eval_period
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.last_training_sample = "
                                                                                    + str(
                                                                                        last_training_sample
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.gamma = "
                                                                                    + str(
                                                                                        gamma
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.clip_param = "
                                                                                    + str(
                                                                                        clip
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.epochs = "
                                                                                    + str(
                                                                                        epoch
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.batch_size = "
                                                                                    + str(
                                                                                        batch
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='PPOAgent.gae_lambda = "
                                                                                    + str(
                                                                                        gae_lambda
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='tf.keras.optimizers.Adam.learning_rate = "
                                                                                    + str(
                                                                                        lr
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='tf.keras.optimizers.Adam.epsilon = "
                                                                                    + str(
                                                                                        epsilon
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Actor.node_state_size = "
                                                                                    + str(
                                                                                        node_state_size
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Actor.hidden_units_readout = \""
                                                                                    + str(
                                                                                        actor_hidden_units_readout
                                                                                    )
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Actor.hidden_units_update = \""
                                                                                    + str(
                                                                                        hidden_units_update
                                                                                    )
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Actor.hidden_units_message = \""
                                                                                    + str(
                                                                                        hidden_units_message
                                                                                    )
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Actor.dropout_rate = "
                                                                                    + str(
                                                                                        dropout
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Actor.message_iterations = "
                                                                                    + str(
                                                                                        message_iteration
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Actor.activation_fn = \""
                                                                                    + activation_fn
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Actor.create_message_nn = \""
                                                                                    + str(
                                                                                        create_message_nn
                                                                                    )
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Critic.node_state_size = "
                                                                                    + str(
                                                                                        node_state_size
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Critic.hidden_units_readout = \""
                                                                                    + str(
                                                                                        critic_hidden_units_readout
                                                                                    )
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Critic.hidden_units_update = \""
                                                                                    + str(
                                                                                        hidden_units_update
                                                                                    )
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Critic.hidden_units_message = \""
                                                                                    + str(
                                                                                        hidden_units_message
                                                                                    )
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Critic.dropout_rate = "
                                                                                    + str(
                                                                                        dropout
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Critic.message_iterations = "
                                                                                    + str(
                                                                                        message_iteration
                                                                                    )
                                                                                    + "'"
                                                                                    " --gin_bindings='Critic.activation_fn = \""
                                                                                    + activation_fn
                                                                                    + "\"'"
                                                                                    " --gin_bindings='Critic.create_message_nn = \""
                                                                                    + str(
                                                                                        create_message_nn
                                                                                    )
                                                                                    + "\"' &"
                                                                                )

                                                                                print(
                                                                                    cmd
                                                                                )
                                                                                os.system(
                                                                                    cmd
                                                                                )
