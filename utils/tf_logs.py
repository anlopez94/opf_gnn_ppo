import numpy as np
import tensorflow as tf


def training_episode_logs(
    writer,
    env,
    episode,
    states,
    assigned_rewards,
    losses=None,
    actor_losses=None,
    critic_losses=None,
    t=0,
):
    with writer.as_default():
        with tf.name_scope("Training"):
            tf.summary.scalar("Reward mean", np.mean(assigned_rewards), step=episode)
            tf.summary.scalar("Reward max", np.max(assigned_rewards), step=episode)
            tf.summary.scalar("Reward min", np.min(assigned_rewards), step=episode)
            min_cost = min(env.cost)
            min_index = env.cost.index(min_cost)

            try:
                loads = env.generators_loads[min_index]
                for gen_load_id, load in enumerate(loads):
                    tf.summary.scalar("Load " + str(gen_load_id), load, step=episode)
                consumption = env.consumption[min_index]
                tf.summary.scalar("consumption " + str(consumption), load, step=episode)
                generation = env.generation[min_index]
                tf.summary.scalar("generation " + str(generation), load, step=episode)
            except:
                print(
                    f"Error in training_episode_logs, cant access env.generators_loads[min_index] with min_index {min_index}"
                )
            tf.summary.scalar("Min cost", min_cost, step=episode)
            tf.summary.scalar("Min cost horizont", min_index, step=episode)
            # tf.summary.scalar("Last cost", env.cost[-1], step=episode)
            tf.summary.scalar("Converged", env.converged, step=episode)
            tf.summary.scalar("Timestep_final", t, step=episode)

            # if env.link_traffic_to_states:
            #     mean_link_utilization = [np.mean(elem[:env.n_links]) for elem in states]
            #     tf.summary.scalar("Mean Link Utilization mean", np.mean(mean_link_utilization), step=episode)
            #     tf.summary.scalar("Mean Link Utilization max", np.max(mean_link_utilization), step=episode)
            #     tf.summary.scalar("Mean Link Utilization min", np.min(mean_link_utilization), step=episode)
            if losses is not None:
                tf.summary.scalar("Loss mean", np.mean(losses), step=episode)
            if actor_losses is not None:
                tf.summary.scalar(
                    "Actor Loss mean", np.mean(actor_losses), step=episode
                )
            if critic_losses is not None:
                tf.summary.scalar(
                    "Critic Loss mean", np.mean(critic_losses), step=episode
                )
        writer.flush()


def eval_step_logs(
    writer, env, eval_step, state, actions=None, reward=None, prob=None, value=None
):
    network = ("+").join(env.network)
    with writer.as_default():
        with tf.name_scope("Eval"):
            if reward is not None:
                tf.summary.scalar("reward", reward, step=eval_step)
            if prob is not None:
                tf.summary.scalar("Prob", prob, step=eval_step)
            if value is not None:
                tf.summary.scalar("Value", value, step=eval_step)
            if actions is not None:
                tf.summary.scalar("Num actions", len(actions), step=eval_step)

        writer.flush()


def eval_final_log(writer, eval_episode, env, network):
    with writer.as_default():
        with tf.name_scope("Eval"):
            pass

            min_cost = min(env.cost)
            min_index = env.cost.index(min_cost)
            try:
                loads = env.generators_loads[min_index]
                for gen_load_id, load in enumerate(loads):
                    tf.summary.scalar("Load " + str(gen_load_id), load, step=episode)
                consumption = env.consumption[min_index]
                tf.summary.scalar("consumption " + str(consumption), load, step=episode)
                generation = env.generation[min_index]
                tf.summary.scalar("generation " + str(generation), load, step=episode)
            except:
                print(
                    f"Error in training_episode_logs, cant access env.generators_loads[min_index] with min_index {min_index}"
                )
            tf.summary.scalar("Min cost", min_cost, step=episode)
            tf.summary.scalar("Min cost horizont", min_index, step=episode)
            # tf.summary.scalar("Last cost", env.cost[-1], step=episode)
            tf.summary.scalar("Converged", env.converged, step=episode)
            tf.summary.scalar("Timestep_final", t, step=episode)
        writer.flush()


def eval_top_log(writer, eval_episode, min_costs, network):
    with writer.as_default():
        with tf.name_scope("Eval"):
            mean_costs = np.mean(min_costs)
            min_cost = np.mean(min_costs)
            tf.summary.scalar(network + " - MEAN costs", mean_costs, step=eval_episode)
            tf.summary.scalar(network + " - Min cost", min_cost, step=eval_episode)

        writer.flush()
