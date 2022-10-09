import numpy as np
import os
import csv
class Results(object):
    def __init__(self, dir, results_base_dir='results'):
        self.results_base_dir = results_base_dir
        self.dir = dir
        self.mean_reward = []
        self.max_reward = []
        self.min_reward = []
        self.min_cost = []
        self.min_index = []
        self.loads = []
        self.converged = []
        self.timestep_final = []
        self.consumption = []
        self.generation = []
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.time = []

    def save_episode(self, env, training_episode, states, rewards, losses, actor_losses, critic_losses, t):
        self.time.append(0)
        self.mean_reward.append(np.mean(rewards))
        self.max_reward.append(np.max(rewards))
        self.min_reward.append(np.min(rewards))
        self.min_cost.append(min(env.cost))
        self.min_index.append(env.cost.index(min(env.cost)))
        min_index = env.cost.index(min(env.cost))
        self.converged.append(env.converged)
        self.timestep_final.append(t)

        try:
            self.loads.append(env.generators_loads[min_index])
            self.consumption.append(env.consumption[min_index])
            self.generation.append(env.generation[min_index])
        except:
            self.loads.append(0)
            self.consumption.append(0)
            self.generation.append(0)
            print(f"Error in training_episode_logs, cant access env.generators_loads[min_index] with min_index {min_index}")

        if losses is not None:
            self.losses.append(np.mean(losses))
        if actor_losses is not None:
            self.actor_losses.append(np.mean(actor_losses))
        if critic_losses is not None:
            self.critic_losses.append(np.mean(critic_losses))
        pass

    def save_eval(self, env, t, time):

        self.min_cost.append(round(min(env.cost),3))
        self.min_index.append(env.cost.index(min(env.cost)))
        min_index = env.cost.index(min(env.cost))
        self.converged.append(env.converged)
        self.timestep_final.append(t)
        try:
            self.consumption.append(round(env.consumption[min_index],3))
        except:
            self.consumption.append(0)
        try:
            self.generation.append(round(env.generation[min_index],3))
        except:
            self.generation.append(0)
        try:
            self.loads.append(env.generators_loads[min_index])
        except:
            self.loads.append(0)
            print(f"Error in training_episode_logs, cant access env.generators_loads[min_index] with min_index {min_index}")
        self.time.append(time)

        self.losses.append(0)
        self.actor_losses.append(0)
        self.critic_losses.append(0)
        self.mean_reward.append(0)
        self.max_reward.append(0)
        self.min_reward.append(0)


    def save_results(self):
        if not os.path.exists(self.results_base_dir + '/' + self.dir):
            os.makedirs(self.results_base_dir + '/' + self.dir)
        with open(self.results_base_dir + '/' + self.dir + '/' + 'results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "time", "mean_reward", "max_reward", "min_reward", "min_cost", "min_index", "converged", "timestep_final", "consumption", "generation", "loads", "losses", "actor_losses", "critic_losses"])
            for i in range(len(self.min_cost)):
                try:
                    loads = [str(x) for x in self.loads[i]]
                    loads = '-'.join(loads)
                except:
                    loads = 0
                writer.writerow([i, self.time[i], self.mean_reward[i], self.max_reward[i], self.min_reward[i], self.min_cost[i], self.min_index[i], self.converged[i], self.timestep_final[i], self.consumption[i], self.generation[i], loads, self.losses[i], self.actor_losses[i], self.critic_losses[i]])
        pass