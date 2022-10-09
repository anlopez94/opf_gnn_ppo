
import pandapower as pp
import os
import networkx as nx
from networkx.readwrite import json_graph
import json
import tqdm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import networkx as nx
import numpy as np
import pandas as pd
import copy
import gin.tf
import sys
import joblib

from utils.scalers import RewardScaler, GenScaler


@gin.configurable
class Environment(object):

    def __init__(self,

                 init_sample=0,
                 seed_init_generation=1,
                 env_type='case30',
                 env_number=0,
                 reward_magnitude='cost',
                 base_reward='min_max',
                 reward_computation='change',
                 generator_slices=25,
                 horizon_percentage=0.3,
                 base_dir=sys.path[0] + '/dataset_generation/dataset/'): # env_type if type(env_type) == list else [env_type]

        env_type = [env for env in env_type.split('+')]
        self.env_type = env_type
        self.env_number = env_number
        self.num_sample = init_sample - 1
        self.seed_init_generation = seed_init_generation
        self.reward_magnitude = reward_magnitude
        self.generator_slices = generator_slices
        self.horizon_percentage = horizon_percentage
        self.base_reward = base_reward
        self.reward_computation = reward_computation
        self.num_features = 4
        self.base_dir = base_dir
        self.dataset_dirs = []

        self.initialize_environment()

    def initialize_environment(self, num_sample=None, random_env=True):
        self.converged = 1
        if num_sample is not None:
            self.num_sample = num_sample
        else:
            self.num_sample += 1

        if random_env:
            num_env = np.random.randint(0, len(self.env_type))
        else:
            num_env = self.num_sample % len(self.env_type)

        self.network = self.env_type[num_env]
        if self.env_number:
            self.file = self.base_dir + self.network + '/' + self.network + '_' + str(self.env_number) + '.xlsx'
        else:
            self.file = self.base_dir + self.network + '/' + self.network + '.xlsx'
        # load excel
        self.net = pp.from_excel(self.file, convert=True)
        self.save_initial_values()
        self.load_scalers()
        self.create_weights_scaler()
        self.update_pp_net()
        pf = self.run_powerflow()
        line_constrains_violated = self.compute_line_constraints(self.net)
        #si inicialmente no converge el problema se suma 1 slice a cada generador hasta que converga el power flow
        # while pf is False or line_constrains_violated is True:
        #     self.aument_one_all_generatos()
        #     self.update_pp_net()
        #     pf = self.run_powerflow()
        #     line_constrains_violated = self.compute_line_constraints(self.net)
        self.create_net_info()
        self.normalize_reward()
        #first reward
        self.reward_measure_prev = self.compute_reward_measure()

    def normalize_reward(self):
        min_reward = - (self._compute_solution_cost(self.net) * 1.3)
        net_opf = self.run_optimalpowerflow(copy.deepcopy(self.net))
        max_reward = - self._compute_solution_cost(net_opf)
        self.reward_scaler = RewardScaler(max_reward, min_reward)
        # test = self.reward_scaler.transform(-14900)
        # test2 = self.reward_scaler.inverse_transform(0.2)

    def compute_line_constraints(self, net):
        # line constraints
        lines_condition = pd.concat([net.line, net.res_line], axis=1)
        lines = lines_condition[lines_condition['loading_percent'] > lines_condition['max_loading_percent']]

        return lines.shape[0] > 0

    def save_initial_values(self, use_min_gen=True):

        self.nodes = self.net['bus'].index.values.tolist()
        self.n_nodes = len(self.nodes)
        self._load_gen_config()
        if use_min_gen== True:
            self.generators_value = {key: value['min'] for key, value in self.gen_config.items()}
        else:
            self.generators_value = {key: 0 for key in self.gen_config.keys()}
        self.generators_value_norm = {key: 0 for key in self.gen_config.keys()}
        self.cost = []
        self.generators_loads = []
        self.generation = []
        self.consumption = []

    def create_weights_scaler(self):
        weights = []
        line = ['r_ohm_per_km', 'x_ohm_per_km']
        for index, row in self.net['line'].iterrows():
            weights.append(row.loc[line].values)
            weights.append(row.loc[line].values)
        self.scaler_weights = MinMaxScaler()
        self.scaler_weights.fit(weights)

    def create_net_info(self):

        node_columns = ['vm_pu', 'va_degree', 'p_mw', 'q_mvar']
        line = ['r_ohm_per_km', 'x_ohm_per_km']
        nodes_info, nodes_neighbour = {}, {}
        weights, edges, gen = [], [], {}

        # node x information
        for index, row in self.net['res_bus'].iterrows():
            nodes_info[index] = row.loc[node_columns].values
        # line information
        for index, row in self.net['line'].iterrows():
            edges.append([row['from_bus'], row['to_bus']])
            edges.append([row['to_bus'], row['from_bus']])
            weights.append(row.loc[line].values)
            weights.append(row.loc[line].values)

        #normalize weights
        weights = np.asarray(weights).astype('float32')
        weights = self.scaler_weights.transform(weights)
        weights = tf.cast(
            weights, dtype=tf.dtypes.float32
        )

        for index, row in self.net['res_gen'].iterrows():
            gen[self.net['gen'].loc[index]['bus']] = row['p_mw']

        #normalize nodes info
        node_features_pre = np.asarray(list(nodes_info.values()))
        node_features_pre = self.scaler_nodes.transform(node_features_pre)
        nodes_info = tf.cast(
            np.asarray(node_features_pre), dtype=tf.dtypes.float32
        )
        #trasnpone edges
        edges = np.asarray(edges).T

        self.edges = edges
        self.nodes_info = nodes_info
        self.weights = weights
        self.gen = gen
        self.graph_info = [nodes_info, edges, weights, self.nodes_gen]
        return [edges, nodes_info, weights, gen]

    def load_scalers(self):
        scalar_x_filename = self.base_dir + self.network + '/scaler_nodes'
        self.scaler_nodes = joblib.load(scalar_x_filename)
        self.scaler_gen = GenScaler(self.gen_config)

    def get_state(self):
        """
        Devuelve el estado actual de la red
        :return:
        :rtype:
        """
        node_features_pre = self.nodes_info.numpy()
        node_features_pre = self.scaler_nodes.transform(node_features_pre)
        node_features_pre = tf.Variable(node_features_pre, dtype=float)
        node_features = tf.reshape(node_features_pre, [node_features_pre.shape[0] * node_features_pre.shape[1]]).numpy().tolist()
        return node_features

    def next_sample(self):
        """

        Returns:
        """
        if len(self.env_type) > 1:
            self.initialize_environment()
        else:
            self.num_sample += 1
            #deverias de dentro del mismo environment cambiar las cargas por ejemplo otras cargas para case 30

    def define_num_sample(self, num_sample):
        """

        Args:
            num_sample ():

        Returns:
        self.num_sample
        """
        self.num_sample = num_sample - 1

    def reset(self, change_sample):
        """

        Args:
            change_sample ():

        Returns:

        """
        if change_sample:
            self.next_sample()

        self.save_initial_values()
        self.update_pp_net()
        pf = self.run_powerflow()
        line_constrains_violated = self.compute_line_constraints(self.net)
        # si inicialmente no converge el problema se suma 1 slice a cada generador hasta que converga el power flow
        # while pf is False or line_constrains_violated:
            # self.aument_one_all_generatos()
            # self.update_pp_net()
            # pf = self.run_powerflow()
            # line_constrains_violated = self.compute_line_constraints(self.net)
        self.create_net_info()
        self.normalize_reward()
        # first reward
        self.reward_measure_prev = self.compute_reward_measure()


    def _load_gen_config(self):
        """

        """
        self.gen_config = {}
        self.gen_dic = {}
        self.nodes_gen = []
        for index, row in self.net['gen'].iterrows():
            self.gen_dic[index] = row['bus']
            self.nodes_gen.append(row['bus'])
            self.gen_config[row['bus']] = {'max': row['max_p_mw'], 'min': row['min_p_mw'], 'bus_id': row['bus']}

    def update_generators(self, gen_id):
        try:
            bus_id = self.gen_dic[gen_id]
            action_value = 1
            if self.generators_value_norm[bus_id] < self.generator_slices:
                self.generators_value_norm[bus_id] += action_value
                self.generators_value[bus_id] = self.generators_value_norm[bus_id]*(self.gen_config[bus_id]['max'] - self.gen_config[bus_id]['min'])/self.generator_slices + self.gen_config[bus_id]['min']
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False



    def aument_one_all_generatos(self):
        self.generators_value_norm = {key: value + 1 for key, value in self.generators_value_norm.items() if value < self.generator_slices }

        for bus_id in self.generators_value_norm.keys():
            self.generators_value[bus_id] = self.generators_value_norm[bus_id]/ self.generator_slices * \
                                                (self.gen_config[bus_id]['max'] - self.gen_config[bus_id]['min'])

    def update_pp_net(self):
        """

        :return:
        :rtype:
        """
        self.net.gen.drop(columns='p_mw', inplace=True)
        self.net.gen['p_mw'] = self.generators_value.values()


    def run_powerflow(self):
        """
        Calcula el power flow en la red self.net
        :return:
        :rtype:
        """
        try:
            pp.runpp(self.net)
            return True
        except pp.powerflow.LoadflowNotConverged:
            return False

    def run_optimalpowerflow(self, net):
        """
        Calcula el power flow en la red self.net
        :return:
        :rtype:
        """
        try:
            pp.runopp(net)
            return net
        except pp.optimal_powerflow.OPFNotConverged:
            return False

    def save_values_logs_powers(self):

        generation_gen_res = self.net.res_gen['p_mw'].sum()
        generation_ext = self.net.res_ext_grid['p_mw'].sum()
        consumption_load = self.net.res_load['p_mw'].sum()
        consumption_loss = self.net.res_line['pl_mw'].sum()
        self.generation.append(generation_gen_res + generation_ext)
        self.consumption.append(consumption_load + consumption_loss)
        self.generators_loads.append([round(self.generators_value[x] / self.gen_config[x]['max'], 2) * 100 for x in
                                      self.generators_value.keys()])

    def step(self, action, step_back=False):
        """

        Args:

        Returns:
        """

        generators_updated = self.update_generators(action)
        self.save_values_logs_powers()
        if generators_updated:
            self.update_pp_net()
            pf = self.run_powerflow()
            line_constrains_violated = self.compute_line_constraints(self.net)
            self.create_net_info()

            state = self.get_state()
            if pf is not False and line_constrains_violated is not True:
                reward = self._compute_reward()
            else:
                reward = -2
                self.converged = 0
                state = False
            return state, reward
        else:
            state = self.get_state()
            reward = -1

            return state, reward


    def step_back(self, action):
        state, reward = self.step(action, step_back=True)
        return state, reward


    """
    ****************************************************************************
                 PRIVATE FUNCTIONS OF THE ENVIRONMENT CLASS
    ****************************************************************************
    """


    def compute_reward_measure(self):
        """

        Args:

        Returns:
        Amount of power losses in the grid o power flow cost
        """

        if self.reward_magnitude == 'losses':
            reward = 0
            for index, row in self.net['res_line'].iterrows():
                reward = reward - row['pl_mw']
        else:
            cost = self._compute_solution_cost(self.net)
            self.cost.append(cost)
            reward = self.reward_scaler.transform(-cost)


        return reward

    def _compute_reward(self, current_reward_measure=None):
        if current_reward_measure is None:
            current_reward_measure = self.compute_reward_measure()

        if self.reward_computation == 'value':
            reward = current_reward_measure
        elif self.reward_computation == 'change':
            #after normalizacion cost we have positives cost rewards
            reward = current_reward_measure - self.reward_measure_prev

        self.reward_measure_prev = current_reward_measure

        return reward

    def _compute_solution_cost(self, net):
        cost = 0
        # external grid cost
        cost_row = net.poly_cost[net.poly_cost['et'] == 'ext_grid']
        cost = cost + cost_row['cp0_eur'].values[0] + net.res_ext_grid['p_mw'].values[0] * \
               cost_row['cp1_eur_per_mw'].values[0] + net.res_ext_grid['p_mw'].values[0] ** 2 * \
               cost_row['cp2_eur_per_mw2'].values[0]
        # itarete over res_gen rows and multiply by cost in poly_cost
        for index, row in net.res_gen.iterrows():
            cost_row = net.poly_cost[(net.poly_cost['et'] == 'gen') & (net.poly_cost['element'] == index)]
            cost = cost + cost_row['cp0_eur'].values[0] + cost_row['cp1_eur_per_mw'].values[0] * row['p_mw'] + \
                   cost_row['cp2_eur_per_mw2'].values[0] * row['p_mw'] ** 2

        return cost


# if __name__ == "__main__":
#     env = Environment()
#     env.step(0)
#     state = env.get_state()
#     pass


