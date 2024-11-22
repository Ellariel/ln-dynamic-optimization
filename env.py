import numpy as np
import random, itertools
from gymnasium import Env, spaces

from proto import *
from utils import *

class LNEnv(Env): 
    def __init__(self, G, transactions, proto_type='LND', observation_size=15, observation_radius=5, global_energy_mix=None, train=True) -> None:
        self.features = ['geodist', 'sum_ghg', 'delay', 'feeratio', 'intercontinental_hops', 'intercountry_hops']
        self.transactions = transactions
        self.global_energy_mix = global_energy_mix
        self.proto_type = proto_type
        self.train = train
        self.g = G
        self.observation_radius = observation_radius
        self.observation_size = observation_size
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.observation_size, len(self.features) * 2), dtype=np.float32)
        self.action_space = spaces.Box(low=-2, high=2, shape=(1, ), dtype=np.float32)
        self.observation_cache = {}     
        self.reward = []
        
    def get_observation(self):
        
        def get_params(id):
            subgraph = nx.ego_graph(self.g, id, radius=self.observation_radius)
            neighbors = list(itertools.islice(subgraph.edges(), self.observation_size))
            params = []
            for i in range(self.observation_size):
                if i < len(neighbors):
                    p = get_path_params(self.g, neighbors[i], self.amount, self.global_energy_mix)
                    params.append([p[f] for f in self.features])
                else:
                    params.append([0] * len(self.features))
            return np.asarray(params, dtype=np.float32)

        if (self.u, self.v) not in self.observation_cache:
            self.observation_cache[(self.u, self.v)] = np.hstack((get_params(self.u), get_params(self.v)))      

        return self.observation_cache[(self.u, self.v)]
        
    def reset(self, seed=None):
        tx = random.choice(self.transactions)
        self.u, self.v, self.amount = tx[0], tx[1], tx[2]   
        self.current_observation = self.get_observation()
        return self.current_observation, {}
            
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = 0

        path = get_shortest_path(self.g, self.u, self.v, self.amount, 
                                 global_energy_mix=self.global_energy_mix,
                                 proto_type=self.proto_type, _e=action)

        if self.train:
            reward += self.compute_reward(path) 
            self.reward.append(reward)          
        
        return self.current_observation, reward, True, False, {}   
  
    def compute_reward(self, path):
        
        params = get_path_params(self.g, path, self.amount, self.global_energy_mix)
        params['succeed'] = perform_payment(self.g, self.u, self.v, self.amount, path)
        reward = int(params['succeed']) / (params['sum_ghg'] / params['dist'] / 1000)
        
        return reward
        
    def get_reward(self):
        return self.reward
    
    def render(self, mode='console'): 
        if mode != 'console':
          raise NotImplementedError()
        pass

    def close(self):
        pass