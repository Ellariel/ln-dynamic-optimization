import random
import networkx as nx
import numpy as np
from gymnasium import Env, spaces
from operator import itemgetter

from proto import *
from utils import *

class LNEnv(Env): 
    def __init__(self, G, transactions, observation_size=5, global_energy_mix=None, train=True) -> None:
        self.features = ['geodist', 'sum_ghg', 'delay', 'feeratio', 'intercontinental_hops', 'intercountry_hops']
        self.transactions = transactions
        self.global_energy_mix = global_energy_mix
        self.train = train
        self.g = G
        self.observation_size = observation_size
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.observation_size, len(self.features) * 2), dtype=np.float32)
        self.action_space = spaces.Box(low=-2, high=2, shape=(1, ), dtype=np.float32)       
        self.reward = []
        
    def get_observation(self):
        
        def get_params(id):
            neighbors = list(self.g.neighbors(id))
            params = []
            for i in range(self.observation_size):
                if i < len(neighbors):
                    p = get_path_params(self.g, [id, neighbors[i]], self.amount, self.global_energy_mix)
                    params.append([p[f] for f in self.features])
                else:
                    params.append([0] * len(self.features))
            return np.asarray(params, dtype=np.float32)
        #print(self.u)
        #print(get_params(self.u))
        #print(self.v)
        #print(get_params(self.v))
        return np.hstack((get_params(self.u), get_params(self.v)))
        
    def reset(self, seed=None):
        tx = random.choice(self.transactions)
        self.u, self.v, self.amount = tx[0], tx[1], tx[2]   
        self.current_observation = self.get_observation()
        #print(self.current_observation)
        return self.current_observation, {}
            
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = 0

        path = get_shortest_path(self.g, self.u, self.v, self.amount, 
                                 global_energy_mix=self.global_energy_mix,
                                 proto_type='LND', _e=action)

        if self.train:
            reward += self._compute_reward(path) 
            self.reward.append(reward)          
        
        return self.current_observation, reward, True, False, {}   
  
    def _compute_reward(self, path):
        reward = 0
        
        params = get_path_params(self.g, path, self.amount, self.global_energy_mix)
        reward = -params['sum_ghg']
        
        return reward
        
    def get_reward(self):
        return self.reward
    
    def render(self, mode='console'): 
        if mode != 'console':
          raise NotImplementedError()
        pass

    def close(self):
        pass