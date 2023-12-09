import os, time, pickle, argparse, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import utils, proto

parser = argparse.ArgumentParser()
parser.add_argument('--approach', default='PPO', type=str)
parser.add_argument('--n_envs', default=4, type=int)
parser.add_argument('--env', default='env', type=str)

#parser.add_argument('--subgraph', default=50, type=int)
#parser.add_argument('--idx', default=0, type=int)
#parser.add_argument('--subset', default='randomized', type=str)

parser.add_argument('--attempts', default=10, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--timesteps', default=1e4, type=int)

args = parser.parse_args()

#idx = args.idx
n_envs = args.n_envs
timesteps = args.timesteps
approach = args.approach
epochs = args.epochs
attempts = args.attempts
#subgraph = args.subgraph
#subset = args.subset

if args.env == 'env':
    version = 'env'
    from env import LNEnv  

def max_neighbors(G):
    def neighbors_count(G, id):
        return len(list(G.neighbors(id)))
    max_neighbors = 0
    for id in G.nodes:
      max_neighbors = max(max_neighbors, neighbors_count(G, id))
    return max_neighbors

base_dir = './'
train_limit = 10000

snapshots_dir = os.path.join(base_dir, 'snapshots')
weights_dir = os.path.join(base_dir, 'weights')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

with open(os.path.join(snapshots_dir, 'ln-graph-prepared.pickle'), 'rb') as f:
    f = pickle.load(f)
    G = f['directed_graph']
    print(f'nodes: {len(G.nodes)} edges: {len(G.edges)}')
    T = f['transactions'][:train_limit]
    print(f'transactions: {len(T)}')
    
with open(os.path.join(snapshots_dir, 'global_energy_mix.json'), 'r') as f:
    global_energy_mix = json.load(f)

train_size = int(len(T) * 0.7)
test_size = int(len(T) * 0.3)
train_set = T[:train_size]
test_set = T[train_size:train_size+test_size]
#valid_set = T[train_size+test_size:]

print(f'graph, n: {len(G.nodes)}, e: {len(G.edges)}, max neighbors: {max_neighbors(G)}')
print(f'transations count: {len(T)}, train_set: {len(train_set)}, test_set: {len(test_set)}')#, valid_set: {len(valid_set)}')
file_mask = f'{approach}-{version}-{n_envs}'

learning_rate = 0.000001

#e = LNEnv(G, train_set, global_energy_mix=global_energy_mix)
#check_env(e)

for a in range(attempts):
    print(f"approach: {approach}, env: {version}, n_envs: {n_envs}")
    print(f"train: {file_mask}")

    E = make_vec_env(lambda: LNEnv(G, train_set, global_energy_mix=global_energy_mix), n_envs=n_envs)

    lf = os.path.join(results_dir, f'{file_mask}.log')
    log = pd.read_csv(lf, sep=';') if os.path.exists(lf) else None
    f = os.path.join(weights_dir, f'{file_mask}.sav')

    if approach == 'PPO':
        model_class = PPO
    else:
        model_class = None
        print(f'{approach} - not implemented!')
        raise ValueError

    if os.path.exists(f) and model_class:
        try:
            model = model_class.load(f, E, force_reset=False, verbose=0, learning_rate=learning_rate)
            print(f"model is loaded {approach}: {f}")
        except:
            model = model_class.load(f+'.tmp', E, force_reset=False, verbose=0, learning_rate=learning_rate)
            print(f"model is loaded {approach}: {f+'.tmp'}")
    else:
        print(f'did not find {approach}: {f}')
        model = model_class("MlpPolicy", E, verbose=0, learning_rate=learning_rate) 

    for epoch in range(1, epochs + 1):
        model.learn(total_timesteps=timesteps, progress_bar=True)

        reward = E.env_method('get_reward')
        mean_reward = np.mean(reward, axis=1)
        max_mean_reward = np.max(mean_reward)
        
        print(f'n_envs: {n_envs}, epoch: {epoch}/{epochs}, attempt: {a}/{attempts}')        
        print(f"max mean reward: {max_mean_reward:.3f}~{mean_reward}")
        if epoch % 2:
            model.save(f)
        else:
            model.save(f+'.tmp')

        log = pd.concat([log, pd.DataFrame.from_dict({'time' : time.time(),
                                                'approach' : approach,
                                                'version' : version,
                                                'n_envs': n_envs,
                                                'max_mean_reward' : max_mean_reward,
                                                'mean_reward' : mean_reward,
                                                'epoch' : epoch,
                                                'epochs' : epochs,
                                                'attempt' : a,
                                                'total_timesteps' : timesteps,
                                                'filename' : f,
#                                                'n': len(G.nodes),
#                                                'e': len(G.edges),
#                                                'max_neighbors':max_neighbors(G),
#                                                'min_pathlen': min(test_total_pathlen + train_total_pathlen),
#                                                'max_pathlen': max(test_total_pathlen + train_total_pathlen),
#                                                'avg_pathlen': np.mean(test_total_pathlen + train_total_pathlen),
                                                }, orient='index').T], ignore_index=True)
        log.to_csv(lf, sep=';', index=False)