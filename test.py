import os, json, pickle, argparse, glob
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC

from proto import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--approach', default='PPO', type=str)
parser.add_argument('--n_envs', default=4, type=int)
parser.add_argument('--env', default='env', type=str)
args = parser.parse_args()

n_envs = args.n_envs
approach = args.approach

if args.env == 'env':
    version = 'env'
    from env import LNEnv

base_dir = './'
train_limit = 10000

opt_e = {
    'H(LND)' : 0.27876,
    'H(CLN)' : 0.25556,
    'H(ECL)' : 0.35784,
}

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

print(f'graph, n: {len(G.nodes)}, e: {len(G.edges)}, max neighbors: {utils.max_neighbors(G)}')
print(f'transations count: {len(T)}, train_set: {len(train_set)}, test_set: {len(test_set)}')#, valid_set: {len(valid_set)}')

E = LNEnv(G, test_set, global_energy_mix=global_energy_mix, train=False)

def load_model(proto_type='LND'):
    model = None
    if proto_type[0] == 'A':
        proto_type = proto_type[2:5]
        file_mask = f'{approach}-{version}-{n_envs}-{proto_type}'
        weights_file = os.path.join(weights_dir, f'{file_mask}.sav')
        weights_file_list = glob.glob(weights_file + '-*')
        if len(weights_file_list):
            weights_file_list = sorted(weights_file_list, key=lambda x: float(''.join([i for i in x.split('-')[-1] if i.isdigit() or i == '.'])))
            if os.path.exists(weights_file_list[-1]):
                weights_file = weights_file_list[-1]
        print(weights_file)
        if approach == 'PPO':
                model_class = PPO
        else:
                model_class = None
                print(f'{approach} - not implemented!')
                raise ValueError
        if os.path.exists(weights_file) and model_class:
                model = model_class.load(weights_file, E, force_reset=False)
                print(f'model is loaded {approach}: {weights_file}')
        else:
                print(f'did not find {approach}: {weights_file}')
                model = model_class("MlpPolicy", E)
    return model
        
def get_tx_params(tx, proto_type='LND'):
    if proto_type[0] == 'A':
        proto_type = 'H' + proto_type[1:]
        E.transactions = [tx]
        obs, _ = E.reset()
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, (list, np.ndarray)):
            action = action[0]
    elif proto_type in opt_e:
        action = opt_e[proto_type]
    else:
        action = 0.5
    path = get_shortest_path(G, tx[0], tx[1], tx[2], 
                                 global_energy_mix=global_energy_mix,
                                 proto_type=proto_type, _e=action)    
    params = get_path_params(G, path, tx[2], global_energy_mix)
    return params

alg = ['LND', 'H(LND)', 'A(LND)',
       'CLN', 'H(CLN)', #'A(CLN)',
       'ECL', 'H(ECL)', #'A(ECL)',
       ]

for a in tqdm(alg):
    file_name = os.path.join(results_dir, f'{a}.json')
    if not os.path.exists(file_name):
        model = load_model(a)
        results = []
        random.seed(13)
        np.random.seed(13)
        for tx in tqdm(test_set, leave=True, desc=a):
            results.append(get_tx_params(tx, proto_type=a))
        with open(file_name, 'w') as f:
            json.dump(results, f)