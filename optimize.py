import numpy as np
import pickle, json, random, os
import ray
# see https://github.com/bayesian-optimization/BayesianOptimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import UtilityFunction

import utils, proto

native_alg = ['LND', 'CLN', 'ECL']
modified_alg = ['H(LND)', 'H(CLN)', 'H(ECL)']
metrics = ['dist', 'geodist', 'sum_ghg', 'delay', 'feeratio',# 'feerate',
           'intercontinental_hops', 'intercountry_hops', 
           'avg_geodist', 'avg_ghg']
train_limit = 10000

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(snapshots_dir, exist_ok=True)

for alg in native_alg:
    os.makedirs(os.path.join(results_dir, alg), exist_ok=True)
    
with open(os.path.join(snapshots_dir, 'ln-graph-prepared.pickle'), 'rb') as f:
    f = pickle.load(f)
    G = f['directed_graph']
    print(f'nodes: {len(G.nodes)} edges: {len(G.edges)}')
    T = f['transactions'][:train_limit]
    print(f'transactions: {len(T)}')
    
with open(os.path.join(snapshots_dir, 'global_energy_mix.json'), 'r') as f:
    global_energy_mix = json.load(f)
    
ray.init()

@ray.remote
def get_alg_results(G, T, alg, e, global_energy_mix, save=False):
    _results = []
    if alg in native_alg:
        f = os.path.join(os.path.join(results_dir, alg), f'{alg}_results.pickle')
    else:
        f = os.path.join(os.path.join(results_dir, alg[2:-1]), f'{alg[2:-1]}_{e}_results.pickle')
    if os.path.exists(f):
        with open(f, 'rb') as f:
            _results = pickle.load(f) 
    else:
        for t in T:
            r = None
            path = proto.get_shortest_path(G, t[0], t[1], t[2], proto_type=alg, 
                                                global_energy_mix=global_energy_mix, _e=e)
            if path:
                r = utils.get_path_params(G, path, t[2], global_energy_mix=global_energy_mix)
            _results.append((t, r)) 
        if save:
            with open(f, 'wb') as f:
                pickle.dump(_results, f)
    return _results[:train_limit]

def get_comparison(G, T, comparison, e, global_energy_mix, opt_metrics):
        a_results, b_results = ray.get([get_alg_results.remote(G, T, comparison[0], e, global_energy_mix),
                                        get_alg_results.remote(G, T, comparison[1], e, global_energy_mix)])
    
        complete = []
        for t in range(len(T)):
            complete.append(bool(a_results[t][1]) and bool(b_results[t][1]))         
        
        metric_results = {m : {} for m in metrics}
        for m in metrics:
            metric_results[m][comparison[0]] = [a_results[i][1][m] for i, t in enumerate(complete) if t and m in a_results[i][1]]
            metric_results[m][comparison[1]] = [b_results[i][1][m] for i, t in enumerate(complete) if t and m in b_results[i][1]]
            
        metric_results['avg_geodist'] = {}
        metric_results['avg_ghg'] = {}
        for a in comparison:
            metric_results['avg_geodist'][a] = np.array(metric_results['geodist'][a]) / np.array(metric_results['dist'][a])
            metric_results['avg_ghg'][a] = np.array(metric_results['sum_ghg'][a]) / np.array(metric_results['dist'][a])
            
        diff = []
        for m in opt_metrics:
            a = np.mean(metric_results[m][comparison[0]])
            b = np.mean(metric_results[m][comparison[1]])
            d = (b - a) * 100 / a
            print(f"{m}, ε={'+' if d > 0 else ''}{d:.1f}%")
            d = -d if d <= 0 else 1/d
            if m == 'avg_ghg' and d > 1:
                d = 1.3 * d # gives the carbon intensity higher importance
            diff.append(d)
        
        return np.prod(diff)

opt_metrics = ['dist', 'avg_ghg', 'intercountry_hops', 'intercontinental_hops']
opt_params = {  ('CLN', 'H(CLN)') : # 0.25556
                    {'optimization_budget' : 30,
                     'bounds' : {'e' : (-1.0, 1.0)},
                     'kappa' : 7.5,
                     'kind' : 'ei',
                     'xi' : 1e-4,
                     'init' : 0.27, 
                    },
                ('ECL', 'H(ECL)') : # 0.35784
                    {'optimization_budget' : 30,
                     'bounds' : {'e' : (-1.0, 1.0)},
                     'kappa' : 2.5,
                     'kind' : 'ei',
                     'xi' : 1e-4, 
                    },
                ('LND', 'H(LND)') : # 0.27876
                    {'optimization_budget' : 30,
                     'bounds' : {'e' : (-1.0, 1.0)}, # Bounded region of parameter space
                     'kappa' : 2.5, # kappa - the balance between exploration and exploitation
                     'kind' : 'ei', # Expected Improvement method
                     'xi' : 1e-4, 
                    },
            }

random.seed(48)
np.random.seed(48)

optimal = []
if G and T:
    for c, v in opt_params.items():
        print(c)  
        acq_function = UtilityFunction(kind = v['kind'], 
                                       kappa = v['kappa'],
        ) 
        optimizer = BayesianOptimization(f = None,
                                        pbounds = v['bounds'],
                                        random_state = 48,
                                        allow_duplicate_points=True,
        )
        logger = JSONLogger(path=os.path.join(results_dir, f"{c[0]}_optlog.json"))
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        next_point = {}
        for i in range(v['optimization_budget']):
            if i == 0 and 'init' in v:
                next_point.update({'e': v['init']})
            else:
                next_point = optimizer.suggest(acq_function)
            target = get_comparison(G, T, c, next_point['e'], global_energy_mix, opt_metrics)
            optimizer.register(params=next_point, target=target)
            print(f"Iteration {i+1}, a point to probe is: {next_point['e']}, corresponded score is: {target}\n")
            
        o = optimizer.max
        optimal.append((c[1], o))
        print(f'Best point for {c[1]}:', o)
        
    print()
    for o in optimal:
        print(f"{o[0]}: {o[1]}")
    
ray.shutdown()  
