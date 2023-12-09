from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
import networkx as nx
import numpy as np
import pickle, os, random, json

import utils

random.seed(13)
np.random.seed(13)

_continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU': 'Europe'
}

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(snapshots_dir, exist_ok=True)

with open(os.path.join(snapshots_dir, 'ln-graph-snapshot.gpickle'), 'rb') as f:
    G = pickle.load(f)
    print(f'nodes: {len(G.nodes)} edges: {len(G.edges)}')

for i in G.nodes:
    if 'locations' in G.nodes[i] and 'country_code_iso3' in G.nodes[i]['locations']:
      try:
        c = G.nodes[i]['locations']['country_code_iso3']
        c = country_alpha2_to_continent_code(country_name_to_country_alpha2(c))
        G.nodes[i]['locations']['continent_code'] = c
      except:
        del G.nodes[i]['locations']['country_code_iso3']
      
if os.path.exists(os.path.join(snapshots_dir, 'global_energy_mix.json')):
    with open(os.path.join(snapshots_dir, 'global_energy_mix.json'), 'r') as f:
      global_energy_mix = json.load(f)
    if not 'world_average' in global_energy_mix:
        world_average = []
        continents = []
        for k, v in global_energy_mix.items():
          try:
            c = country_alpha2_to_continent_code(country_name_to_country_alpha2(k))
            global_energy_mix[k]['continent'] = c
            continents.append(c)
          except:
            pass
          if 'carbon_intensity' in v:
              world_average.append(v['carbon_intensity'])

        continents = list(set(continents))
        continent_average = {}    
        for c in continents:
          g = []
          for k, v in global_energy_mix.items():
            if 'continent' in v and c == v['continent']:
              g.append(v['carbon_intensity'])
          continent_average[c] = np.mean(g)
          
        global_energy_mix['continent_average'] = continent_average
        global_energy_mix['world_average'] = np.mean(world_average)
    with open(os.path.join(snapshots_dir, 'global_energy_mix.json'), 'w') as f:
        json.dump(global_energy_mix, f)
  
T = utils.generate_tx(G, 10000)
for i in T:
    if nx.shortest_path_length(G, i[0], i[1]) <= 1:
        print('path error!')

with open(os.path.join(snapshots_dir, 'ln-graph-prepared.pickle'), 'wb') as f:
        pickle.dump({'directed_graph' : G,
                     'transactions' : T}, f)
        print('ln-graph-prepared.pickle saved.')