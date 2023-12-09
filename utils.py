
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from geopy.distance import geodesic

def not_na(x):
    return pd.notna(x)

def drop_none(x):
    return [i for i in x if not_na(i)]

def get_node_location(G, id):
    if 'locations' in G.nodes[id]:
      return G.nodes[id]['locations']

def get_coords(G, id):
     c = get_node_location(G, id)
     if c and 'latitude' in c:
        return (c['latitude'], c['longitude']) 

def get_continent(G, id):
     c = get_node_location(G, id)
     if c and 'continent_code' in c:
        return c['continent_code'] 

def get_country(G, id):
     c = get_node_location(G, id)
     if c and 'country_code_iso3' in c:
        return c['country_code_iso3'] 
    
def get_continents(G, path):
    loc = [get_continent(G, p) for p in path]
    loc = drop_none(loc)
    return loc

def get_countries(G, path):
    loc = [get_country(G, p) for p in path]
    loc = drop_none(loc)
    return loc

def get_geodist(G, path):
        loc = [get_coords(G, p) for p in path]
        loc = drop_none(loc)
        dist = []
        for c in range(len(loc)-1):
            dist.append(geodesic(loc[c], loc[c+1]).km)
        return np.sum(dist)

def get_continent_hops(G, path):
        hops = 0
        loc = get_continents(G, path)
        for c in range(len(loc)-1):
            if loc[c] != loc[c+1]:
                  hops += 1
        return hops

def get_country_hops(G, path):
        hops = 0
        loc = get_countries(G, path)
        for c in range(len(loc)-1):
            if loc[c] != loc[c+1]:
                hops += 1
        return hops

# Carbon intensity of electricity (gCO2/kWh)
# https://github.com/mlco2/codecarbon/blob/master/codecarbon/data/private_infra/global_energy_mix.json
def get_carbon_intensity(G, id, global_energy_mix):
    country, continent = get_country(G, id), get_continent(G, id)
    ci = global_energy_mix[country]['carbon_intensity'] if country in global_energy_mix and 'carbon_intensity' in global_energy_mix[country] else False
    ci = global_energy_mix['continent_average'][continent] if not ci and continent and 'continent_average' in global_energy_mix and continent in global_energy_mix['continent_average'] else ci
    ci = global_energy_mix['world_average'] if not ci and 'world_average' in global_energy_mix else ci
    ci = 436 if not ci else ci
    return ci

def get_total_carbon_intensity(G, path, global_energy_mix):
        ci = 0
        for p in path:
            ci += get_carbon_intensity(G, p, global_energy_mix)
        return ci

def get_carbon_costs(G, u, v, global_energy_mix, e=0.5):
    u = get_carbon_intensity(G, u, global_energy_mix)
    v = get_carbon_intensity(G, v, global_energy_mix)
    h = e*(u + v)/2 + (1 - e)*(v - u)
    return h

def get_path_params(G, path, amount, global_energy_mix=None):
    a = amount
    p = path
    delay = 0     
    for i in range(len(p) - 1):
        if G.has_edge(p[i], p[i + 1]):
            e = G.edges[p[i], p[i + 1]]
            a += a * e['fee_rate_sat'] + e['fee_base_sat']
            delay += e['delay']
    return {'path' : p,
            'dist' : len(p),
            'geodist' : get_geodist(G, p),
            'sum_ghg' : get_total_carbon_intensity(G, p, global_energy_mix),
            'delay' : delay,
            'feeratio' : a / amount,
            'feerate' : a / amount - 1,
            'amount' : a,
            'intercontinental_hops' : get_continent_hops(G, p),
            'intercountry_hops' : get_country_hops(G, p),
            'continents': get_continents(G, p),
            'countries': get_countries(G, p),
            }

def generate_tx(G, transacitons_count=1000, seed=2, centralized=False):
    log_space = np.logspace(0, 7, 10**6)
    ##
    def random_amount(): # SAT
        # Возвращает массив значений от 10^0 = 1 до 10^7, равномерно распределенных на логарифмической шкале
        # https://coingate.com/blog/post/lightning-network-bitcoin-stats-progress
        # The highest transaction processed is 0.03967739 BTC, while the lowest is 0.000001 BTC. The average payment size is 0.00508484 BTC;
        # highest: 3967739.0 SAT
        # average: 508484.0 SAT
        # lowest: 100.0 SAT
        return log_space[random.randrange(0, 10**6)] + 100
    ##
    def shortest_path_len(u, v):
        path_len = 0
        try:
              path_len = nx.shortest_path_length(G, u, v)
        except:
              pass
        return path_len
    ##
    random.seed(seed)
    tx_set = []
    nodes = list(G.nodes)
    max_path_length = 0
    if not centralized:
      for _ in tqdm(range(1, transacitons_count + 1)):
            while True:
              u = nodes[random.randrange(0, len(nodes))]
              v = nodes[random.randrange(0, len(nodes))]
              p = shortest_path_len(u, v)
              max_path_length = max(max_path_length, p)
              if v != u and p >= 2 and (u, v) not in tx_set:
                break
            tx_set.append((u, v))
    else:
      u = nodes[random.randrange(0, len(nodes))]
      for v in tqdm(nodes):
          p = shortest_path_len(u, v)
          max_path_length = max(max_path_length, p)
          if v != u and p >= 2 and (u, v) not in tx_set:
            tx_set.append((u, v))
    tx_set = [(tx[0], tx[1], random_amount()) for tx in tx_set]
    print(f'max_path_length: {max_path_length}')
    return tx_set