"""
Created on Tue Apr 14 2020

@author: Laura Iacovissi
"""

import time
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
from multiprocessing import Pool

# resistances
res = [ 0.25, 0.5, 0.75, 1 ] # phase 1 values
# number of nodes (list)
N = 1000
# directed or not
directed = True
# compute FLTR on a sample or on all nodes
do_sample = False
# number of nodes to sample
sample = 5000
# number of graph samples
K = 500


def expand_influence(G, x, t):
    '''
    This function computes the FLTR metric for the x node in the G graph.

    INPUT
    G : networkx Graph/DiGraph, G = (V,E)
    x : int, node of interest
    t : float, resistance values (constant on nodes)

    OUTPUT
    total : int, FLTR(x)
    max(exp_level): int, maximum expantion level reached during the computation
    '''

    # save nodes in an numpy array
    nodes = np.array(G.nodes)
    # convert the percentage in a number
    T = t * N
    # compute the activation set for the node of interest
    if directed: X = [y for y in G.predecessors(x)] + [x]
    else: X = [y for y in G.neighbors(x)] + [x]
    # initialize counter for the active nodes
    total = len(X)
    # list (queue) of active nodes
    Q = sorted(X)
    # node states (active = True, not active = False)
    state = np.array([v in X for v in nodes])
    # node influence (starting from zero, at most n)
    influence = np.array([0] * N)
    # node expantion level (starting from 0 if in X, else -1. worst case: n)
    exp_level = np.array([-int(not v in X) for v in nodes])

    # vectorized version of the influence expantion
    while Q != []:
        # dequeue
        v = Q.pop(0)
        # define neighborhood mask
        if directed: neigh = np.isin(nodes, list(G.predecessors(v)))
        else: neigh = np.isin(nodes, list(G.neighbors(v)))
        # update expantion levels
        exp_level[~state & neigh] = exp_level[v] + 1
        # update influence values
        influence[~state & neigh] += 1
        # define activation mask
        activated = ~state & neigh & (influence > T)
        # update state values
        state[activated] = True
        # update counter of activated nodes
        total += sum(activated)
        # enqueue the activated nodes
        Q.extend(nodes[activated])

    return  total, max(exp_level)


def run_simulation_parallel():
    # info
    start_time = time.time()

    # check the directed value
    if directed: lab = '_dir'
    else: lab = '_und'
    # load graphs G(N,p_i)
    G = np.load('data/graphs/graphs_{}_{}'.format(N, lab))
    # load probabilities p_i
    with open('data/out/keys{}.txt'.format(N), 'r') as f:
        prob = eval(f.read())
    # select the nodes of interest
    if do_sample: nodes = np.floor(n * np.random.rand(sample)).astype(int)  # pick randomly some nodes
    else: nodes = np.arange(N) # use all available nodes

    # define containers for data, dict {prob : data}
    out = {}
    raw_data = {}
    data_per_node = {}
    data_per_prob = {}
    # run in parallel the expantion on a fixed value of p_i and save the outputs
    pool = Pool() # initialize the constructor
    for i, p in enumerate(prob):
        # associate processes to args
        out[p] = pd.DataFrame.from_records({'args' : list(product(range(K), nodes, res)) ,
                                            'output' : pool.starmap(expand_influence, product(G[i], nodes, res))
                                            })
        # output converted in a dataframe
        raw_data[p] = pd.DataFrame.from_records(out[p].apply(lambda x: [x.args[0],x.args[1],x.args[2],x.output[0],x.output[1]],axis=1),
                                  columns= ['realization','node','resistance', 'metric','max_level'])
        raw_data[p].to_csv('data/out/data{}_{}_{}'.format(lab, N, p))
        # statistics per node (double index: resistance and node)
        data_per_node[p] = pippo[p].groupby('resistance').apply(lambda x: x[['metric','max_level','node']].groupby('node').mean())
        data_per_node[p].to_csv('data/out/data_node{}_{}_{}'.format(lab, N, p))
        # statistics per graph G(n,p,t) (single index: resistance)
        data_per_prob[p] = pippo[p].groupby('resistance').mean()[['metric', 'max_level']]
        data_per_prob[p].to_csv('data/out/data_graph{}_{}_{}'.format(lab, N, p))
    pool.close() # close the constructor

    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("Size: {} \n Total uptime: {} \n".format(N, human_uptime))


if __name__ == "__main__":
    run_simulation_parallel()
