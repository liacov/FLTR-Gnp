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


def expand_influence(G, x, t, directed):
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
    # node incoming influence (starting from zero, at most n)
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


def run_simulation_parallel(params):

    # load probabilities p_i
    with open('data/out/keys{}.txt'.format(params.n), 'r') as f:
        prob = eval(f.read())
    # pick the probability of interest
    p = prob[params.p]
    del prob
    # load resistance values
    res = np.load('data/out/res_phase1.npy')
    # check the directed value
    if params.directed: lab = '_dir'
    else: lab = '_und'
    # load graphs G(N,p_i)
    G = np.load('data/graphs/graphs_{}_{}'.format(params.n, lab))
    # select the nodes of interest
    if params.do_sample: nodes = np.floor(n * np.random.rand(sample)).astype(int)  # pick randomly some nodes
    else: nodes = np.arange(N) # use all available nodes

    # info
    start_time = time.time()

    # run in parallel the expantion on a fixed value of p_i and save the outputs
    pool = Pool() # initialize the constructor
    # associate processes to args
    out = pd.DataFrame.from_records({'args' : list(product(range(K), nodes, res)) ,
                                     'output' : pool.starmap(expand_influence, product(G[i], nodes, res, params.directed))
                                    })
    # output converted in a dataframe
    raw_data = pd.DataFrame.from_records(out.apply(lambda x: [x.args[0],x.args[1],x.args[2],x.output[0],x.output[1]],axis=1),
                              columns= ['realization','node','resistance', 'metric','max_level'])
    del out
    raw_data.to_csv('data/out/data{}_{}_{}.csv'.format(lab, params.n, p))
    # statistics per node (double index: resistance and node)
    data_per_node = raw_data.groupby('resistance').apply(lambda x: x[['metric','max_level','node']].groupby('node').mean())
    data_per_node.to_csv('data/out/data_node{}_{}_{}.csv'.format(lab, params.n, p))
    del data_per_node
    # statistics per graph G(n,p,t) (single index: resistance)
    data_per_prob = raw_data.groupby('resistance').mean()[['metric', 'max_level']]
    data_per_prob.to_csv('data/out/data_graph{}_{}_{}.csv'.format(lab, params.n, p))
    del data_per_prob
    del raw_data
    # close the constructor
    pool.close()

    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("Size: {} \n Total uptime: {} \n".format(N, human_uptime))


if __name__ == "__main__":

    # define arguments
    parser = argparse.ArgumentParser()
    # graph size
    parser.add_argument('--n', type=int)
    # index of the probability to use
    parser.add_argument('--p', type=int)
    # directed or not
    parser.add_argument('--directed', type=bool)
    # do node sample or not
    parser.add_argument('--do_sample', type=bool, default=True)
    # node sample size
    parser.add_argument('--sample', type=int, default=5000)
    # number of samples for Gnp
    parser.add_argument('--k', type=int, default=500)
    '''
    # Not impremented in the FLTR
    # if weighted
    parser.add_argument('--weighted', type=bool, default=False)
    # weight interval
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=1)
    '''
    # parse arguments to dictionary
    args = parser.parse_args()

    run_simulation_parallel(args)
