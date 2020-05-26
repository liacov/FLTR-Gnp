"""
Created on Wed Apr 29 2020

@author: Laura Iacovissi
"""

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from numba import jit
from itertools import product
from multiprocessing import Pool

@jit(nopython=True)
def FLTM(T, Q, exp_level, influence, state, total, nodes, n, G):
    # vectorized version of the influence expantion
    for i in range(n):
        neigh = np.array([False]*n)
        # dequeue
        v = Q[0]
        Q = Q[1:]
        # define neighborhood mask
        for j in list(np.nonzero(G[v,:])[0]):
            neigh[j] = True
        # update expantion levels
        exp_level[~state & neigh] = exp_level[v] + 1
        # update influence values
        influence[~state & neigh] += 1
        # define activation mask
        activated = ~state & neigh & (influence > T)
        # update state values
        state[activated] = True
        # update counter of activated nodes
        total += np.sum(activated)
        # enqueue the activated nodes
        act = nodes[activated]
        Q = np.concatenate((Q, act))
        if Q.size == 0:
            break
    return  total, max(exp_level), np.mean(exp_level)

def expand_influence(n_job, args):
    '''
    This function computes the FLTR metric for the x node in the G graph.

    INPUT
    G : numpy, adjacency matrix of a graph (n x n). i -> j iff A{i,j} = 1
    x : int, node of interest
    t : float, resistance values (constant on nodes)
    n : int, graph size

    OUTPUT
    total : int, FLTR(x)
    max(exp_level): int, maximum expantion level reached during the computation
    mean(exp_level): int, mean expantion level reached during the computation
    '''

    G, x, t, n, jobs = args

    # info
    print('Jobs {}/{}'.format(int(n_job), int(jobs)))
    # save nodes in an numpy array
    nodes = np.arange(n)
    # convert the percentage in a number
    T = t * (n - 1)
    # compute the activation set for the node of interest
    X = list(np.nonzero(G[x,:])[0]) + [x]
    # initialize counter for the active nodes
    total = len(X)
    # list (queue) of active nodes
    Q = np.array(sorted(X))
    # node states (active = True, not active = False)
    state = np.array([v in X for v in nodes])
    # node incoming influence (starting from zero, at most n)
    influence = np.array([0] * n)
    # node expantion level (starting from 0 if in X, else -1. worst case: n)
    exp_level = np.array([-int(not v in X) for v in nodes])

    return FLTM(T, Q, exp_level, influence, state, total, nodes, n, G)


def run_simulation_parallel(params):

    # reading probabilities
    with open('data/keys{}.txt'.format(args.n), 'r') as f:
        # original probs
        prob = eval(f.read())
    with open('data/keys_ref.txt', 'r') as f:
        # refinement probs
        prob2 = eval(f.read())
    prob.extend(prob2)
    prob.sort(reverse=True)
    # pick the probability of interest
    p = prob[params.p]
    del prob
    # load resistance values
    res = np.load('data/res_phase4.npy')
    # check the directed value
    if params.d:
        lab = 'dir'
    else:
        lab = 'und'
    # load graphs G(N,p_i) from adjacency matrices
    matrices = np.load('data/graphs/graph_{}_{}_{}.npy'.format(params.n, p, lab))
    matrices = [ matrices[i, :, :] for i in range(params.k) ] # convert the data in an iterable
    # select the nodes of interest
    if params.do_sample:
        nodes = np.floor(params.n * np.random.rand(params.sample)).astype(int)  # pick randomly some nodes
    else:
        nodes = np.arange(params.n) # use all available nodes

    # info
    start_time = time.time()

    # run in parallel the expantion on a fixed value of p_i and save the outputs
    pool = Pool() # initialize the constructor
    # compute number of jobs
    n_jobs = params.k * len(nodes) * len(res)
    # associate processes to args
    out = pd.DataFrame.from_records({
                                    'args' : list(product(range(params.k), nodes, res)) ,
                                    'output' : pool.starmap(expand_influence,
                                                            enumerate(product(matrices, nodes, res, [params.n], [n_jobs])))
                                    })
    # output converted in a dataframe
    raw_data = pd.DataFrame.from_records(out.apply(lambda x: [x.args[0],x.args[1],x.args[2],x.output[0],x.output[1],x.output[2]],axis=1),
                              columns= ['realization', 'node', 'resistance', 'metric', 'max_level', 'avg_level'])
    del out
    raw_data.to_csv('data/maxpred/data_{}_{}_{}.csv'.format(lab, params.n, p))
    # statistics per node (double index: resistance and node)
    data_per_node = raw_data.groupby('resistance').apply(lambda x: x[['metric', 'max_level', 'avg_level', 'node']].groupby('node').mean())
    data_per_node.to_csv('data/maxpred/data_node_{}_{}_{}.csv'.format(lab, params.n, p))
    del data_per_node
    # statistics per graph G(n,p,t) (single index: resistance)
    data_per_prob = raw_data.groupby('resistance').mean()[['metric', 'max_level', 'avg_level']]
    data_per_prob.to_csv('data/maxpred/data_graph_{}_{}_{}.csv'.format(lab, params.n, p))
    del data_per_prob
    del raw_data
    # close the constructor
    pool.close()

    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print()
    print("Size: {} \n Total uptime: {} \n".format(params.n, human_uptime))


if __name__ == "__main__":

    # define arguments
    parser = argparse.ArgumentParser()
    # graph size
    parser.add_argument('--n', type=int)
    # index of the probability to use
    parser.add_argument('--p', type=int)
    # directed or not
    parser.add_argument('--dir', dest='d', action='store_true')
    parser.add_argument('--und', dest='d', action='store_false')
    parser.set_defaults(d=True)
    # do node sample or not
    parser.add_argument('--yes_sample', dest='do_sample', action='store_true')
    parser.add_argument('--no_sample', dest='do_sample', action='store_false')
    parser.set_defaults(do_sample=False)
    # node sample size
    parser.add_argument('--sample', type=int, default=5000)
    # number of samples for Gnp
    parser.add_argument('--k', type=int, default=50)
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
