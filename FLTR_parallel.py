"""
Created on Mon Mar 16 2020

@author: Laura Iacovissi

The following code generate Erdos-Renyi random graphs G(V,E) from a list of
n = |V| values and a list of probabilities prob of having an edge (i,j) varying
in a given set. To each edge in E a random weight in [a,b) is associated.
Then the FLTR metric is computed on all the nodes (if do_sample = F) or on a
random sample of nodes (if do_sample = T, size of the sample = sample).
The results (full output and statistics) are saved in the results folder as files.

For the same couple of paramenters n,p a both a directed and an undirected G
are generated (with different samplings).

A level of verbosity for the algorithm can be selected changing the variable
'verbose'.
"""

import numpy as np
import pandas as pd
import networkx as nx
import time
import datetime

from multiprocessing import Pool

# weight interval
b = 1
a = 0
# wheter or not assign weights
weighted = False
# resistance
res = [ 1, 0.75, 0.5, 0.4, 0.3, 0.2 ]
# number of nodes (list)
# N = list(range(10**3 * 2, 10**5, 10**3 * 5))
N = [1000]
# compute FLTR on a sample or on all nodes
do_sample = False
# number of nodes to sample
sample = 1000
# verbosity of the program : {0,1,2}
verbose = 0

def generate_graphs(n, prob, directed):
    '''
    This function generates Erdos-Renyi weighted random graphs/digraphs of the
    same size (# nodes) with different probabilities to have an edge among two
    vertices.

    INPUT
    n : int, # nodes
    prob : list of floats, different probability values
    a, b : float, extremes of the interval from which the weights are sampled
    directed: bool, whether the graph is directed or not

    OUTPUT
    G : networkx graph
    '''
    G = []

    # pick the faster algorithm
    for p in prob:
        if p > np.floor(np.log10(n))*10**(-4):
             G.append(nx.gnp_random_graph(n, p, directed = directed))
        else:
             G.append(nx.fast_gnp_random_graph(n, p, directed = directed))
        # assign random weights
        if weighted:
            if directed:
                for e in list(G[-1].edges) + list(G[-1].in_edges):
                    G[-1][e[0]][e[1]]['weight'] = (b - a) * np.random.random_sample() + a
            if not directed:
                for e in list(G[-1].edges):
                    G[-1][e[0]][e[1]]['weight'] = (b - a) * np.random.random_sample() + a
    return G


def expand_influence(G, x, n, t, directed, verbose = 0):
    '''
    This function computes the FLTR metric for the x node in the G graph.

    INPUT
    G : networkx Graph/DiGraph, G = (V,E)
    x : int, node of interest
    n : int, n = |V|
    t : float, resistance values (constant on nodes)
    verbose: {0,1,2}, level of verbosity of the function

    OUTPUT
    total : int, FLTR(x)
    max(exp_level): int, maximum expantion level reached during the computation
    '''

    start_time_exp = time.time()

    if not weighted:
        # convert the percentage in a number
        T = t * n
    # compute the activation set for the node of interest
    if directed: X = [y for y in G.predecessors(x)] + [x]
    else: X = [y for y in G.neighbors(x)] + [x]
    # initialize counter for the active nodes
    total = len(X)
    if verbose == 2: print("Starting nodes :", total)
    # list (queue) of active nodes
    Q = sorted(X)
    # list of the node states
    state = [v in X for v in G.nodes]
    # add activation as a label
    # nx.set_node_attributes(G, state, 'activation') # changing state the labels will change
    # list of the node influence
    influence = [0] * n
    # add current level of influence
    # nx.set_node_attributes(G, influence, 'influence')  # changing influence the labels will change
    # list of the node expantion level
    exp_level = [-int(not v in X) for v in G.nodes]
    # add current level of expantion level
    # nx.set_node_attributes(G, exp_level, 'level')  # changing influence the labels will change

    while Q != []:
        # dequeue
        v = Q.pop(0)
        if directed: neigh = G.predecessors(v)
        else: neigh = G.neighbors(v)
        for u in G.neighbors(v):
            # pick the inactive neighbors
            if not state[u]:
                # update the influence value
                if weighted:
                    influence[u] += G[v][u]['weight']
                else: influence[u] += 1
                if influence[u] > T:
                    # the node is activated
                    state[u] = True
                    # update the counter
                    total += 1
                    # enqueue
                    Q.append(u)
                # update expantion level
                exp_level[u] = exp_level[v] + 1

    end_time_exp = time.time()
    uptime_exp = end_time_exp - start_time_exp
    human_uptime_exp = datetime.timedelta(seconds=uptime_exp)
    if verbose == 2:
        print("Uptime: {}, Active nodes: {}, Max expantion level: {}".format(
        human_uptime_exp, total, max(exp_level)))
    return  total, max(exp_level)


def saver(stats, data, directed, n):

    # check the directed value
    if directed: lab = ''
    else: lab = '_und'

    for key, val in stats.items():
        # sigle dataframe stores in stats_{key}
        val.to_csv("results/stats{}_{}_{}.csv".format(lab, str(n), str(key)), index = False)

    # keys : probabilities
    with open("results/keys{}.txt".format(str(n)), "w") as f:
        #saving keys to file
        f.write(str(list(stats.keys())))

    for key, val in data.items():
        # sigle dataframe stores in data_{key}
        val[0].to_csv("results/data{}_{}_{}_metrics.csv".format(lab, str(n), str(key)), index = False)
        val[1].to_csv("results/data{}_{}_{}_levels.csv".format(lab, str(n), str(key)), index = False)


def run_simulation(n, prob, directed):
    # info
    start_time = time.time()
    # list of networkx graphs
    G = generate_graphs(n, prob, directed)
    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("G({},p)'s generation  uptime: {} \n".format(n, human_uptime))

    # info
    start_time = time.time()

    # dict: { prob : [(avg FLTR , avg expantion level) for t in res] }
    stats = {}
    # dict: { prob : [FLTRs dataframe, levels dataframe] }
    # dataframes : columns = resistance, rows = nodes (no correspondance with index)}
    data = {}

    for i, graph in enumerate(G):

        # info
        if verbose == 1:
            print("Graph p:", prob[i])
            # check the giant component size
            if not directed:
                print('connected components: \n', [len(c) for c in sorted(
                nx.connected_components(graph), key=len, reverse=True)])
        # node selection
        if do_sample:
            # pick randomly some nodes
            nodes = np.floor(n * np.random.rand(sample)).astype(int)
        else: nodes = graph.nodes
        # initialize the dict entry as DataFrame
        df = pd.DataFrame(columns = ['res','avg_FLTR','avg_exp_level'])

        metrics = {}
        levels = {}
        for t in res:
            # info
            if verbose == 2:
                print("#### Graph #{} with p = {} ant threshold = {} ####".format(
                i, prob[i], t), end = "\n\n")
            # lists of partial result - one per (graph, t)
            metrics[t] = []
            levels[t] = []

            for x in nodes:
                # info
                if verbose == 2: print("Node", x)
                # compute expantion
                FLTR, level = expand_influence(graph, x, n, t, directed)
                # save partial results
                metrics[t].append(FLTR)
                levels[t].append(level)
                # info
                if verbose == 2: print('FLTR: {}, level: {}'.format(FLTR, level))

            # save results for (graph, t)
            stat_dict = {'res': t, 'avg_FLTR': np.mean(metrics[t]),
             'avg_exp_level': np.mean(levels[t])}
            # info
            if verbose == 2:
                print(stat_dict)
            df = df.append(stat_dict, ignore_index = True)
            # info
            if verbose == 2: print("", end = "\n\n\n")

        # info
        if verbose == 1: print(df, end = "\n\n")
        # save results in the dicts
        stats[prob[i]] = df
        data[prob[i]] = [pd.DataFrame(metrics), pd.DataFrame(levels)]
        del df
        del metrics
        del levels
    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("Size: {} \n Total uptime: {} \n".format(n, human_uptime))

    # save results on a csv file
    saver(stats, data, directed, n)


def run_in_parallel():
    args = []
    for n in N:
        # Define probability associated to the size
        prob = [ 5e-1, 1e-2, 4e-3, 2e-3, 1/999, 1/(2*n), 1/(10*n) ]
        # update list of args
        args.append((n, prob, True)) # directed
        args.append((n, prob, False)) # undirected

    # Define the parallel processes
    # 'processes' is the number of worker processes to use. If processes is None
    # then the number returned by os.cpu_count() is used.
    pool = Pool()
    # Associate processes to args
    pool.starmap(run_simulation, args)

    # save the explored values of n
    np.save('results/sizes.npy', N)


if __name__ == "__main__":
    run_in_parallel()
