"""
Created on Tue Apr 14 2020

@author: Laura Iacovissi
"""

import time
import datetime
import argparse
import numpy as np
import networkx as nx
from multiprocessing import Pool


def generate_graphs(n, p, directed, weighted, a, b):
    '''
    This function generates Erdos-Renyi weighted random graphs/digraphs returning
    as output the adjacency matrix in numpy format.

    INPUT
    n : int, # nodes
    p : float, probability value
    directed: bool, whether the graph is directed or not

    OUTPUT
    G : networkx graph (adj_matrix)
    '''

    # pick the faster algorithm
    if p > np.floor(np.log10(n))*10**(-4):
         G = nx.gnp_random_graph(n, p, directed = directed)
    else:
         G = nx.fast_gnp_random_graph(n, p, directed = directed)
    # assign random weights
    if weighted:
        if directed:
            for e in list(G[-1].edges) + list(G.in_edges):
                G[e[0]][e[1]]['weight'] = (b - a) * np.random.random_sample() + a
        if not directed:
            for e in list(G.edges):
                G[e[0]][e[1]]['weight'] = (b - a) * np.random.random_sample() + a
    return nx.to_numpy_matrix(G)

def main():
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
    # append or not
    parser.add_argument('--app', dest='append', action='store_true')
    parser.add_argument('--no_app', dest='append', action='store_false')
    parser.set_defaults(append=False)
    # numberof samples
    parser.add_argument('--k', type=int, default=50)
    # if weighted
    parser.add_argument('--weighted', type=bool, default=False)
    # weight interval
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=1)
    # parse arguments to dictionary
    args = parser.parse_args()

    print(args)

    # load probabilities p_i
    with open('data/out/keys{}.txt'.format(args.n), 'r') as f:
        prob = eval(f.read())
    # pick the probability of interest
    p = prob[args.p]
    del prob

    # info
    start_time = time.time()
    pool = Pool() # initialize the constructor
    # generate K samples for the (N,p) couple
    graphs = pool.starmap(generate_graphs, [ (args.n, p, args.d, args.weighted, args.a, args.b) ] * args.k)
    # close constructor
    pool.close()
    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("\n\nG({},{})'s samples generation uptime: {}".format(args.n, p, human_uptime))
    print("Directed:", args.d)
    print()

    # check the directed value
    if args.d: lab = 'dir'
    else: lab = 'und'


    # save list of list of graphs
    path = 'data/graphs/graph_{}_{}_{}.npy'.format(args.n, p, lab)
    if args.append:
        if os.path.isfile(path):
            old_g = np.load(path)
            old_g = np.append(old_g, graphs)
            np.save(path, old_g)
        else:
            np.save(path, g)
    else:
        np.save(path.format(args.n, p, lab), graphs)


if __name__ == "__main__":
    main()
