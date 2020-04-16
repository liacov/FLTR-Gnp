import time
import datetime
import argparse
import numpy as np
import networkx as nx
from multiprocessing import Pool


def generate_graphs(n, p, directed, weighted, a, b):
    '''
    This function generates Erdos-Renyi weighted random graphs/digraphs of the
    same size (# nodes) with different probabilities to have an edge among two
    vertices.

    INPUT
    n : int, # nodes
    p : float, probability value
    directed: bool, whether the graph is directed or not

    OUTPUT
    G : list of networkx graphs
    '''

    G = []

    # pick the faster algorithm
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

def main():
    # define arguments
    parser = argparse.ArgumentParser()
    # graph size
    parser.add_argument('--n', type=int)
    # index of the probability to use
    parser.add_argument('--p', type=int)
    # directed or not
    parser.add_argument('--directed', type=bool)
    # numberof samples
    parser.add_argument('--k', type=int, default=500)
    # if weighted
    parser.add_argument('--weighted', type=bool, default=False)
    # weight interval
    parser.add_argument('--a', type=int, default=0)
    parser.add_argument('--b', type=int, default=1)
    # parse arguments to dictionary
    args = parser.parse_args()

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
    G = pool.starmap(generate_graphs, [ (args.n, p, args.directed, args.weighted, args.a, args.b) ] * args.k)
    # close constructor
    pool.close()
    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("\n\nG({},{})'s samples generation uptime: {}".format(args.n, p, human_uptime))
    print("Directed:", args.directed)
    print()

    # check the directed value
    if args.directed: lab = 'dir'
    else: lab = 'und'

    # save list of list of graphs
    np.save("data/graphs/graph_{}_{}_{}".format(args.n, p, lab), G)


if __name__ == "__main__":
    main()
