import time
import datetime
import numpy as np
import networkx as nx
from multiprocessing import Pool

# weight interval
b = 1
a = 0
# wheter or not to assign weights
weighted = False
# directed or not
directed = False
# number of nodes (list)
N = 10**3
# number of sample to generate for each G
K = 500

def generate_graphs(n, p, directed):
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
    # define probabilities of interest
    prob = [ 8e-1, 7e-1, 6e-1, 5e-1, 1e-2, 4e-3, 2e-3, 1/999, 1/(2*N), 1/(10*N) ]
    # initialize the graphs container (array)
    G = []
    # info
    start_time = time.time()
    # run in parallel the graph generator for each value of p_i
    pool = Pool() # initialize the constructor
    for i, p in enumerate(prob):
        # generate K samples for the (N,p) couple
        G.append(pool.starmap(generate_graphs, [ (N, p, directed) ] * K))
    # close constructor
    pool.close()
    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("\n\nG({},p)'s samples generation uptime: {}".format(N, human_uptime))
    print("Probabilities:\n", prob)
    print("Directed:", directed)
    print()

    # check the directed value
    if directed: lab = 'dir'
    else: lab = 'und'

    # save list of list of graphs
    np.save("data/graphs/graph_{}_{}".format(N, lab), G)
    # keys : probabilities
    with open("data/out/keys{}.txt".format(str(N)), "w") as f:
        #saving keys to file
        f.write(str(list(prob)))

if __name__ == "__main__":
    main()
