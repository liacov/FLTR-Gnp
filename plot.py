"""
Created on Tue Apr 20 2020

@author: Laura Iacovissi
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data_nodes(prob, directed, n, res):
    # check the directed value
    if directed:
        lab = 'dir'
    else : lab = 'und'

    fig, axs = plt.subplots(len(prob), 2, figsize = (20,40))
    for i, p in enumerate(prob):
        # load data
        data = pd.read_csv('data/out/data_node_{}_{}_{}.csv'.format(lab, n, p))
        # plot data
        axs[i,0].set_title('Metrics with p = {}'.format(p), fontsize=15)
        _ = axs[i,0].boxplot([data.loc[data.resistance == k, 'metric'] for k in res], positions = [1,2,3,4], labels=res)
        axs[i,1].set_title('Levels with p = {}'.format(p), fontsize=15)
        _ = axs[i,1].boxplot([data.loc[data.resistance == k, 'max_level'] for k in res], positions = [1,2,3,4], labels=res)
        # delete from memory
        del data

    plt.suptitle('Data visualization for n = {}, {}'.format(n, lab), y=1.01, fontsize=20)
    plt.tight_layout()
    plt.savefig('images/stats_per_node_{}_{}.jpeg'.format(lab, n))


def plot_data_graphs(prob, directed, n):
    # check the directed value
    if directed:
        lab = 'dir'
    else : lab = 'und'

    fig, axs = plt.subplots(len(prob), 2, figsize = (20,40))
    for i, p in enumerate(prob):
        # load data
        data = pd.read_csv('data/out/data_graph_{}_{}_{}.csv'.format(lab, n, p))
        # plot data
        axs[i,0].set_title('Metrics with p = {}'.format(p), fontsize=15)
        data['metric'].transpose().plot(kind='line',ax=axs[i,0])
        axs[i,1].set_title('Levels with p = {}'.format(p), fontsize=15)
        data['max_level'].transpose().plot(kind='line',ax=axs[i,1])
        # delete from memory
        del data

    plt.suptitle('Data visualization for n = {}, {}'.format(n, lab), y=1.01, fontsize=20)
    plt.tight_layout()
    plt.savefig('images/stats_per_gnp_{}_{}.jpeg'.format(lab, n))


def main():
    # define arguments
    parser = argparse.ArgumentParser()
    # graph size
    parser.add_argument('--n', type=int)
    # directed or not
    parser.add_argument('--dir', dest='d', action='store_true')
    parser.add_argument('--und', dest='d', action='store_false')
    parser.set_defaults(d=True)
    # probability interval
    parser.add_argument('--from_p', type=int, default=0)
    parser.add_argument('--to_p', type=int, default=9)
    # parse arguments to dictionary
    args = parser.parse_args()

    # reading probabilities
    with open('data/out/keys{}.txt'.format(args.n), 'r') as f:
        prob = eval(f.read())
    # filtering probabilities
    prob = prob[args.from_p : args.to_p + 1]
    print('Used probabilities:', prob)
    # reading resistances
    res = np.load('data/out/res_phase1.npy')

    # compute thresholds for phase transitions
    gc_threshold = 1/args.n # giant component arising
    print('GC critical point: ', gc_threshold)
    conn_threshold = np.log(args.n)/args.n # connected regime arising
    print('Connection critical point: ', conn_threshold)
    # show data on nodes per resistance value, for each gnp (mean on sample)
    plot_data_nodes(prob, args.d, args.n, res)
    # show data on graph per resistance values, for each gnp (mean on sample and nodes)
    plot_data_graphs(prob, args.d, args.n)


if __name__ == "__main__":
    main()
