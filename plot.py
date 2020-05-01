"""
Created on Tue Apr 20 2020

@author: Laura Iacovissi
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data_nodes(prob, directed, n, res, t):
    # check the directed value
    if directed:
        lab = 'dir'
    else : lab = 'und'
    # check the t type
    if t:
        thr = 'maxpred'
    else : thr = 'pred'

    fig, axs = plt.subplots(len(prob), 3, figsize = (20,40))
    for i, p in enumerate(prob):
        # load data
        data = pd.read_csv('data/{}/data_node_{}_{}_{}.csv'.format(thr, lab, n, p))
        # plot data
        axs[i,0].set_title('Metrics with p = {}'.format(p), fontsize=15)
        _ = axs[i,0].boxplot([data.loc[data.resistance == k, 'metric'] for k in res], positions = [1,2,3,4], labels=res)
        axs[i,1].set_title('Max levels with p = {}'.format(p), fontsize=15)
        _ = axs[i,1].boxplot([data.loc[data.resistance == k, 'max_level'] for k in res], positions = [1,2,3,4], labels=res)
        axs[i,2].set_title('Avg levels with p = {}'.format(p), fontsize=15)
        _ = axs[i,2].boxplot([data.loc[data.resistance == k, 'avg_level'] for k in res], positions = [1,2,3,4], labels=res)
        # delete from memory
        del data

    plt.suptitle('Data visualization for n = {}, {}'.format(n, lab), y=1.01, fontsize=20)
    plt.tight_layout()
    plt.savefig('images/stats_per_node_{}_{}_{}.jpeg'.format(thr, lab, n))


def plot_data_graphs(prob, directed, n, t):
    # check the directed value
    if directed:
        lab = 'dir'
    else: lab = 'und'
    # check the t type
    if t:
        thr = 'maxpred'
    else: thr = 'pred'

    fig, axs = plt.subplots(len(prob), 3, figsize = (20,40))
    for i, p in enumerate(prob):
        # load data
        data = pd.read_csv('data/{}/data_graph_{}_{}_{}.csv'.format(thr, lab, n, p)).set_index('resistance')
        # plot data
        axs[i,0].set_title('Metrics with p = {}'.format(p), fontsize=15)
        data['metric'].transpose().plot(kind='line',ax=axs[i,0])
        axs[i,1].set_title('Max levels with p = {}'.format(p), fontsize=15)
        data['max_level'].transpose().plot(kind='line',ax=axs[i,1])
        axs[i,2].set_title('Avg levels with p = {}'.format(p), fontsize=15)
        data['avg_level'].transpose().plot(kind='line',ax=axs[i,2])
        # delete from memory
        del data

    plt.suptitle('Data visualization for n = {}, {}'.format(n, lab), y=1.01, fontsize=20)
    plt.tight_layout()
    plt.savefig('images/stats_per_gnp_{}_{}_{}.jpeg'.format(thr, lab, n))


def main():
    # define arguments
    parser = argparse.ArgumentParser()
    # graph size
    parser.add_argument('--n', type=int)
    # directed or not
    parser.add_argument('--dir', dest='d', action='store_true')
    parser.add_argument('--und', dest='d', action='store_false')
    parser.set_defaults(d=True)
    # type of t
    parser.add_argument('--maxpred', dest='t', action='store_true')
    parser.add_argument('--pred', dest='t', action='store_false')
    parser.set_defaults(t=True)
    # parse arguments to dictionary
    args = parser.parse_args()

    # reading probabilities
    with open('data/keys{}.txt'.format(args.n), 'r') as f:
        prob = eval(f.read())
    print('Used probabilities:', prob)
    # reading resistances
    res = np.load('data/res_phase1.npy')

    # show data on nodes per resistance value, for each gnp (mean on sample)
    plot_data_nodes(prob, args.d, args.n, res, args.t)
    # show data on graph per resistance values, for each gnp (mean on sample and nodes)
    plot_data_graphs(prob, args.d, args.n, args.t)


if __name__ == "__main__":
    main()
