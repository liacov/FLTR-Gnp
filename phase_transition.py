"""
Created on May 5 2020

@author: Laura Iacovissi
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_p0(prob, directed, n, res, t):
    # check the directed value
    if directed:
        lab = 'dir'
    else : lab = 'und'
    # check the t type
    if t:
        thr = 'maxpred'
    else : thr = 'pred'

    fig, axs = plt.subplots(len(res), 1, figsize = (8,12))
    for i, t in enumerate(res):
        data = {}
        for p in prob:
            # load data
            temp = pd.read_csv('data/{}/data_{}_{}_{}.csv'.format(thr, lab, n, p), index_col=0)
            # on each realization check for which nodes the property is realized
            temp2 = temp[temp.resistance == t].groupby('realization').apply(lambda x: x.max_level < 1)
            data[p] = (pd.DataFrame(temp2).groupby('realization').sum() > 0).mean()
            del temp, temp2
        # plot data
        axs[i].set_title('Phase transition for t = {}'.format(t), fontsize=15)
        _ = axs[i].plot(list(data.keys()), list(data.values()), 'o-')
        _ = axs[i].set_xlim(xmax=0.2)

        # delete from memory
        del data
    #plt.suptitle('Phase transition for maxlevel < 1, n = {}, {}'.format(n, lab), y=1.01, fontsize=20)
    plt.tight_layout()
    plt.savefig('images/p0_{}_{}_{}.jpeg'.format(thr, lab, n), bbox_inches='tight')



def plot_p1(prob, directed, n, res, t):
    # check the directed value
    if directed:
        lab = 'dir'
    else : lab = 'und'
    # check the t type
    if t:
        thr = 'maxpred'
    else : thr = 'pred'

    fig, axs = plt.subplots(len(res), 1, figsize = (8,12))
    for i, t in enumerate(res):
        data = {}
        for p in prob:
            # load data
            temp = pd.read_csv('data/{}/data_{}_{}_{}.csv'.format(thr, lab, n, p), index_col=0)
            # on each realization check for which nodes the property is realized
            temp2 = temp[temp.resistance == t].groupby('realization').apply(lambda x: (x.max_level <= 1) & (x.avg_level < 0))
            data[p] = (pd.DataFrame(temp2).groupby('realization').sum() > 0).mean()
            del temp, temp2
        # plot data
        axs[i].set_title('Phase transition for t = {}'.format(t), fontsize=15)
        _ = axs[i].plot(list(data.keys()), list(data.values()), 'o-')
        _ = axs[i].set_xlim(xmax=0.2)

        # delete from memory
        del data
    #plt.suptitle('Phase transition for maxlevel < 1, n = {}, {}'.format(n, lab), y=1.01, fontsize=20)
    plt.tight_layout()
    plt.savefig('images/p1_{}_{}_{}.jpeg'.format(thr, lab, n), bbox_inches='tight')


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
    with open('data/keys_ref{}.txt'.format(args.n), 'r') as f:
        prob2 = eval(f.read())
    prob.extend(prob2)
    prob.sort()
    print('Used probabilities:', prob)
    # reading resistances
    res = np.load('data/res_phase1.npy')

    # show data on nodes per resistance value, for each gnp (mean on sample)
    plot_p0(prob, args.d, args.n, res, args.t)
    # show data on graph per resistance values, for each gnp (mean on sample and nodes)
    plot_p1(prob, args.d, args.n, res, args.t)


if __name__ == "__main__":
    main()
