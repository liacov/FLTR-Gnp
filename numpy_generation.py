"""
Created on Tue Apr 20 2020

@author: Laura Iacovissi
"""

import time
import datetime
import argparse
import numpy as np


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

    # info
    start_time = time.time()
    # graph path
    path = 'data/graphs/graph_{}_{}_{}.npy'.format(args.n, p, lab)
    if args.d:
        # not symmetric random adj matrix with zero diagonal
        np.save(path, np.fill_diagonal(np.random.binomial(1, args.p, (args.k, args.n, args.p)), 0))
    else:
        # not symmetric random adj matrix
        A = np.random.binomial(1, args.p, (args.k, args.n, args.p), 0)
        # make symmetric and with zero diagonal
        for i in range(A.shape[0]):
            # triangularize
            A[i,:,:] = np.tril(A[i,:,:])
            # make symm
            A[i,:,:] = A[i,:,:] + np.transpose(A[i,:,:])
        np.save(path, A)
    # info
    end_time = time.time()
    uptime = end_time - start_time
    human_uptime = datetime.timedelta(seconds=uptime)
    print("\n\nG({},{})'s samples generation uptime: {}".format(args.n, p, human_uptime))
    print("Directed:", args.d)
    print()

if __name__ == "__main__":
    main()
