#! /usr/bin/env python

from optparse import OptionParser
import numpy as np
import time

from naive import knn_naive
from extras.parsers import netflix, generic
from extras.helper import debug
import lsh.lsh as lsh
import cover_tree.cover_tree as cover_tree

def init():
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="tfile", help="training file")
    parser.add_option("-q", "--query", dest="qfile", help="query file")
    parser.add_option("-r", "--radius", dest="r", help="radius for LSH", type="float", default=0.6)
    parser.add_option("-i", "--ntrains", dest="i", help="#of trainings in training file to be included", type="int")
    parser.add_option("-j", "--nquerys", dest="j", help="#of queries in query file to be included", type="int")
    parser.add_option("-n", "--netflix", dest="netflix", help="run with netflix data", action='store_true')
    parser.add_option("-s", "--stackex", dest="stackex", help="run with stackexchange data", action='store_true')
    parser.add_option("-g", "--generic", dest="generic", help="run with generic", action='store_true')
    (options, args) = parser.parse_args()

    # netflix block #
    if options.netflix:
        debug('running with netflix...')
        D = netflix(options.tfile)
        print D.shape
        X = D[:options.i]
        if not options.qfile:
            Q = D[:options.j]
        else:
            Q = netflix(options.qfile)[:options.j]

    # stackoverflow block #
    elif options.stackex:
        debug('running with stackoverflow...')

    # generic block #
    elif options.generic:
        debug('running with generic dataset...')
        D = generic(options.tfile)
        X = D[:options.i]
        if not options.qfile:
            Q = D[:options.j]
        else:
            Q = generic(options.qfile)[:options.j]
    else:
        debug('data format not specified')
    
    #root = cover_tree.create(X)
    #nn = lsh.start(X, Q, options.r)
    #cover_tree.dfs(root)
    
    ct_timings = []
    lsh_timings = []
    sizes = []
    n_nghs = 5
    for s in range(500, options.i, 1000):
        root = cover_tree.create(X[:s])
        ct_tot_t = 0
        lsh_tot_t = 0
        for q in Q:
            debug('querying cover tree')
            start = time.time()
            for j in range(100):
                nghs = cover_tree.knn(n_nghs, q, root)
            ct_tot_t += (time.time() - start) / 100
            root = None
            
            # use the distance from cover tree to initialize lsh
            nn = lsh.start(X[:s], Q, nghs[-1].dist)
            
            debug('querying LSH')
            start = time.time()
            for j in range(100):
                nghs = lsh.get_ngh_struct(nn, q)
            lsh_tot_t += (time.time() - start) / 100
            nghs = None
            nn = None

        ct_timings.append(ct_tot_t/Q.shape[0])
        lsh_timings.append(lsh_tot_t/Q.shape[0])
        sizes.append(s)

if __name__ == "__main__":
    init()

