#! /usr/bin/env python3.4

from optparse import OptionParser
import numpy as np
import time
from guppy import hpy

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
    parser.add_option("-g", "--generic", dest="generic", help="run with generic", action='store_true')
    (options, args) = parser.parse_args()
    
    t = 0
    # netflix block #
    if options.netflix:
        debug('running with netflix...')
        t = 0
        D = netflix(options.tfile)
        debug(D.shape)
        X = D[:options.i]
        if not options.qfile:
            Q = D[:options.j]
        else:
            Q = netflix(options.qfile)[:options.j]

    # generic block #
    elif options.generic:
        t = 1
        debug('running with generic dataset...')
        D = generic(options.tfile)
        X = D[:options.i]
        if not options.qfile:
            Q = D[:options.j]
        else:
            Q = generic(options.qfile)[:options.j]
    else:
        debug('data format not specified')
    
    root = cover_tree.create(X)
    debug('querying cover tree')
    for i in range(Q.shape[0]):
        nghs = cover_tree.knn(5, Q[i], root)
        debug("NNs for query %d returned %d neighbours \n" % (i, len(nghs)))
        for n in nghs:
            debug("NN at dist=%f\n" % (n.dist))
    
    #avg = np.average(knn_naive.knn_naive(10, Q[0], X))
    #print avg

    lsh.seed() 
    nn = lsh.start(X, options.r, t)
    debug('querying lsh')
    for i in range(Q.shape[0]):
        nghs = lsh.get_ngh_struct(nn, Q[i])
        debug("NNs for query %d returned %d neighbours \n" % (i, len(nghs)))
        for n in nghs:
            debug("NN at dist=%f\n" % n[1])

    #cover_tree.dfs(root)
    #collect_timings(X, Q)
    #do_profiling(X)

def do_profiling(X):
    n_nghs = 5
    debug(X.shape)
    lsh.seed()
    out = open('dense_mem.txt', 'w')
    ct_size = 0
    lsh_size = 0
    sizes = [1000, 5000, 10000, 20000, 40000, 60000]
    h = hpy()
    for s in sizes:
        h.setrelheap()
        root = cover_tree.create(X[:s])
        ct_size = h.heap().indisize
        
        debug('querying cover tree')
        nghs = cover_tree.knn(n_nghs, X[0], root)
        
        # use the distance from cover tree to initialize lsh
        h.setrelheap()
        nn = lsh.start(X[:s], nghs[-1].dist)
        lsh_size = h.heap().indisize
        out.write("%f, %f, %f\n" % (ct_size, lsh_size, s))
        del nghs 
        del root
        del nn
        out.flush()
    out.close()
    

def collect_timings(X, Q):
    ct_timings = []
    lsh_timings = []
    n_nghs = 5
    n_q_repeats = 10
    lsh.seed()
    out = open('dense.txt', 'w')
    sizes = [1000, 5000, 10000, 20000, 40000, 60000]
    for s in sizes:
        root = cover_tree.create(X[:s])
        ct_tot_t = 0
        lsh_tot_t = 0
        for q in Q:
            debug('querying cover tree')
            start = time.time()
            for j in range(n_q_repeats):
                nghs = cover_tree.knn(n_nghs, q, root)
            ct_tot_t += (time.time() - start) / n_q_repeats
            
            # use the distance from cover tree to initialize lsh
            nn = lsh.start(X[:s], nghs[-1].dist)
            
            debug('querying LSH')
            start = time.time()
            for j in range(n_q_repeats):
                nghs = lsh.get_ngh_struct(nn, q)
            lsh_tot_t += (time.time() - start) / n_q_repeats
            del nghs
            del nn
        out.write("%f,%f,%f\n" % (ct_tot_t/Q.shape[0], lsh_tot_t/Q.shape[0], s))
        ct_timings.append(ct_tot_t/Q.shape[0])
        lsh_timings.append(lsh_tot_t/Q.shape[0])
        out.flush()
    out.close()

if __name__ == "__main__":
    init()

