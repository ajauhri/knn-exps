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
    ''' 
    root = cover_tree.create(X)
    #lsh.start(X, Q, options.r)
        #cover_tree.dfs(root)
    ct_timings = []
    sizes = []
    for i in range(10,800,20):
        start = time.time()
        _ = [cover_tree.knn(i, Q[0], root) for j in range(100)]
        end = time.time() - start
        ct_timings.append(end/100)
        sizes.append(i)
   '''
    res = knn_naive.knn_naive(500, X[0], X)
    print np.average(res)

if __name__ == "__main__":
    init()

