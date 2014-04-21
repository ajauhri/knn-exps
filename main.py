#! /usr/bin/env python

from optparse import OptionParser

from naive import knn_naive
from extras.parsers import netflix, generic
from extras.helper import debug
import lsh.lsh as lsh
def init():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="ifile", help="input file")
    parser.add_option("-t", "--train", dest="tfile", help="train file")
    parser.add_option("-n", "--netflix", dest="netflix", help="run with netflix data", action='store_true')
    parser.add_option("-s", "--stackex", dest="stackex", help="run with stackexchange data", action='store_true')
    parser.add_option("-g", "--generic", dest="generic", help="run with generic", action='store_true')
    (options, args) = parser.parse_args()

    # netflix block #
    if options.netflix:
        debug('running with netflix...')
        X = netflix(options.ifile)[:200,:]
        debug('loaded input')
        #root = create_cover_tree(X[:200,:])
        #T = input_parsers.netflix(options.tfile)
        #for row in T:
        #    knn(1, ...
        #dfs(root)
        #knn_naive.knn_naive(2, X[201,:], X[:100,:])
        #debug('done with naive')
        lsh.compute_opt(X, X[:10,:])
        debug('done with lsh')


    # stackoverflow block #
    elif options.stackex:
        debug('running with stackoverflow...')

    # generic block #
    elif options.generic:
        debug('running with generic dataset...')
        X = generic(options.ifile)
        lsh.compute_opt(X[:100,:], X[:40])
        #knn_naive.knn_naive(500, X[1], X)
        debug('input loaded')

    else:
        debug('data format not specified')

if __name__ == "__main__":
    init()

