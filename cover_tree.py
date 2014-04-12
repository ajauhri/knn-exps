#! /usr/bin/env python

# py libraries
from __future__ import  division
import sys
import math
from operator import attrgetter
import random
import numpy as np
import scipy.sparse as ss

# user-defined libs
import parsers
import cover_tree_helper as helper
from cover_tree_helper import debug
import const

def insert(p, root, max_scale):
    scale = max_scale
    Q_p_ds = [helper.ds_node(root, helper.distance(root.point, p))]
    
    while True:
        helper.get_children(p, Q_p_ds, scale)
        min_Q_p_ds = min(Q_p_ds, key=attrgetter('dist'))
        
        if min_Q_p_ds.dist == 0:
            return None
        elif min_Q_p_ds.dist > math.pow(const.base, scale):
            break;
        else:
            scale_dist = math.pow(const.base, scale)
            if min_Q_p_ds.dist <= scale_dist:
                while True:
                    pos = random.randint(0, len(Q_p_ds) - 1)
                    if Q_p_ds[pos].dist <= scale_dist:
                        parent = Q_p_ds[pos].node
                        parent_scale = scale
                        break
            for i in reversed(range(0, len(Q_p_ds) - 1)):
                if Q_p_ds[i].dist > scale_dist:
                    del Q_p_ds[i]
            
            scale -= 1
    
    new_child = helper.tree_node(p)
    try:
        if new_child not in parent.children[parent_scale]:
            parent.children[parent_scale].append(new_child)
    except KeyError:
            parent.children[parent_scale] = [new_child]

    root.min_scale = min(root.min_scale, parent_scale - 1)
                

def create_cover_tree(X):
    dist = ss.csr_matrix(X[1:,:])
    # make copies of the first vector s.t. sparse matrix substraction. Need to find a better way...
    stacked_x = ss.vstack([X[0] for n in range(dist.get_shape()[0])])
    dist = dist - stacked_x
    dist = np.sqrt(dist.multiply(dist).sum(1))

    # make the first element as root
    root = helper.tree_node(X[0])
    root.max_scale = helper.get_scale(dist.max())
    root.min_scale = root.max_scale
    debug('insertion started')
    for i in xrange(1, X.shape[0]):
        insert(X[i], root, root.max_scale)
    debug('insertion done')

def init(fname):
    X = parsers.netflix(fname)[:1000,:]
    debug('loaded input')
    create_cover_tree(X)
    max_dist = 0

if __name__ == "__main__":
    init(sys.argv[1])

        
