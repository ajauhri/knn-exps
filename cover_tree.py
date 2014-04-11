#! /usr/bin/env python

import numpy as np
import sys

class tree_node:
    def __init__(self):
        self.point = None
        self.max_scale = 0
        self.min_scale = 0

class ds_node:
    def __init__(self):
        self.tree_node = None
        self.dist = 0

def create_cover_tree(fname):
    X = process_netflix(fname)
    for row in X:
        print row

if __name__ == "__main__":
    create_tree(sys.argv[1])

        
