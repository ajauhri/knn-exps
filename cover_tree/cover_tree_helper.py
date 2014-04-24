from __future__ import division
import numpy as np
import math
import scipy.sparse as ss

import const

const.base = 1.3
const.il2 = 1/math.log(const.base)

class tree_node:
    def __init__(self, point):
        self.point = point 
        self.max_scale = 0
        self.min_scale = 0
        self.children = {}

class ds_node:
    def __init__(self, node, dist):
        self.node = node 
        self.dist = dist 

def get_scale(dist):
    return int(math.ceil(const.il2 * math.log(dist))) 

def distance(p1, p2):
    diff = p1 - p2
    if ss.issparse(diff):
        return np.sqrt(diff.multiply(diff).sum(1))[0,0]
    else:
        return np.linalg.norm(diff)


# to get all children at a scale and returns a vector with all points at that scale 
def get_children(p, Qi_p_ds, scale):
    for Qi in Qi_p_ds:
        if scale in Qi.node.children:
            for node in Qi.node.children[scale]:
                Qi_p_ds.append(ds_node(node, distance(node.point, p)))

