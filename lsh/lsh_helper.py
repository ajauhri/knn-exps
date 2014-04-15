from __future__ import division
import math
import numpy as np

import lsh_structs
import const

const.max_hash_rnd = 536870912

def compute_product_mod_default_prime(a, b, size):
    h = 0
    h = a.multiply(b).sum(1)
    h = (h & const.two_to_32_minus_1) + 5 * (h >> 32)
    if h >= const.prime_default:
        h = h - const.prime_default
    assert (h < const.prime_default)
    return h

'''
probability of collision of 2 points in a LSH function
'''
def compute_p(w, c):
    x = w / c
    return 1 - math.erfc(x / np.sqrt(2)) - 2 * np.sqrt(1/np.pi) / np.sqrt(2) / x * (1 - np.exp(-(x**2) / 2))

def compute_m(k, p, w):
    assert((k & 1) == 0) # k should be even in order to use ULSH
    mu = 1 - math.pow(compute_p(w, 1), k / 2)
    d = (1-mu) / (1-p) * 1 / math.log(1/mu) * math.pow(mu, -1/(1-mu))
    y = math.log(d)
    m = math.ceil(1 - y/math.log(mu) - 1/(1-mu))
    while math.pow(mu, m - 1) * (1 + m * (1 - mu)) > (1 - p):
        m += 1
    return m

def create_ht(t, n, k):
    uhash = lsh_structs.uh_struct()
    uhash.table_size = n 
    uhash.hashed_data_length = k
    uhash.buckets = 0
    uhash.points = 0
    if t == 1:
        uhash.ll_hash_table = []
    elif t == 2:
        #something
        print 'df'

    uhash.main_hash_a = np.random.random_integers(1, const.max_hash_rnd, uhash.hashed_data_length)
    uhash.control_hash = np.random.random_integers(1, const.max_hash_rnd, uhash.hashed_data_length)
    return uhash


def init_hash_functions(params):
    nn = lsh_structs.nn_struct(params.k, params.l)
    nn.r = params.r
    nn.l = params.l
    nn.k = params.k
    for i in xrange(params.l):
        for j in xrange(params.k):
            nn.funcs[i][j].a = np.random.normal(0, 1, (1, params.d))
            nn.funcs[i][j].b = np.random.uniform(0, params.w) 
    return nn 
    
def compute_ulsh(nn, g, reduced_p):
    hashes = []
    for k in xrange(nn.k):
        s = reduced_p.multiply(nn.funcs[g][k].a).sum(1)
        hashes.append(math.floor((s + nn.funcs[g][k].b) / nn.w))

    nn.computed_ulshs.append(hashes)


