from __future__ import division
import math
import numpy as np

import lsh_structs
import const

const.max_hash_rnd = 536870912
const.prime_default = 4294967291
def compute_product_mod_default_prime(a, b, size):
    h = int(np.dot(a[:size], np.transpose(b[:size])))
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
    nn = lsh_structs.nn_struct(int(params.k / 2), int(params.m))
    nn.r = params.r
    nn.l = params.l
    nn.k = params.k
    nn.n_hf_tuples = int(params.m)
    nn.hf_tuples_length = int(params.k / 2)
    nn.w = params.w
    for i in xrange(nn.n_hf_tuples):
        for j in xrange(nn.hf_tuples_length):
            nn.funcs[i][j].a = np.random.normal(0, 1, (1, params.d))
            nn.funcs[i][j].b = np.random.uniform(0, params.w) 
    return nn 
    
def compute_ulsh(nn, g, reduced_p):
    hashes = []
    for k in xrange(nn.hf_tuples_length):
        s = reduced_p * nn.funcs[g][k].a.transpose()
        hashes.append(math.floor((s + nn.funcs[g][k].b) / nn.w))

    nn.computed_ulshs.append(hashes)

def compute_uhf_of_ulsh(uhash, ulsh, length):
    assert(length * 2 == uhash.hashed_data_length)
    arr = [] 
    arr.append(compute_product_mod_default_prime(uhash.main_hash_a, ulsh, length))
    arr.append(compute_product_mod_default_prime(uhash.control_hash, ulsh, length))
    arr.append(compute_product_mod_default_prime(uhash.main_hash_a + length, ulsh, length))
    arr.append(compute_product_mod_default_prime(uhash.control_hash + length, ulsh, length))
    return arr

def add_bucket_entry(uhash, pieces, first_bucket_vector, second_bucket_vector, point_index):
    #print first_bucket_vector, second_bucket_vector
    h_index = first_bucket_vector[0] + second_bucket_vector[0 + 2]
    if h_index >= const.prime_default:
        h_index -= const.prime_default
    control = first_bucket_vector[1] + second_bucket_vector[1 + 2]
    if control >= const.prime_default:
        control -= const.prime_default
    h_index = h_index % uhash.table_size
