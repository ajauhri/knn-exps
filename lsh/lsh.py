
from __future__ import division
import math
import numpy as np


import const
from lsh_func import lsh_func
const.w = 4
const.two_to_32_minus_1 = 4294967295
const.prime_default = 4294967291

# SelfTuning.cpp:275
def compute_p(w, c):
    x = w / c
    return 1 - math.erfc(x / np.sqrt(2)) - 2 * np.sqrt(1/np.pi) / np.sqrt(2) / x * (1 - np.exp(-math.pow(x, 2) / 2))

def comute_l(k, success_prob):
    p = compute_p(const.w, 1)
    return math.ceil( math.log(1 - success_prob) / math.log(1 - math.pow(p, k)))

def determine_rt_coefficients():
    n = tot_points / 50
    if n < 100:
        n = tot_points
    elif n > 10000:
        n = 10000
    
    #get the dataset based of the size
    # useUfunctions is some optimization for calculating the hash - look at LocalitySensitiveHashing.h for RNNParametersT
   alg_params = params()
   alg_params.k = 16
   alg_params.success_prob = 0.9
   alg_params.points_arr_sizse = n

   alg_params.m = compute_m(k, success_prob)
   alg_params.l = alg_params.m
    
   init_lsh_with_dataset(alg_params, n, X)

    
def init_lsh_with_dataset(alg_params, n, X):
    # initiate hash functions
    init_hash_functions(alg_params)
    alg_params.n_points = n # not sure about the difference of n_points and points_arr_size
    alg_params.points = X
    for x in X:
        prepare_point()

    
def compute_ulsh(alg_params, g, reduced_p):
    hashes = []
    for k in xrange(alg_params.k):
        s = reduced_g.multiply(alg_params.lsh_funcs[g][k].a).sum(1)
        hashes.append(math.floor((s + alg_params.lsh_funcs[g][k].b) / alg_params.w))

    alg_params.computed_ulshs.append(hashes)

def compute_uhf(alg_params):


def prepare_point(alg_params, p):
    # you may have time this
    reduced_p = p / alg_params.r # works for sparse matrix!!!

    for i in xrange(alg_params.l):
        compute_ulsh(alg_params, i, reduced_p)
    
    for i in xrange(alg_params.l):
        compute_uhf(alg_params)


def init_hash_functions(params):
    params.lsh_funcs = [[0 for j in xrange(params.k)] for i in xrange(params.l)]
    for i in xrange(params.l):
        for j in xrange(params.k):
            lsh_funcs[i][j] = lsh_func()
            lsh_funcs[i][j].a = np.random.normal(0, 1, (1, params.d))
            lsh_funcs[i][j].b = np.random.unif(0, params.w) 





