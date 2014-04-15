# standard libs
from __future__ import division
import math
import numpy as np
import psutil 

# user defined libs
import const
from extras.helper import debug
from lsh_func import lsh_func
from alg_params import alg_params
import lsh_helper

# some constants as descibed in the paper
const.success_pr = 0.9
const.w = 4
const.HYBRID = True
const.two_to_32_minus_1 = 4294967295
const.prime_default = 4294967291

# SelfTuning.cpp:275
def comute_l(k, success_prob):
    p = compute_p(const.w, 1)
    return math.ceil( math.log(1 - success_prob) / math.log(1 - math.pow(p, k)))
'''
    
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

'''
def determine_rt_coeffs(params, X):
    n = X.shape[0] / 50
    if n < 100:
        n = X.shape[0] 
    elif n > 10000:
        n = 10000
     
    params.k = 16
    params.t = n

    params.m = lsh_helper.compute_m(params.k, params.success_pr, params.w) 
    alg_params.l = params.m * (params.m - 1) / 2
    
    #init_lsh_with_dataset(alg_params, n, X)


'''
X - training data
Q - query data
r - radii
'''
def compute_opt(X, Q, r=0.6):
    params = alg_params()

    ''' setup algo params''' 
    params.success_pr = const.success_pr
    params.w = const.w
    params.type_ht = const.HYBRID
    #params.t = X.shape[0] # setting to the size of the training set
    params.d = X.shape[1]
    params.r = r
    params.r2 = r**2

    available_mem = psutil.phymem_usage().available

    '''determine coefficients for efficient computation '''
    for i in xrange(10):
        determine_rt_coeffs(params, X )  
