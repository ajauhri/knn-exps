# standard libs
from __future__ import division
import math
import numpy as np
import psutil 

# user defined libs
import const
from extras.helper import debug
from lsh_structs import alg_params, lsh_func, nn_struct
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
def compute_uhf(alg_params):

'''
def compute_uhf_of_ulsh(uhash, ulsh, length):
# need to do some assertions here
    compute_product_mod_default_prime(uhash.main_hash_a, ulsh, length)
    compute_product_mod_default_prime(uhash.control_hash, ulsh, length)
    compute_product_mod_default_prime(uhash.main_hash_a + length, ulsh, length)
    compute_product_mod_default_prime(uhash.control_hash + length, ulsh, length)

def construct_point(nn, uhash, p):
    # you have time this
    reduced_p = p / nn.r 

    for i in xrange(nn.l):
        lsh_helper.compute_ulsh(nn, i, reduced_p)
    
    for i in xrange(nn.l):
        lsh_helper.compute_uhf_of_ulsh(uhash, np.array(nn.computed_ulshs[i]), nn.k)


def init_lsh_with_dataset(params, n, X):
    # initialize hash functions
    debug("initializing hash functions")
    nn = lsh_helper.init_hash_functions(params)
    nn.points = X[:n,:]

    # initialize second level hashing (bucket hashing) 232
    uhash = lsh_helper.create_ht(1, n, params.k) #1 - for linked lists; 2 - for hybrid chains
    for x in X:
        construct_point(nn, uhash, x)

def determine_rt_coeffs(params, X):
    n = X.shape[0] / 50
    if n < 100:
        n = X.shape[0] 
    elif n > 10000:
        n = 10000
     
    params.k = 16
    params.t = n

    params.m = lsh_helper.compute_m(params.k, params.success_pr, params.w) 
    params.l = int(params.m * (params.m - 1) / 2)
    
    init_lsh_with_dataset(params, n, X) #SelfTuning:202


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
