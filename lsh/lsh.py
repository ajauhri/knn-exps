# standard libs
from __future__ import division
import sys
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

# SelfTuning.cpp:275
def comute_l(k, success_prob):
    p = compute_p(const.w, 1)
    return math.ceil( math.log(1 - success_prob) / math.log(1 - math.pow(p, k)))
'''
def compute_uhf(alg_params):

'''
def construct_point(nn, uhash, p):
    # you have time this
    reduced_p = p / nn.r 

    for i in xrange(nn.n_hf_tuples):
        lsh_helper.compute_ulsh(nn, i, reduced_p)
    
    for i in xrange(nn.n_hf_tuples):
        nn.computed_hashes_of_ulshs.append(lsh_helper.compute_uhf_of_ulsh(uhash, np.array(nn.computed_ulshs[i]), nn.hf_tuples_length))

def init_lsh_with_dataset(params, n, X):
    # initialize hash functions
    debug("initializing hash functions")
    nn = lsh_helper.init_hash_functions(params)
    nn.points = X

    # initialize second level hashing (bucket hashing) 232
    uhash = lsh_helper.create_ht(1, n, nn.k) #1 - for linked lists; 2 - for hybrid chains
    count = 0
    computed_hashes_of_ulshs = eval(`[[[0]*4]*X.shape[0]]*nn.l`)
    for i in xrange(X.shape[0]):
        sys.stdout.write("\rloading hashes for point %d out of %d" % (count,X.shape[0]))
        construct_point(nn, uhash, X[i])
        count += 1
        sys.stdout.flush()
        for j in xrange(nn.n_hf_tuples):
            for k in xrange(4):
                computed_hashes_of_ulshs[j][i][k] = nn.computed_hashes_of_ulshs[j][k]
            print computed_hashes_of_ulshs[j][i]
    
    print 
    for i in xrange(nn.l):
        for j in xrange(X.shape[0]):
            lsh_helper.add_bucket_entry(uhash, 2, computed_hashes_of_ulshs[0][j], computed_hashes_of_ulshs[1][j], j)

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
    
    init_lsh_with_dataset(params, n, X[:n,:]) #SelfTuning:202


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
    for i in xrange(1):
        determine_rt_coeffs(params, X )  
