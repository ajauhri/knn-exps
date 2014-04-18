# standard libs
from __future__ import division
import sys
import math
import numpy as np
import psutil 
from random import randrange

# user defined libs
import const
from extras.helper import debug
from lsh_structs import alg_params, lsh_func, nn_struct
import lsh_helper

# some constants as descibed in the paper
const.success_pr = 0.9
const.w = 4
const.HYBRID = True

# SelfTuning.cpp:275
def comute_l(k, success_prob):
    p = compute_p(const.w, 1)
    return math.ceil( math.log(1 - success_prob) / math.log(1 - math.pow(p, k)))

def init_lsh_with_dataset(params, n, X):
    # initialize hash functions
    debug("initializing hash functions")
    nn = lsh_helper.init_hash_functions(params, n)
    nn.points = X

    # initialize second level hashing (bucket hashing) 232
    uhash = lsh_helper.create_ht(1, n, nn.k) #1 - for linked lists; 2 - for hybrid chains
    count = 0
    computed_hashes_of_ulshs = eval(`[[[0]*4]*X.shape[0]]*nn.l`)
    for i in xrange(X.shape[0]):
        sys.stdout.write("\rloading hashes for point %d out of %d" % (count + 1, X.shape[0]))
        lsh_helper.construct_point(nn, uhash, X[i])
        count += 1
        sys.stdout.flush()
        for j in xrange(nn.n_hf_tuples):
            for k in xrange(4):
                computed_hashes_of_ulshs[j][i][k] = nn.computed_hashes_of_ulshs[j][k]
    print
    first_u_comp = 0
    second_u_comp = 1
    for i in xrange(nn.l):
        sys.stdout.write("\rL = %d of %d" % (i + 1, nn.l))
        for j in xrange(X.shape[0]):
            lsh_helper.add_bucket_entry(uhash, 2, computed_hashes_of_ulshs[first_u_comp][j], computed_hashes_of_ulshs[second_u_comp][j], j)

        second_u_comp += 1
        if second_u_comp == nn.n_hf_tuples:
            first_u_comp += 1
            second_u_comp = first_u_comp + 1
       
        nn.hashed_buckets.append(lsh_helper.create_ht(2, n, nn.k, True, uhash.control_hash, uhash.main_hash_a, uhash))
        uhash.ll_hash_table = [None for x in range(uhash.table_size)]
        uhash.points = 0
        uhash.buckets = 0
        sys.stdout.flush()
    print
    return nn


def get_ngh_struct(nn, q):
    lsh_helper.construct_point(nn, nn.hashed_buckets[0], q)
    computed_hashes_of_ulshs = eval(`[[0]*4]*nn.n_hf_tuples`)
    for i in xrange(nn.n_hf_tuples):
        for j in xrange(4):
            computed_hashes_of_ulshs[i][j] = nn.computed_hashes_of_ulshs[i][j]
    first_u_comp = 0
    second_u_comp = 1
    
    neighbours = []
    n_marked_points = 0

    for i in xrange(nn.l):
        hybrid_hash_table = nn.hashed_buckets[i].hybrid_hash_table
        b_index = lsh_helper.get_bucket(nn.hashed_buckets[i], 2, computed_hashes_of_ulshs[first_u_comp], computed_hashes_of_ulshs[second_u_comp])

        second_u_comp += 1
        if second_u_comp == nn.n_hf_tuples:
            first_u_comp += 1
            second_u_comp = first_u_comp + 1

        if nn.hashed_buckets[i].t == 2:
            if b_index and hybrid_hash_table[b_index]:
                offset = 0
                if hybrid_hash_table[b_index].point.bucket_length == 0:
                    offset = 0
                    for j in range(const.n_fields_per_index_of_overflow):
                        offset += ((hybrid_hash_table[b_index + 1 + j].point.bucket_length) << (j * const.n_bits_for_bucket_length))
                index = 0
                done = False
                while not done:
                    if index == const.max_nonoverflow_points_per_bucket:
                        index += offset
                    candidate_index = hybrid_hash_table[b_index + index].point.point_index
                    assert(candidate_index >= 0 and candidate_index < nn.n)
                    done = True if hybrid_hash_table[b_index + index].point.is_last_point else False
                    index += 1

                    if nn.marked_points[candidate_index] == False:
                        nn.marked_points_indices[n_marked_points] = candidate_index
                        nn.marked_points[candidate_index] = True
                        n_marked_points += 1

                        candidate_point = nn.points[candidate_index]
                        
                        if np.linalg.norm(q - candidate_point) <= nn.r**2:
                            neighbours.append((candidate_point, candidate_index))

    for i in range(n_marked_points):
        nn.marked_points[nn.marked_points_indices[i]] = False

    return neighbours
            
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

    nn = init_lsh_with_dataset(params, n, X[:n,:])
    for i in range(20):
        r = randrange(n)
        print len(get_ngh_struct(nn, X[r]))

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
