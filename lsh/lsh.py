# standard libs
from __future__ import division
import sys
import math
import numpy as np
import psutil 
from random import randrange
import ctypes as C 
import lsh_globals
from lsh_globals import timers

# user defined libs
import const
from extras.helper import debug
from lsh_structs import alg_params, lsh_func, nn_struct
import lsh_helper
import time

# some constants as descibed in the paper
const.success_pr = 0.9
const.w = 4
const.HYBRID = True

def comute_l(k, success_prob):
    p = compute_p(const.w, 1)
    return math.ceil( math.log(1 - success_prob) / math.log(1 - math.pow(p, k)))

def init_lsh_with_dataset(params, n, X):
    # initialize hash functions
    debug("initializing hash functions")

    nn = lsh_helper.init_hash_functions(params, n, X)

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
                computed_hashes_of_ulshs[j][i][k] = np.uint32(nn.computed_hashes_of_ulshs[j][k])
    print
    first_u_comp = 0
    second_u_comp = 1
    for i in range(nn.l):
        sys.stdout.write("\rL = %d of %d" % (i + 1, nn.l))
        for j in range(X.shape[0]):
            lsh_helper.add_bucket_entry(uhash, 2, computed_hashes_of_ulshs[first_u_comp][j], computed_hashes_of_ulshs[second_u_comp][j], j)
        
        second_u_comp += 1
        if second_u_comp == nn.n_hf_tuples:
            first_u_comp += 1
            second_u_comp = first_u_comp + 1
           
        nn.hashed_buckets.append(lsh_helper.create_ht(2, n, nn.k, True, uhash.main_hash_a, uhash.control_hash, uhash))
        uhash.ll_hash_table = [None for x in range(uhash.table_size)]
        uhash.points = 0
        uhash.buckets = 0
        sys.stdout.flush()
    print
    return nn


def get_ngh_struct(nn, q):
    ##print 'query', 
    lsh_helper.construct_point(nn, nn.hashed_buckets[0], q)
    computed_hashes_of_ulshs = eval(`[[0]*4]*nn.n_hf_tuples`)
    for i in xrange(nn.n_hf_tuples):
        for j in xrange(4):
            computed_hashes_of_ulshs[i][j] = np.uint32(nn.computed_hashes_of_ulshs[i][j])
    first_u_comp = 0
    second_u_comp = 1
    
    neighbours = []
    n_marked_points = 0

    for i in xrange(nn.l):
        start_time = time.time()
        hybrid_point = lsh_helper.get_bucket(nn.hashed_buckets[i], 2, computed_hashes_of_ulshs[first_u_comp], computed_hashes_of_ulshs[second_u_comp])
        #print 'p=',hybrid_point

        second_u_comp += 1
        if second_u_comp == nn.n_hf_tuples:
            first_u_comp += 1
            second_u_comp = first_u_comp + 1
        timers.get_bucket_time += time.time() - start_time

        if nn.hashed_buckets[i].t == 2:
            start_time = time.time()
            if hybrid_point:
                offset = 0
                if hybrid_point.point.bucket_length == 0:
                    offset = 0
                    for j in range(const.n_fields_per_index_of_overflow):
                        offset += np.uint32((C.pointer(hybrid_point)[1+j].point.bucket_length) << (j * const.n_bits_for_bucket_length))

                index = 0
                done = False
                while not done:
                    if index == const.max_nonoverflow_points_per_bucket:
                        index += offset
                    candidate_index = C.pointer(hybrid_point)[index].point.point_index
                    #print 'candidate_index = ', candidate_index
                    assert(candidate_index >= 0 and candidate_index < nn.n)
                    done = True if C.pointer(hybrid_point)[index].point.is_last_point else False
                    index += 1

                    if nn.marked_points[candidate_index] == False:
                        nn.marked_points_indices[n_marked_points] = candidate_index
                        nn.marked_points[candidate_index] = True
                        n_marked_points += 1

                        candidate_point = nn.points[candidate_index]

                        lsh_globals.n_dist_comps += 1
                        if np.linalg.norm(q - candidate_point) <= nn.r:
                            print 'candidate index =', candidate_index
                            neighbours.append((candidate_point, candidate_index))
            timers.bucket_cycle_time += time.time() - start_time
    for i in range(n_marked_points):
        nn.marked_points[nn.marked_points_indices[i]] = False

    return neighbours
            
def determine_rt_coeffs(params, X, Q):
    n = X.shape[0] / 50
    if n < 100:
        n = X.shape[0] 
    elif n > 10000:
        n = 10000

    params.k = 16
    params.t = n

    params.m = lsh_helper.compute_m(params.k, params.success_pr, params.w) 
    params.l = int(params.m * (params.m - 1) / 2)
    
    n_suc_reps = 0 
    while n_suc_reps < 5: 
        nn = init_lsh_with_dataset(params, n, X[:n,:])

        n_suc_reps = 0

        lsh_pre_comp = 0
        u_hash_comp = 0
        dist_comp = 0
        n_queries = 20
        for i in range(n_queries):
            timers.compute_lsh_time = 0
            timers.get_bucket_time = 0
            timers.bucket_cycle_time = 0
            lsh_globals.n_dist_comps = 0

            q_index = const.prng.randint(0, Q.shape[0] - 1)
            print 'query =' , q_index
            print 'nNNs=', len(get_ngh_struct(nn, Q[q_index]))

            if lsh_globals.n_dist_comps >= min(n/10, 100):
                n_suc_reps += 1
                lsh_pre_comp += timers.compute_lsh_time / params.k / params.m
                u_hash_comp += timers.get_bucket_time / params.l
                dist_comp += timers.bucket_cycle_time / lsh_globals.n_dist_comps

            if n_suc_reps >= 5:
                lsh_pre_comp /= n_suc_reps
                u_hash_comp /= n_suc_reps
                dist_comp /= n_suc_reps
            else:
                params.r = params.r*2
    
    print 'out'
     
'''
X - training data
Q - query data
r - radii
'''
def compute_opt(X, Q, r=0.6):
    prng = np.random.RandomState()
    const.prng = prng
    params = alg_params()

    ''' setup algo params''' 
    params.success_pr = const.success_pr
    params.w = const.w
    params.type_ht = const.HYBRID
    #params.t = X.shape[0] # setting to the size of the training set
    params.d = X.shape[1]
    params.r = r

    available_mem = psutil.phymem_usage().available

    '''determine coefficients for efficient computation '''
    for i in xrange(1):
        determine_rt_coeffs(params, X, Q)  
