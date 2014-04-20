from __future__ import division
import ctypes as C
import math
import numpy as np

import lsh_structs
import const


def compute_product_mod_default_prime(a, b, size):
    h = 0
    shifts = np.uint64(32)
    for i in range(size):
        h = h + np.uint64(a[i]) * np.uint64(b[i])
        h = np.uint64(h)
        h = (h & const.two_to_32_minus_1) + 5 * (h >> shifts)
        if h >= const.prime_default:
            h = h - const.prime_default
        assert (h < const.prime_default)
    return h
'''
    h = int(np.dot(a[:size], np.transpose(b[:size])))
    if h >= const.prime_default:
        h = h - const.prime_default
    return h
'''
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

def create_ht(t, table_size, k, use_external=False, main_hash_a=None, control_hash=None, model_ht=None):
    last = 0
    uhash = lsh_structs.uh_struct(t, table_size, k)
    if t == 1:
        uhash.ll_hash_table = [None for x in range(table_size)]
    elif t == 2:
        assert (model_ht != None)
        uhash.hybrid_hash_table = (lsh_structs.hybrid_chain_entry * table_size)()
        uhash.hybrid_chains_storage = (lsh_structs.hybrid_chain_entry * (model_ht.points + model_ht.buckets))()

        index_in_storage = 0
        last_index_in_storage = model_ht.points + model_ht.buckets - 1
        for i in range(table_size):
            b = model_ht.ll_hash_table[i]
            if b:
                uhash.hybrid_hash_table[i] = uhash.hybrid_chains_storage[index_in_storage]
            else:
                uhash.hybrid_hash_table[i].is_null = 1
            while b:
                points_in_b = 1
                b_entry = b.first_entry.next_entry
                while b_entry:
                    points_in_b += 1
                    b_entry = b_entry.next_entry

                uhash.hybrid_chains_storage[index_in_storage].control_value = b.control_value
                index_in_storage += 1
                uhash.hybrid_chains_storage[index_in_storage].point.is_last_bucket = 1 if (not b.next_bucket_in_chain) else 0
                last += uhash.hybrid_chains_storage[index_in_storage].point.is_last_bucket
                #print 'L', uhash.hybrid_chains_storage[index_in_storage].point.is_last_bucket
                uhash.hybrid_chains_storage[index_in_storage].point.bucket_length = points_in_b if (points_in_b <= const.max_nonoverflow_points_per_bucket) else 0

                uhash.hybrid_chains_storage[index_in_storage].point.is_last_point = 1 if (points_in_b == 1) else 0
                uhash.hybrid_chains_storage[index_in_storage].point.point_index = b.first_entry.point_index
                index_in_storage += 1

                curr_index = index_in_storage
                n_overflow = 0
                overflow_start = last_index_in_storage

                if points_in_b <= const.max_nonoverflow_points_per_bucket:
                    index_in_storage += points_in_b - 1
                else:
                    n_overflow = points_in_b - const.max_nonoverflow_points_per_bucket
                    overflow_start = last_index_in_storage - n_overflow + 1
                    last_index_in_storage = overflow_start - 1

                    value = overflow_start - (curr_index - 1 + const.max_nonoverflow_points_per_bucket)
                    for j in range(const.n_fields_per_index_of_overflow):
                        uhash.hybrid_chains_storage[curr_index + j].point.bucket_length = value & ((1 << const.n_bits_for_bucket_length) - 1)
                        value = value >> const.n_bits_for_bucket_length
                    index_in_storage = index_in_storage + const.max_nonoverflow_points_per_bucket - 1
                    assert (index_in_storage <= last_index_in_storage + 1)

                b_entry = b.first_entry.next_entry
                while b_entry:
                    uhash.hybrid_chains_storage[curr_index].point.point_index = b_entry.point_index
                    uhash.hybrid_chains_storage[curr_index].point.is_last_point = 0
                    b_entry = b_entry.next_entry

                    curr_index += 1
                    if curr_index == index_in_storage and points_in_b > const.max_nonoverflow_points_per_bucket:
                        curr_index = overflow_start
                uhash.hybrid_chains_storage[curr_index - 1].point.is_last_point = 1
                b = b.next_bucket_in_chain
        
        assert(index_in_storage == last_index_in_storage + 1)
        uhash.points = model_ht.points
        uhash.buckets = model_ht.buckets


    if use_external:
        uhash.control_hash = control_hash
        uhash.main_hash_a = main_hash_a
    else:
        uhash.main_hash_a = const.prng.random_integers(1, const.max_hash_rnd, uhash.data_length)
        uhash.control_hash = const.prng.random_integers(1, const.max_hash_rnd, uhash.data_length)
    #print 'last=', last 
    return uhash

# box-muller transform
def gen_normal():
    while True:
        x1 = const.prng.uniform(0, 1)
        if x1 != 0:
            break
    x2 = const.prng.uniform(0, 1)
    return (np.sqrt(-2.0 * np.log(x1)) * np.cos(2 * np.pi * x2))

def init_hash_functions(params, n, d):
    nn = lsh_structs.nn_struct(int(params.k / 2), int(params.m))
    nn.n = n
    nn.r = params.r
    nn.l = params.l
    nn.k = params.k
    nn.n_hf_tuples = int(params.m)
    nn.hf_tuples_length = int(params.k / 2)
    nn.w = params.w
    for i in range(nn.n_hf_tuples):
        for j in range(nn.hf_tuples_length):
            for k in range(d):
                nn.funcs[i][j].a.append(gen_normal())
            nn.funcs[i][j].a = np.array(nn.funcs[i][j].a).reshape(1, d)
            nn.funcs[i][j].b = const.prng.uniform(0, params.w) 
    nn.marked_points = [False for x in range(n)] 
    nn.marked_points_indices = [None for x in range(n)]
    return nn 
    
def compute_ulsh(nn, g, reduced_p):
    hashes = []
    for k in xrange(nn.hf_tuples_length):
        s = reduced_p * nn.funcs[g][k].a.transpose()
        hashes.append(np.uint32(math.floor((s[0][0] + nn.funcs[g][k].b) / nn.w)))
    return hashes 

def compute_uhf_of_ulsh(uhash, ulsh, length):
    #print ulsh, uhash.main_hash_a[:length], '*', np.dot(ulsh[:length], uhash.main_hash_a[:length])
    assert(length * 2 == uhash.data_length)
    arr = [] 
    arr.append(compute_product_mod_default_prime(uhash.main_hash_a, ulsh, length))
    arr.append(compute_product_mod_default_prime(uhash.control_hash, ulsh, length))
    arr.append(compute_product_mod_default_prime(uhash.main_hash_a + length, ulsh, length))
    arr.append(compute_product_mod_default_prime(uhash.control_hash + length, ulsh, length))
    return arr

def construct_point(nn, uhash, p):
    # you have to time this
    reduced_p = p / nn.r 
    
    nn.computed_ulshs = []
    for i in range(nn.n_hf_tuples):
        nn.computed_ulshs.append(compute_ulsh(nn, i, reduced_p))

    nn.computed_hashes_of_ulshs = []
    for i in range(nn.n_hf_tuples):
        nn.computed_hashes_of_ulshs.append(compute_uhf_of_ulsh(uhash, np.array(nn.computed_ulshs[i]), nn.hf_tuples_length))


def add_bucket_entry(uhash, pieces, first_bucket_vector, second_bucket_vector, point_index):
    #print first_bucket_vector, second_bucket_vector
    h_index = np.uint64(first_bucket_vector[0]) + np.uint64(second_bucket_vector[0 + 2])
    if h_index >= const.prime_default:
        h_index -= const.prime_default
    assert(h_index < const.prime_default)
    h_index = np.uint32(h_index)
    h_index = h_index % uhash.table_size
    
    control = np.uint64(first_bucket_vector[1]) + np.uint64(second_bucket_vector[1 + 2])
    if control >= const.prime_default:
        control -= const.prime_default
    assert(control < const.prime_default)
    control = np.uint32(control)

    if uhash.t == 1:
        b = uhash.ll_hash_table[h_index] 
        while b and b.control_value != control:
            b = b.next_bucket_in_chain
        # if bucket does not exist
        if b is None:
            uhash.buckets += 1
            #print "HO ", uhash.buckets, point_index
            uhash.ll_hash_table[h_index] = lsh_structs.bucket(control, point_index, uhash.ll_hash_table[h_index])
        else:
            bucket_entry = lsh_structs.bucket_entry(point_index, b.first_entry.next_entry)
            b.first_entry.next_entry = bucket_entry
    uhash.points += 1

def get_bucket(uhash, pieces, first_bucket_vector, second_bucket_vector):
    h_index = np.uint64(first_bucket_vector[0]) + np.uint64(second_bucket_vector[0 + 2])
    if h_index >= const.prime_default:
        h_index -= const.prime_default
    assert(h_index < const.prime_default)
    h_index = np.uint32(h_index)
    h_index = h_index % uhash.table_size

    control = np.uint64(first_bucket_vector[1]) + np.uint64(second_bucket_vector[1 + 2])
    if control >= const.prime_default:
        control -= const.prime_default
    assert(control < const.prime_default)
    control = np.uint32(control)

    if uhash.t == 2:
        index_hybrid = uhash.hybrid_hash_table[h_index]
        while not index_hybrid.is_null:
            #print 'enter',
            if index_hybrid.control_value == control:
                index_hybrid = C.pointer(index_hybrid)[1]
                return index_hybrid
            else:
                index_hybrid = C.pointer(index_hybrid)[1]
                if index_hybrid.point.is_last_bucket:
                    #print 'leave 2 ', hybrid_hash_table[h_index].point.is_last_bucket
                    return None
                l = index_hybrid.point.bucket_length
                index_hybrid = C.pointer(index_hybrid)[l]
        #print h_index
        #print 'leave 3'
        return None
