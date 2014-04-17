from __future__ import division
import math
import numpy as np

import lsh_structs
import const

const.max_hash_rnd = 536870912
const.prime_default = 4294967291
const.two_to_32_minus_1 = 4294967295
const.n_bits_per_point_index = 20
const.n_bits_for_bucket_length = 32 - 2 - const.n_bits_per_point_index
const.max_nonoverflow_points_per_bucket = ((1 << const.n_bits_for_bucket_length) - 1)

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

def create_ht(t, table_size, k, use_external=False, main_hash_a=None, control_hash=None, model_ht=None):
    uhash = lsh_structs.uh_struct(t, table_size, k)
    if t == 1:
        uhash.ll_hash_table = [None for x in range(table_size)]
    elif t == 2:
        assert (model_ht != None)
        uhash.hybrid_hash_table = [lsh_structs.hybrid_chain_entry() for x in xrange(table_size)]
        uhash.hybrid_chains_storage = [lsh_structs.hybrid_chain_entry() for x in xrange(model_ht.points + model_ht.buckets)]

        index_in_storage = 0
        last_index_in_storage = model_ht.points + model_ht.buckets - 1
        for i in xrange(table_size):
            b = model_ht.ll_hash_table[i]
            if b:
                uhash.hybrid_hash_table[i] = uhash.hybrid_chains_storage[index_in_storage]
            while b:
                points_in_b = 1
                b_entry = b.first_entry.next_entry
                while b_entry:
                    points_in_b += 1
                    b_entry = b_entry.next_entry

                uhash.hybrid_chains_storage[index_in_storage].control_value = b.control_value
                index_in_storage += 1
                uhash.hybrid_chains_storage[index_in_storage].point.is_last_bucket = 1 if (not b.next_bucket_in_chain) else 0
                uhash.hybrid_chains_storage[index_in_storage].point.bucket_length = points_in_b if (points_in_b <= const.max_nonoverflow_points_per_bucket) else 0
                uhash.hybrid_chains_storage[index_in_storage].point.is_last_point = 1 if (points_in_b == 1) else 0
                uhash.hybrid_chains_storage[index_in_storage].point_index = b.first_entry.point_index
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
                    for j in xrange(const.n_fields_per_index_of_overflow):
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
        uhash.main_hash_a = np.random.random_integers(1, const.max_hash_rnd, uhash.data_length)
        uhash.control_hash = np.random.random_integers(1, const.max_hash_rnd, uhash.data_length)
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

def add_bucket_entry(uhash, pieces, first_bucket_vector, second_bucket_vector, point_index):
    #print first_bucket_vector, second_bucket_vector
    h_index = first_bucket_vector[0] + second_bucket_vector[0 + 2]
    if h_index >= const.prime_default:
        h_index -= const.prime_default
    assert(h_index < const.prime_default)
    h_index = h_index % uhash.table_size
    
    control = first_bucket_vector[1] + second_bucket_vector[1 + 2]
    if control >= const.prime_default:
        control -= const.prime_default
    assert(control < const.prime_default)

    if uhash.t == 1:
        b = uhash.ll_hash_table[h_index] 
        while b and b.control_value != control:
            b = b.next_bucket_in_chain

        # if bucket does not exist
        if not b:
            uhash.buckets += 1
            uhash.ll_hash_table[h_index] = lsh_structs.bucket(control, point_index, uhash.ll_hash_table[h_index])
        else:
            bucket_entry = lsh_structs.bucket_entry(point_index, b.first_entry.next_entry)
            b.first_entry.next_entry = bucket_entry
    uhash.points += 1

