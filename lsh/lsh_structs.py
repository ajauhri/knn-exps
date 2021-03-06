import const
import ctypes as C
import numpy as np
const.max_hash_rnd = 536870912
const.prime_default = 4294967291
const.two_to_32_minus_1 = np.uint64(4294967295)
const.n_bits_per_point_index = 20
const.n_bits_for_bucket_length = 32 - 2 - const.n_bits_per_point_index
const.n_fields_per_index_of_overflow = ((32 + const.n_bits_for_bucket_length - 1) / const.n_bits_for_bucket_length)
const.max_nonoverflow_points_per_bucket = ((1 << const.n_bits_for_bucket_length) - 1)


class alg_params:
    def __init__(self):
        self.r = None
        self.w = None
        self.l = None 
        self.k = None 
        self.d = None
        self.m = None
        self.type_ht = None
        self.success_pr = None

class nn_struct:
    def __init__(self, k, l):
        self.funcs =  [[lsh_func() for j in xrange(k)] for i in xrange(l)]
        self.points = None
        self.r = None
        self.l = None
        self.k = None
        self.w = None
        self.n = None
        
        self.n_hf_tuples = None
        self.hf_tuples_length = None
        self.computed_ulshs = None
        self.computed_hashes_of_ulshs = None
        
        self.hashed_buckets = []
        self.marked_points = None
        self.marked_points_indices = None

class lsh_func:
    def __init__(self):
        self.a = []
        self.b = 0
        
class uh_struct:
    def __init__(self, t, table_size, data_length):
        self.t = t
        self.table_size = table_size
        self.data_length = data_length
        self.buckets = 0
        self.points = 0
        
        self.chain_sizes = None
        self.hybrid_chains_storage = None
        
        self.ll_hash_table = None 
        self.hybrid_hash_table = None

        self.main_hash_a = None
        self.control_hash = None

class bucket:
    def __init__(self, control, point_index, next_bucket):
        self.control_value = control 
        self.first_entry = bucket_entry(point_index, None) 
        self.next_bucket_in_chain = next_bucket

class bucket_entry:
    def __init__(self, point_index, next_entry):
        self.point_index = point_index 
        self.next_entry = next_entry 

class hybrid_entry(C.Structure):
    _fields_ = [("is_last_bucket", C.c_uint),
                ("bucket_length", C.c_uint),
                ("is_last_point", C.c_uint),
                ("point_index", C.c_uint)]
    
    def __init__(self, is_last = 1, length = const.n_bits_for_bucket_length, last_point = 1, point_index = const.n_bits_per_point_index):
        super(hybrid_entry, self).__init__(is_last, length, last_point, point_index)

    
class hybrid_chain_entry(C.Union):
    _fields_ = [("control_value", C.c_uint),
                ("point", hybrid_entry),
                ("is_null", C.c_int)]

