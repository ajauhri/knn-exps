import const
class alg_params:
    def __init__(self):
        self.r = None
        self.r2 = None
        self.w = None
        self.l = None #nHFTuples
        self.k = None #hfTuplesLength
        self.d = None
        self.m = None
        self.t = None
        self.type_ht = None
        self.success_pr = None
        #self.points_arr_size = None
        self.points = None
        self.n_points = 0
        self.computed_ulshs = []

class nn_struct:
    def __init__(self, k, l):
        self.funcs =  [[lsh_func() for j in xrange(k)] for i in xrange(l)]
        self.points = None
        self.r = None
        self.l = None
        self.k = None
        self.w = None
        self.n_hf_tuples = None
        self.hf_tuples_length = None
        self.computed_ulshs = None
        self.computed_hashes_of_ulshs = None
        self.hashed_buckets = []

class lsh_func:
    def __init__(self):
        self.a = []
        self.b = 0
        
class uh_struct:
    def __init__(self, t, table_size, data_length):
        self.t = t
        self.table_size = table_size
        self.buckets = 0
        self.points = 0
        
        self.data_length = data_length
        self.chain_sizes = None
        self.hybrid_chains_storage = None
        
        self.ll_hash_table = None 
        self.hybrid_hash_table = []

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

class hybrid_chain_entry:
    def __init__(self):
        self.control_value = None
        self.point = hybrid_entry()

class hybrid_entry:
    def __init__(self):
        self.is_last_bucket = 1
        self.bucket_length = const.n_bits_for_bucket_length
        self.is_last_point = 1
        self.point_index = const.n_bits_per_point_index
