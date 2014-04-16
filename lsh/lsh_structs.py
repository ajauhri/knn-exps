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
        self.computed_ulshs = []
        self.computed_hashes_of_ulshs = []



class lsh_func:
    def __init__(self):
        self.a = []
        self.b = 0
        
class uh_struct:
    def __init__(self):
        self.table_size = None
        self.buckets = None
        self.hashed_data_length = None
        self.points = None
        self.main_hash_a = None
        self.control_hash = None
       
