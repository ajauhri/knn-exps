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

class lsh_func:
    def __init__(self):
        self.a = []
        self.b = 0
