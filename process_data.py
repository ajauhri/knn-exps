import scipy.io

def process_netflix(fname):
    mat = scipy.io.loadmat(fname)
    return mat.X


