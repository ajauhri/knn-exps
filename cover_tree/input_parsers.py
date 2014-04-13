# hope to have data in same format...
import scipy.io

def netflix(fname):
    mat = scipy.io.loadmat(fname)
    return mat['X'].transpose()

