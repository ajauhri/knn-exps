# hope to have data in same format...
import scipy.io
import numpy as np

def netflix(fname):
    mat = scipy.io.loadmat(fname)
    return mat['X'].transpose()

def generic(fname):
    X = np.loadtxt(fname)
    return X
