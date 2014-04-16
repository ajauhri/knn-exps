# hope to have data in same format...
import scipy.io
import numpy as np

def netflix(fname):
    mat = scipy.io.loadmat(fname)
    return mat['X'].transpose()

def generic(fname):
    X = np.array([])
    with open(fname, 'rt') as f:
        content = f.readlines()
        X = np.matrix(content[0].split(), dtype=float)
        for line in content[1:]:
            X = np.vstack([X, np.matrix(line.split(), dtype=float)])
    return X
