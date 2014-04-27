import numpy as np
import scipy.sparse as ss

def knn_naive(k, p, X):
    #stacked_p = ss.vstack([p for n in range(X.get_shape()[0])])
    #dist = X - stacked_p
    #dist = np.sqrt(dist.multiply(dist).sum(1))
    dist = []
    dists = np.linalg.norm(X - p, axis = 1)
    assert (dists.shape[0] == X.shape[0])
    return dists
