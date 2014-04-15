from __future__ import division
import math
import numpy as np

def compute_product_mod_default_prime(a, b, size):
    h = 0
    h = a.multiply(b).sum(1)
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

