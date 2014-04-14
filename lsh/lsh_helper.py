from random import SystemRandom

def compute_product_mod_default_prime(a, b, size):
    h = 0
    h = a.multiply(b).sum(1)
    h = (h & const.two_to_32_minus_1) + 5 * (h >> 32)
    if h >= const.prime_default:
        h = h - const.prime_default
    assert (h < const.prime_default)
    return h
