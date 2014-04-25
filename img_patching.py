#! /usr/bin/env python
import sys
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
from vlfeat.phow import vl_phow
from random import randrange

import lsh.lsh as lsh

def init(fname):
    patch_size = 24
    bucket_size = 6
    img = imread(fname)
    copy_img = img
    (height, width) = img.shape[:2]
    print 'img height = %d, img widhth = %d' % (height, width)
    
    (F, D) = vl_phow(copy_img, sizes = [bucket_size], step = 6, color = 'hsv')
    # darken some patches of images
    patches = []
    for i in range(20):
        mid = randrange(0, F.shape[0])
        start_x = F[mid][0] - bucket_size / 2
        start_y = F[mid][1] - bucket_size / 2
        patches.append(mid)
        copy_img[start_x:start_x+patch_size, start_y:start_y+patch_size] = [0, 0, 0]
    
    (F, D) = vl_phow(copy_img, sizes = [bucket_size], step = 6, color = 'hsv')
    #plt.imshow(copy_img, interpolation='nearest')
    #plt.show()
    print D.shape
    lsh.start(D, D, float(100))
    
if __name__ == "__main__":
    init(sys.argv[1])

