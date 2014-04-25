#! /usr/bin/env python
from __future__ import division
import sys
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
from vlfeat.phow import vl_phow
from random import randrange
import math

import lsh.lsh as lsh
from extras.helper import debug

def init(fname):
    patch_size = 24
    bucket_size = 6
    s = 6
    img = imread(fname)
    train_img = img.copy()
    query_img = img.copy()
    (height, width) = img.shape[:2]
    debug('img height = %d, img width = %d' % (height, width))
    
    (F, D) = vl_phow(train_img, sizes = [bucket_size], step = 6, color = 'hsv')
    # darken some patches of images
    patches = []
    for i in range(10):
        mid_ind = randrange(0, F.shape[0])
        patches.append(mid_ind)
        
        start_x = F[mid_ind][0] - bucket_size / 2
        start_y = F[mid_ind][1] - bucket_size / 2
        train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] = [0, 0, 0]
        
    
    (F, D) = vl_phow(train_img, sizes = [bucket_size], step = s, color = 'hsv')
    trimmed_height = height - 2 * ((bucket_size) * 3/2)
    limit = int(math.ceil(trimmed_height / s)) 

    for val in patches:
        # bottom neighbour
        mid_ind = val + 4*limit
        if mid_ind < F.shape[0]:
            start_x = F[mid_ind][0] - bucket_size / 2
            start_y = F[mid_ind][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            print 'not found bottom neigbour'


        # top neighbour
        mid_ind = val - 4*limit

        if mid_ind >= 0:
            start_x = F[mid_ind][0] - bucket_size / 2
            start_y = F[mid_ind][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            print 'not found top neigbour'



        # right neighbour
        mid_ind = val + 4 
        if mid_ind < F.shape[0]:
            start_x = F[mid_ind][0] - bucket_size / 2
            start_y = F[mid_ind][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            print 'not found right neigbour'

        # left neighbour
        mid_ind = val - 4
        if mid_ind >= 0:
            start_x = F[mid_ind][0] - bucket_size / 2
            start_y = F[mid_ind][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            print 'not found left neighbour'


    #print F[ind+1,0], F[ind+1, 1]
    #print (width/6), F.shape
    plt.subplot(211)
    plt.imshow(train_img, interpolation='nearest')
    plt.subplot(212)
    plt.imshow(query_img, interpolation='nearest')
    plt.show()

    #lsh.start(D, D, float(100))
    
if __name__ == "__main__":
    init(sys.argv[1])

