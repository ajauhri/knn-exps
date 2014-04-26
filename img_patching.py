#! /usr/bin/env python
from __future__ import division
import sys
from scipy.misc import imread, imsave
import numpy as np
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
    
    (F, D) = vl_phow(train_img, sizes = [bucket_size], step = s, color = 'hsv')
    # darken some patches of images
    patches = []
    for i in range(4):
        patch_i = randrange(0, F.shape[0])
        patches.append(patch_i)
        
        start_x = F[patch_i][0] - bucket_size / 2
        start_y = F[patch_i][1] - bucket_size / 2
        train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] = [0, 0, 0]
        
    
    (F, X) = vl_phow(train_img, sizes = [bucket_size], step = s, color = 'hsv')
    trimmed_height = height - 2 * ((bucket_size) * 3/2)
    limit = int(math.ceil(trimmed_height / s)) 
    
    apprx_img = train_img.copy() 
    patch_nghs = []
    for val in patches:
        # bottom neighbour
        patch_i = val + 4*limit
        if patch_i < F.shape[0]:
            patch_nghs.append(patch_i)
            start_x = F[patch_i][0] - bucket_size / 2
            start_y = F[patch_i][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            patch_nghs.append(-1)
            print 'not found bottom neigbour'


        # top neighbour
        patch_i = val - 4*limit
        if patch_i >= 0:
            patch_nghs.append(patch_i)
            start_x = F[patch_i][0] - bucket_size / 2
            start_y = F[patch_i][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            patch_nghs.append(-1)
            print 'not found top neigbour'

        # right neighbour
        patch_i = val + 4
        if patch_i < F.shape[0]:
            patch_nghs.append(patch_i)
            start_x = F[patch_i][0] - bucket_size / 2
            start_y = F[patch_i][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            patch_nghs.append(-1)
            print 'not found right neigbour'

        # left neighbour
        patch_i = val - 4
        if patch_i >= 0:
            patch_nghs.append(patch_i)
            start_x = F[patch_i][0] - bucket_size / 2
            start_y = F[patch_i][1] - bucket_size / 2
            query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
        else:
            patch_nghs.append(-1)
            print 'not found left neighbour'

    Q = np.vstack([X[patch_i] if patch_i != -1 else np.ones((1, X.shape[1])) for patch_i in patch_nghs])
    nghs = lsh.start(X, Q, float(100))
    
    ones = np.ones(1, X.shape[1])
    for i in range(len(patches)):
        bottom_b_ngh = None
        top_b_ngh = None
        left_b_ngh = None
        right_b_ngh = None
        j = i*4

        # bottom's neighbours
        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghbs[j]:
                if F[ngh[0]][0] < F[patch_nghs[j]][0]:
                    diff = F[patch_nghs[j]][0] - F[ngh[0]][0]
                    if min(diff, min_dist):
                        min_dist = diff
                        bottom_b_ngh = ngh[0]
            if not bottom_b_ngh:
                bottom_b_ngh = patch_nghs[j]

        # top's neighbours
        j = j+1
        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghbs[j]:
                if F[ngh[0]][0] > F[patch_nghs[j]][0]:
                    diff = F[patch_nghs[j]][0] - F[ngh[0]][0]
                    if min(diff, min_dist):
                        min_dist = diff
                        top_b_ngh = ngh[0]
            if not top_b_ngh:
                top_b_ngh = patch_nghs[j]

        # right's neighbours
        j = j+1
        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghbs[j]:
                if F[ngh[0]][1] < F[patch_nghs[j]][1]:
                    diff = F[patch_nghs[j]][1] - F[ngh[0]][1]
                    if min(diff, min_dist):
                        min_dist = diff
                        right_b_ngh = ngh[0]
            if not right_b_ngh:
                right_b_ngh = patch_nghs[j]

        # left's neighbours
        j = j+1
        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghbs[j]:
                if F[ngh[0]][1] > F[patch_nghs[j]][1]:
                    diff = F[patch_nghs[j]][1] - F[ngh[0]][1]
                    if min(diff, min_dist):
                        min_dist = diff
                        left_b_ngh = ngh[0]
            if not left_b_ngh:
                left_b_ngh = patch_nghs[j]
         
        start_x = F[patches[i]][0] - bucket_size / 2
        start_y = F[patches[i]][1] - bucket_size / 2
        count = 0 
        if bottom_b_ngh:
            count += 1
            start_ngh_x = F[bottom_b_ngh][0] - bucket_size / 2
            start_ngh_y = F[bottom_b_ngh][1] - bucket_size / 2
            train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] += train_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]
 
        if top_b_ngh:
            count += 1
            start_ngh_x = F[top_b_ngh][0] - bucket_size / 2
            start_ngh_y = F[top_b_ngh][1] - bucket_size / 2
            train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] += train_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]

 
        if right_b_ngh:
            count += 1
            start_ngh_x = F[right_b_ngh][0] - bucket_size / 2
            start_ngh_y = F[right_b_ngh][1] - bucket_size / 2
            train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] += train_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]
 
        if left_b_ngh:
            count += 1
            start_ngh_x = F[left_b_ngh][0] - bucket_size / 2
            start_ngh_y = F[left_b_ngh][1] - bucket_size / 2
            train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] += train_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]
        
        train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] /= count
         
    #plt.subplot(221)
    #plt.imshow(train_img, interpolation='nearest')
    #plt.subplot(222)
    #plt.imshow(query_img, interpolation='nearest')
    #plt.show()
   

          

if __name__ == "__main__":
    init(sys.argv[1])
