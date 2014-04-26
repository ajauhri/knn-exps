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
    img = imread(fname)
    train_img = img.copy()
    query_img = img.copy()
    (width, height) = img.shape[:2]
    debug('img height = %d, img width = %d' % (height, width))
    
    (F, D) = vl_phow(train_img, sizes = [bucket_size], step = bucket_size, color = 'hsv')
    # darken some patches of images
    patches = []
    for i in range(10):
        while True:
            patch_i = randrange(0, F.shape[0])
            start_x = F[patch_i][1] - (bucket_size) * 2
            start_y = F[patch_i][0] - (bucket_size) * 2
            if start_x >= 0 and start_y >= 0 and start_x + patch_size < width and start_y + patch_size < height:
                break
        print patch_i 
        train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] = [0, 0, 0]
        patches.append(patch_i)
        
    
    (F, X) = vl_phow(train_img, sizes = [bucket_size], step = bucket_size, color = 'hsv')
    trimmed_width = width - 2 * ((bucket_size) * 3/2)
    limit = int(math.ceil(trimmed_width / bucket_size)) 
    patch_nghs = []
    # get 4 nghs for each patch
    for patch_i in patches:
        patch_x = F[patch_i][1] - (bucket_size) * 2 
        patch_y = F[patch_i][0] - (bucket_size) * 2 

        center = patch_i + 4*limit
        if center < F.shape[0]:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if start_x == patch_x and start_y - patch_y == patch_size:
                query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found right neigbour'
        else:
            patch_nghs.append(-1)
            print 'not found right neigbour'

        center = patch_i - 4*limit
        if center >= 0:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if start_x == patch_x and patch_y - start_y == patch_size:
                query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found left neigbour'
        else:
            patch_nghs.append(-1)
            print 'not found left neigbour'

        center = patch_i + 4 
        if center < F.shape[0]:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if start_x - patch_x == patch_size and start_y == patch_y:
                query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found bottom neigbour'
        else:
            patch_nghs.append(-1)
            print 'not found bottom neigbour'

        center = patch_i - 4
        if center >= 0:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if patch_x - start_x == patch_size and start_y == patch_y:
                query_img[start_x:start_x + patch_size, start_y:start_y+patch_size] = [255, 255, 255]
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found top neighbour'
        else:
            patch_nghs.append(-1)
            print 'not found top neighbour'

    approx_img = train_img.copy() 
    
    Q = np.vstack([X[patch_i] if patch_i != -1 else np.ones((1, X.shape[1])) for patch_i in patch_nghs])
    nghs = lsh.start(X, Q, float(200))
    ones = np.ones((1, X.shape[1]))
    for i in range(len(patches)):
        bottom_b_ngh = None
        top_b_ngh = None
        left_b_ngh = None
        right_b_ngh = None
        j = i*4

        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                if F[ngh[0]][1] < F[patch_nghs[j]][1]:
                    diff = F[patch_nghs[j]][1] - F[ngh[0]][1]
                    if diff < min_dist and patch_nghs[j] != ngh[0]:
                        min_dist = diff
                        right_b_ngh = ngh[0]
            if not right_b_ngh:
                right_b_ngh = patch_nghs[j]

        j = j+1
        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                if F[ngh[0]][1] > F[patch_nghs[j]][1]:
                    diff = F[ngh[0]][0] - F[patch_nghs[j]][0]
                    if diff < min_dist and ngh[0] != patch_nghs[j]:
                        min_dist = diff
                        left_b_ngh = ngh[0]
            if not left_b_ngh:
                left_b_ngh = patch_nghs[j]

        j = j+1
        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                if F[ngh[0]][0] < F[patch_nghs[j]][0]:
                    diff = F[patch_nghs[j]][1] - F[ngh[0]][1]
                    if diff < min_dist and patch_nghs[j] != ngh[0] :
                        min_dist = diff
                        bottom_b_ngh = ngh[0]
            if not bottom_b_ngh:
                bottom_b_ngh = patch_nghs[j]

        # left's neighbours
        j = j+1
        min_dist = 10000
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                if F[ngh[0]][0] > F[patch_nghs[j]][0]:
                    diff = F[ngh[0]][1] - F[patch_nghs[j]][1]
                    if diff < min_dist and patch_nghs[j] != ngh[0]:
                        min_dist = diff
                        top_b_ngh = ngh[0]
            if not top_b_ngh:
                top_b_ngh = patch_nghs[j]
         
        patch_x = F[patches[i]][1] - bucket_size * 2
        patch_y = F[patches[i]][0] - bucket_size * 2
        count = 0 
        if right_b_ngh:
            start_ngh_x = F[right_b_ngh][1] - bucket_size * 2
            start_ngh_y = F[right_b_ngh][0] - bucket_size * 2

            if start_ngh_x > 0 and (start_ngh_x + patch_size) < approx_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < approx_img.shape[1]):
                count += 1
                approx_img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] += approx_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]

        if left_b_ngh:
            start_ngh_x = F[left_b_ngh][1] - bucket_size * 2
            start_ngh_y = F[left_b_ngh][0] - bucket_size * 2

            if start_ngh_x > 0 and (start_ngh_x + patch_size) < approx_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < approx_img.shape[1]):
                count += 1
                approx_img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] += approx_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]
        if bottom_b_ngh:
            count += 1
            start_ngh_x = F[bottom_b_ngh][1] - bucket_size * 2
            start_ngh_y = F[bottom_b_ngh][0] - bucket_size * 2
            if start_ngh_x > 0 and (start_ngh_x + patch_size) < approx_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < approx_img.shape[1]):
                count += 1
                approx_img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] += approx_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]
        if top_b_ngh:
            start_ngh_x = F[top_b_ngh][1] - bucket_size * 2
            start_ngh_y = F[top_b_ngh][0] - bucket_size * 2
            if start_ngh_x > 0 and (start_ngh_x + patch_size) < approx_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < approx_img.shape[1]):
                count += 1
                approx_img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] += approx_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]

        approx_img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] /= count
    
    plt.subplot(221)
    plt.imshow(img, interpolation='nearest')
    plt.subplot(222)
    plt.imshow(train_img, interpolation='nearest')
    plt.subplot(223)
    plt.imshow(query_img, interpolation='nearest')
    plt.subplot(224)
    plt.imshow(approx_img, interpolation='nearest')
    #plt.show()
    plt.savefig("approx_img.png", format="png")
   
if __name__ == "__main__":
    init(sys.argv[1])
