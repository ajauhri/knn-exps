#! /usr/bin/env python
from __future__ import division
import sys
from scipy.misc import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
from vlfeat.phow import vl_phow
from random import randrange
import random
import math

import lsh.lsh as lsh
from extras.helper import debug
from naive import knn_naive

def make_border(img, x, y, size, color, w=2):
    img[x:x+w, y:y+size] = color
    img[x:x+size, y:y+w] = color
    img[x+size:x + size+w, y:y+size] = color
    img[x:x+size, y + size:y+size+w] = color

def init(fname):
    random.seed(8)
    patch_size = 12
    bucket_size = 3
    step_size = 12
    n_patches = 20 
    img = imread(fname)
    train_img = img.copy()
    query_img = img.copy()
    nn_img = img.copy()
    (width, height) = img.shape[:2]
    debug('img height = %d, img width = %d' % (height, width))
    
    (F, D) = vl_phow(train_img, sizes = [bucket_size], step = step_size, color = 'hsv')
    # darken some patches of images
    patches = []
    for i in range(n_patches):
        while True:
            patch_i = randrange(0, F.shape[0])
            start_x = F[patch_i][1] - (bucket_size) * 2
            start_y = F[patch_i][0] - (bucket_size) * 2
            if start_x >= 0 and start_y >= 0 and start_x + patch_size < width and start_y + patch_size < height:
                break
        print patch_i 
        train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] = [0, 0, 0]
        patches.append(patch_i)
        
    
    (F, X) = vl_phow(train_img, sizes = [bucket_size], step = step_size, color = 'rgb')
    trimmed_width = width - 2 * ((step_size) * 1/4)
    limit = int(math.ceil(trimmed_width / step_size)) 
    print F[:limit,:2] 
    print F[limit: 2*limit, :2]
    # necessary since vl_phow returns uint8 which is problematic for calculating norms
    X = X.astype(float)

    # get 4 nghs for each patch
    patch_nghs = []
    for patch_i in patches:
        patch_x = F[patch_i][1] - (bucket_size) * 2 
        patch_y = F[patch_i][0] - (bucket_size) * 2 

        center = patch_i + limit
        if center < F.shape[0]:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if start_x == patch_x and start_y - patch_y == patch_size:
                make_border(query_img, start_x, start_y, patch_size, [10, 255, 1])
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found right neigbour'
        else:
            patch_nghs.append(-1)
            print 'not found right neigbour'

        center = patch_i - limit
        if center >= 0:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if start_x == patch_x and patch_y - start_y == patch_size:
                make_border(query_img, start_x, start_y, patch_size, [255, 1, 10])
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found left neigbour'
        else:
            patch_nghs.append(-1)
            print 'not found left neigbour'

        center = patch_i + 1 
        if center < F.shape[0]:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if start_x - patch_x == patch_size and start_y == patch_y:
                make_border(query_img, start_x, start_y, patch_size, [239, 255, 0])
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found bottom neigbour'
        else:
            patch_nghs.append(-1)
            print 'not found bottom neigbour'

        center = patch_i - 1 
        if center >= 0:
            start_x = F[center][1] - bucket_size * 2
            start_y = F[center][0] - bucket_size * 2
            if patch_x - start_x == patch_size and start_y == patch_y:
                make_border(query_img, start_x, start_y, patch_size, [17, 0, 255])
                patch_nghs.append(center)
            else:
                patch_nghs.append(-1)
                print 'not found top neighbour'
        else:
            patch_nghs.append(-1)
            print 'not found top neighbour'
    res_img = train_img.copy() 

    Q = np.vstack([X[patch_i] if patch_i != -1 else np.ones((1, X.shape[1])) for patch_i in patch_nghs])

    ones = np.ones((1, X.shape[1]))
    '''
    for q in Q:
        if not (ones == q).all():
            res = knn_naive.knn_naive(10, q, X)
            print np.average(res)
    assert(len(patch_nghs) == n_patches*4)
    '''

    nghs = lsh.start(X, Q, float(850))
    for i in range(len(patches)):
        bottom_b_ngh = None
        top_b_ngh = None
        left_b_ngh = None
        right_b_ngh = None

        j = i*4
        min_angle = 3*np.pi 
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                d_y = F[patch_nghs[j]][1] - F[ngh[0]][1]
                d_x = F[patch_nghs[j]][0] - F[ngh[0]][0]
                ang = np.arctan2(d_y, d_x)
                if F[ngh[0]][0] < F[patch_nghs[j]][0] and np.fabs(np.pi - ang) < min_angle and patch_nghs[j] != ngh[0]:
                    min_angle = np.abs(np.pi - ang)
                    right_b_ngh = (ngh[0], min_angle)

        j = j+1
        min_angle = 3*np.pi 
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                d_y = F[patch_nghs[j]][1] - F[ngh[0]][1]
                d_x = F[patch_nghs[j]][0] - F[ngh[0]][0]
                ang = np.arctan2(d_y, d_x)
                if F[ngh[0]][0] > F[patch_nghs[j]][0] and np.fabs(ang) < min_angle and ngh[0] != patch_nghs[j]:
                    min_angle = np.fabs(ang)
                    left_b_ngh = (ngh[0], min_angle)

        j = j+1
        min_angle = 3*np.pi 
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                d_y = F[patch_nghs[j]][1] - F[ngh[0]][1]
                d_x = F[patch_nghs[j]][0] - F[ngh[0]][0]
                ang = np.arctan2(d_y, d_x)
                if F[ngh[0]][1] < F[patch_nghs[j]][1] and np.fabs(np.pi/2 - ang) < min_angle and patch_nghs[j] != ngh[0] :
                    min_angle = np.fabs(np.pi/2 - ang)
                    bottom_b_ngh = (ngh[0], min_angle)

        j = j+1
        min_angle = 3*np.pi 
        if not (ones == Q[j]).all():
            for ngh in nghs[j]:
                d_y = F[patch_nghs[j]][1] - F[ngh[0]][1]
                d_x = F[patch_nghs[j]][0] - F[ngh[0]][0]
                ang = np.arctan2(d_y, d_x)
                if F[ngh[0]][1] > F[patch_nghs[j]][1] and np.fabs((np.pi*3/2) - ang) < min_angle and patch_nghs[j] != ngh[0]:
                    min_angle = np.fabs((np.pi*3/2) - ang)
                    top_b_ngh = (ngh[0], min_angle)
         
         
        nn_img = train_img.copy()
        patch_x = F[patches[i]][1] - bucket_size * 2
        patch_y = F[patches[i]][0] - bucket_size * 2
        best_ngh = (-1, 0)
        if right_b_ngh:
            start_ngh_x = F[right_b_ngh[0]][1] - bucket_size * 2
            start_ngh_y = F[right_b_ngh[0]][0] - bucket_size * 2
            if start_ngh_x > 0 and (start_ngh_x + patch_size) < res_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < res_img.shape[1]):
                if best_ngh[0] == -1 or right_b_ngh[1] < best_ngh[1]:
                    print 'r',best_ngh[1], right_b_ngh[1]
                    best_ngh = right_b_ngh
                make_border(nn_img, start_ngh_x, start_ngh_y, patch_size, [10, 255, 1])

        if left_b_ngh:
            start_ngh_x = F[left_b_ngh[0]][1] - bucket_size * 2
            start_ngh_y = F[left_b_ngh[0]][0] - bucket_size * 2
            if start_ngh_x > 0 and (start_ngh_x + patch_size) < res_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < res_img.shape[1]):
                if best_ngh[0] == -1 or left_b_ngh[1] < best_ngh[1]:
                    print 'l',best_ngh[1], left_b_ngh[1]
                    best_ngh = left_b_ngh
                make_border(nn_img, start_ngh_x, start_ngh_y, patch_size, [255, 1, 10])
        
        if bottom_b_ngh:
            start_ngh_x = F[bottom_b_ngh[0]][1] - bucket_size * 2
            start_ngh_y = F[bottom_b_ngh[0]][0] - bucket_size * 2
            if start_ngh_x > 0 and (start_ngh_x + patch_size) < res_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < res_img.shape[1]):
                if best_ngh[0] == -1 or bottom_b_ngh[1] < best_ngh[1]:
                    print 'b',best_ngh[1], bottom_b_ngh[1]
                    best_ngh = bottom_b_ngh
                make_border(nn_img, start_ngh_x, start_ngh_y, patch_size, [239, 255, 0])
        
        if top_b_ngh:
            start_ngh_x = F[top_b_ngh[0]][1] - bucket_size * 2
            start_ngh_y = F[top_b_ngh[0]][0] - bucket_size * 2
            if start_ngh_x > 0 and (start_ngh_x + patch_size) < res_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size < res_img.shape[1]):
                if best_ngh[0] == -1 or top_b_ngh[1] < best_ngh[1]:
                    print 't',best_ngh[1], top_b_ngh[1]
                    best_ngh = top_b_ngh
                make_border(nn_img, start_ngh_x, start_ngh_y, patch_size, [17, 0, 255])
        
        if best_ngh:
            start_ngh_x = F[best_ngh[0]][1] - bucket_size * 2
            start_ngh_y = F[best_ngh[0]][0] - bucket_size * 2
            res_img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = res_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]
            #make_border(nn_img, start_ngh_x, start_ngh_y, patch_size, [247, 247, 347])
            make_border(nn_img, patch_x, patch_y, patch_size, [175, 1, 1])


            plt.subplot(221)
            plt.imshow(res_img, interpolation='nearest')
            plt.subplot(222)
            plt.imshow(train_img, interpolation='nearest')
            plt.subplot(223)
            plt.imshow(query_img, interpolation='nearest')
            plt.subplot(224)
            plt.imshow(nn_img, interpolation='nearest')
            plt.show()
    '''
    plt.subplot(221)
    plt.imshow(res_img, interpolation='nearest')
    plt.subplot(222)
    plt.imshow(train_img, interpolation='nearest')
    plt.subplot(223)
    plt.imshow(query_img, interpolation='nearest')
    plt.subplot(224)
    plt.imshow(nn_img, interpolation='nearest')
    #plt.show()
    plt.savefig("res_img.png", format="png")
    '''
   
if __name__ == "__main__":
    init(sys.argv[1])
