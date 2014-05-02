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
 
def get_best_ngh(nghs, patch_ngh):
    best_ngh = None
    min_dist = 10000
    if patch_ngh == -1:
        return None
    else:
        for ngh in nghs:
            if ngh[1] < min_dist and patch_ngh != ngh[0]:
                min_dist = ngh[1]
                best_ngh = ngh
    return best_ngh

def init(fname):
    lsh.seed()
    patch_size = 12
    bucket_size = 3
    step_size = 12
    n_patches = [5, 10, 15, 20, 25, 30]
    out = open('rms.txt', 'w')
    img = imread(fname)
    for n in n_patches:
        train_img = img.copy()
        query_img = img.copy()
        (width, height) = img.shape[:2]
        debug('img height = %d, img width = %d' % (height, width))
        
        (F, D) = vl_phow(train_img, sizes = [bucket_size], step = step_size, color = 'hsv')
        # darken <n> patches of images
        patches = []
        for i in range(n):
            while True:
                patch_i = randrange(0, F.shape[0])
                start_x = F[patch_i][1] - (bucket_size) * 2
                start_y = F[patch_i][0] - (bucket_size) * 2
                if start_x >= 0 and start_y >= 0 and start_x + patch_size < width and start_y + patch_size < height:
                    break
            train_img[start_x:start_x+patch_size, start_y:start_y+patch_size] = [0, 0, 0]
            patches.append(patch_i)
        #train_img = np.concatenate((train_img, img2), axis=0)    
        (F, X) = vl_phow(train_img, sizes = [bucket_size], step = step_size, color = 'hsv')
        
        # get the offset in the X or F matrix such each offset/steps is equivalent to a row in the image
        trimmed_width = width - 2 * ((step_size) * 1/4)
        limit = int(math.ceil(trimmed_width / step_size)) 
        
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
                    debug('not found right neigbour')
            else:
                patch_nghs.append(-1)
                debug('not found right neigbour')

            center = patch_i - limit
            if center >= 0:
                start_x = F[center][1] - bucket_size * 2
                start_y = F[center][0] - bucket_size * 2
                if start_x == patch_x and patch_y - start_y == patch_size:
                    make_border(query_img, start_x, start_y, patch_size, [255, 1, 10])
                    patch_nghs.append(center)
                else:
                    patch_nghs.append(-1)
                    debug('not found left neigbour')
            else:
                patch_nghs.append(-1)
                debug('not found left neigbour')

            center = patch_i + 1 
            if center < F.shape[0]:
                start_x = F[center][1] - bucket_size * 2
                start_y = F[center][0] - bucket_size * 2
                if start_x - patch_x == patch_size and start_y == patch_y:
                    make_border(query_img, start_x, start_y, patch_size, [239, 255, 0])
                    patch_nghs.append(center)
                else:
                    patch_nghs.append(-1)
                    debug('not found bottom neigbour')
            else:
                patch_nghs.append(-1)
                debug('not found bottom neigbour')

            center = patch_i - 1 
            if center >= 0:
                start_x = F[center][1] - bucket_size * 2
                start_y = F[center][0] - bucket_size * 2
                if patch_x - start_x == patch_size and start_y == patch_y:
                    make_border(query_img, start_x, start_y, patch_size, [17, 0, 255])
                    patch_nghs.append(center)
                else:
                    patch_nghs.append(-1)
                    debug('not found top neighbour')
            else:
                patch_nghs.append(-1)
                debug('not found top neighbour')
        
        res_img = train_img.copy() 
        Q = np.vstack([X[patch_i] if patch_i != -1 else np.ones((1, X.shape[1])) for patch_i in patch_nghs])
        nn = lsh.start(X, float(750))

        nghs = []
        for q in Q:
            if not (q == 1).all():
                nghs.append(lsh.get_ngh_struct(nn, q))
            else:
                nghs.append([])
         
        for i in range(len(patches)):
            bottom_b_ngh = None
            top_b_ngh = None
            left_b_ngh = None
            right_b_ngh = None
            
            j = i*4
            right_b_ngh = get_best_ngh(nghs[j], patch_nghs[j])
            
            j = j+1
            left_b_ngh = get_best_ngh(nghs[j], patch_nghs[j])

            j = j+1
            bottom_b_ngh = get_best_ngh(nghs[j], patch_nghs[j])

            j = j+1
            top_b_ngh = get_best_ngh(nghs[j], patch_nghs[j])
             
            patch_x = F[patches[i]][1] - bucket_size * 2
            patch_y = F[patches[i]][0] - bucket_size * 2

            def dist(t):
                return t[-1]
            
            def valid(k):
                if k >= F.shape[0]:
                    return False

                start_ngh_x = F[k][1] - bucket_size * 2
                start_ngh_y = F[k][0] - bucket_size * 2
                if k not in patches and start_ngh_x > 0 and (start_ngh_x + patch_size) < res_img.shape[0] and start_ngh_y > 0 and (start_ngh_y + patch_size) < res_img.shape[1]:
                    return True
                else:
                    return False
            
            best_ngh = (-1, 0)
            j = i*4
            if right_b_ngh:
                if valid(right_b_ngh[0] - limit):
                    right_b_ngh = (right_b_ngh[0] - limit, 
                                   np.linalg.norm(X[right_b_ngh[0] - limit] - X[patch_nghs[j]]))
                    best_ngh = right_b_ngh
                elif valid(right_b_ngh[0] + limit):
                    right_b_ngh = (right_b_ngh[0] + limit, 
                                   np.linalg.norm(X[right_b_ngh[0] + limit] - X[patch_nghs[j]]))
                    best_ngh = right_b_ngh

            j = j + 1
            if left_b_ngh:
                if valid(left_b_ngh[0] + limit):
                    left_b_ngh = (left_b_ngh[0] + limit, 
                                  np.linalg.norm(X[left_b_ngh[0] + limit] - X[patch_nghs[j]]))
                    if best_ngh[0] == -1 or left_b_ngh[1] < best_ngh[1]:
                        best_ngh = left_b_ngh
                elif valid(left_b_ngh[0] - limit):
                    left_b_ngh = (left_b_ngh[0] - limit, 
                                  np.linalg.norm(X[left_b_ngh[0] - limit] - X[patch_nghs[j]]))
                    if best_ngh[0] == -1 or left_b_ngh[1] < best_ngh[1]:
                        best_ngh = left_b_ngh
                       
            j = j + 1
            if bottom_b_ngh:
                if valid(bottom_b_ngh[0] - 1):
                    bottom_b_ngh = (bottom_b_ngh[0] - 1, 
                                    np.linalg.norm(X[bottom_b_ngh[0] - 1] - X[patch_nghs[j]]))
                    if best_ngh[0] == -1 or bottom_b_ngh[1] < best_ngh[1]:
                        best_ngh = bottom_b_ngh
                elif valid(bottom_b_ngh[0] + 1):
                    bottom_b_ngh = (bottom_b_ngh[0] + 1,
                                    np.linalg.norm(X[bottom_b_ngh[0] + 1] - X[patch_nghs[j]]))

                    if best_ngh[0] == -1 or bottom_b_ngh[1] < best_ngh[1]:
                        best_ngh = bottom_b_ngh

            j = j + 1
            if top_b_ngh:
                if valid(top_b_ngh[0] + 1):
                    top_b_ngh = (top_b_ngh[0] + 1, 
                                 np.linalg.norm(X[top_b_ngh[0] + 1] - X[patch_nghs[j]]))
                    if best_ngh[0] == -1 or top_b_ngh[1] < best_ngh[1]:
                        best_ngh = top_b_ngh
                elif valid(top_b_ngh[0] - 1):
                    top_b_ngh = (top_b_ngh[0] - 1,
                                 np.linalg.norm(X[top_b_ngh[0] - 1] - X[patch_nghs[j]]))
                    if best_ngh[0] == -1 or top_b_ngh[1] < best_ngh[1]:
                        best_ngh = top_b_ngh
            

            start_ngh_x = F[best_ngh[0]][1] - bucket_size * 2
            start_ngh_y = F[best_ngh[0]][0] - bucket_size * 2
            res_img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = train_img[start_ngh_x:start_ngh_x+patch_size, start_ngh_y:start_ngh_y+patch_size]
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_title('Approx. Image')
        ax2.set_title('Query Image')
        ax1.xaxis.set_visible(False) 
        ax1.yaxis.set_visible(False) 
        ax2.xaxis.set_visible(False) 
        ax2.yaxis.set_visible(False) 

        ax1.imshow(res_img, interpolation='nearest')
        ax2.imshow(train_img, interpolation='nearest')

        plt.savefig("res_img.eps", format="eps", dpi=1000)
        out.write("%f,%f\n" % (np.sqrt(np.mean((img - res_img)**2)), n))
   
if __name__ == "__main__":
    init(sys.argv[1])
