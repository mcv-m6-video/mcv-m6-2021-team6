import cv2 as cv
import os
import glob
import fnmatch
import numpy as np
import time
import pickle
from numpy.linalg import norm
from scipy.stats import multivariate_normal as gaussian

# POST: Given a PATH and a file pattern (e.g: *jpg) it creates you a list of image_paths
def list_images_from_path(path, pattern):
    im_paths = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):          # fnmatch can be tricky in Win environments
                im_paths.append(os.path.join(path, name))
    print( len(im_paths), " have been found in " + path + " with a given extension: "+ pattern)
    return im_paths

# Helper to avoid getting lots of for inside the code
def create_imgs_list(paths):
    result=[]
    for path in paths:
        im = cv.imread(path)
        result.append(im)
    return result


## Method 1: Use of the median of the first 25th file
# https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/
# POST: It calculate and shows the image of the median
def method1(im_paths):
    train = []
    for index, path in enumerate(im_paths):
        if index < round(len(im_paths)/4):
            train.append(path)
            print(index)
        else:
            print('BREAK AT: ' + str(index))
            break

    imgs = create_imgs_list(train)
    medianFrame = np.median(imgs, axis=0).astype(dtype=np.uint8)

    # Display median frame
    cv.imshow('frame', medianFrame)
    cv.waitKey(0)
    cv.imwrite('../datasets/medianFrame25th.jpg')


def match(pixel, mu, sigma):
    x = pixel
    u = np.mat(mu).T
    sigma = np.mat(sigma)
    d = np.sqrt((x - u).T * sigma.I * (x - u))
    if d < 2.5:
        return True
    else:
        return False


def method2(im_paths):
    train = []
    for index, path in enumerate(im_paths):
        train.append(path)
        print("Calculating img: " + str(path))
        if index > round(len(im_paths) / 4):
            print("I'm done at: " + str(index))
            break

    #Copying values of paper
    T = round(len(im_paths) / 4)
    img = cv.imread(train[0])
    dim = (640, 360)
    img = cv.resize(img, dim)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    H, W = img.shape
    print("H = " + str(H))
    alpha = 1.0 / T
    K = 5
    D = 2.5
    cf = 0.75
    C=1

    B = np.ones((H, W), dtype=np.int)
    weight = np.array([[[1, 0, 0, 0, 0] for j in range(W)] for i in range(H)])
    mu = np.array([[[np.zeros(3) for k in range(K)] for j in range(W)] for i in range(H)])
    sd = np.array([[[225 * np.eye(3) for k in range(K)] for j in range(W)] for i in range(H)])


    # IF YOU DON'T HAVE PKL FILES, UNCOMMENT
    # Very small size
    #dim = (128,72)
    start = time.time()
    for ind, train in enumerate(train):
        # Just in case you want still less images, add here: if ind < XXX:
        img = cv.imread(train)
        #img = cv.resize(img,dim)
        print("processing img.{0}".format(ind) + " with size (H,W,C): (" + str(H) + ", " + str(W) + ", " + str(C) + ")")
        for i in range(H):
            for j in range(W):
                match_flag = -1
                for k in range(K):
                    # once match break
                    if match(img[i][j], mu[i][j][k], sd[i][j][k]):
                        match_flag = k
                        break
                x = np.array(img[i][j]).reshape(1, 3)
                if match_flag != -1:
                    # update
                    m = mu[i][j][match_flag]
                    s = sd[i][j][match_flag]
                    x = img[i][j].astype(np.float)
                    delta = x - m
                    p = gaussian.pdf(img[i][j], m, s)
                    weight[i][j] = (1 - alpha) * weight[i][j]
                    weight[i][j][match_flag] += alpha
                    mu[i][j][match_flag] += delta * p
                    sd[i][j][match_flag] += p * (np.matmul(delta, delta.T) - s)
                if match_flag == -1:
                    # replace the least probable distribution
                    w = [weight[i][j][k] for k in range(K)]
                    min_id = w.index(min(w))
                    mu[i][j][min_id] = x
                    sd[i][j][min_id] = 225 * np.eye(3)
        # sort
        for i in range(H):
            for j in range(W):
                rank = weight[i][j] * 1.0 / [norm(np.sqrt(sd[i][j][k])) for k in range(K)]
                rank_ind = [k for k in range(K)]
                rank_ind.sort(key=lambda x: -rank[x])
                weight[i][j] = weight[i][j][rank_ind]
                mu[i][j] = mu[i][j][rank_ind]
                sd[i][j] = sd[i][j][rank_ind]
                cum = 0
                for ind, order in enumerate(rank_ind):
                    cum += weight[i][j][ind]
                    if cum > cf:
                        B[i][j] = ind + 1
                        break

    end = time.time()
    
    #save B
    f = open ('B_matrix.pckl', 'wb')
    pickle.dump(B,f)
    f.close()

    f = open ('Mu_matrix.pckl', 'wb')
    pickle.dump(mu,f)
    f.close()

    f = open('SD_matrix.pckl', 'wb')
    pickle.dump(sd, f)
    f.close()

    f = open('weight_matrix.pckl', 'wb')
    pickle.dump(weight, f)
    f.close()

    print("the cost of train is {0}min".format((end - start) / 60))
    '''

    # LOAD
    with open('B_matrix.pckl', 'rb') as f:
        B= pickle.load(f)
    with open('Mu_matrix.pckl', 'rb') as f:
        mu= pickle.load(f)
    with open('SD_matrix.pckl', 'rb') as f:
        sd= pickle.load(f)
    with open('weight_matrix.pckl', 'rb') as f:
        weight= pickle.load(f)
    '''

    for ind, test in enumerate(im_paths):
        img = cv.imread(test)
        img = cv.resize(img, dim)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        result = np.array(img)
        for i in range(H):
            for j in range(W):
                for k in range(B[i][j]):
                    if match(img[i][j], mu[i][j][k], sd[i][j][k]):
                        result[i][j] = 255
                    else:
                        result[i][j] = 0
        cv.imwrite(r'./result/' + '%05d' % ind + '.png', result)


if __name__ == '__main__':
    print("option 1")
    backSub = cv.createBackgroundSubtractorMOG2()
    paths = list_images_from_path('../datasets/frames', '*.jpg')
    # Create constructor:
    #bg_remover = cv.BackgroundSubtractorMOG2(500, 16, True)
    i = 0
    '''
    for path in paths:
        if i<10:
            print(i)
            print(path)
            im = cv.imread(path)
            fgmask = backSub.apply(im)
            cv.imshow('frame', fgmask)
            c = cv.waitKey(0)
            i+=1
        else:
            break
    '''
    method2(paths)

