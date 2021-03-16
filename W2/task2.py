import cv2 as cv
import os
import glob
import fnmatch
import numpy as np

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

if __name__ == '__main__':
    print("option 1")
    backSub = cv.createBackgroundSubtractorMOG2()
    paths = list_images_from_path('../datasets/frames', '*.jpg')
    # Create constructor:
    bg_remover = cv.BackgroundSubtractorMOG2(500, 16, True)
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
    method1(paths)

