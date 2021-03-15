import cv2 as cv
import os
import glob
import fnmatch

# POST: Given a PATH and a file pattern (e.g: *jpg) it creates you a list of image_paths
def list_images_from_path(path, pattern):
    im_paths = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):          # fnmatch can be tricky in Win environments
                im_paths.append(os.path.join(path, name))
    print( len(im_paths), " have been found in " + path + " with a given extension: "+ pattern)
    return im_paths


if __name__ == '__main__':
    print("option 1")
    backSub = cv.createBackgroundSubtractorMOG2()
    paths = list_images_from_path('../datasets/frames', '*.jpg')
    # Create constructor:
    bg_remover = cv.BackgroundSubtractorMOG2(500, 16, True)
    i = 0
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

