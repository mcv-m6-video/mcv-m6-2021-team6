import cv2
import os
import glob
import pickle as pkl
import imageio
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from BoundingBox import *

def GetGaussianModel(frames_path, number_frames ,color_space=cv2.COLOR_BGR2GRAY, mu_file = f"task1_1/mu.pkl",sigma_file=  f"task1_1/sigma.pkl"):


    if os.path.isfile(mu_file) and os.path.isfile(sigma_file):
        img = cv2.imread(frames_path + '/frame_0001.jpg')
        img = cv2.cvtColor(img, color_space)
        cv2.imwrite(f"imagenamespace_{color_space}.png", img)
        mu = pkl.load(open(mu_file, "rb"))
        sigma = pkl.load(open(sigma_file, "rb"))
        print(f"Loading {mu_file} and {sigma_file}")
        return mu, sigma

    p25_frames = int(number_frames * 0.25)
    img = cv2.imread(frames_path + '/frame_0001.jpg')
    img = cv2.cvtColor(img, color_space)
    cv2.imwrite(f"imagenamespace_{color_space}", img)
    #cv2.waitKey()

    img = np.expand_dims(img, -1)

    rng = round(p25_frames)

    frames = []
    i = 0
    print('Reading frames...')
    imga = np.zeros((p25_frames, *img.shape)).astype(np.float32)
    for x in range(0, p25_frames):
        if i < rng:
            img = cv2.imread(frames_path + ('/frame_{:04d}.jpg'.format(i + 1)))
            frames.append(img)
            imga[i, ...] = np.expand_dims(cv2.cvtColor(img, color_space).astype(np.float32),-1)
            i = i + 1

    print('Done.')
    # Estimate the median of the images to substract the background
    print('Calculating mean image...')
    mu = np.mean(imga, axis=(0, -1), dtype=np.float32)
    print('Calculating std image...')
    sigma = np.std(imga, axis=(0,-1), dtype=np.float32)

    pkl.dump(mu, open(mu_file, "wb"))
    pkl.dump(sigma, open(sigma_file, "wb"))

    return mu, sigma

def remove_background(
        mu,
        sigma,
        alpha,
        frames_path,
        initial_frame,
        final_frame,
        animation=False,
        rho=0.2,
        color_space=cv2.COLOR_BGR2GRAY, channels=(0)):

    c = 0
    detected_bb = []
    for i in tqdm(range(initial_frame, final_frame)):
        # read image
        img = cv2.imread(frames_path + ('/frame_{:04d}.jpg'.format(i + 1)))

        img = cv2.cvtColor(img, color_space)

        if i == initial_frame and animation:
            sx, sy = np.int(np.shape(img)[0] / 4), np.int(np.shape(img)[1] / 4)
            frames = np.zeros((final_frame - initial_frame, sx, sy))




        frame = np.zeros(np.shape(img))

        frame[np.abs(img - mu) >= alpha * (sigma + 2)] = 1
        frame[np.abs(img - mu) < alpha * (sigma + 2)] = 0

        #cv2.imshow("s", frame)
        #cv2.waitKey()
        if len(frame.shape) != 2:

            frame = frame[:, :, channels]

        frame = np.ascontiguousarray(frame)


        if animation:
            rframe = cv2.resize(frame, (sy, sx))
            frames[c, ...] = rframe

        c += 1
        detected_bb.append(fg_segmentation_to_boxes(frame, i, img))

    if animation:
        imageio.mimsave('bg_removal_a{}_p{}_{}.gif'.format(alpha, rho, color_space), frames)

    return detected_bb

def denoise_bg(frame):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))
    frame = cv2.medianBlur(frame, 7)
    filled = frame
    # Flood fill
    filled = ndimage.binary_fill_holes(filled).astype(np.uint8)
    # Erode
    filled = cv2.erode(filled, kernel1, iterations=1)
    # Dilate
    filled = cv2.dilate(filled, kernel, iterations=1)
    return (filled * 255).astype(np.uint8)


def bg_estimation(mode, **kwargs):
    if mode == 'mog':
        return cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    if mode == 'knn':
        return cv2.createBackgroundSubtractorKNN(detectShadows=True)
    if mode == 'gmg':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if mode == 'LSBP':
        return cv2.bgsegm.createBackgroundSubtractorLSBP()

def fg_segmentation_to_boxes(frame, i,img, box_min_size=(10, 10), cls='car'):
    frame = np.ascontiguousarray(frame * 255).astype(np.uint8)
    _, contours,_ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    frame_dets = []
    foreground_mask_bbs = np.zeros(np.shape(frame))
    j = 1
    for con in contours:
        (x, y, w, h) = cv2.boundingRect(con)
        frame_dets.append(BoundingBox(int(i), None, 'car', x, y, x + w, y + h, 1))
        j = j + 1
    return frame_dets






