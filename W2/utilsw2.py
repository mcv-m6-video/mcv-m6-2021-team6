import cv2
import os
import glob
import pickle as pkl
import imageio
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from BoundingBox import *

def GetGaussianModel(frames_path, number_frames ,color_space=cv2.COLOR_BGR2GRAY):
    mu_file = f"task1_1/mu.pkl"
    sigma_file = f"task1_1/sigma.pkl"

    if os.path.isfile(mu_file) and os.path.isfile(sigma_file):
        mu = pkl.load(open(mu_file, "rb"))
        sigma = pkl.load(open(sigma_file, "rb"))
        print(f"Loading {mu_file} and {sigma_file}")
        return mu, sigma

    p25_frames = int(number_frames * 0.25)
    img = cv2.imread(frames_path + '/frame_0001.jpg')
    img = cv2.cvtColor(img, color_space)
    cv2.imshow("test", img)
    cv2.waitKey()
    rng = round(p25_frames)

    frames = []
    i = 0
    print('Reading frames...')

    for x in range(0, p25_frames):
        if i < rng:
            img = cv2.imread(frames_path + ('/frame_{:04d}.jpg'.format(i + 1)))
            frames.append(img)
            i = i + 1

    print('Done.')
    # Estimate the median of the images to substract the background
    print('Calculating mean image...')
    mu = np.mean(frames, axis=(0, -1)).astype(dtype=np.uint8)
    print('Calculating std image...')
    sigma = np.std(frames, axis=(0,-1)).astype(dtype=np.uint8)
    print('Done.')
    # cv2.imshow('median', frames_median)
    cv2.imwrite('task1_1/mu.png', mu)
    cv2.imwrite('task1_1/std.png', sigma)
    # cv2.waitKey(5000)




    #img = unsqueeze(img)

    #imga = np.zeros((p25_frames, *img.shape)).astype(np.float32)
    #print('Reading frames ')
    #for i in range(0, p25_frames):
    #    img = cv2.imread(frames_path + ('/frame_{:04d}.jpg'.format(i + 1)))
    #    imga[i, ...] = unsqueeze(cv2.cvtColor(img, color_space).astype(np.float32))

    # mean
    #print('Calculating mean .... (takes a while)')
    #mu = np.mean(imga, axis=(0, -1), dtype=np.float32)
    # variance
    #print('Calculating variance .... (takes a while)')
    #sigma = np.std(imga, axis=(0, -1), dtype=np.float32)

    #print('End')
    pkl.dump(mu, open(mu_file, "wb"))
    pkl.dump(sigma, open(sigma_file, "wb"))

    return mu, sigma

def remove_bg(
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
        img = cv2.cvtColor(img, color_space).astype(np.float32)

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
            frame = frame.sum(-1)
            max_v = frame.max()
            frame[frame == max_v] = 255
            frame[frame != 255] = 0

        #frame = np.ascontiguousarray(frame).astype("uint8")


        detected_bb += fg_segmentation_to_boxes(frame, i, img)
        if animation:
            #cv2.imshow("s", frame)
            #cv2.waitKey()
            rframe = cv2.resize(frame, (sy, sx))

            frames[c, ...] = rframe
        c += 1

    #cv2.imshow("s", frame)
    #cv2.waitKey()
    if animation:
        frames_to_gif('bg_removal_a{}_p{}_{}.gif'.format(alpha, rho, color_space), frames)

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
    detections = []
    framee = np.ascontiguousarray(frame* 255).astype(np.uint8)
    #cv2.imshow("ss", framee)
    #cv2.waitKey()
    _, contours,_ = cv2.findContours(framee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:

        (x, y, w, h) = cv2.boundingRect(contour)
        #if w > box_min_size[0] and h > box_min_size[1]:
        detections.append(BoundingBox(
                frame=int(i),
                id=int(0),
                label=cls,
                xtl=float(x),
                ytl=float(y),
                xbr=float(x+w),
                ybr=float(y+h),
                confidence=None
            )
          #  [i, cls, 0, x, y, x + w, y + h]
        )
        xy, x2y2 = (int(float(x)), int(float(y))), (int(float(x+w)), int(float(y+h)))
        color = (0, 255, 0)


        cv2.rectangle(img, xy, x2y2, color, 3)
        #print(xy)
        #print(x2y2)

    img= cv2.resize(img, (int(1920 / 2), int(1080 / 2)))

    # Show result
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections

def frames_to_gif(filename, frames):
    #frames = frames.astype('uint8')

    imageio.mimsave(filename, frames)

def animation_2bb(name, format, gt_bb, bb_cords, frame_path, fps=10, seconds=10, ini=0, width=480, height=270):
    """
    This function records a video of some frames with both the GT (green)
    and detection (blue) bounding boxes. If we have a confidence value that number
    is added on top of the bounding box.
    Input
        Name: Name of the file to save
        format: format of the file, it can be .avi or .gif (. must be included)
        gt_bb: ground truth bounding boxes in the same format as reed
        bb_cords: bounding box for the detection
    """

    # in case we have a confidence value in the detection
    if len(bb_cords) == 7:
        confid = True
    else:
        confid = False

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('./' + name + format, fourcc, float(fps), (width, height))

    lst_gt = [item for item in gt_bb]
    lst_nogt = [item for item in bb_cords]
    images = []

    for i in range(fps * seconds):
        f_val = i + ini
        frame1 = cv2.imread((frame_path + '/frame_{:04d}.jpg').format(f_val))

        args_gt = [i for i, num in enumerate(lst_gt) if num == f_val]
        for ar in args_gt:
            # Ground truth bounding box in green
            for x in gt_bb[ar]:
                cv2.rectangle(frame1, (int(x.bbox[0]), int(x.bbox[1])),
                              (int(x.bbox[2]), int(x.bbox[3])), (0, 255, 0), 2)



        args_nogt = [i for i, num in enumerate(lst_nogt) if num == f_val]
        for ar in args_nogt:
            # guessed GT in blue
            cv2.rectangle(frame1, (int(bb_cords[ar][3]), int(bb_cords[ar][4])),
                          (int(bb_cords[ar][5]), int(bb_cords[ar][6])), (255, 0, 0), 2)

            if confid:
                cv2.putText(frame1, str(bb_cords[ar][6]) + " %",
                            (int(bb_cords[i][2]), int(bb_cords[i][3]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        frame1 = cv2.resize(frame1, (width, height))

        if format == '.gif':
            images.append(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        else:
            video.write(frame1)

    if format == '.gif':
        imageio.mimsave(name + format, images)
    else:
        video.release()
