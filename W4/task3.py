import cv2
import os
import glob
import numpy as np
import itertools

path_to_frames = '../datasets/frames/'
path_to_video = '../datasets/AICity_data/train/S03/c010/vdo.avi'

lk_params = dict( winSize  = (10, 10),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 3000,
                       qualityLevel = 0.5,
                       minDistance = 3,
                       blockSize = 3)

# 1st Source tested.
# It was not working by default,
# and maybe needs to be readed the code form the 24min video
# https://pysource.com/2018/05/14/optical-flow-with-lucas-kanade-method-opencv-3-4-with-python-3-tutorial-31/

# SOURCE: https://gist.github.com/jayrambhia/3295631
def lk():
    cam = cv2.VideoCapture(0)
    _, img = cam.read()
    oldg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bb = [200, 200, 300, 300]
    old_pts = []
    while True:
        try:
            _, img = cam.read()
            img1 = img[bb[0]:bb[2], bb[1]:bb[3]]
            # print img1.shape
            g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # print g.shape
            pt = cv2.goodFeaturesToTrack(g, **feature_params)
            #for i in range(len(pt)):
            #    pt[i][0][0] = pt[i][0][0] + bb[0]
            #    pt[i][0][1] = pt[i][0][1] + bb[1]
            newg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            p0 = np.float32(pt).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(oldg, newg, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(newg, oldg, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_pts = []
            for pts, val in zip(p1, good):
                if val:
                    new_pts.append([pts[0][0], pts[0][1]])
                    cv2.circle(img, (pts[0][0], pts[0][1]), 5, thickness=3, color=(255, 255, 0))
            bb = predictBB(bb, old_pts, new_pts)
            if bb[0] + bb[2] >= img.shape[0]:
                bb[0] = img.shape[0] - bb[2] - 1
            if bb[1] + bb[3] >= img.shape[1]:
                bb[1] = img.shape[1] - bb[3] - 1
            old_pts = new_pts
            oldg = newg
            cv2.rectangle(img, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), color=(255, 0, 0))
            cv2.imshow("LK", img)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            break


def predictBB(bb0, pt0, pt1):
    if not pt0:
        pt0 = pt1
    dx = []
    dy = []
    for p1, p2 in zip(pt0, pt1):
        dx.append(p2[0] - p1[0])
        dy.append(p2[1] - p1[1])
    if not dx or not dy:
        return bb0
    cen_dx = round(sum(dx) / len(dx))
    cen_dy = round(sum(dy) / len(dy))
    print(cen_dx, cen_dy)
    print ("shift")
    bb = [int(bb0[0] + cen_dx), int(bb0[1] + cen_dy), int(bb0[2]), int(bb0[3])]
    if bb[0] <= 0:
        bb[0] = 1
    if bb[1] <= 0:
        bb[1] = 1
    return bb


def task3(path_to_video, save_frames=False, path_to_frames='../datasets/frames/'):
    # Reading inputs.
    # If you need to save the frames --> save_frames = True. False == reading from path
    if save_frames:
        vidcap = cv2.VideoCapture(path_to_video)
        success, image = vidcap.read()
        count = 1
        if not os.path.exists(path_to_frames):
            os.makedirs(path_to_frames)
        while success:
            cv2.imwrite(path_to_frames + "frame_{:04d}.jpg".format(count), image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1

        print("Finished saving")

    video_n_frames = len(glob.glob1(path_to_frames, "*.jpg"))

    print("Readed: ", video_n_frames, " files with extension *.jpg")


if __name__ == '__main__':
    display = False
    #task3(path_to_video, save_frames=False,
    #            path_to_frames=path_to_frames)
    lk()