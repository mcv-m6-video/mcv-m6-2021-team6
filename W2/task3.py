import os
import matplotlib as plt
from utilsw2 import *
from Reader import *
from Adapted_voc_evaluation import *
import cv2 as cv2
import numpy as np
from docopt import docopt

path_to_video = 'datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = 'datasets/frames/'
results_path = 'Results/Task3'
path_to_roi = 'datasets/AICity_data/train/S03/c010/roi.jpg'

# read arguments

def task3 (results_path, path_to_video, save_frames = True, Method ='KNN', Filter = 'yes',color_space=cv2.COLOR_BGR2GRAY):

    if (save_frames):
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

    backSub = bg_estimation(Method)

    roi = cv2.imread(path_to_roi, cv2.IMREAD_GRAYSCALE)

    reader = AnnotationReader(path='datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml',
                                  initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames * 0.25 + 100))
    gt_bb = reader.get_bboxes_per_frame(classes=['car'])

    det_bb = remove_bg3(roi,
                       Filter,
                       backSub,
                       path_to_frames,
                       int(video_n_frames * 0.25),
                       int(video_n_frames * 0.25 + 100),
                       color_space=color_space)

    bb_gt = []
    for frame in range(int(video_n_frames * 0.25), int(video_n_frames * 0.25 + 100)):
        bb_gt.append(gt_bb[frame])

    ap, prec, rec = average_precision(bb_gt, det_bb)
    print(ap)

if __name__ == '__main__':
    task3(results_path, path_to_video, save_frames = False, Method ='KNN', Filter = 'yes')
