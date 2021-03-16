import numpy as np
import os
from utilsw2 import *
from Reader import *
from Adapted_voc_evaluation import *
path_to_video = '../datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = '../datasets/frames/'
results_path = '../Results/Task1_1'
def task4(color_space=cv2.COLOR_BGR2GRAY, mu_file = f"task1_1/mu.pkl",sigma_file=  f"task1_1/sigma.pkl"):
    video_n_frames = len(glob.glob1(path_to_frames, "*.jpg"))

    # this is very time consuming, we should avoid comuting it more than once.
    mu, sigma = GetGaussianModel(path_to_frames, video_n_frames,color_space,mu_file,sigma_file)

    lowLimit = int(video_n_frames * 0.25)
    highLimit = int(video_n_frames * 0.25 + 100)
    det_bb = remove_background(mu,
                       sigma,
                       5.4,
                       path_to_frames,
                       lowLimit,
                       highLimit,
                       animation=True,
                       color_space=color_space)

    reader = AnnotationReader(path='../datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml', initFrame=lowLimit, finalFrame=highLimit)
    gt_bb = reader.get_bboxes_per_frame(classes=['car'])
    bb_gt = []
    # for frame in gt.keys():
    for frame in range(lowLimit,  highLimit):
        bb_gt.append(gt_bb[frame])

    ap, prec, rec = mean_average_precision(bb_gt , det_bb)
    print (ap)

if __name__ == '__main__':
    colors = [ cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2YCrCb, cv2.COLOR_BGR2LAB]
    for c in colors:
        task4(c,f"task1_1/mu{str(c)}.pkl",f"task1_1/sigma{str(c)}.pkl")
        #task4(c)