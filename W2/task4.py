from utilsw2 import *
from Reader import *
from Adapted_voc_evaluation import *
import glob
path_to_video = 'datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = 'datasets/frames/'
results_path = 'Results/Task1_1'


def task4(color_space=cv2.COLOR_BGR2GRAY, mu_file = f"W2/task1_1/mu.pkl",sigma_file=  f"W2/task1_1/sigma.pkl"):
    video_n_frames = len(glob.glob1(path_to_frames, "*.jpg"))

    mu, sigma = GetGaussianModel(path_to_frames, video_n_frames,color_space,mu_file,sigma_file)

    lowLimit = int(video_n_frames * 0.25)
    highLimit = int(video_n_frames)
    det_bb = remove_background(mu,
                       sigma,
                       6,
                       path_to_frames,
                       lowLimit,
                       highLimit,
                       animation=True,
                       color_space=color_space)

    reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/train/S03/c010/gt/gt.txt',initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames))
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)
    bb_gt = []
    # for frame in gt.keys():
    for frame in range(int(video_n_frames * 0.25),  int(video_n_frames)):
        annotations = gt.get(frame, [])
        bb_gt.append(annotations)

    ap, prec, rec = mean_average_precision(bb_gt , det_bb)
    print (ap)

if __name__ == '__main__':
    colors = [cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2YCrCb, cv2.COLOR_BGR2LAB]
    for c in colors:
        task4(c,f"W2/task4_1/mu{str(c)}.pkl",f"W2/task4_1/sigma{str(c)}.pkl")
