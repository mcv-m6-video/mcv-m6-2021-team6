import matplotlib as plt
from utilsw2 import *
from Reader import *
import cv2
from Adapted_voc_evaluation import *
path_to_video = '../datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = '../datasets/frames/'
results_path = '../Results/Task1_1'
path_to_roi = '../datasets/AICity_data/train/S03/c010/roi.jpg'

def task1_1(result_path, path_video, save_frames, color_space=cv2.COLOR_BGR2GRAY):

    if(save_frames):
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

    # this is very time consuming, we should avoid comuting it more than once.
    mu, sigma = GetGaussianModel(path_to_frames, video_n_frames,color_space)

    det_bb = remove_bg(mu,
                       sigma,
                       2,
                       path_to_frames,
                       int(video_n_frames * 0.25),
                       int(video_n_frames*0.25 +100),
                       animation=True,
                       color_space=color_space)

    reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/train/S03/c010/gt/gt.txt',initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames*0.25 +100))
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    bb_gt = []
    # for frame in gt.keys():
    for frame in range(int(video_n_frames * 0.25),  int(video_n_frames*0.25 +100)):
        annotations = gt.get(frame, [])
        bb_gt.append(annotations)


    ap, prec, rec = mean_average_precision(bb_gt , det_bb)
    print(ap)

if __name__ == '__main__':
    task1_1(results_path, path_to_video, save_frames = False)

