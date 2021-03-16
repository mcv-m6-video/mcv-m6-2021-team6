import matplotlib as plt
from utilsw2 import *
from Reader import *
import cv2
from Adapted_voc_evaluation import *
path_to_video = '../datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = '../datasets/frames/'
results_path = '../Results/Task1_1'

def task1_1(save_frames, color_space=cv2.COLOR_BGR2GRAY):
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

    alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    aps7 = []
    det_bb = remove_background(mu,
                       sigma,
                       3.5,
                       path_to_frames,
                       int(video_n_frames * 0.25),
                       int(video_n_frames * 0.25 + 100),
                       animation=True,
                       color_space=color_space)

    reader = AnnotationReader(path='../datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml', initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames * 0.25 + 100))
    gt_bb = reader.get_bboxes_per_frame(classes=['car'])
    bb_gt = []
    # for frame in gt.keys():
    for frame in range(int(video_n_frames * 0.25),  int(video_n_frames * 0.25 + 100)):
        bb_gt.append(gt_bb[frame])

    ap, prec, rec = mean_average_precision(bb_gt , det_bb)
    print (ap)
    #print (prec)
    #print(rec)
    # ap = calculate_ap(det_bb, gt_bb, int(video_n_frames * 0.25), video_n_frames, mode='area')
    #animation_2bb('try_dnoise', '.gif', gt_bb, det_bb, path_to_frames, 10, 10, int(video_n_frames * 0.25),
    #              int(1920 / 4), int(1080 / 4))


    #plt.title('Median Filter')
    #    plt.plot(alphas, aps3, label = 'Window size 3')
    #    plt.plot(alphas, aps5, label = 'Window size 5')
    #plt.plot(alphas, aps7, label='Window size 7')
    #plt.xlabel(r'$\alpha$')
    #plt.ylabel('mAP')
    #plt.legend()



if __name__ == '__main__':
    task1_1( save_frames = False)