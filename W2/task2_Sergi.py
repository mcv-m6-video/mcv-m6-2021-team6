from utilsw2 import *
from Reader import *
import cv2
from Adapted_voc_evaluation import *

path_to_video = '../datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = '../datasets/frames/'
results_path = 'Results/Task1_1'
path_to_roi = '../datasets/AICity_data/train/S03/c010/roi.jpg'

def remove_bg2(
        mu,
        sigma,
        alpha,
        frames_path,
        initial_frame,
        final_frame,
        animation=False,
        adaptive = True,
        rho=0.2,
        color_space=cv2.COLOR_BGR2GRAY, channels=(0)):
    roi = cv2.imread('datasets/AICity_data/train/S03/c010/roi.jpg', cv2.IMREAD_GRAYSCALE)
    c = 0
    det_bb = []
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

        if adaptive:
            mu = (1 - rho) * mu + (rho * img)
            sigma = np.sqrt((rho * (img - mu) ** 2) + ((1 - rho) * sigma ** 2))

        # cv2.imshow("s", frame)
        # cv2.waitKey()
        if len(frame.shape) != 2:
            frame = frame[:, :, channels]
            frame = frame.sum(-1)
            max_v = frame.max()
            frame[frame == max_v] = 255
            frame[frame != 255] = 0
            frame = frame & roi

        # frame = np.ascontiguousarray(frame).astype("uint8")

        if animation:
            # cv2.imshow("s", frame)
            # cv2.waitKey()
            rframe = cv2.resize(frame, (sy, sx))

            frames[c, ...] = rframe
        c += 1

        det, im = fg_segmentation_to_boxes(frame, i, img)
        det_bb.append(det)

    # cv2.imshow("s", frame)
    # cv2.waitKey()
    if animation:
        frames_to_gif('bg_removal_a{}_p{}_{}.gif'.format(alpha, rho, color_space), frames)

    return det_bb

def task2():

    mu = pkl.load(open('task1_1/mu.pkl', 'rb'))
    sigma = pkl.load(open('task1_1/sigma.pkl', 'rb'))

    video_n_frames = len(glob.glob1(path_to_frames, "*.jpg"))

    det_bb = remove_bg2(mu,
                       sigma,
                       2,
                       path_to_frames,
                       int(video_n_frames * 0.25),
                       int(video_n_frames * 0.25 + 100),
                       animation=True,
                       adaptive=True,
                       color_space=cv2.COLOR_BGR2GRAY)


    reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/train/S03/c010/gt/gt.txt',initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames*0.25 +100))
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    bb_gt = []
    # for frame in gt.keys():
    for frame in range(int(video_n_frames * 0.25),  int(video_n_frames*0.25 +100)):
        annotations = gt.get(frame, [])
        bb_gt.append(annotations)


    ap, prec, rec = mean_average_precision(bb_gt , det_bb)
    print(ap)


if __name__ == '__main__':
    task2()