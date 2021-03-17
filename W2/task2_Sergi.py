from utilsw2 import *
from Reader import *
import cv2
import pickle
import os
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
        frames_to_gif('bg_removal_Adaptive_a{}_p{}_{}.gif'.format(alpha, rho, color_space), frames)

    return det_bb

def task2():

    mu_origin = pkl.load(open('task1_1/mu.pkl', 'rb'))
    sigma_origin = pkl.load(open('task1_1/sigma.pkl', 'rb'))

    video_n_frames = len(glob.glob1(path_to_frames, "*.jpg"))


    #alpha = [0.5, 0.8, 1, 1.5, 2, 2.3, 2.5, 3, 3.5, 4, 5, 7, 8]
    #r = [0.1, 0.2, 0.5, 1, 1.2, 1.5, 2]

    alpha_bis = [5, 0.7, 2.5]
    r_bis = [0.000001, 0.00005, 0.05, 0.7, 1]

    # Snippet of code if do you want to use in range. Used to perform the grid search
    #alpha_bis = np.arange(start=1.3, stop=10, step=0.3 )
    #r_bis = np.arange(start=0.8, stop=5, step=0.1)

    #alpha = alpha_bis
    #r = r_bis
    print("List of alpha values:" , alpha_bis)
    print("List of rho values:", r_bis)

    map = []
    for al in alpha_bis:
        a = []
        for ro in r_bis:
            '''
            # checking previous results are not better than last ones
            # This was used for assisting a human-grid search
            if len(map)>5:
                print ("MAP [-1] = " + str(map[-1]))
                print("MAP [-2] = " + str(map[-2]))
                bool1 = map[-2] > map[-1]
                bool2 = map[-3] > map[-1]
                bool3 = map[-4] > map[-1]

                if (bool1 and bool2 and bool3):
                    break
            '''
            mu = mu_origin
            sigma = sigma_origin
            print('Alpha & rho = ', al, ro)
            det_bb = remove_bg2(mu,
                               sigma,
                               al,
                               path_to_frames,
                               int(video_n_frames * 0.25),
                               int(video_n_frames * 0.25 + 100),
                               animation=False,
                               adaptive=True,
                               rho=ro,
                               color_space=cv2.COLOR_BGR2GRAY)


            reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/train/S03/c010/gt/gt.txt',initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames*0.25 +100))
            gt = reader.get_annotations(classes=['car'], only_not_parked=True)

            bb_gt = []
            # for frame in gt.keys():
            for frame in range(int(video_n_frames * 0.25),  int(video_n_frames*0.25 +100)):
                annotations = gt.get(frame, [])
                bb_gt.append(annotations)


            ap, prec, rec = mean_average_precision(bb_gt , det_bb)

            a.append(ap)
            print("El valors obtingut son: " + str(a))
        print(max(a))
        map.append(max(a))

    # SAVING:
    f = open('map_results.pckl', 'wb')
    pickle.dump(map, f)
    f.close()
    print(len(map))




if __name__ == '__main__':
    task2()