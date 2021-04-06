import os
import cv2
from Adapted_voc_evaluation import *
from track import *
from Reader import *
import glob
import motmetrics as mm
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

#Class from group two of 2020.
class MOTAcumulator:

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, y_true, y_pred):
        X = np.array([det.center for det in y_true])
        Y = np.array([det.center for det in y_pred])

        if len(X) > 0 and len(Y) > 0:
            dists = pairwise_distances(X, Y, metric='euclidean')
        else:
            dists = np.array([])

        self.acc.update(
            [det.id for det in y_true],
            [det.id for det in y_pred],
            dists
        )

    def get_idf1(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1'], name='acc')
        return summary['idf1']['acc']

    def get_metrics(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1', 'idp', 'precision', 'recall'], name='acc')
        return summary


path_to_video = 'datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = 'datasets/frames/'


def task2_1(path_to_video, save_frames=False, path_to_frames='datasets/frames/', neural_network=1):
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

    # Reading the groundtruth
    reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml')
    gt_file = reader.get_annotations(classes=['car'])

    gt_bb = []
    for frame in range(int(video_n_frames * 0.25), int(video_n_frames)):
        annotations = gt_file.get(frame, [])
        gt_bb.append(annotations)

    # Reading our BB
    if neural_network == 1:
        reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/train/S03/c010/det/det.txt')
    elif neural_network == 2:
        reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/train/S03/c010/det/det_ssd512.txt')
    elif neural_network == 3:
        reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    elif neural_network == 4:
        reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')

    det_file = reader.get_annotations(classes=['car'])

    bb_frames = []
    tracks = []
    max_track = 0
    moc_gt = []
    moc_pred = []
    # Create an accumulator that will be updated during each frame
    acc = MOTAcumulator()
    
    i = int(video_n_frames * 0.25)
    for frame in range(int(video_n_frames * 0.25), int(video_n_frames)):
        if neural_network == 1:
            dets = det_file.get((frame - int(video_n_frames * 0.25)), [])
            img = cv2.imread(path_to_frames + ('/frame_{:04d}.jpg'.format(frame)),0)
            mot_gt = gt_file.get((frame - int(video_n_frames * 0.25)), [])
        else:
            dets = det_file.get(frame, [])
            img = cv2.imread(path_to_frames + ('/frame_{:04d}.jpg'.format(frame + 1)),0)
            mot_gt = gt_file.get(frame, [])

        # Getting the optical flow
        frame_tracks = []
        if i == int(video_n_frames):
            break
        if i == int(video_n_frames * 0.25):
            flow = None
        else:
            # get points on which to detect the flow
            points = []
            for d in dets:
                points.append([d.xtl, d.ytl])
                points.append([d.xbr, d.ybr])
            #Dense method (Farneback) from open cv
            flow = cv2.calcOpticalFlowFarneback(previous_frame, img, None, 0.5, 3, 30, 6, 5, 1.2, 0)

        i += 1
        previous_frame = img.copy()

        for track in tracks:
            # Comparing the detections of the current frame with the previous frame and choose the better matched
            matched_det = match_bbox_flow(track.last_detection(), dets, flow)

            # Needs five consecutive matches (five previous frames)
            if matched_det:
                track.buffer += 1
                dets.remove(matched_det)
                if track.buffer >= 4:
                    track.add_detection(matched_det)
                    frame_tracks.append(track)
            # Removing the cars that disappear from the frames
            if not matched_det and track.id < max_track:
                track.count = track.count + 1
                if track.count >= 10:
                    tracks.remove(track)
                    try:
                        frame_tracks.remove(track)
                    except:
                        continue

        # New track detection
        for new_bb in dets:
            new_bb.id = max_track + 1
            new_track = Track(max_track + 1, [new_bb])
            tracks.append(new_track)
            max_track += 1


        frame_det = []
        for track in frame_tracks:
            det = track.last_detection()
            frame_det.append(det)
            if True:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 4)
                cv2.putText(img, str(track.id), org=(int(det.xtl),int(det.ytl)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = track.color, thickness = 2)
                for c in track.detections:
                    cv2.circle(img, c.center, 5, track.color, -1)

        #IDF1
        moc_pred.append(frame_det)
        moc_gt.append(gt_file.get(frame, []))
        acc.update(moc_gt[-1], moc_pred[-1])

        if True:
            cv2.imshow('tracking detections', cv2.resize(img, (900, 600)))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        bb_frames.append(frame_det)

    iou = compute_iou_over_time(gt_bb, bb_frames)

    ap, prec, rec = mean_average_precision(gt_bb, bb_frames)

    if neural_network == 1:
        print('Mask_FineTune AP: ', ap)
        print('Mask_FineTune IoU: ', iou[0])
        print('\nAdditional metrics:')
        print(acc.get_idf1())
    elif neural_network == 2:
        print('SSD512 AP: ', ap)
        print('SSD512 IoU: ', iou[0])
        print('\nAdditional metrics:')
        print(acc.get_idf1())
    elif neural_network == 3:
        print('Mask_RCNN AP: ', ap)
        print('Mask_RCNN IoU: ', iou[0])
        print('\nAdditional metrics:')
        print(acc.get_idf1())
    elif neural_network == 4:
        print('YOLO3 AP: ', ap)
        print('YOLO3 IoU: ', iou[0])
        print('\nAdditional metrics:')
        print(acc.get_idf1())

cv2.destroyAllWindows()

if __name__ == '__main__':
    # Neural Network : 1 = Mask_r_cnn FineTune,  2 = SSD, 3 = Mask_r_cnn, 4 = Yolo
    for n in range(1, 5):
        task2_1(path_to_video, save_frames=False, path_to_frames=path_to_frames, neural_network=n)
