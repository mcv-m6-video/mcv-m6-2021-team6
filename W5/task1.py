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


def task1(path_v, path_f='datasets/frames/', save_frames=False, th = 1, mask = [0, 0], op = False, cam = 'c010', wz = 0, model='yolo3', seq = 'S03'):
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
    print(video_n_frames)
    # Reading the groundtruth and getting the Bounding Boxes per frame
    reader = AICityChallengeAnnotationReader(path='datasets/aic19-track1-mtmc-train/train/{}/{}/gt/gt.txt'.format(seq, cam))
    gt_file = reader.get_annotations(classes=['car'])
    gt_bb = []
    for frame in range(int(video_n_frames * 0.25), int(video_n_frames)):
        annotations = gt_file.get(frame, [])
        gt_bb.append(annotations)

    # Reading the detections
    reader = AICityChallengeAnnotationReader(path='datasets/aic19-track1-mtmc-train/train/{}/{}/det/det_{}.txt'.format(seq, cam, model))
    det_file = reader.get_annotations(classes=['car'])

    bb_frames = []
    tracks = []
    max_track = 0
    moc_gt = []
    moc_pred = []
    # Create an accumulator that will be updated during each frame
    acc = MOTAcumulator()

    a = int(video_n_frames * 0.25)
    for frame in range(int(video_n_frames * 0.25), int(video_n_frames)):

        det = det_file.get(frame, [])
        img = cv2.imread(path_to_frames + ('/frame_{:04d}.jpg'.format(frame + 1)))
        img0 = cv2.imread(path_to_frames + ('/frame_{:04d}.jpg'.format(frame + 1)), 0)

        if op == True:
            # Getting the optical flow
            if a == int(video_n_frames):
                break
            if a == int(video_n_frames * 0.25):
                flow = None
            else:
                # Dense method (Farneback) from open cv
                flow = cv2.calcOpticalFlowFarneback(previous_frame, img0, None, 0.5, 3, 30, 6, 5, 1.2, 0)
            a += 1
            previous_frame = img0.copy()

        frame_tracks = []
        for track in tracks:
            # Comparing the detections of the current frame with the previous frame and choose the better matched
            if op == True:
                matched_det, id_remove = match_bbox_flow_mov(track.last_detection(), det, flow, th)
            else:
                matched_det, id_remove = matched_bbox_mov(track.last_detection(), det, th)

            # Needs five consecutive matches (five previous frames)
            if matched_det:
                if matched_det.bbox[3] > mask[0] and matched_det.bbox[2] > mask[1]:
                    track.buffer += 1
                    det.remove(matched_det)
                    if track.buffer >= 4:
                        if id_remove == False:
                            track.add_detection(matched_det)
                            frame_tracks.append(track)
                else:
                    tracks.remove(track)
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
        for new_bb in det:
            if new_bb.bbox[3] > mask[0] and new_bb.bbox[2] > mask[1] and ((new_bb.bbox[3]-new_bb.bbox[1]) > wz):
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

    print('{} AP {}: '.format(model, cam), ap)
    print('{} IoU {}: '.format(model, cam), iou[0])
    print('\nAdditional metrics IDF1 {}:'.format(cam))
    print(acc.get_idf1())



if __name__ == '__main__':

    th = 0.96
    mask = [0, 0]
    wz = 100
    #seq 01
    #cam = ['c001', 'c002', 'c003', 'c004', 'c005']
    #seq 03
    #cam = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    #seq 04
    #cam = ['c016', 'c017', 'c018', 'c019', 'c020', 'c021','c022','c023','c024','c025','c026','c027','c028','c029','c030','c031','c032','c033','c034','c035','c036','c037','c038','c039','c040',]
    # model --> mask_rcnn, ssd512, yolo3
    model = 'yolo3'
    cam = ['c026','c027','c028','c029','c030','c031','c032','c033','c034','c035','c036','c037','c038','c039','c040',]
    # seq --> s01, s03, s04
    seq = 'S04'
    for c in cam:
        if c == 'c010':
            mask = [150, 0]
        elif c == 'c011':
            mask = [900, 1300]
        elif c == 'c012':
            mask = [600, 0]
        elif c == 'c014':
            mask = [150, 0]
        elif c == 'c015':
            mask = [800, 400]

        path_to_video = 'datasets/aic19-track1-mtmc-train/train/{}/{}/vdo.avi'.format(seq, c)
        path_to_frames = 'datasets/{}/'.format(c)

        task1(path_v = path_to_video, path_f = path_to_frames, save_frames=True, th = th, mask = mask, op = False, cam = c, wz = wz, model = model, seq = seq)
