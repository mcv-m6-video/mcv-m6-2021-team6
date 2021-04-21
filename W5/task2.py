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
        summary = mh.compute(self.acc, metrics=['idf1', 'idp', 'idr', 'precision', 'recall'], name='acc')
        return summary

def detector(image_A, image_B, detector = 'sift'):
    if detector == 'sift':
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        print(detector)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(image_A, None)
        kp2, des2 = sift.detectAndCompute(image_B, None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)

    elif detector == 'orb':
        # Initiate STAR detector
        orb = cv2.ORB_create()
        print(detector)

        # compute the descriptors with ORB
        kp1, des1 = orb.detectAndCompute(image_A, None)
        kp2, des2 = orb.detectAndCompute(image_B, None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)

    elif detector == 'surf':
        # Initiate STAR detector
        surf = cv2.xfeatures2d.SURF_create(400)
        print(detector)

        # compute the descriptors with ORB
        kp1, des1 = surf.detectAndCompute(image_A, None)
        kp2, des2 = surf.detectAndCompute(image_B, None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)

    return matches


def matches(imageA, imageB, bboxA, bboxB):

    '''
    try:
        cv2.imshow('tracking detections 1', cv2.resize(imageA, (900, 600)))
        cv2.imshow('tracking detections 2', cv2.resize(imageB, (900, 600)))
        cv2.waitKey(0)

        h1 = cv2.calcHist([imageA], [1], None, [256], [0, 256])
        h1 = cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX, -1)

        h2 = cv2.calcHist([imageB], [1], None, [256], [0, 256])
        h2 = cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX, -1)

        good = cv2.compareHist(h1, h2, 0)
    except:
        good = 0
    '''
    try:

        imageA = imageA[(int(bboxA.bbox[1])):(int(bboxA.bbox[3])),
                 (int(bboxA.bbox[0])):(int(bboxA.bbox[2]))]
        imageB = imageB[(int(bboxB.bbox[1])):(int(bboxB.bbox[3])),
                 (int(bboxB.bbox[0])):(int(bboxB.bbox[2]))]
        imageA = cv2.resize(imageA, (100, 100))
        # imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2YCrCb)
        imageB = cv2.resize(imageB, (100, 100))
        # imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2YCrCb)

        '''cv2.imshow('tracking detections 1', cv2.resize(imageA, (900, 600)))
        cv2.imshow('tracking detections 2', cv2.resize(imageB, (900, 600)))
        cv2.waitKey(0)'''

        matches = detector(imageA, imageB, detector='sift')
        '''# Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(imageA, None)
        kp2, des2 = sift.detectAndCompute(imageB, None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(des1, des2, k=2)'''
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append([m])
    except:
        good = []

    return good


def task2(path_to_frames2,save_frames=False, th = 1, mask = [0, 0],cam = ['c010', 'c011', 'c012'], op = False, wz = 0, model='yolo3', seq = 'S03'):
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
    # Reading the groundtruth and getting the Bounding Boxes per frame cam1
    reader1 = AICityChallengeAnnotationReader(path='../datasets/aic19-track1-mtmc-train/train/{}/{}/gt/gt.txt'.format(seq, cam[0]))
    gt_file1 = reader1.get_annotations(classes=['car'])
    gt_bb1 = []
    for frame in range(int(500), int(video_n_frames)):
        annotations = gt_file1.get(frame, [])
        gt_bb1.append(annotations)

    # Reading the groundtruth and getting the Bounding Boxes per frame cam1
    reader2 = AICityChallengeAnnotationReader(path='../datasets/aic19-track1-mtmc-train/train/{}/{}/gt/gt.txt'.format(seq, cam[1]))
    gt_file2 = reader1.get_annotations(classes=['car'])
    gt_bb2 = []
    for frame in range(int(500), int(video_n_frames)):
        annotations = gt_file2.get(frame, [])
        gt_bb2.append(annotations)

    # Reading the detections cam1
    reader1 = AICityChallengeAnnotationReader(path='../datasets/aic19-track1-mtmc-train/train/{}/{}/mtsc/mtsc_tc_{}.txt'.format(seq, cam[0], model))
    det_file1 = reader1.get_annotations(classes=['car'])

    # Reading the detections cam2
    reader2 = AICityChallengeAnnotationReader(path='../datasets/aic19-track1-mtmc-train/train/{}/{}/mtsc/mtsc_tc_{}.txt'.format(seq, cam[1], model))
    det_file2 = reader2.get_annotations(classes=['car'])

    bb_frames1, bb_frames2 = [], []
    tracks1, tracks2 = [], []
    max_track = 0
    moc_gt1, moc_gt2 = [], []
    moc_pred1, moc_pred2 = [], []
    # Create an accumulator that will be updated during each frame
    acc1 = MOTAcumulator()
    acc2 = MOTAcumulator()
    track1_tot, track2_tot = [],[]

    rng = [len(os.listdir(path_to_frames)), len(os.listdir(path_to_frames2))]

    a = min(rng)

    for frame in range(int(0), a):

        det1 = det_file1.get(frame, [])
        path1 = path_to_frames + ('/frame_{:04d}.jpg'.format(frame + 1))
        img1 = cv2.imread(path1)

        det2 = det_file2.get(frame, [])
        path2 = path_to_frames2 + ('/frame_{:04d}.jpg'.format(frame + 1))
        img2 = cv2.imread(path2)

        frame_tracks1, frame_tracks2 = [], []
        for track in tracks1:
            # Comparing the detections of the current frame with the previous frame and choose the better matched
            matched_det1, id_remove1 = matched_bbox_mov(track.last_detection(), det1, th)

            if matched_det1:
                if (matched_det1.bbox[3] > mask[0]\
                    and matched_det1.bbox[2] > mask[1]) \
                        and (matched_det1.bbox[3] - matched_det1.bbox[1]) > wz[0] \
                        and (matched_det1.bbox[2] - matched_det1.bbox[0]) > wz[0]:

                    track.add_detection(matched_det1)
                    if track.buffer == 0:
                        track1_tot.append([track.id, matched_det1, img1])
                    track.buffer += 1
                    det1.remove(matched_det1)
                    if track.buffer == 4:
                        for i, t in enumerate(track1_tot):
                            if t[0] == track.id:
                                track1_tot[i] = [t[0], matched_det1, img1]
                        max_good = []
                        for t in track2_tot:
                            good = matches(img1, t[2], matched_det1, t[1])
                            print(len(good))
                            if len(good) > len(max_good):
                                max_good = good
                                save_t = t
                        if len(max_good) > 25:
                            remove_id = track.id
                            track.id = save_t[0]
                            for i, t in enumerate(track2_tot):
                                if t[0] == save_t[0]:
                                    track2_tot.pop(i)
                            for i, t in enumerate(track1_tot):
                                if t[0] == remove_id:
                                    track1_tot.pop(i)
                    if track.buffer > 4:
                        if id_remove1 == False:
                            track.add_detection(matched_det1)
                            frame_tracks1.append(track)
                else:
                    tracks1.remove(track)
            # Removing the cars that disappear from the frames
            if not matched_det1 and track.id < max_track:
                track.count = track.count + 1
                if track.count >= 10:
                    tracks1.remove(track)
                    try:
                        frame_tracks1.remove(track)
                    except:
                        continue

        # New track detection
        for new_bb in det1:
            if (new_bb.bbox[3] > mask[0] and new_bb.bbox[2] > mask[1])\
                    and (new_bb.bbox[3]-new_bb.bbox[1]) > wz[0] \
                    and (new_bb.bbox[3]-new_bb.bbox[1]) < wz[1] \
                    and (new_bb.bbox[2] - new_bb.bbox[0]) > wz[0] \
                    and (new_bb.bbox[2] - new_bb.bbox[0]) < wz[1]:
                new_bb.id = max_track + 1
                new_track = Track(max_track + 1, [new_bb])
                tracks1.append(new_track)

                max_track += 1

        for track in tracks2:
            # Comparing the detections of the current frame with the previous frame and choose the better matched
            matched_det2, id_remove2 = matched_bbox_mov(track.last_detection(), det2, th)

            if matched_det2:
                if (matched_det2.bbox[3] > mask[2] \
                    and matched_det2.bbox[2] > mask[3]) \
                        and (matched_det2.bbox[3] - matched_det2.bbox[1]) > wz[2] \
                        and (matched_det2.bbox[2] - matched_det2.bbox[0]) > wz[2]:

                    track.add_detection(matched_det2)
                    if track.buffer == 0:
                        track2_tot.append([track.id, matched_det2, img2])

                    track.buffer += 1
                    det2.remove(matched_det2)
                    if track.buffer == 4:
                        for i, t in enumerate(track2_tot):
                            if t[0] == track.id:
                                track2_tot[i] = [t[0], matched_det2, img2]
                        max_good = []
                        for t in track1_tot:
                            good = matches(t[2], img2, t[1], matched_det2)
                            print(len(good))
                            if len(good) > len(max_good):
                                max_good = good
                                save_t = t
                        if len(max_good) > 25:
                            remove_id2 = track.id
                            track.id = save_t[0]
                            for i, t in enumerate(track1_tot):
                                if t[0] == save_t[0]:
                                    track1_tot.pop(i)
                            for i, t in enumerate(track2_tot):
                                if t[0] == remove_id2:
                                    track2_tot.pop(i)
                    if track.buffer > 4:
                        if id_remove2 == False:
                            track.add_detection(matched_det2)
                            frame_tracks2.append(track)
                else:
                    tracks2.remove(track)
            # Removing the cars that disappear from the frames
            if not matched_det2 and track.id < max_track:
                track.count = track.count + 1
                if track.count >= 10:
                    tracks2.remove(track)
                    try:
                        frame_tracks2.remove(track)
                    except:
                        continue

        # New track detection
        for new_bb in det2:
            if (new_bb.bbox[3] > mask[2] and new_bb.bbox[2] > mask[3])\
                    and (new_bb.bbox[3]-new_bb.bbox[1]) > wz[2] \
                    and (new_bb.bbox[3]-new_bb.bbox[1]) < wz[3] \
                    and (new_bb.bbox[2] - new_bb.bbox[0]) > wz[2] \
                    and (new_bb.bbox[2] - new_bb.bbox[0]) < wz[3]:
                new_bb.id = max_track + 1
                new_track = Track(max_track + 1, [new_bb])
                tracks2.append(new_track)

                max_track += 1

        frame_det1, frame_det2 = [], []
        for track in frame_tracks1:
            det = track.last_detection()
            frame_det1.append(det)
            if True:
                cv2.rectangle(img1, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 4)
                cv2.putText(img1, str(track.id), org=(int(det.xtl), int(det.ytl)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=4, color=(256, 256, 256), thickness=4)
                for c in track.detections:
                    cv2.circle(img1, c.center, 5, track.color, -1)
        for track in frame_tracks2:
            det = track.last_detection()
            frame_det2.append(det)
            if True:
                cv2.rectangle(img2, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 4)
                cv2.putText(img2, str(track.id), org=(int(det.xtl), int(det.ytl)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=4, color=(256, 256, 256), thickness=4)
                for c in track.detections:
                    cv2.circle(img2, c.center, 5, track.color, -1)
        # IDF1
        moc_pred1.append(frame_det1)
        moc_gt1.append(gt_file1.get(frame, []))
        acc1.update(moc_gt1[-1], moc_pred1[-1])
        # IDF1
        moc_pred2.append(frame_det2)
        moc_gt2.append(gt_file2.get(frame, []))
        acc2.update(moc_gt2[-1], moc_pred2[-1])

        if True:
            a = cv2.resize(img1, (700, 400))
            b = cv2.resize(img2, (700, 400))
            #cv2.imshow('tracking detections 1', cv2.resize(img1, (700, 400)))
            #cv2.imshow('tracking detections 2', cv2.resize(img2, (700, 400)))
            total = np.concatenate((a, b), axis=1)
            cv2.imshow('Tracking', cv2.resize(total, (1000, 500)))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        bb_frames1.append(frame_det1)

    iou = compute_iou_over_time(gt_bb1, bb_frames1)

    ap, prec, rec = mean_average_precision(gt_bb1, bb_frames1)

    metrics = acc1.get_metrics()
    print('IDF1: ', metrics.idf1['acc'])
    print('IDP: ', metrics.idp['acc'])
    print('IDR: ', metrics.idr['acc'])
    print('Precision: ', metrics.precision['acc'])
    print('Recall: ', metrics.recall['acc'])




    print('{} AP {}: '.format(model, cam), ap)
    print('{} IoU {}: '.format(model, cam), iou[0])
    '''print('\nAdditional metrics IDF1 {}:'.format(cam))
    print(acc1.get_idf1())'''

if __name__ == '__main__':

        th = [0.96]
        model = 'yolo3'
        cam = ['c002', 'c001']
        #cam = ['c039', 'c040']
        #cam = ['c010', 'c011']
        seq = 'S01'
        #seq = 'S03
        #seq = 'S04'
        
        '''
        'c010':
            mask = [150, 0]
            wz = [75, 1500]
        'c011':
            mask = [800, 1300]
            wz = [100, 1500]
        'c012':
            mask = [600, 0]
            wz = [75, 2000]
        'c013':
            mask = [100, 0]
            wz = [70, 1100]
        'c014':
            mask = [150, 900]
            wz = [75, 1500]
        'c015':
            mask = [850, 500]
        '''
        #cam1 --> mask[0],mask[1]
        #cam2 --> mask[2], mask[3]
        #same wz
        #mask = [1920, 700]
        # Si seq =! S03 mask = [0,0,0,0]
        mask = [150, 500, 800, 1300]
        wz = [75, 1500, 100, 1500]

        path_to_video = '../datasets/aic19-track1-mtmc-train/train/{}/{}/vdo.avi'.format(seq, cam[0])
        path_to_frames = '../datasets/{}/'.format(cam[0])
        path_to_frames2 = '../datasets/{}/'.format(cam[1])

        for t in th:
            task2(path_to_frames2= path_to_frames2,save_frames=False, th=t, mask=[0, 0, 0, 0], op=False, cam=cam, wz=wz, model=model, seq=seq)
