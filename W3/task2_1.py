import os
import cv2
from Adapted_voc_evaluation import *
from track import *
from Reader import *
import glob

path_to_video = '../datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = '../datasets/frames/'


def task2_1(path_to_video, save_frames=False, path_to_frames='../datasets/frames/', neural_network=1):
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
    reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml')
    gt_file = reader.get_annotations(classes=['car'])

    gt_bb = []
    for frame in range(int(video_n_frames * 0.25), int(video_n_frames)):
        annotations = gt_file.get(frame, [])
        gt_bb.append(annotations)

    # Reading our BB
    if neural_network == 1:
        reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/train/S03/c010/det/det.txt')
    elif neural_network == 2:
        reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/train/S03/c010/det/det_ssd512.txt')
    elif neural_network == 3:
        reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    elif neural_network == 4:
        reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/train/S03/c010/det/det_yolo3.txt')

    det_file = reader.get_annotations(classes=['car'])

    bb_frames = []
    tracks = []
    max_track = 0
    for frame in range(int(video_n_frames * 0.25), int(video_n_frames)):
        if neural_network == 1:
            det = det_file.get((frame - int(video_n_frames * 0.25)), [])
            img = cv2.imread(path_to_frames + ('/frame_{:04d}.jpg'.format(frame)))
        else:
            det = det_file.get(frame, [])
            img = cv2.imread(path_to_frames + ('/frame_{:04d}.jpg'.format(frame + 1)))

        # Getting all the detections on the frame
        frame_tracks = []

        for track in tracks:
            # Comparing the detections of the current frame with the previous frame and choose the better matched
            matched_det = matched_bbox(track.last_detection(), det)

            # Needs five consecutive matches (five previous frames)
            if matched_det:
                track.buffer += 1
                det.remove(matched_det)
                if track.buffer >= 4:
                    track.add_detection(matched_det)
                    frame_tracks.append(track)
            # Removing the cars that disappear from the frames
            if not matched_det and track.id < max_track:
                track.count = track.count + 1
                if track.count >= 4:
                    tracks.remove(track)
                    try:
                        frame_tracks.remove(track)
                    except:
                        continue

        # New track detection
        for new_bb in det:
            new_bb.id = max_track + 1
            new_track = Track(max_track + 1, [new_bb])
            tracks.append(new_track)
            max_track += 1

        frame_det = []
        for track in frame_tracks:
            det = track.last_detection()
            frame_det.append(det)
            if True:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 2)
                cv2.putText(img, str(track.id), org=(int(det.xtl),int(det.ytl)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = track.color, thickness = 2)
                for c in track.detections:
                    cv2.circle(img, c.center, 5, track.color, -1)
        if False:
            cv2.imshow('tracking detections', cv2.resize(img, (900, 600)))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        bb_frames.append(frame_det)

    ap, prec, rec = mean_average_precision(gt_bb, bb_frames)
    iou_general, iou_per_frame = compute_iou_over_time(gt_bb, bb_frames)

    frames = []
    for i in range(0, len(iou_per_frame)):
        frames.append(i)

    if neural_network == 3:
        gif = 'IOU_Mask_RCNN.gif'
        print('Mask_RCNN AP: ', ap)
        print('IOU Mask_RCNN: ', iou_general)
        visualization(frames, iou_per_frame, x_max_lim=len(frames), y_max_lim=1, n_of_frames=len(frames)
                      , speed=20, name=gif)
        # plt.plot(frames, iou_per_frame)
        # fig1 = plt.gcf()
        # plt.ylabel('IoU Mask_RCNN')
        # plt.xlabel('Frames')
        # plt.show()
        # fig1.savefig('Mask_RCNN_IoU.png')
    elif neural_network == 2:
        gif = 'IOU_SSD512.gif'
        print('SSD512 AP: ', ap)
        print('IOU SSD512: ', iou_general)
        visualization(frames, iou_per_frame, x_max_lim=len(frames), y_max_lim=1, n_of_frames=len(frames)
                      , speed=20, name=gif)
    elif neural_network == 4:
        gif = 'IOU_YOLO3.gif'
        print('YOLO3 AP: ', ap)
        print('IOU YOLO3: ', iou_general)
        visualization(frames, iou_per_frame, x_max_lim=len(frames), y_max_lim=1, n_of_frames=len(frames)
                      , speed=20, name=gif)
    elif neural_network == 1:
        gif = 'IOU_FineTuned.gif'
        print('FineTuned AP: ', ap)
        print('IOU FineTuned: ', iou_general)
        visualization(frames, iou_per_frame, x_max_lim=len(frames), y_max_lim=1, n_of_frames=len(frames)
                      , speed=20, name=gif)


cv2.destroyAllWindows()

if __name__ == '__main__':
    # Neural Network : 1 = Mask_r_cnn FineTune,  2 = SSD, 3 = Mask_r_cnn, 4 = Yolo
    for n in range(1, 5):
        task2_1(path_to_video, save_frames=False,
                path_to_frames=path_to_frames, neural_network=n)
