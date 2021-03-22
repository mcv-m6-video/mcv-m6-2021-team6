import os
import cv2 as cv2
from Adapted_voc_evaluation import *
from track import *
from Reader import *
import glob

path_to_video = 'datasets/AICity_data/train/S03/c010/vdo.avi'
path_to_frames = 'datasets/frames/'

def task2_1(path_to_video, save_frames = False, path_to_frames = 'datasets/frames/'):
    # Reading inputs.
    #If you need to read the frames --> save_frames = True
    if (save_frames):
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
    reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml',initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames))
    gt_file = reader.get_annotations(classes=['car'])
    # Reading our BB
    reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/train/S03/c010/det/det_yolo3.txt',initFrame=int(video_n_frames * 0.25), finalFrame=int(video_n_frames))
    det_file = reader.get_annotations(classes=['car'])

    tracks = []
    max_track = 0
    i = 0
    for frame in det_file.keys():
        img = cv2.imread(path_to_frames + ('/frame_{:04d}.jpg'.format(i + 1)))
        i = i + 1
        # Getting all the detections on the frame
        det = det_file.get(frame, [])
        frame_tracks = []

        for track in tracks:
            #Comparing the detections of the current frame with the previous frame and choose the better matched
            matched_det = matched_bbox(track.last_detection(), det)

            #Needs five consecutive matches (five previous frames)
            if matched_det:
                track.buffer += 1
                det.remove(matched_det)
                if track.buffer > 4:
                    track.add_detection(matched_det)
                    frame_tracks.append(track)
            #Removing the cars that disappear from the frames
            if not matched_det and track.id < max_track:
                track.count = track.count + 1
                if track.count > 5:
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
            frame_tracks.append(new_track)
            max_track += 1

        frame_det = []
        for track in frame_tracks:
            det = track.last_detection()
            frame_det.append(det)
            if True:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 2)
                for c in track.detections:
                    cv2.circle(img, c.center, 5, track.color, -1)

        cv2.imshow('tracking detections', cv2.resize(img, (900, 600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    task2_1(path_to_video, save_frames = False, path_to_frames = path_to_frames)