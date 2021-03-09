import cv2
import glob
import os
from Reader import *
from Adapted_voc_evaluation import  *
SELF_PATH = os.getcwd()
PATH_DATASET = os.path.normpath(os.path.join(os.path.dirname(__file__), '../datasets'))
PATH_RESULTS = os.path.normpath(os.path.join(os.path.dirname(__file__), '../Results'))

def task1_1(show):
    reader = Reader(PATH_DATASET, ['car'], "perFrame")
    gt = reader.get_annotations()
    det = reader.get_det("yolo")

    grouped = defaultdict(list)
    for box in gt:
        grouped[box.frame].append(box)
    ordered_gt= OrderedDict(sorted(grouped.items()))


    # if we want to replicate results
    # np.random.seed(10)

    cap = cv2.VideoCapture(PATH_DATASET+"/AICity_data/train/S03/c010/vdo.avi")
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #if noise_params['add']:
    #    noisy_gt = add_noise(gt, noise_params, num_frames)
    #    grouped_noisy_gt = group_by_frame(noisy_gt)

    for frame_id in range(num_frames):
        _, frame = cap.read()

        if show == 'gt':
            frame = draw_boxes(frame, ordered_gt[frame_id])

        if show == 'det':
            frame = draw_boxes(frame,[(i) for i in det if i.frame == frame_id], det=True)

        #if show['noisy']:
        #    frame = draw_boxes(frame, frame_id, grouped_noisy_gt[frame_id], color='r')

        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:  # press q to quit
            break

        frame_id += 1

    cv2.destroyAllWindows()

    return

def task1_2(ap=0.5):
    reader = Reader(PATH_DATASET, ['car'], "perFrame")
    gt = reader.get_annotations()
    det = reader.get_det("yolo")

    grouped = defaultdict(list)
    for box in gt:
        grouped[box.frame].append(box)
    orderedict = OrderedDict(sorted(grouped.items()))
    rec, prec, ap = voc_eval(det, orderedict, ap, is_confidence=True)
    print(ap)

    return

def draw_boxes(image, boxes, det=False):
    rgb = (0, 255, 0)
    for box in boxes:
        image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), rgb, 2)
        if det:
            cv2.putText(image, "s", (int(box.xtl), int(box.ytl) - 5), cv2.FONT_ITALIC, 0.6, rgb, 2)
    return image

if __name__ == '__main__':
    #task1_1("gt")
    task1_2()
