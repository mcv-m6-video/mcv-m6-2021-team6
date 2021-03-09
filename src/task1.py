import cv2
from Reader import *
from Adapted_voc_evaluation import  *
PATH_DATASET = '../datasets/AICity_data/'
PATH_RESULTS = './Results/'

def task1_1(show):
    reader = Reader(PATH_DATASET, ['car'], "perFrame")
    gt = reader.get_annotations()
    det = reader.get_det("yolo")

    grouped = defaultdict(list)
    for box in gt:
        grouped[box.frame].append(box)
    ordered_gt= OrderedDict(sorted(grouped.items()))

    cap = cv2.VideoCapture("../datasets/AICity_data/train/S03/c010/vdo.avi")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    for frame_id in range(num_frames):
        _, frame = cap.read()

        if show == 'gt':
            frame = draw_boxes(frame, ordered_gt[frame_id])

        if show == 'det':
            frame = draw_boxes(frame,[(i) for i in det if i.frame == frame_id], det=True)

        cv2.imshow('frame', frame)
        if cv2.waitKey() == 113:
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
    task1_1("gt")
    task1_2()
