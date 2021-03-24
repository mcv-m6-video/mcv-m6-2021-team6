import argparse
import os
import time
from distutils.util import strtobool

import cv2

from deep_sort import DeepSort
from Detectron2_2 import Detectron2
from UtilsW3 import draw_bboxes, Detection
from Reader import *
from evaluation.idf1 import MOTAcumulator
from evaluation.average_precision import mean_average_precision
#from utils.detection import Detection


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        predictions_out=[]              # Predictions that will be output of the method
                                        # for posterior analysis
        i=0                             # frame counter

        # Reading the groundtruth from given files
        reader = AICityChallengeAnnotationReader(
            path='../datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml')
        gt_file = reader.get_annotations(classes=['car'])

        y_true = []
        y_pred = []
        acc = MOTAcumulator()

        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)

            '''
            detection(frame=i,
                      id=None,
                      label='car',
                      xtl=float(det[1][0]),
                      ytl=float(det[1][1]),
                      xbr=float(det[1][2]),
                      ybr=float(det[1][3]),
                      score=det[2])
                      '''
            # Once we have predicted what do we have in the frame
            # load to the lists our GT
            y_true.append(gt_file.get(i, []))           # Remember: i == frame counter
            '''
            y_pred.append(Detection(frame=i,
                                    id=None,
                                    label='car',
                                    xtl=float(bbox_xcycwh[1][0]),
                                    ytl=float(bbox_xcycwh[1][1]),
                                    xbr=float(bbox_xcycwh[1][2]),
                                    ybr=float(bbox_xcycwh[1][3]),
                                    score=cls_conf))           # TODO: Maybe displacement between y_true (one per frame) & bbox. What happen if not same length? [Shifted]
            '''
            y_pred.append(bbox_xcycwh)
            ypred2=[]
            ypred2.append(Detection(i,None,'car',bbox_xcycwh[1][0],bbox_xcycwh[1][1],bbox_xcycwh[1][2],bbox_xcycwh[1][3],cls_conf))
            acc.update(y_true[-1], ypred2[-1])
            if bbox_xcycwh is not None:
                # select class CAR....
                # in Detectron is classified by class 2. Note that other similar like :: truck can be of interest
                mask = cls_ids == 2
                try:
                    bbox_xcycwh = bbox_xcycwh[mask]
                    # Before making it 20% bigger:
                    predictions_out.append(bbox_xcycwh)     # Save the predicted bbox
                    bbox_xcycwh[:, 3:] *= 1.2

                    cls_conf = cls_conf[mask]
                    outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        im = draw_bboxes(im, bbox_xyxy, identities)
                except:
                    print("No elements have been found - passing")
            end = time.time()
            print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(im)
                #cv2.imwrite(self.args.save_path, im)

            # before leaving the loop
            i+=1
        # exit(0)

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
        summary = acc.compute()
        print(f"AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, IDF1: {summary['idf1']['acc']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()