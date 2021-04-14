import argparse
import os
import time
from distutils.util import strtobool

import cv2

from deep_sort import DeepSort
from Detectron2_2 import Detectron2
from UtilsW5 import draw_bboxes


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))

        # Display arg management
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2()

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        # check if in the given parameter there's a file.
        assert os.path.isfile(self.args.vid_path), "Error: vid_path error. The given path is not a file"
        self.vdo.open(self.args.vid_path)
        # obtain width & height of video
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()

        # ROI addition
        if self.args.roi_path:
            assert os.path.isfile(self.args.roi_path), "Error: roi_path error. The given path is not a file"
            self.roi = cv2.imread(self.args.roi_path, cv2.IMREAD_GRAYSCALE)
            if self.args.display:
                cv2.imshow("ROI", self.roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)

            if bbox_xcycwh is not None:
                # select class CAR....
                # in Detectron is classified by class 2. Note that other similar like :: truck can be of interest
                mask = cls_ids == 2
                try:
                    bbox_xcycwh = bbox_xcycwh[mask]
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
            # exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_path", type=str)
    parser.add_argument("--roi_path", type=str)
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