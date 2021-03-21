from torchvision import models
import torch
import cv2
from torchvision.models import detection
from torchvision.transforms import transforms
import numpy as np
import sys
sys.path.append('..')
from W3.UtilsW3 import *
from W3.BoundingBox import *

class DetectionModel:
    def __init__(self, model, dataPath, pretained = True, limit_frames = (0, 50)):
        self.name_model = model
        self.model = self.buildTorchModel(model, pretained)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.limit_frames = limit_frames
        self.videoPath = dataPath
        self.videoCapture = cv2.VideoCapture(dataPath)
        self.resultsPath = '../results'
        self.ground_true = []
        self.predictions = []
        self.detections = {}

    def buildTorchModel(self, model, pretained):
        if (model == 'mask'):
            return detection.maskrcnn_resnet50_fpn(pretrained = pretained)

    def evaluation(self, gt):
        print('Transfer the model')
        self.model.to(self.device)
        print('Evaluating Model')
        self.model.eval()
        tensor = transforms.ToTensor()

        for i in range(self.limit_frames[0], self.limit_frames[1]):
            self.videoCapture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = self.videoCapture.read()

            print(f"Transform frame {i} to tensor")
            tensor_transforms = [tensor(img).to(self.device)]
            predictions = self.model(tensor_transforms)[0]

            joint_preds = list(zip(predictions['labels'], predictions['boxes'], predictions['scores']))
            car_det = list(filter(lambda x: x[0] == 3, joint_preds))

            boxes = [box[1].cpu().detach().numpy() for box in car_det]
            picked_boxes = np.array(boxes).astype("int")
            detections_int = [(det[0].cpu().detach().item(), list(map(int, det[1].cpu().detach().numpy())), det[2].cpu().detach().item()) for det
                              in car_det]
            detections = list(filter(lambda x: x[1] in picked_boxes, detections_int))

            self.detections[i] = []
            for det in detections:
                self.detections[i].append(BoundingBox(frame=i,
                                      id=None,
                                      label='car',
                                      xtl=float(det[1][0]),
                                      ytl=float(det[1][1]),
                                      xbr=float(det[1][2]),
                                      ybr=float(det[1][3]),
                                      score=det[2]))

            self.predictions.append(self.detections[i])
            self.ground_true.append(gt.get(i, []))
    def train(self):
        pass

    def get_metrics(self):
        ap, prec, rec = mean_average_precision(self.ground_true, self.predictions, classes=['car'])
        return ap, prec, rec

    def get_qualitative_metrics(self, gt):

        util_gt = dict()
        for i in range(self.limit_frames[0], self.limit_frames[1]):
            util_gt[i] = gt[i]
        video_iou_plot(util_gt, self.detections, video_path=self.videoPath,
                       title=f'{self.name_model} detections')

