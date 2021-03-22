from torchvision import models
import torch
import cv2
import random

import torchvision
from torchvision.models import detection
from torchvision.transforms import transforms
import numpy as np
from W3.BoundingBox import *
from W3.UtilsW3 import *
from object_detection.CustomTorchDataset import *
from object_detection.UtilsDetection import *
class DetectionModel:
    def __init__(self, model, dataPath, pretained = True,finetune = False, limit_frames = (0, 50)):
        self.name_model = model
        self.model = self.buildTorchModel(model, pretained,finetune=finetune)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.limit_frames = limit_frames
        self.videoPath = dataPath
        self.videoCapture = cv2.VideoCapture(dataPath)
        self.resultsPath = '../results'
        self.ground_true = []
        self.predictions = []
        self.detections = {}
        self.class_car = 3

    def buildTorchModel(self, model_name, pretained, finetune = False):
        model = None
        if (model_name == 'mask'):
            model = detection.maskrcnn_resnet50_fpn(pretrained = pretained)
        elif model_name == 'fast':
            model = detection.fasterrcnn_resnet50_fpn(pretrained=pretained)
        if finetune:
            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,4)
            model.roi_heads.mask_roi_pool = None
            model.roi_heads.mask_head = None
            model.roi_heads.mask_predictor = None
        return model

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
            car_det = list(filter(lambda x: x[0] == self.class_car, joint_preds))

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
    def train(self,k_fold = 1, num_epochs=1):
        transform = torchvision.transforms.ToTensor()
        dataset = CustomTorchDataset(gt_file= '../datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml', root_dir=self.videoPath, transform=transform)

        if k_fold == 1:
            indices = range(len(dataset))
            split = int(len(dataset) * 0.25)
            train_sampler = torch.utils.data.SubsetRandomSampler(indices[:split])
            test_sampler = torch.utils.data.SubsetRandomSampler(indices[split:])
        elif k_fold == 2:
            indices = range(len(dataset))
            indices = list(indices)
            random.shuffle(indices)
            split = int(len(dataset) * 0.25)
            train_sampler = torch.utils.data.SubsetRandomSampler(indices[:split])
            test_sampler = torch.utils.data.SubsetRandomSampler(indices[split:])

        elif k_fold == 3:
            pass

        total_size = len(dataset)
        fraction = 1 / k_fold
        seg = int(total_size * fraction)
        # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
        # index: [trll,trlr],[vall,valr],[trrl,trrr]
        for i in range(k_fold):
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = i * seg + seg
            trrl = valr
            trrr = total_size
            train_left_indices = list(range(trll, trlr))
            train_right_indices = list(range(trrl, trrr))

            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(vall, valr))

            # define training and validation data loaders
            self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=train_indices, num_workers=1,
                                                       collate_fn=collate_fn)
            self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=val_indices, num_workers=1,
                                                      collate_fn=collate_fn)
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

            y_prediction = None
            for epoch in range(num_epochs):
                train_one_epoch(self.model, optimizer, self.train_loader, self.device, epoch, print_freq=10)
                lr_scheduler.step()
                y_prediction = evaluate(self.model, self.test_loader, self.device)

            cpu_device = torch.device("cpu")
            for x in y_prediction:
                if len(x) != 0:
                    self.detections[x[0].frame] = [z for z in x]

    def collate_fn(batch):
        return tuple(zip(*batch))

    def get_metrics(self, is_cpu = True):
        ap, prec, rec = mean_average_precision(self.ground_true, self.predictions, classes=['car'], is_cpu=is_cpu)
        return ap, prec, rec

    def get_qualitative_metrics(self, gt):

        util_gt = dict()
        for i in range(self.limit_frames[0], self.limit_frames[1]):
            util_gt[i] = gt[i]
        video_iou_plot(util_gt, self.detections, video_path=self.videoPath,
                       title=f'{self.name_model} detections')

