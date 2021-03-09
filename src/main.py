import cv2
import glob
import os
from Reader import *
from detectron2.data import DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import inference_on_dataset, PascalVOCDetectionEvaluator
from detectron2.data import build_detection_test_loader
from kitti_load import *

PATH_DATASET = '../datasets/AICity_data/'
PATH_RESULTS = './Results/'

def main():
    print("Loading database ...")
    dsname = 'my_dataset'  # any custom dataset
    dictAIcity = get_AICity_dicts(PATH_DATASET, ['car'], "perFrame")
    DatasetCatalog.register(dsname, dictAIcity)
    print("Database registered")

    #reader = Reader(PATH_DATASET, ["car"], "perFrame")

    #print("Get gt")
    #gt = reader.get_annotations()

    #print("Get det")
    #det = reader.get_det("yolo")

    print("Get cfg...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    print("merge model ...")
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    print("create evaluator")
    evaluator = PascalVOCDetectionEvaluator(dsname)
    predictor = DefaultPredictor(cfg)
    inference_on_dataset(predictor.model, data_loader, evaluator)
    i = 0

if __name__ == '__main__':
    main()
