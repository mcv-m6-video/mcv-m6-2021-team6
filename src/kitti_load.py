import glob
import os
import cv2
from detectron2.structures import BoxMode
import numpy as np
from Reader import *

def get_AICity_dicts(path_dataset,classes, sortStrategy):
    categories = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
    }

    reader = Reader(path_dataset, classes, "perFrame")
    gt = reader.get_annotations()

    video_path = "../datasets/AICity_data/train/S03/c010/vdo.avi"
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidcap = cv2.VideoCapture('../datasets/AICity_data/train/S03/c010/vdo.avi')

    success,image = vidcap.read()
    count = 0
    images = list()
    images_id = list()
    while success:
      #cv2.imshow("frame%d.jpg" % count, image)     # save frame as JPEG file
      success,image = vidcap.read()
      images.append(image)
      images_id.append(count)
      count += 1
      #for d in gt[count][1]:
      #    cv2.rectangle(image, (int(d[1]), int(d[2])), (int(d[3]), int(d[4])), (0, 255, 0), 2)
      #cv2.imshow('image', image)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()



    #obtain
    #label_dir='/home/mcv/datasets/KITTI/training/label_2'
    #image_path = glob.glob(working_folder + '/*.png')
    #label_path = glob.glob(label_dir + '/*.txt')
    #label_file = sorted(label_path)
    #image_file = sorted(image_path)

    #for file in images :
    #    splitd = file.split(os.sep)
    #    img_name = splitd[-1]
    #    img_id = img_name.split('.')[0]
    #    if set_type is not 'testing':
    #        label_file.append(label_dir + img_id + '.txt')

    dataset_dicts = []


    #iteration

    for id in images_id:
        # for d in gt[count][1]:
        record = {}
        #height, width = cv2.imread(image_file[i]).shape[:2]
        record["file_name"] = "frame%d.jpg" % id
        record["image_id"] = id
        record["height"] = height
        record["width"] = width

        objs = []
        for frame in gt[id][1]:

            obj = {
                "bbox": [frame[1], frame[2], frame[3], frame[4]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": frame[0]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == '__main__':
    get_AICity_dicts('../datasets/AICity_data/',['car'],"perFrame")