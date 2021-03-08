import cv2
import glob
import os
from Reader import *

setup_logger()

PATH_DATASET = '../datasets/AICity_data/'
PATH_RESULTS = './Results/'

def main():
    reader = Reader(PATH_DATASET, ["car"], "perFrame")
    gt = reader.get_annotations()
    det = reader.get_det("yolo")
    i = 0

if __name__ == '__main__':
    main()
