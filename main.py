## Hey, starting this first snipped of Detectron with a bit of code examples
## extractred from here:
## https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=dq9GY37ml1kr
#
## And of course, deleting the interactive Python part.....
## Because I love .py plain text executions :)

## DEPENDENCIES:
## A good guide to follow: https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c



import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# opencv is pre-installed on colab

# install detectron2: (Colab has CUDA 10.1 + torch 1.7)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

im = cv2.imread("example/000000439715.jpg")
cv2.imshow('A horse', im)
cv2.waitKey(0)