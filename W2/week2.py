"""
Usage:
  week2.py <Task> [<Method>] [<Filters>]  [--testDir=<td>]
  week2.py -h | --help

  <Task> --> Choose the task to evaluate 1, 2, 3 or 4
  <Method> --> Method to subtract the background (KKN or MOG2) (only for task 3, predetermined (KKN and False)
  <Filters> --> Bolean to decide if we want to remove the noise or use the original function from openCV (only for task 3)
  Example of use --> python week2.py 3 KKN True
"""

import os
import cv2 as cv2
import numpy as np
from docopt import docopt

def task3(Method = 'KNN', Filter = False):
  kernel = np.ones((3, 3), np.uint8)
  Video_Path = '..data/vdo.avi'

  if Method == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
  elif Method == 'KNN':
    backSub = cv2.createBackgroundSubtractorKNN()

  capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(Video_Path))
  roi = cv2.imread('..data/roi.jpg', cv2.IMREAD_GRAYSCALE)
  
  if not capture.isOpened():
    print('Unable to open: ' + Video_Path)
    exit(0)

  while True:
    ret, frame = capture.read()
    
    if frame is None:
        break
        
    if Filter == True:
        fgMask = backSub.apply(frame)
        fgMask = fgMask & roi
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_ERODE, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_ERODE, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, kernel)
        _, fgMask = cv2.threshold(fgMask, 170, 255, cv2.THRESH_BINARY)
    else:
        fgMask = backSub.apply(frame)

    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)

    if keyboard == 'q' or keyboard == 27:
        break
        
if __name__ == '__main__':
  # read arguments
  args = docopt(__doc__)
  Task = int(args['<Task>'])
  # Optional arguments for task three (default: Method = KKN, Filter = False)
  Method = str(args['<Method>']) 
  Filter = bool(args['<Filters>'])
  if Task == 1:
    task1_1()
  elif Task == 2:
    task2()
  elif Task == 3: 
    task3(Method, Filter)
  
