import os
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from BoundingBox import *

class Reader:
    def __init__(self, path, classes, sortStrategy):
        self.path = path
        self.classes = list(classes)
        self.sortStrategy = sortStrategy

    def get_annotations(self):

        tree = ET.parse(self.path+"ai_challenge_s03_c010-full_annotation.xml")

        root = tree.getroot()

        groundTruth = defaultdict()
        for child in root[2:]:
            if child.attrib['label'] in self.classes:
                for c in child:
                    frameNumber = int(c.attrib['frame'])
                    lista = [child.attrib['label'],
                             float(c.attrib['xtl']),
                             float(c.attrib['ytl']),
                             float(c.attrib['xbr']),
                             float(c.attrib['ybr'])]
                    if(frameNumber in groundTruth.keys()):
                        groundTruth[frameNumber].append(lista)
                    else:
                        groundTruth[frameNumber] = [lista]

        orderedDict = OrderedDict(groundTruth)

        if self.sortStrategy == "perFrame":
            return sorted(orderedDict.items())
        return orderedDict



    def get_det(self, detfile):
        detfilePath = self.path +"train/S03/c010/det/"
        if(detfile == "rccn"):
            detfilePath = detfilePath + "det_mask_rcnn"+".txt"
        if (detfile == "ssd"):
            detfilePath = detfilePath + "det_ssd512" + ".txt"
        if (detfile == "yolo"):
            detfilePath = detfilePath + "det_yolo3" + ".txt"
        with open(detfilePath) as f:
            lines = f.readlines()

        annotations = []
        for line in lines:
            data = line.split(',')
            annotations.append(BoundingBox(
                frame=int(data[0]) - 1,
                id=int(data[1]),
                label='car',
                xtl=float(data[2]),
                ytl=float(data[3]),
                xbr=float(data[2]) + float(data[4]),
                ybr=float(data[3]) + float(data[5]),
                score=float(data[6])
            ))

        return annotations


