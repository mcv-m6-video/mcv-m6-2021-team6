import os
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from BoundingBox import *
import xmltodict

class Reader:
    def __init__(self, path, classes, sortStrategy):
        self.path = path
        self.classes = list(classes)
        self.sortStrategy = sortStrategy

    def get_annotations(self):

        abs_path_to_xml = os.path.abspath(self.path+"/AICity_data/train/S03/c010/Anotation_40secs_AICITY_S03_C010.xml")
        with open(abs_path_to_xml) as f:
            tracks = xmltodict.parse(f.read())['annotations']['track']

        annotations = []
        for track in tracks:
            id = track['@id']
            label = track['@label']

            if label != 'car':
                continue

            for box in track['box']:
                annotations.append(BoundingBox(
                    id=int(id),
                    label=label,
                    frame=int(box['@frame']),
                    xtl=float(box['@xtl']),
                    ytl=float(box['@ytl']),
                    xbr=float(box['@xbr']),
                    ybr=float(box['@ybr'])
                ))


        return annotations



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
                confidence=float(data[6])
            ))

        return annotations


