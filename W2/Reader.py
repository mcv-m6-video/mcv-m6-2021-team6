import os
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from copy import deepcopy
import numpy as np
from BoundingBox import *
import xmltodict


class AICityChallengeAnnotationReader:

    def __init__(self, path,  initFrame, finalFrame):
        self.annotations = parse_annotations(path,  initFrame, finalFrame)
        self.classes = np.unique([det.label for det in self.annotations])

    def get_annotations(self, classes=None, noise_params=None, do_group_by_frame=True, only_not_parked=False):
        """
        Returns:
            detections: {frame: [Detection,...]} if group_by_frame=True
        """

        if classes is None:
            classes = self.classes

        detections = []
        for det in self.annotations:
            if det.label in classes:  # filter by class
                d = deepcopy(det)
                if noise_params:  # add noise
                    if np.random.random() > noise_params['drop']:
                        box_noisy = d.bbox + np.random.normal(noise_params['mean'], noise_params['std'], 4)
                        d.xtl = box_noisy[0]
                        d.ytl = box_noisy[1]
                        d.xbr = box_noisy[2]
                        d.ybr = box_noisy[3]
                        detections.append(d)
                else:
                    detections.append(d)

        if do_group_by_frame:
            detections = group_by_frame(detections)

        return detections


def parse_annotations(path, initFrame, finalFrame):
    root, ext = os.path.splitext(path)

    if ext == ".xml":
        with open(path) as f:
            tracks = xmltodict.parse(f.read())['annotations']['track']

        annotations = []
        for track in tracks:
            id = track['@id']
            label = track['@label']
            boxes = track['box']
            for box in boxes:
                if label == 'car':
                    parked = box['attribute']['#text'].lower() == 'true'
                else:
                    parked = None
                annotations.append(BoundingBox(
                    frame=int(box['@frame']),
                    id=int(id),
                    label=label,
                    xtl=float(box['@xtl']),
                    ytl=float(box['@ytl']),
                    xbr=float(box['@xbr']),
                    ybr=float(box['@ybr']),
                    #score=None
                ))

    if ext == ".txt":
        """
        MOTChallenge format [frame, ID, left, top, width, height, conf, -1, -1, -1]
        """

        with open(path) as f:
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


def group_by_frame(bboxes):
    grouped = defaultdict(list)
    for bb in bboxes:
        grouped[bb.frame].append(bb)
    return OrderedDict(sorted(grouped.items()))


def group_by_id(bboxes):
    grouped = defaultdict(list)
    for bb in bboxes:
        grouped[bb.id].append(bb)
    return OrderedDict(sorted(grouped.items()))


class AnnotationReader:

    """
    Creates AnnotationReader object that reads the annotations
    """
    def __init__(self, path,  initFrame, finalFrame):
        # Read XML file
        self.annotations = parse_annotations(path,  initFrame, finalFrame)
        self.classes = np.unique([bb.label for bb in self.annotations])

    def get_bboxes_per_frame(self, classes=None, noise_params=None):
        """
        This function returns the bounding boxes sorted by frame, in the format {frame: [BB, BB, BB...]}
        """

        if classes is None:
            classes = self.classes

        bboxes = []
        for bb in self.annotations:
            if bb.label in classes:  # filter by class
                current_box = deepcopy(bb)
                if noise_params:  # add noise
                    if np.random.random() > noise_params['drop']:
                        box_noisy = current_box.bbox + np.random.normal(noise_params['mean'], noise_params['std'], 4)
                        current_box.xtl = box_noisy[0]
                        current_box.ytl = box_noisy[1]
                        current_box.xbr = box_noisy[2]
                        current_box.ybr = box_noisy[3]
                        bboxes.append(current_box)
                else:
                    bboxes.append(current_box)

        bboxes = group_by_frame(bboxes)

        return bboxes

