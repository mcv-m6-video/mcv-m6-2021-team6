import numpy as np
from copy import deepcopy

def mean_average_precision(gt_list, pred_list, confidence_score=True, classes=["car"]):
    """
    Mean Average Precision calculation
    Parameters:
        gt_list: [[Detection,...],...]
        pred_list: [[Detection,...],...]
        confidence_score: indicates the method to compute the map
        classes: indicates the classes of the dataset
    """
    precs = []
    recs = []
    aps = []
    for c in classes:
        gt_list_class = [[det for det in boxlist if det.label == c] for boxlist in gt_list]
        pred_list_class = [[det for det in boxlist if det.label == c] for boxlist in pred_list]
        ap, prec, rec = average_precision(gt_list_class, pred_list_class, confidence_score)
        precs.append(prec)
        recs.append(rec)
        aps.append(ap)
    prec = np.mean(precs)
    rec = np.mean(recs)
    map = np.mean(aps)

    return map, prec, rec


def average_precision(gt_list, pred_list, confidence_score=True):
    """
    Average Precision with or without confidence scores.
    Params:
        gt_list: [[Detection,...],...]
        pred_list: [[Detection,...],...]
    """
    pred_list = [(i, det) for i in range(len(pred_list)) for det in pred_list[i]]
    if len(pred_list) == 0:
        return 0

    if confidence_score :
        sorted_ind = np.argsort([-det[1].confidence for det in pred_list])
        pred_list_sorted = [pred_list[i] for i in sorted_ind]
        ap, prec, rec = voc_ap(gt_list, pred_list_sorted)
    else:
        n = 10
        precs = []
        recs = []
        aps = []
        for _ in range(n):
            shuffled_ind = np.random.permutation(len(pred_list))
            pred_list_shuffled = [pred_list[i] for i in shuffled_ind]
            ap, prec, rec = voc_ap(gt_list, pred_list_shuffled)
            precs.append(prec)
            recs.append(rec)
            aps.append(ap)
        prec = np.mean(precs)
        rec = np.mean(recs)
        ap = np.mean(aps)
    return ap, prec, rec


# Below code is adapted from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py

def voc_ap(gt_list, pred_list, ovthresh=0.5):
    """
    Average Precision as defined by PASCAL VOC (11-point tracking).
    Params:
        gt_list: [[Detection,...],...]
        pred_list: [Detection,...]
        ovthresh: overlap threshold.
    """
    class_recs = []
    npos = 0
    for R in gt_list:
        bbox = np.array([det.bbox for det in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs.append({"bbox": bbox, "det": det})
    # pred_list.sort(key=lambda x: x[0])
    image_ids = [det[0] for det in pred_list]
    BB = np.array([det[1].bbox for det in pred_list]).reshape(-1, 4)
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        if len(class_recs) > image_ids[d]:
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R["bbox"].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                overlaps = iou_vectorized(BBGT, bb[None, :])
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(npos)+0.000000001)

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # compute VOC AP using 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap, prec, rec

def iou_bbox(box1, box2):
    """
    Input format is [xtl, ytl, xbr, ybr] per bounding box, where
    tl and br indicate top-left and bottom-right corners of the bbox respectively
    """
    #determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union
    iou = interArea / float(box1Area + box2Area - interArea)

    # return the intersection over union value
    return iou

def compute_iou_over_time(gt_list, pred_list):
    """
    Compute the mean IOU over time (for each frame)
    Parameters:
        gt_list: [[Detection,...],...]
        pred_list: [[Detection,...],...]
    """
    frame_ious = []
    for dets_gt, dets_pred in zip(gt_list, pred_list):
        box_ious = []
        for det in dets_gt:
            ious = []
            for det_p in dets_pred:
                iou = iou_bbox(det.bbox,det_p.bbox)
                ious.append(iou)
            if ious == []:
                ious.append(0.0)
            box_ious.append(max(ious))
        frame_ious.append(np.mean(box_ious))
    mean_iou_global = np.mean(frame_ious)
    return mean_iou_global, frame_ious


def iou_vectorized(boxes1, boxes2):
    """
    Compute overlaps between two sets of boxes.
    Params:
        boxes1: [[xtl,ytl,xbr,ybr],...]
        boxes2: [[xtl,ytl,xbr,ybr],...]
    Returns:
        overlaps: matrix of pairwise overlaps.
    """

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # intersection
    ixmin = np.maximum(x11, np.transpose(x21))
    iymin = np.maximum(y11, np.transpose(y21))
    ixmax = np.minimum(x12, np.transpose(x22))
    iymax = np.minimum(y12, np.transpose(y22))
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    area1 = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    area2 = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)
    uni = area1 + np.transpose(area2) - inters

    overlaps = inters / uni

    return overlaps

def matched_bbox(prev_det, det):
    max_iou = 0
    for d in det:
        iou = iou_bbox(prev_det.bbox, d.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = d
    if max_iou > 0:
        best_match.id = prev_det.id
        return best_match
    else:
        return None

def match_bbox_flow(last_detection, det, flow):
    last_detection_copy = deepcopy(last_detection)

    # Compensate last_detection
    if flow is not None:
        last_detection_copy.xtl += flow[int(last_detection_copy.ytl)-1, int(last_detection_copy.xtl)-1, 0]
        last_detection_copy.ytl += flow[int(last_detection_copy.ytl)-1, int(last_detection_copy.xtl)-1, 1]
        last_detection_copy.xbr += flow[int(last_detection_copy.ybr)-1, int(last_detection_copy.xbr)-1, 0]
        last_detection_copy.ybr += flow[int(last_detection_copy.ybr)-1, int(last_detection_copy.xbr)-1, 1]

    max_iou = 0
    for detection in det:
        iou = iou_bbox(last_detection_copy.bbox, detection.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = detection
    if max_iou > 0:
        best_match.id = last_detection_copy.id
        return best_match
    else:
        return None

def matched_bbox_mov(prev_det, det, th):
    max_iou = 0
    id_remove = False
    for d in det:
        iou = iou_bbox(prev_det.bbox, d.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = d
    if max_iou > 0:
        best_match.id = prev_det.id
        if max_iou > th:
            id_remove = True
        else:
            id_remove = False
        return best_match, id_remove
    else:
        return None, id_remove

def match_bbox_flow_mov(last_detection, det, flow, th):
    last_detection_copy = deepcopy(last_detection)

    # Compensate last_detection
    if flow is not None:
        try:
            last_detection_copy.xtl += flow[int(last_detection_copy.ytl), int(last_detection_copy.xtl), 0]
            last_detection_copy.ytl += flow[int(last_detection_copy.ytl), int(last_detection_copy.xtl), 1]
            last_detection_copy.xbr += flow[int(last_detection_copy.ybr), int(last_detection_copy.xbr), 0]
            last_detection_copy.ybr += flow[int(last_detection_copy.ybr), int(last_detection_copy.xbr), 1]
        except:
            last_detection_copy = deepcopy(last_detection)

    max_iou = 0
    id_remove = False
    for detection in det:
        iou = iou_bbox(last_detection_copy.bbox, detection.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = detection
    if max_iou > 0:
        best_match.id = last_detection_copy.id
        if max_iou > th:
            id_remove = True
        else:
            id_remove = False
        return best_match, id_remove
    else:
        return None, id_remove
