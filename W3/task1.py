from Reader import *
from object_detection.DetectionModel import *

def task1_1():
    reader = AICityChallengeAnnotationReader(path='../datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    detectionModel = DetectionModel('mask', '../datasets/AICity_data/train/S03/c010/vdo.avi')
    detectionModel.evaluation(gt)
    ap, prec, rec = detectionModel.get_metrics()
    print(f'Ap is {ap}')
    print(f'Precision is {prec}')
    print(f'Recall is {rec}')
    detectionModel.get_qualitative_metrics(gt)

if __name__ == '__main__':
    task1_1()
