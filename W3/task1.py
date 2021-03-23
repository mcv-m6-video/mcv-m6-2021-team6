from Reader import *
from object_detection.DetectionModel import *
from object_detection.UtilsDetection import *
def task1_1(model = None):
    reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    detectionModel = None
    if model == None:
        detectionModel = DetectionModel('fast', 'datasets/AICity_data/train/S03/c010/vdo.avi')
    else:
        pass
    detectionModel.evaluation(gt)
    ap, prec, rec = detectionModel.get_metrics(False)
    print(f'Ap is {ap}')
    print(f'Precision is {prec}')
    print(f'Recall is {rec}')
    detectionModel.get_qualitative_metrics(gt)

def task1_2(k_fold = 1 ):
    detectionModel = DetectionModel('mask', 'datasets/AICity_data/train/S03/c010/vdo.avi',  finetune=True)
    detectionModel.train(k_fold)
    reader = AICityChallengeAnnotationReader(path='datasets/AICity_data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    #detectionModel.get_qualitative_metrics(gt)
    with torch.no_grad():
        i = 0
        f = open("Results/det.txt", "w")
        for images, targets in detectionModel.test_loader:
            images = [image.to(detectionModel.device) for image in images]
            predictions = detectionModel.model(images)
            for image, prediction in zip(images, predictions):
                image = (image.to('cpu').numpy() * 255).astype(np.uint8).transpose((1, 2, 0))
                image = np.ascontiguousarray(image)
                boxes = prediction['boxes'].to('cpu').numpy().astype(np.int32)
                for box in boxes:
                    f.write(f"{i}, -1, {box[0]}, {box[1]}, {box[2]-box[0]}, {box[3]-box[1]}, 0, -1, -1, -1 \n")
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                i = i + 1
                cv2.imshow('predictions', image)
                cv2.waitKey(0)
        f.close()
if __name__ == '__main__':
    #task1_1()
    task1_2(2)
