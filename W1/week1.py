import os

import numpy as np
import matplotlib.pyplot as plt

from utils.aicity_reader import AICityChallengeAnnotationReader
from evaluation.average_precision import mean_average_precision
from utils.plotutils import video_iou_plot

from utils.flow_reader import read_flow_field, read_grayscale_image
from evaluation.optical_flow_evaluation import get_msen_pepn
from utils.plotutils import optical_flow_arrow_plot, optical_flow_magnitude_plot, histogram_with_mean_plot


def task1_1(save_path=None):
    reader = AICityChallengeAnnotationReader(path='../data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    # add probability to delete bounding boxes
    drop_values = np.linspace(0, 1, 11)
    maps = []
    for drop in drop_values:
        noise_params = {'drop': drop, 'mean': 0, 'std': 0}
        gt_noisy = reader.get_annotations(classes=['car'], noise_params=noise_params)

        y_true = []
        y_pred = []
        for frame in gt.keys():
            y_true.append(gt.get(frame))
            y_pred.append(gt_noisy.get(frame, []))

        map, _ = mean_average_precision(y_true, y_pred)
        maps.append(map)

    plt.plot(drop_values, maps)
    plt.xticks(drop_values)
    plt.xlabel('drop prob')
    plt.ylabel('mAP')
    plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'map_drop_bbox.png'))

    # add noise to the size and position of bounding boxes
    std_values = np.linspace(0, 100, 11)
    maps = []
    for std in std_values:
        noise_params = {'drop': 0, 'mean': 0, 'std': std}
        gt_noisy = reader.get_annotations(classes=['car'], noise_params=noise_params)

        y_true = []
        y_pred = []
        for frame in gt.keys():
            y_true.append(gt.get(frame))
            y_pred.append(gt_noisy.get(frame, []))

        map, _ = mean_average_precision(y_true, y_pred)
        maps.append(map)

    plt.xlabel('std')
    plt.ylabel('mAP')
    plt.xticks(std_values)
    plt.plot(std_values, maps)
    plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'map_noisy_bbox.png'))


def task1_2():
    reader = AICityChallengeAnnotationReader(path='../datasets/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    for detector in ['mask_rcnn', 'ssd512', 'yolo3']:
        reader = AICityChallengeAnnotationReader(path=f'../datasets/AICity_data/train/S03/c010/det/det_{detector}.txt')
        det = reader.get_annotations(classes=['car'])

        y_true = []
        y_pred = []
        for frame in gt.keys():
            y_true.append(gt.get(frame))
            y_pred.append(det.get(frame, []))

        map, prec, _ = mean_average_precision(y_true, y_pred)
        print(f'{detector} mAP: {map:.4f}')


def task2(start=0, length=100, save_path=None):
    reader = AICityChallengeAnnotationReader(path='../data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    gt = {frame: gt[frame] for frame in range(start, start+length)}

    noise_params = {'drop': 0.05, 'mean': 0, 'std': 10}
    gt_noisy = reader.get_annotations(classes=['car'], noise_params=noise_params)
    gt_noisy = {frame: gt_noisy[frame] for frame in range(start, start+length)}
    video_iou_plot(gt, gt_noisy, video_path='data/AICity_data/train/S03/c010/vdo.avi', title='noisy annotations',
                   save_path=save_path)

    for detector in ['mask_rcnn', 'ssd512', 'yolo3']:
        reader = AICityChallengeAnnotationReader(path=f'data/AICity_data/train/S03/c010/det/det_{detector}.txt')
        det = reader.get_annotations(classes=['car'])
        det = {frame: det[frame] for frame in range(start, start+length)}
        video_iou_plot(gt, det, video_path='../data/AICity_data/train/S03/c010/vdo.avi', title=f'{detector} detections',
                       save_path=save_path)


def task3_4(save_path=None):
    pred_path = '..data/results_opticalflow_kitti'
    kitti_data = '..data/data_stereo_flow/training'

    # Task 3
    dilate = True
    for frame_id in ['045', '157']:
        filename = f'000{frame_id}_10.png'

        flow_pred = read_flow_field(os.path.join(pred_path, f'LKflow_{filename}'))
        flow_gt = read_flow_field(os.path.join(kitti_data, f'flow_noc/{filename}'))

        error_flow, non_occ_err_flow, msen, pepn = get_msen_pepn(flow_pred, flow_gt, th=3)
        print(f'SEQ-{frame_id}\n  MSEN: {round(msen, 4)}\n  PEPN: {round(pepn, 4)}%')

        plt.imshow(error_flow)
        plt.title(f'Error_Flow-{frame_id}')
        plt.axis('off')
        plt.show()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'error_flow_{frame_id}.png'))

        histogram_with_mean_plot(title='Error Histogram', idx=frame_id, values=non_occ_err_flow, mean_value=msen, save_path=save_path)
        optical_flow_magnitude_plot(flow_pred, frame_id, save_path, title="Predicted_Flow", dilate=dilate)
        optical_flow_magnitude_plot(flow_gt, frame_id, save_path, title="GT_Flow", dilate=dilate)

        # Task 4
        image_gray = read_grayscale_image(os.path.join(kitti_data, 'image_0', filename))
        optical_flow_arrow_plot(image_gray, flow_gt[:, :, 0:2], 'GT', frame_id, 10, path=save_path)
        optical_flow_arrow_plot(image_gray, flow_pred[:, :, 0:2], 'PRED', frame_id, 10, path=save_path)


if __name__ == '__main__':
    #task1_1()
    task1_2()
    #task2(start=550)
    #task3_4()
