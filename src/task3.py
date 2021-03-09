import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from OpticalFlow import read_flow, optical_flow_magnitude_plot, msen_pepn
from display import plt_flow_error, histogram_with_mean_plot

# Task 3: Optical flow evaluation metrics
for frame_id in ['045', '157']:
    filename = f'000{frame_id}_10.png'
    flow_predicted = read_flow(os.path.join('./databases/Kitti/', f'LKflow_{filename}'))
    flow_groundtruth = read_flow(os.path.join('./databases/Kitti/gt/', f'{filename}'))

    error_flow, non_occ_err_flow, msen, pepn = msen_pepn(flow_predicted, flow_groundtruth, th=5)
    print(f'SEQ-{frame_id}\n  MSEN: {round(msen, 4)}\n  PEPN: {round(pepn, 4)}%')

    plt_flow_error(error_flow, frame_id)

    histogram_with_mean_plot(title='Error Histogram', idx=frame_id, values=non_occ_err_flow, mean_value=msen)
    optical_flow_magnitude_plot(flow_predicted, frame_id, title="Predicted_Flow")
    optical_flow_magnitude_plot(flow_groundtruth, frame_id, title="GroundTruth_Flow")
