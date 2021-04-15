import cv2
import numpy as np
from display import plt_flow_magnitude

def read_flow(path: str) -> np.ndarray:
    """
    Method based on the provided code from KITTI
    """
    I = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    F_col = (I[:, :, 2] - 2 ** 15) / 64
    F_row = (I[:, :, 1] - 2 ** 15) / 64

    F_valid = I[:, :, 0]
    F_valid[F_valid > 1] = 1

    F_col[F_valid == 0] = 0
    F_row[F_valid == 0] = 0
    flow = np.dstack((F_col, F_row, F_valid))
    return flow

def msen_pepn(flow_predicted: np.ndarray, flow_groundtruth: np.ndarray, th: int = 5) -> (np.ndarray, np.ndarray, float, float):
    # compute mse
    u_diff = flow_groundtruth[:, :, 0] - flow_predicted[:, :, 0]
    v_diff = flow_groundtruth[:, :, 1] - flow_predicted[:, :, 1]
    sq_error = np.sqrt(u_diff**2 + v_diff**2)

    non_occluded_idx = flow_groundtruth[:, :, 2] != 0
    err_non_occluded = sq_error[non_occluded_idx]

    msen = np.mean(err_non_occluded)
    pepn = get_pepn(err_non_occluded, len(err_non_occluded), th)

    return sq_error, err_non_occluded, msen, pepn


def get_pepn(err: np.ndarray, n_pixels: int, th: int) -> float:

    return (np.sum(err > th) / n_pixels) * 100


def optical_flow_magnitude_plot(flow_image: np.ndarray, frame_id: str, path: str = "", title="Predicted_Flow"):
    if len(flow_image.shape) > 2:
        magnitude, angle = cv2.cartToPolar(flow_image[:, :, 0], flow_image[:, :, 1])
        flow_image = magnitude

    kernel = np.ones((5, 5), np.uint8)
    flow_image = cv2.dilate(flow_image, kernel, iterations=1)

    plt_flow_magnitude(flow_image, title, frame_id, path)
