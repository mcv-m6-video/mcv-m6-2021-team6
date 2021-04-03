from Video import *
import cv2
import numpy as np
from tqdm import trange


def task2_1():
    video = Video("videos/video_test.mp4", 500, 250)
    revideo = video.get_result_video()
    previous_frame = None
    acc_t = np.zeros(2)
    acc_list = []
    for i in trange(0, video.num_frames):
        success, frame = video.capture.read()
        frame = cv2.resize(frame, (500, 250), interpolation=cv2.INTER_AREA)
        if not success:
            break

        if i == 0:
            frame_stabilized = frame
        else:
            optical_flow = block_matching(previous_frame, frame, block_size=32, search_area=16, motion_type='forward')
            average_optical_flow = - np.array(optical_flow.mean(axis=0).mean(axis=0), dtype=np.float32)
            acc_t += average_optical_flow
            H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
            frame_stabilized = cv2.warpAffine(frame, H, (500, 250))


        previous_frame = frame
        acc_list.append(acc_t)

        revideo.write(frame_stabilized)


def distance(x1: np.ndarray, x2: np.ndarray, metric='euclidean'):
    if metric == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif metric == 'sad':
        return np.sum(np.abs(x1 - x2))
    elif metric == 'mad':
        return np.mean(np.abs(x1 - x2))
    elif metric == 'ssd':
        return np.sum((x1 - x2) ** 2)
    elif metric == 'mse':
        return np.mean((x1 - x2) ** 2)
    else:
        raise ValueError(f'Unknown distance metric: {metric}')

def block_matching(img_prev: np.ndarray, img_next: np.ndarray, block_size=16, search_area=16,
                        motion_type='backward', metric='euclidean', algorithm='es'):

    if motion_type == 'forward':
        reference = img_prev
        target = img_next
    elif motion_type == 'backward':
        reference = img_next
        target = img_prev
    else:
        raise ValueError(f'Unknown motion type: {motion_type}')

    assert (reference.shape == target.shape)
    height, width = reference.shape[:2]
    motion_field = np.zeros((height, width, 2), dtype=float)
    # Get block in the first image:
    for row in range(0, height - block_size, block_size):

        for col in range(0, width - block_size, block_size):

            # block matching
            dist_min = np.inf
            rowb = max(row - search_area, 0)
            colb = max(col - search_area, 0)
            # Get search area and compare the candidate blocks in the image 2 with the previous block in the image 1
            r = 0
            c = 0
            referenceb = reference[row:row + block_size, col:col + block_size]
            targetb = target[rowb: min(row + block_size + search_area, height),
                      colb: min(col + block_size + search_area, width)]
            for row_s in range(targetb.shape[0]-referenceb.shape[0]):

                for col_s in range(targetb.shape[1]-referenceb.shape[1]):


                    # Compute the distance between blocks
                    dist = distance(referenceb, targetb[row_s:row_s+referenceb.shape[0], col_s:col_s+referenceb.shape[1]], metric)
                    if dist < dist_min:
                        r = row_s
                        c = col_s
                        dist_min = dist

            # Get the flow
            v = r - (row - rowb)
            u = c - (col - colb)
            motion_field[row:row + block_size, col:col + block_size, :] = [u, v]

    return motion_field

if __name__ == '__main__':
    task2_1()
