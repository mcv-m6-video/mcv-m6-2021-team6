import cv2
import numpy as np
from OpticalFlow import read_flow, msen_pepn
from display import plt_flow_error, histogram_with_mean_plot, draw_flow, get_plot_legend, colorflow_black, visualize_3d_plot
import time

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

def task1(img1, img2, block_size = 8, search_area = 8, motion_type='forward', metric='euclidean'):

    if motion_type == 'forward':
        ref = img1
        tar = img2
    elif motion_type == 'backward':
        ref = img2
        tar = img1
    else:
        raise ValueError(f'Unknown motion type: {motion_type}')

    assert (img1.shape == img2.shape)
    h = img1.shape[0]
    w = img1.shape[1]
    # predicted frame.
    #pred_img = np.empty((h, w, 3), dtype=np.uint8)
    motion_field = np.zeros((h, w, 2), dtype=float)
    for row in range(0, h - block_size, int(block_size)):

        for col in range(0, w - block_size, int(block_size)):
            # block matching
            block = ref[row:row + block_size, col:col + block_size]
            # minimum distance
            dist_min = np.inf
            rowb = max(row - search_area, 0)
            colb = max(col - search_area, 0)
            r, c = 0, 0
            target = tar[rowb: min(row + block_size + search_area, h),
                      colb: min(col + block_size + search_area, w)]
            for row_s in range(target.shape[0]-block.shape[0]):

                for col_s in range(target.shape[1]-block.shape[1]):

                    # Compute the distance between blocks
                    dist = distance(block, target[row_s:row_s+block.shape[0], col_s:col_s+block.shape[1]], metric)
                    if dist == 0:
                        r = row_s
                        c = col_s
                        dist_min = dist
                        break
                    elif dist < dist_min:
                        r = row_s
                        c = col_s
                        dist_min = dist
                if dist == 0:
                    break
            # Get the flow
            v = r - (row - rowb)
            u = c - (col - colb)

            motion_field[row:row + block_size, col:col+block_size, :] = [u, v]
    if motion_type == 'forward':
        return motion_field
    elif motion_type == 'backward':
        return motion_field * -1

if __name__ == '__main__':
    display = False
    path_img = "datasets/Kitti/"
    file_img1 = f'{path_img}000045_10.png'
    file_img2 = f'{path_img}000045_11.png'
    img1 = cv2.imread(file_img1, 0)
    img2 = cv2.imread(file_img2, 0)
    flow_gt = read_flow(f'{path_img}000045_10_noc.png')
    block_size = [36]
    search_area = [30]
    msens = np.zeros((len(search_area), len(block_size)))
    psens = np.zeros((len(search_area), len(block_size)))
    X, Y = np.meshgrid(search_area, block_size)
    j = 0
    for x in block_size:
        i = 0
        for y in search_area:
            s = time.time()
            motion_field = task1(img1, img2, block_size = x, search_area = y, motion_type='backward', metric='euclidean')
            e = time.time()
            error_flow, non_occ_err_flow, msen, pepn = msen_pepn(motion_field, flow_gt, th=5)
            msens[j, i] = msen
            psens[j, i] = pepn
            i += 1
            print('MSEN: ', msen, 'PEPN: ', pepn,'Block size: ',x, 'Search windows: ', y, 'runtime: ', e-s)
            if display == True:
                file_img1 = f'{path_img}000045_10.png'
                img1 = cv2.imread(file_img1, 0)
                cv2.imshow("flow on image", draw_flow(img1, motion_field))
                cv2.waitKey(0)
                flow_legend = get_plot_legend(256, 256)
                color_flow_legend = colorflow_black(flow_legend)
                cv2.imshow("color wheel", color_flow_legend)
                cv2.waitKey(0)
                cv2.imshow('flow predicted', colorflow_black(motion_field))
                cv2.waitKey(0)
                cv2.imshow('flow gt', colorflow_black(flow_gt))
                cv2.waitKey(0)
                # Plot msen grid search
                visualize_3d_plot(X, Y, msens, 'Area size', 'Block size', 'MSEN')
                # Plot psen grid search
                visualize_3d_plot(X, Y, psens, 'Area size', 'Block size', 'PEPN')
        j += 1
