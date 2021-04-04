from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from OpticalFlow import read_flow, msen_pepn
from display import draw_flow, get_plot_legend, colorflow_black
import numpy as np
import time
from HornSchunck import HornSchunck

def task1_2(img1, img2, method='pyflow', block_size = None):

    if method == 'pyflow':
        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        s = time.time()
        u, v, im2W = coarse2fine_flow( img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
        e = time.time()

        motion_field = np.dstack((u, v))

    elif method == 'farneback':

        '''
        nputArray 	prev, 
        InputArray 	next,
        InputOutputArray 	flow,
        double 	pyr_scale,
        int 	levels,
        int 	winsize,
        int 	iterations,
        int 	poly_n,
        double 	poly_sigma,
        int 	flags	
        '''

        s = time.time()
        motion_field = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, block_size, 3, 5, 1.2, 0)
        e = time.time()
        
    elif method == 'horn':
        s = time.time()
        U, V = HornSchunck(img1, img2, alpha=5.0, Niter=100)
        motion_field = np.dstack((U, V))
        e = time.time()

    return motion_field, e, s

if __name__ == '__main__':
    path_img = "datasets/Kitti/"
    flow_gt = read_flow(f'{path_img}000045_10_gt.png')
    file_img1 = f'{path_img}000045_10.png'
    file_img2 = f'{path_img}000045_11.png'
    img1 = cv2.imread(file_img1, 0)
    img2 = cv2.imread(file_img2, 0)
    block_size = [32]
    for blk in block_size:
        motion_field, e, s = task1_2(img1, img2, method= 'horn', block_size = None)
        error_flow, non_occ_err_flow, msen, pepn = msen_pepn(motion_field, flow_gt, th=5)
        print(f'MSEN: {msen:.4f}, PEPN: {pepn:.4f}, runtime: {e - s:.3f}s, BS:{blk}')

    flow_legend = get_plot_legend(256, 256)
    color_flow_legend = colorflow_black(flow_legend)
    cv2.imshow("color wheel", color_flow_legend)
    cv2.imshow('flow predicted', colorflow_black(motion_field))
