import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OpticalFlow import read_flow, optical_flow_magnitude_plot, msen_pepn
from display import plt_flow_error, histogram_with_mean_plot, visualize_flow, get_plot_legend, colorflow_black

def euclidean_dist(x,y):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance

def task1(path_img, block_size = 8, search_area = 8 ):
    # Reading images
    file_img1 = f'{path_img}000045_10.png'
    file_img2 = f'{path_img}000045_11.png'
    img1 = cv2.imread(file_img1, 0)
    img2 = cv2.imread(file_img2, 0)
    h = img1.shape[0]
    w = img1.shape[1]
    # predicted frame.
    #pred_img = np.empty((h, w, 3), dtype=np.uint8)
    
    # Get block in the first image:
    for row in range(int(block_size/2), h-int(block_size/2), block_size):
        print('here')
        for col in range(int(block_size/2), w-int(block_size/2), block_size):
            # block
            block = img1[row - int(block_size/2):row + int(block_size/2) + 1,
                    col - int(block_size/2):col + int(block_size/2) + 1]
            # minimum distance
            dist_min = float('inf')
            motion_field = np.zeros((h, w, 2), dtype=float)
            pi, pj = 0, 0
            # Get search area and compare the candidate blocks in the image 2 with the previous block in the image 1
            for row_s in range(max(int(block_size/2), row - search_area),
                        min(h - int(block_size/2), row + search_area)):
                for col_s in range(max(int(block_size/2), col - search_area),
                            min(w - int(block_size/2), col + search_area), block_size):
                    candidate_blk = img2[row_s - int(block_size/2): row_s + int(block_size/2)+1,
                            col_s - int(block_size/2):col_s + int(block_size/2) + 1]
                    # Compute the distance between blocks
                    dist = euclidean_dist(np.hstack(np.hstack(block)) , np.hstack(np.hstack(candidate_blk)))
                    if dist == 0:
                        dist_min = dist
                        r = row_s
                        c = col_s
                        break
                    elif dist < dist_min:
                        dist_min = dist
                        r = row_s
                        c = col_s
                    rowb = max(row - search_area, 0)
                    colb = max(col - search_area, 0)
                if dist == 0:
                    break
            #Get the flow
            v = r - (row - rowb)
            u = c - (col - colb)
            motion_field[row- int(block_size/2):row + int(block_size/2),
            col - int(block_size/2):col + int(block_size/2), :] = [u, v]
            
    return motion_field

if __name__ == '__main__':
    path_img = "datasets/Kitti/"
    flow_gt = read_flow("datasets/Kitti/000045_10_gt.png")
    block_size = [6, 10]
    search_area = [24, 30]
    metric_msen = []
    metric_pepn = []
    bkl_s = []
    sear_a = []
    # Run task 1 using different combinations of block_size and search_area
    for x in block_size:
        for y in search_area:
            motion_field = task1(path_img, block_size = x, search_area = y)
            error_flow, non_occ_err_flow, msen, pepn = msen_pepn(motion_field, flow_gt, th=5)
            metric_msen.append(msen)
            metric_pepn.append(pepn)
            bkl_s.append(x)
            sear_a.append(y)
            print(msen, x, y)
            print(pepn, x, y)
            
    #Visualizations
    file_img1 = f'{path_img}000045_10.png'
    img1 = cv2.imread(file_img1, 0)
    optical_flow_magnitude_plot(motion_field, frame_id= '045', title="Predicted_Flow")
    visualize_flow(img1, motion_field)

    flow_legend = get_plot_legend(256, 256)
    color_flow_legend = colorflow_black(flow_legend)
    cv2.imshow("color wheel", color_flow_legend)
    #cv2.imshow('image predicted', pred_img)
    cv2.imshow('flow predicted', colorflow_black(motion_field))

    cv2.waitKey(0)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    z = metric_msen
    x = bkl_s
    y = sear_a
    vertices = [list(zip(x, y, z))]
    poly = Poly3DCollection(vertices, alpha=0.8)
    ax.add_collection3d(poly)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, 50)
    ax.savefig('datasets/Kitti/msen.png')

    # closing all open windows
    cv2.destroyAllWindows()

'''
    bx = plt.axes(projection='3d')
    # Data for a three-dimensional line
    zline = metric_pepn
    xline = bkl_s
    yline = sear_a
    bx.plot3D(xline, yline, zline, 'gray')
    bx.savefig('datasets/Kitti/pepn.png')
'''
