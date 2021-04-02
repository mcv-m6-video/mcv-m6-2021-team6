import matplotlib.pyplot as plt
import os
import cv2 as cv2
import numpy as np

def plt_flow_error(I, frame_id):
    plt.imshow(I, cmap='PRGn', vmin=0, vmax=255)
    plt.colorbar()
    plt.title(f'Error_Flow-{frame_id}')
    plt.axis('off')
    plt.show()

def plt_flow_magnitude(I, title, frame_id, path):
    plt.imshow(I, cmap='Reds')
    plt.axis('off')
    plt.title(f'{title}-{frame_id}')
    plt.savefig(os.path.join(path, f'{title.lower()}_{frame_id}.png')) if path else plt.show()
    plt.close()


def histogram_with_mean_plot(title: str, idx: str, values: float, mean_value: float, save_path=None):
    plt.figure()
    plt.title(f'{title}-{idx}')
    plt.hist(values, 25, color="skyblue")
    plt.axvline(mean_value, color='g', linestyle='dashed', linewidth=1, label=f'MSEN {round(mean_value, 1)}')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'histogram_{idx}.png')) if save_path else plt.show()
    plt.close()

def visualize_flow(I, flow, suffix=""):
    resized = cv2.resize(I, (250, 80), interpolation=cv2.INTER_AREA)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = mag[::5, ::5]
    ang = ang[::5, ::5]
    plt.figure()
    plt.imshow(resized)
    plt.title("Flow" + suffix)
    plt.quiver(mag * np.cos(ang), mag * np.sin(ang))
    plt.show()

def get_plot_legend(size_x, size_y):
    nx, ny = size_x, size_y
    x = range(nx)
    y = range(ny)
    xv, yv = np.meshgrid(x, y)

    xv = xv - (nx / 2)
    yv = yv - (ny / 2)

    return np.dstack((xv[:, :], yv[:, :]))

def colorflow_black(flow):
    w, h = flow.shape[:2]
    hsv = np.zeros((w, h, 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb
