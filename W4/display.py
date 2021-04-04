import os
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
    resized = cv2.resize(I, (250, 75), interpolation=cv2.INTER_AREA)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = mag[::5, ::5]
    ang = ang[::5, ::5]
    plt.figure()
    plt.imshow(resized)
    plt.title("Flow" + suffix)
    plt.quiver(mag * np.cos(ang), mag * np.sin(ang))
    plt.show()

    
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

  
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

  
def visualize_3d_plot(X, Y, Z, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, edgecolor='none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
