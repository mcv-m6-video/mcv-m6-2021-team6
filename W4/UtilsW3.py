from Metrics import *
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import os.path as path

def video_iou_plot(gt, det, video_path, title='', save_path='results'):
    frames = list(gt.keys())
    overlaps = []

    for frame in frames:
        boxes1 = [d.bbox for d in gt.get(frame)]
        boxes2 = [d.bbox for d in det.get(frame, [])]
        iou = mean_intersection_over_union(boxes1, boxes2)
        overlaps.append(iou)

    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    image = ax[0].imshow(np.zeros((height, width)))
    line, = ax[1].plot(frames, overlaps)
    artists = [image, line]

    def update(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, img = cap.read()
        for d in gt[frames[i]]:
            cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 255, 0), 2)
        for d in det[frames[i]]:
            cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 0, 255), 2)
        artists[0].set_data(img[:, :, ::-1])
        artists[1].set_data(frames[:i + 1], overlaps[:i + 1])
        return artists

    ani = animation.FuncAnimation(fig, update, len(frames), interval=2, blit=True)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('#frame')
    ax[1].set_ylabel('mean IoU')
    fig.suptitle(title)
    if save_path is not None:
        if not path.exists(save_path):
            os.makedirs(save_path)
        ani.save(os.path.join(save_path, 'video_iou.gif'), writer='ffmpeg')
    else:
        plt.show()