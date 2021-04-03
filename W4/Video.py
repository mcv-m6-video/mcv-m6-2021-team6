import cv2
import os
class Video:
    def __init__(self, path, reheight, rewidht):
        self.pathVideo = path
        self.capture = cv2.VideoCapture(path)
        self.resultsPath = "results"
        if not os.path.exists(self.resultsPath):
            os.makedirs(self.resultsPath)
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.rewidth = rewidht
        self.reheight = reheight



if __name__ == '__main__':
    video = Video("videos/video_test.mp4")
