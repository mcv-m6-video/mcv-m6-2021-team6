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

    def get_result_video(self, name):
        # Stabilize video sequence
        return cv2.VideoWriter(f'{self.resultsPath}/{name}_result.mp4',
                              cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))

if __name__ == '__main__':
    video = Video("videos/video_test.mp4")
