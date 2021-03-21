from W3.Reader import *
import cv2
import torch
class CustomTorchDataset:
    """Face Landmarks dataset."""

    def __init__(self, gt_file, root_dir, transform=None):
        """
           Args:
               gt_file (string): Path to the file with annotations.
               root_dir (string): Directory with all the images.
               transform (callable, optional): Optional transform to be applied
                   on a sample.
           """
        self.gt_file = gt_file
        self.root_dir = root_dir
        self.transform = transform
        reader = AICityChallengeAnnotationReader(path=gt_file)
        self.annotations = reader.get_annotations(classes=['car'])

    def __len__(self):
        videoCapture = cv2.VideoCapture(self.root_dir)
        return int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        videoCapture = cv2.VideoCapture(self.root_dir)
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = videoCapture.read()

        if self.transform is not None:
            img = self.transform(img)
        currentBB = None
        for x in self.annotations:
            if(x == idx):
                currentBB = self.annotations[x]
                break
        currentBoxes = []
        for x in currentBB:
            currentBoxes.append(x.bbox)
        boxes = torch.as_tensor(currentBoxes, dtype=torch.float32)
        labels = torch.full((len(boxes),), 3, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}

        return img, target
