import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

class HistogramEqualization:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        elif len(img.shape) == 2:
            img = cv2.equalizeHist(img)
        return Image.fromarray(img)
