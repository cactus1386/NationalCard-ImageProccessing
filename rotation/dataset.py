import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import os

class HomographyDataset(Dataset):
    def __init__(self, annotations_csv, images_dir, output_size=(320, 240)):
        self.data = pd.read_csv(annotations_csv)
        self.images_dir = images_dir
        self.output_size = output_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row['filename'])
        image = cv2.imread(image_path)

        # Resize image to fixed size
        image_resized = cv2.resize(image, self.output_size)
        h, w = self.output_size

        # Original points in the original image
        pts = np.array([
            [row['x1'], row['y1']],
            [row['x2'], row['y2']],
            [row['x3'], row['y3']],
            [row['x4'], row['y4']]
        ], dtype=np.float32)

        # Map original points to resized space
        scale_x = w / image.shape[1]
        scale_y = h / image.shape[0]
        pts_resized = pts * [scale_x, scale_y]

        # Define ideal rectangle (fixed points)
        ideal = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        # Offset prediction target
        target = (pts_resized - ideal).reshape(-1)
        input_tensor = self.transform(image_resized)

        return input_tensor, torch.tensor(target, dtype=torch.float32)
