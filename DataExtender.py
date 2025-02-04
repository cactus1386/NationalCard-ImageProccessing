import cv2
import numpy as np
import albumentations as A
import os
from glob import glob

input_folder = "dataset/original/"
output_folder = "dataset/augmented/"
os.makedirs(output_folder, exist_ok=True)

augment = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
])

image_paths = glob(os.path.join(input_folder, "*.jpg"))

for img_path in image_paths:
    image = cv2.imread(img_path)

    for i in range(5):
        augmented = augment(image=image)["image"]
        output_path = os.path.join(
            output_folder, f"{os.path.basename(img_path).split('.')[0]}_aug_{i}.jpg")
        cv2.imwrite(output_path, augmented)

print(output_folder)
