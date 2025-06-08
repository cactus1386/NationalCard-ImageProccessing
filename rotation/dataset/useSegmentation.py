import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO

def extract_card_corners_from_mask(mask_xy, image_shape):
    pts = np.array(mask_xy, dtype=np.int32)
    height, width = image_shape[:2]
    mask_binary = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask_binary, [pts], 255)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype("int32")
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        ordered = np.array([
            corners[np.argmin(s)],
            corners[np.argmin(diff)],
            corners[np.argmax(s)],
            corners[np.argmax(diff)]
        ])
        return ordered
    
    else:
        return None

images_folder = "../../archive/iranian_national_id_card_images/iranian_national_id_card_images"
model_path = "../../models/card_segmentation.pt"
output_csv_path = "./card_corners.csv"
failed_images_path = "./failed_images.txt"

model = YOLO(model_path)
csv_data = [["filename", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]]
failed_images = []

for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)
        results = model(image)[0]
        found = False

        for mask_xy in results.masks.xy:
            corners = extract_card_corners_from_mask(mask_xy, image.shape)
            if corners is not None:
                flat_corners = corners.flatten().tolist()
                csv_data.append([filename] + flat_corners)
                found = True
                break

        if not found:
            failed_images.append(filename)

with open(output_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

with open(failed_images_path, "w") as f:
    for name in failed_images:
        f.write(name + "\n")
