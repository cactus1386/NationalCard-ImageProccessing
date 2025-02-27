import cv2
import numpy as np
from ultralytics import YOLO
from mtcnn import MTCNN

model = YOLO("CardDetection.pt")

image = cv2.imread("./test_image_phase1/2975.PNG")

results = model(image)
boxes = results[0].boxes

if len(boxes) == 0:
    print("No card detected!")
    exit()

x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])

cropped_card = image[y1:y2, x1:x2]

mtcnn = MTCNN()

faces = mtcnn.detect_faces(cropped_card)

if len(faces) == 0:
    print("No face detected on the card!")
    exit()

x, y, w, h = faces[0]['box']  
face_center = (int(x + w // 2), int(y + h // 2))

card_center = (cropped_card.shape[1] // 2, cropped_card.shape[0] // 2)
angle = np.arctan2(face_center[1] - card_center[1],
                   face_center[0] - card_center[0]) * (180 / np.pi)

angle += 180

(h, w) = cropped_card.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D(card_center, angle, 1.0)
rotated = cv2.warpAffine(cropped_card, rotation_matrix, (w, h))

cv2.imwrite("aligned_card_mtcnn.jpg", rotated)

print(f"Detected Rotation Angle: {angle:.2f} degrees")
