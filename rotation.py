import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("CardDetection.pt")

image = cv2.imread("./test_image_phase1/2993.PNG")

results = model(image)
boxes = results[0].boxes

if len(boxes) == 0:
    print("No card detected!")
    exit()

x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])

cropped_card = image[y1:y2, x1:x2]


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

detected_faces = face_cascade.detectMultiScale(cropped_card, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

if len(detected_faces) == 0:
    print("No face detected on the card!")
    exit()

(x, y, w, h) = detected_faces[0]
face_center = (x + w // 2, y + h // 2)

card_center = (cropped_card.shape[1] // 2, cropped_card.shape[0] // 2)
angle = np.arctan2(face_center[1] - card_center[1], face_center[0] - card_center[0]) * (180 / np.pi)

(h, w) = cropped_card.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D(card_center, angle, 1.0)
rotated = cv2.warpAffine(cropped_card, rotation_matrix, (w, h))

cv2.imwrite("aligned_card.jpg", rotated)

print(f"Detected Rotation Angle: {angle:.2f} degrees")
