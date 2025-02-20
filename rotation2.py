import cv2
import numpy as np
from ultralytics import YOLO
from mtcnn import MTCNN

# بارگذاری مدل YOLO برای تشخیص کارت ملی
model = YOLO("CardDetection.pt")

# خواندن تصویر
image = cv2.imread("./test_image_phase1/2975.PNG")

# اجرای YOLO برای تشخیص کارت ملی
results = model(image)
boxes = results[0].boxes

if len(boxes) == 0:
    print("No card detected!")
    exit()

# دریافت مختصات کارت
x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])

# برش کارت ملی
cropped_card = image[y1:y2, x1:x2]

# بارگذاری مدل MTCNN برای شناسایی چهره
mtcnn = MTCNN()

# شناسایی چهره‌ها
faces = mtcnn.detect_faces(cropped_card)

if len(faces) == 0:
    print("No face detected on the card!")
    exit()

# گرفتن اولین چهره (چون در کارت فقط یک چهره هست)
x, y, w, h = faces[0]['box']  # در اینجا اولین چهره را می‌گیریم
face_center = (int(x + w // 2), int(y + h // 2))

# محاسبه زاویه چرخش کارت
card_center = (cropped_card.shape[1] // 2, cropped_card.shape[0] // 2)
angle = np.arctan2(face_center[1] - card_center[1],
                   face_center[0] - card_center[0]) * (180 / np.pi)

# اضافه کردن 180 درجه به زاویه برای تصحیح چرخش
angle += 180

# چرخش تصویر
(h, w) = cropped_card.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D(card_center, angle, 1.0)
rotated = cv2.warpAffine(cropped_card, rotation_matrix, (w, h))

# ذخیره کارت اصلاح‌شده
cv2.imwrite("aligned_card_mtcnn.jpg", rotated)

print(f"Detected Rotation Angle: {angle:.2f} degrees")
