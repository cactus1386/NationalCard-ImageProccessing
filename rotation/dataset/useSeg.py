import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 📌 تابع گوشه‌یابی با کانتور + approxPolyDP
def extract_card_corners_from_mask(mask_xy, image_shape):
    # تبدیل نقاط ماسک به قالب OpenCV
    pts = np.array(mask_xy, dtype=np.int32)
    
    # ساخت ماسک باینری
    height, width = image_shape[:2]
    mask_binary = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_binary, [pts], 255)

    # پیدا کردن کانتور و تقریب به چهارضلعی
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        corners = approx.reshape(4, 2).astype("int32")

        # مرتب‌سازی نقاط: tl, tr, br, bl
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        ordered = np.array([
            corners[np.argmin(s)],     # top-left
            corners[np.argmin(diff)],  # top-right
            corners[np.argmax(s)],     # bottom-right
            corners[np.argmax(diff)]   # bottom-left
        ])
        return ordered
    else:
        return None

# 🧠 مدل YOLOv8 segmentation
model = YOLO("../../models/card_segmentation.pt")  # جایگزین با مسیر مدل خودت

# 📷 بارگذاری تصویر
image_path = "../../archive/sample1.jpg"
image = cv2.imread(image_path)

# اجرای مدل
results = model(image)[0]

# استخراج و رسم گوشه‌ها
for mask_xy in results.masks.xy:
    corners = extract_card_corners_from_mask(mask_xy, image.shape)
    if corners is not None:
        # رسم دایره روی گوشه‌ها
        for (x, y) in corners:
            cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
        # اتصال گوشه‌ها با خط
        cv2.polylines(image, [corners.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 0), thickness=2)
    else:
        print("⛔ نتوانست ۴ گوشه پیدا کند.")

# نمایش تصویر نهایی
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Card Corners from Segmentation")
plt.axis("off")
plt.show()
