import cv2
import numpy as np
import albumentations as A
import os
from glob import glob

# 📌 1. مسیر تصاویر ورودی و خروجی
input_folder = "dataset/original/"
output_folder = "dataset/augmented/"
os.makedirs(output_folder, exist_ok=True)

# 📌 2. تعریف تکنیک‌های افزایش داده (Augmentation)
augment = A.Compose([
    A.Rotate(limit=15, p=0.5),  # چرخش تا ۱۵ درجه
    A.RandomBrightnessContrast(p=0.5),  # تغییر روشنایی و کنتراست
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),  # نویز گاوسی
    A.MotionBlur(blur_limit=3, p=0.3),  # محو شدگی حرکتی
    A.Perspective(scale=(0.02, 0.05), p=0.3),  # تغییر پرسپکتیو
])

# 📌 3. پردازش تمام تصاویر در فولدر
image_paths = glob(os.path.join(input_folder, "*.jpg"))

for img_path in image_paths:
    # خواندن تصویر
    image = cv2.imread(img_path)

    # اعمال افزایش داده
    for i in range(5):  # هر تصویر ۵ بار تغییر کند
        augmented = augment(image=image)["image"]
        output_path = os.path.join(
            output_folder, f"{os.path.basename(img_path).split('.')[0]}_aug_{i}.jpg")
        cv2.imwrite(output_path, augmented)

print(f"✅ افزایش داده انجام شد. تصاویر جدید در {output_folder} ذخیره شدند.")
