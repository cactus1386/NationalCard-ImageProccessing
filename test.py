import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import pandas as pd
import torch

# تنظیم GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# مدل‌ها برای تشخیص
objects_model = YOLO('TextDetection.pt')  # مدل تشخیص اجزای کارت
objects_model.to(device)  # انتقال مدل به GPU
card_model = YOLO('CardDetection.pt')  # مدل تشخیص کارت
card_model.to(device)  # انتقال مدل به GPU

# OCR برای تشخیص متن فارسی
ocr = easyocr.Reader(['fa'], gpu=torch.cuda.is_available())

# تابع تشخیص زاویه کارت


def detect_card_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # تبدیل به تصویر خاکستری
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # تشخیص لبه‌ها

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # تشخیص خطوط

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90  # تبدیل زاویه به درجه
            angles.append(angle)

        return np.median(angles)  # بازگشت زاویه میانگین

    return 0  # اگر خطی پیدا نشود، فرض می‌کنیم کارت صاف است

# تابع تصحیح زاویه کارت


def deskew_image(img):
    angle = detect_card_angle(img)  # تشخیص زاویه کارت

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(
        center, -angle, 1.0)  # ماتریس چرخش
    deskewed_img = cv2.warpAffine(
        img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed_img

# تابع برای برش کارت ملی


def crop_card(path):
    img = cv2.imread(path)  # خواندن تصویر
    img = deskew_image(img)  # تصحیح زاویه کارت

    results = card_model(img)  # تشخیص کارت با مدل

    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]
            # تبدیل مختصات به اعداد صحیح
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            cropped_img = img[y1:y2, x1:x2]  # برش کارت
            return cropped_img

# تابع پردازش کارت برای استخراج اطلاعات


def process_img(img):
    ignore_class = ['FatherName', 'LastName',
                    'Name']  # کلاس‌های نادیده گرفته شده
    data = {}

    results = objects_model(img)  # اجرای مدل تشخیص اجزا

    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            label = result.names[int(box.cls)]  # نام کلاس

            if label in ignore_class:  # حذف کلاس‌های نادیده گرفته شده
                continue

            cropped_img = img[(y1 + 7):(y2 + 7), (x1 + 7)                              :(x2 + 7)]  # برش ناحیه متن

            ocr_result = ocr.readtext(cropped_img)  # تشخیص متن

            for (bbox, text, conf) in ocr_result:
                print(f"Text: {text}, Confidence: {conf}")
                data[label] = text  # ذخیره متن استخراج شده بر اساس کلاس

    return data

# تابع اصلی برای پردازش تصاویر موجود در یک پوشه


def detect(folder):
    detected = []

    for img in os.listdir(folder):
        print(img)
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.heic')):
            img_path = os.path.join(folder, img)

            data = {'image_id': '', 'national_id': '', 'birth_year': '',
                    'birth_month': '', 'birth_day': '', 'expiry_year': '',
                    'expiry_month': '', 'expiry_day': ''}

            data['image_id'] = img.split('.')[0]

            card = crop_card(img_path)  # برش کارت ملی
            extracted_data = process_img(card)  # استخراج اطلاعات از کارت

            for key, value in extracted_data.items():
                value = value.replace(' ', '')

                if key == 'Expire':
                    try:
                        y, m, d = value.split('/')
                        data['expiry_year'] = int(y)
                        data['expiry_month'] = int(m)
                        data['expiry_day'] = int(d)
                    except:
                        pass

                if key == 'Birth':
                    try:
                        y, m, d = value.split('/')
                        data['birth_year'] = int(y)
                        data['birth_month'] = int(m)
                        data['birth_day'] = int(d)
                    except:
                        pass

                if key == 'National':
                    data['national_id'] = value

            detected.append(data)

    pd.DataFrame(detected).to_csv(
        'image_phase2.csv', index=False, encoding='utf-8')


# اجرای برنامه
folder_path = './test_image_phase1'  # مسیر پوشه تصاویر
detect(folder_path)
