import cv2
import numpy as np
from ultralytics import YOLO
from hezar.models import Model
import mediapipe as mp
import re

def load_models():
    """بارگذاری مدل‌ها با MediaPipe"""
    models = {}
    
    # MediaPipe Face Detection
    models['face_detection'] = mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # 1 for full-range detection
        min_detection_confidence=0.5
    )
    models['mp_drawing'] = mp.solutions.drawing_utils
    
    # YOLO models - مسیرها رو تغییر بدید
    models['objects_model'] = YOLO('models/TextDetection.pt')
    models['card_model'] = YOLO('models/CardDetection.pt') 
    models['seg_model'] = YOLO('models/card_segmentation.pt')
    
    # Hezar OCR
    models['hezar_ocr'] = Model.load("hezarai/crnn-base-fa-v2")
    
    return models

def detect_face_mediapipe(image, face_detection):
    """تشخیص چهره با MediaPipe"""
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        
        if results.detections:
            detection = results.detections[0]  # اولین چهره
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = image.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            # اطمینان از مقادیر صحیح
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            return {'bbox': [x1, y1, x2, y2]}
        
        return None
    except Exception as e:
        print(f"خطا در تشخیص چهره: {e}")
        return None

def rotation(image_path, models):
    """چرخش و تصحیح کارت با MediaPipe"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("خطا در خواندن تصویر")
            return None
            
        results = models['card_model'](image)

        if len(results[0].boxes) == 0:
            print("کارت ملی تشخیص داده نشد!")
            return None

        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
        cropped_card = image[y1:y2, x1:x2]

        # استفاده از MediaPipe به جای InsightFace
        face_result = detect_face_mediapipe(cropped_card, models['face_detection'])

        if face_result:
            x1, y1, x2, y2 = face_result['bbox']
            w, h = x2 - x1, y2 - y1
            face_center = (x1 + w // 2, y1 + h // 2)
            card_center = (cropped_card.shape[1] // 2, cropped_card.shape[0] // 2)

            angle = np.arctan2(face_center[1] - card_center[1], face_center[0] - card_center[0]) * (180 / np.pi) + 180
            rotation_matrix = cv2.getRotationMatrix2D(card_center, angle, 1.0)
            rotated = cv2.warpAffine(cropped_card, rotation_matrix, (cropped_card.shape[1], cropped_card.shape[0]))

            rotated_per = segmentation_perspective(rotated, models['seg_model'])
            return rotated_per

        corrected_card = segmentation_perspective(cropped_card, models['seg_model'])
        return corrected_card
        
    except Exception as e:
        print(f"خطا در چرخش کارت: {e}")
        return None

def process_img(img, models, confidence_threshold=0.5):
    """استخراج متن از تصویر"""
    try:
        data = {}
        results = models['objects_model'](img)
        box_confidences = {}

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                label = result.names[int(box.cls)]
                conf = float(box.conf)
                
                if conf < confidence_threshold:
                    continue

                cropped_img = img[max(0, y1-3):y2+3, max(0, x1-3):x2+3]
                
                if cropped_img.size == 0:
                    continue
                    
                ocr_result = models['hezar_ocr'].predict(cropped_img)

                if ocr_result:
                    text_list = [item['text'] for item in ocr_result]
                    text = " ".join(text_list)
                else:
                    text = ""

                if label in box_confidences:
                    if conf > box_confidences[label][1]:
                        box_confidences[label] = (text, conf)
                else:
                    box_confidences[label] = (text, conf)

        for label, (text, conf) in box_confidences.items():
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            data[label] = cleaned_text

        return data
        
    except Exception as e:
        print(f"خطا در پردازش تصویر: {e}")
        return {}

def segmentation_perspective(image, seg_model):
    """تصحیح پرسپکتیو - همان کد قبلی"""
    try:
        results = seg_model(image)

        for r in results:
            masks = r.masks
            if masks is not None:
                for mask in masks.xy:
                    pts = np.array(mask, dtype=np.int32)
                    height, width = image.shape[:2]
                    mask_binary = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask_binary, [pts], 255)

                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                        
                    contour = max(contours, key=cv2.contourArea)

                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) == 4:
                        corners = approx.reshape(4, 2)
                        corners = sorted(corners, key=lambda x: x[1])
                        top_corners = sorted(corners[:2], key=lambda x: x[0])
                        bottom_corners = sorted(corners[2:], key=lambda x: x[0])
                        ordered_corners = [top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]]

                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        if aspect_ratio > 1:
                            width = 1180
                            height = int(width / aspect_ratio)
                        else:
                            height = 750
                            width = int(height * aspect_ratio)

                        ideal_coords = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
                        matrix = cv2.getPerspectiveTransform(np.array(ordered_corners, dtype="float32"), ideal_coords)

                        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                        sharpened = cv2.filter2D(image, -1, kernel)
                        corrected_image = cv2.warpPerspective(sharpened, matrix, (width, height), flags=cv2.INTER_LANCZOS4)

                        return corrected_image
                    else:
                        return image
            else:
                return image
        
        return image
        
    except Exception as e:
        print(f"خطا در تصحیح پرسپکتیو: {e}")
        return image
