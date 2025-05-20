import albumentations as A
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

image_dir = "dataset/images"
label_dir = "dataset/labels"
aug_image_dir = "dataset/images_aug"
aug_label_dir = "dataset/labels_aug"

os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

transform = A.Compose([
    A.Affine(
        scale=(0.8, 1.2),
        rotate=(-10, 10),
        shear=(-7, 7),
        p=0.7
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MedianBlur(blur_limit=5, p=0.5),
        A.MotionBlur(blur_limit=(3, 7), p=0.5),
    ], p=0.3), # Apply one type of blur
    A.Perspective(scale=(0.02, 0.05), p=0.35),
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])


for img in tqdm(image_files, desc="Applying augmentations"):
    base_name = os.path.splitext(img)[0]
    img = os.path.join(image_dir, img)
    label = os.path.join(label_dir, base_name + '.txt')

    image = Image.open(img)
    image_np = np.array(image)

    with open(label, 'r', encoding='utf-8') as f:
        label_text = f.read()

    cv2.imwrite(os.path.join(aug_image_dir, f"{base_name}_orig.png"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    with open(os.path.join(aug_label_dir, f"{base_name}_orig.txt"), 'w', encoding='utf-8') as f:
        f.write(label_text)

    for i in range(5):
        aug = transform(image=image_np)
        aug_image = aug['image']
        aug_name = f"{base_name}_aug{i+1:02d}"
        cv2.imwrite(os.path.join(aug_image_dir, f"{aug_name}.png"), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        with open(os.path.join(aug_label_dir, f"{aug_name}.txt"), 'w', encoding='utf-8') as f:
            f.write(label_text)


print('augmentation complete!')