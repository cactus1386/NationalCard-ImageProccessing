import cv2
import numpy as np
import random

background = cv2.imread("aa.jpg")


def random_national_id():
    base = [random.randint(0, 9) for _ in range(9)]
    check = sum(d * (10 - i) for i, d in enumerate(base)) % 11
    check_digit = check if check < 2 else 11 - check
    return "".join(map(str, base)) + str(check_digit)


def add_text(image, text, position, font_scale=1, color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, 2)


for i in range(100):
    img = background.copy()
    national_id = random_national_id()
    birth_date = f"{random.randint(1, 31)}/{random.randint(1, 12)}/{random.randint(1300, 1405)}"

    add_text(img, national_id, (120, 150))
    add_text(img, birth_date, (200, 200))

    cv2.imwrite(f"synthetic_dataset/synthetic_{i}.jpg", img)
