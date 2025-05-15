from PIL import Image, ImageDraw, ImageFont
import os
import random

fonts = [
    './Font/B Titr Bold_0.ttf',
    './Font/B-NAZANIN.TTF',
    './Font/Vazir-Bold.ttf',
    './Font/Vazir-Medium.ttf',
    './Font/Vazir.ttf'
]

def generate_national_code():
    code = [random.randint(0, 9) for _ in range(9)]
    s = sum([code[i] * (10 - i) for i in range(9)])
    r = s % 11
    check_digit = r if r < 2 else 11 - r
    code.append(check_digit)
    return ''.join(map(str, code))

def generate_date(start_year=1300, end_year=1450):
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    if month <= 6:
        day = random.randint(1, 31)
    elif month == 12:
        day = random.randint(1, 29)
    else:
        day = random.randint(1, 30)
    return f"{year:04d}/{month:02d}/{day:02d}"


os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

for i, text in enumerate(texts):
    img = Image.new('RGB', (400, 100), color=(255, 255, 255))
    font_size = random.randint(20, 35)
    font = ImageFont.truetype(random.choice(fonts), font_size)
    draw = ImageDraw.Draw(img)

    draw.text((10, 25), text, font=font, fill=(0, 0, 0), direction='rtl')

    image = f"dataset/images/{i:04d}.png"
    img.save(image)

    label = f"dataset/labels/{i:04d}.txt"
    with open(label, 'w', encoding='utf-8') as f:
        f.write(text)