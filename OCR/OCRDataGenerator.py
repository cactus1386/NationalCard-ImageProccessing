from PIL import Image, ImageDraw, ImageFont
import os
import random
import arabic_reshaper
from bidi.algorithm import get_display

fonts = [
    './Font/B Titr Bold_0.ttf',
    './Font/B-NAZANIN.TTF',
    './Font/Vazir-Bold.ttf',
    './Font/Vazir-Medium.ttf',
    './Font/Vazir.ttf'
]
font_size = random.randint(25, 35)
width = 200
height = 70

def generate_national():
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

names = [
    "علی", "محمد", "رضا", "حسین", "مهدی", "امیر", "یوسف", "ابوالفضل", "سینا", "مبین",
    "پارسا", "کیان", "نیما", "شایان", "عرفان", "سهیل", "طاها", "آرمین", "آریا", "نریمان",
    "زهرا", "فاطمه", "نگار", "مریم", "سارا", "ریحانه", "مهسا", "نسترن", "آزاده", "الهام",
    "حدیث", "یلدا", "شادی", "ساناز", "نازنین", "ثریا", "پریسا", "رعنا", "نسیم", "آیدا",
    "باران", "هستی", "نرگس", "رها", "ماندانا", "ساحل", "کتایون", "ترانه", "پرستو", "بهاره",
    "سمیرا", "شکوفه", "گلناز", "ملیحه", "سپیده", "شیرین", "لعیا", "پونه", "شیما", "فرناز",
    "سحر", "مهناز", "آرزو", "شفق", "نغمه", "فرزانه", "مهشید", "لیلا", "نیلوفر", "بهناز",
    "رزیتا", "مرجان", "مونا", "مهتاب", "بهار", "پگاه", "میترا", "افسانه", "الهه", "عاطفه",
    "نگین", "ثنا", "شبنم", "ندا", "مرضیه", "مژده", "راحله", "نوشین", "کیمیا", "فرشته",
    "رویا", "حنا", "سوده", "ساره", "یاسمن", "فرنوش", "رکسانا", "صنم", "آتنا", "سولماز",
]

lasts = [
    "رضایی", "کاظمی", "کریمی", "قاسمی", "احمدی", "موسوی", "شریفی", "مرادی", "یوسفی", "جعفری",
    "محمدی", "ابراهیمی", "اکبری", "حیدری", "نعمتی", "نصیری", "صادقی", "عزیزی", "نجفی", "خسروی",
    "حسینی", "رستمی", "عبدی", "امیری", "زارعی", "ملکی", "طالبی", "فتحی", "یعقوبی", "سلیمانی",
    "رجبی", "سلطانی", "قنبری", "راستی", "حکیمی", "توکلی", "انصاری", "اسدی", "شجاعی", "نوری",
    "کاملی", "دولتی", "طاهری", "باقری", "مهدوی", "نیکو", "نعمتی", "انوری", "امجدی", "آذری",
    "همتی", "ربیعی", "جهانی", "نعمتی", "فیضی", "پرهام", "دشتی", "سعیدی", "اسماعیلی", "بیاتی",
    "فخیمی", "محجوب", "صارمی", "میرزایی", "قنبری", "میرزاخانی", "حسنی", "چگنی", "فدوی", "عظیمی",
]

fathers = [
    "حسن", "حسین", "جعفر", "اکبر", "ناصر", "حبیب", "کریم", "صادق", "یوسف", "محمود",
    "غلام", "کاظم", "رمضان", "رحیم", "سعید", "حمید", "قاسم", "مرتضی", "ابراهیم", "رضا",
    "فرهاد", "فرخ", "ایرج", "فرزاد", "جمشید", "تورج", "فریدون", "فرید", "علی‌اکبر", "علی‌اصغر",
    "شهرام", "مسعود", "محسن", "مهدی", "مجید", "بهمن", "پرویز", "کیوان", "شهریار", "افشین",
]

def convert_to_persian_numbers(text):
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    return ''.join(persian_digits[int(ch)] if ch.isdigit() else ch for ch in text)

with open('data.txt', 'w', encoding='utf-8') as f:
    for _ in range(1500):
        first = random.choice(names)
        last = random.choice(lasts)
        father = random.choice(fathers)
        birth = generate_date(1290, 1410)
        expire = generate_date(1350, 1450)
        national = generate_national()

        birth = convert_to_persian_numbers(birth)
        expire = convert_to_persian_numbers(expire)
        national = convert_to_persian_numbers(national)

        data = f'{first} \n {last} \n {father} \n {birth} \n {expire} \n {national}'

        f.write(data + '\n')

print('generate data complete and save in data.txt')
print('______________________________________________________________')

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

with open('data.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()


for i, text in enumerate(lines):
    img = Image.new('RGB', (width, height), color=(153,212,230))
    font = ImageFont.truetype(random.choice(fonts), font_size)
    reshaped_text = arabic_reshaper.reshape(text.strip())
    bidi_text = get_display(reshaped_text)
    draw = ImageDraw.Draw(img)


    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), bidi_text, font=font, fill=(0, 0, 0))

    image = f"dataset/images/{i:04d}.png"
    img.save(image)

    label = f"dataset/labels/{i:04d}.txt"
    with open(label, 'w', encoding='utf-8') as f:
        f.write(text)

print('well done!!! dataset is ready.')