from PIL import Image, ImageDraw, ImageFont
import random

background = Image.open("./Dataset/C/aa.jpg")

draw = ImageDraw.Draw(background)
font = ImageFont.truetype("arial.ttf", 35)

draw.text((750, 195), "۱۲۳۴۵۶۷۸۹۰", font=font, fill="black")
draw.text((760, 425), "۱۳۶۰/۱۲/۰۹", font=font, fill="black")
draw.text((760, 570), '۱۳۶۰/۱۲/۰۹', font=font, fill="black")

background.save("./Dataset/S/synthetic_card.jpg")
