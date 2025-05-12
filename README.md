
# National ID Card Detection and OCR

NationalCard-ImageProcessing is a Python-based tool for automatically detecting and extracting information from Iranian national ID cards — regardless of their orientation, angle, or perspective. The system processes input images and returns the extracted data (like National ID number and birthdate) in a structured table format.




## Installation

1. Clone the repository:

```bash
git clone https://github.com/cactus1386/NationalCard-ImageProccessing.git
cd NationalCard-ImageProccessing
```

2. Set images directory path:
```bash
detect('your-path')
```

3. Run cells in main.ipynb file!
## Example Image

Place an example ID card image here:

![98](https://github.com/user-attachments/assets/f6c0b2c6-7068-452a-885a-b7fb3f593ad7)


## Example Output

```json
{
  "image_id": "98",
  "national_id": "۳۶۳۱۶۹۸۳۵۸",
  "first_name": "ساحل",
  "last_name": "روانخش",
  "birth_year": "۱۳۶۵",
  "birth_month": "۰۵",
  "birth_day": "۲۱",
  "father_name": "بهمن",
  "expiry_year": "۱۴۰۹",
  "expiry_month": "۱۰",
  "expiry_day": "۲۱"
}
```
## Contact
Radin
- [radinam1386@gmail.com](mailto:radinam1386@gmail.com)
- Telegram: [@KhodeRadinam](https://t.me/KhodeRadinam)

Ali
- [ali.najafpour07@gmail.com](ali.najafpour07@gmail.com)
- Telegram: [@Ali_Najafpour07](https://t.me/Ali_Najafpour07)
## Developers

code by Radin Almasi and Ali Najafpour
