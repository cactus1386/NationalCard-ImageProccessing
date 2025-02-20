# National ID Card Detection and OCR

This project detects and extracts information from Iranian national ID cards using a combination of object detection and OCR techniques.

## Installation

1. Clone the repository:
   ```bash
   https://github.com/cactus1386/NationalCard-ImageProccessing.git
   cd NationalCard-ImageProccessing
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place images of national ID cards in the `images` folder.
2. Set folder in your main path like this:

```python
folder = '/content/drive/MyDrive/images'
```


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

## Example Image

Place an example ID card image here:

![98](https://github.com/user-attachments/assets/f6c0b2c6-7068-452a-885a-b7fb3f593ad7)

## Developers:

code by Radin Almasi and Ali Najafpour

