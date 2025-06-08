import cv2
import os
import csv

image_folder = "../../archive/iranian_national_id_card_images/iranian_national_id_card_images"
output_csv = "./corner_labels.csv"
failed_images_path = "./failed_images.txt"

if not os.path.exists(output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])

with open(failed_images_path, "r") as f:
    failed_files = [line.strip() for line in f.readlines()]

scale = 0.25
points = []

def click_event(event, x, y, flags, param):
    global points, current_display, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x, real_y = int(x / scale), int(y / scale)
        points.append((real_x, real_y))
        cv2.circle(current_display, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("Image", current_display)

for filename in failed_files:
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        print(f"⛔ File not found: {filename}")
        continue

    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    display_image = cv2.resize(image, (int(width * scale), int(height * scale)))
    current_display = display_image.copy()
    points = []

    print(f"Labeling: {filename}")
    print("Order: top-left → top-right → bottom-right → bottom-left")

    cv2.imshow("Image", display_image)
    cv2.setMouseCallback("Image", click_event)

    while True:
        cv2.imshow("Image", current_display)
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4:
            break
        elif key == ord('q'):
            print("⛔ Skipped:", filename)
            break

    cv2.destroyAllWindows()

    if len(points) == 4:
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            flat_points = [coord for pt in points for coord in pt]
            writer.writerow([filename] + flat_points)
