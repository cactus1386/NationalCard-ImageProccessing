import cv2
import os
import csv

image_folder = "../../archive/iranian_national_id_card_images/iranian_national_id_card_images"

output_csv = "./corner_labels.csv"

if not os.path.exists(output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])

image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
points = []

def click_event(event, x, y, flags, param):
    global points, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", current_image)

for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    current_image = image.copy()
    points = []

    print(f"Labeling: {filename}")
    print("top-left → top-right → bottom-right → bottom-left")
    
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    while True:
        cv2.imshow("Image", current_image)
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4:
            break
        elif key == ord('q'):
            print("⛔ Skipped")
            break

    cv2.destroyAllWindows()

    if len(points) == 4:
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            flat_points = [coord for pt in points for coord in pt]
            writer.writerow([filename] + flat_points)
