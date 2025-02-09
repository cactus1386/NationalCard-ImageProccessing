from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('best.pt')

image_path = "Dataset/D/1.jpg"
image = cv2.imread(image_path)

results = model.predict(source=image, conf=0.3)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls)]
        confidence = box.conf.item()

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.axis("off")
plt.show()
