import cv2
from ultralytics import YOLO
import time
import serial

model = YOLO("pendeteksisampah.pt")

STREAM_URL = "http://192.168.4.102:81/stream"
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Gagal membuka stream ESP32-CAM!")
    exit()

frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

ser = serial.Serial('COM3', 9600)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)

    results = model.predict(source=frame, conf=0.3, verbose=False)

    detected_class = None

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Kelas: 0 = organik, 1 = anorganik, 2 = B3
            if cls == 0:
                detected_class = '1'
            elif cls == 1:
                detected_class = '2'
            elif cls == 2:
                detected_class = '3'

    if detected_class:
        ser.write(detected_class.encode())
        print(f"Deteksi kelas: {detected_class}")
    else:
        ser.write(b'0')

    cv2.imshow("Deteksi Sampah", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
