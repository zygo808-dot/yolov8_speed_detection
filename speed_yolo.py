import cv2
import time
from ultralytics import YOLO
import math

# Load model YOLOv8
model = YOLO("yolov8n.pt")  # model ringan

# Buka kamera laptop (0 = default webcam)
cap = cv2.VideoCapture(0)

# Simpan posisi sebelumnya
prev_positions = {}
prev_time = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Deteksi + tracking
    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id

        if ids is not None:
            ids = ids.cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                speed = 0

                if obj_id in prev_positions:
                    prev_x, prev_y = prev_positions[obj_id]
                    dt = current_time - prev_time[obj_id]

                    if dt > 0:
                        dist = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                        speed = dist / dt  # pixel per second

                # Simpan posisi sekarang
                prev_positions[obj_id] = (cx, cy)
                prev_time[obj_id] = current_time

                # Gambar bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

                # Tampilkan speed
                cv2.putText(frame, f"ID {int(obj_id)} Speed: {int(speed)} px/s",
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Speed Detection YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()