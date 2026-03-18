import cv2
import time
import math
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Buka kamera laptop
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Menyimpan posisi dan waktu sebelumnya
prev_positions = {}
prev_times = {}

# ===== KALIBRASI PIXEL KE METER =====
# contoh: 100 pixel = 1 meter
PIXEL_TO_METER = 1/100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # YOLO tracking
    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id

        if ids is not None:
            ids = ids.cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box

                # titik tengah objek
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                speed_kmh = 0

                if obj_id in prev_positions:
                    prev_x, prev_y = prev_positions[obj_id]
                    prev_time = prev_times[obj_id]

                    dt = current_time - prev_time

                    if dt > 0:
                        # jarak pixel
                        dist_pixel = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)

                        # konversi meter
                        dist_meter = dist_pixel * PIXEL_TO_METER

                        # meter per second
                        speed_mps = dist_meter / dt

                        # km per hour
                        speed_kmh = speed_mps * 3.6

                # simpan posisi terbaru
                prev_positions[obj_id] = (cx, cy)
                prev_times[obj_id] = current_time

                # ===== BOUNDING BOX MERAH =====
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 0, 255), 2)

                # titik tengah
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # teks kecepatan
                text = f"ID {int(obj_id)} : {speed_kmh:.2f} km/h"

                cv2.putText(frame,
                            text,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

    cv2.imshow("YOLOv8 Speed Detection", frame)

    # tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()