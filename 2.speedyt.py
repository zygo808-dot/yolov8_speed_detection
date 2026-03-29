import cv2
import numpy as np
import argparse
import time
import math
import subprocess
import json
from collections import defaultdict, deque
from datetime import datetime

# ================= YOUTUBE STREAM =================
def get_youtube_stream(url):
    try:
        cmd = ["yt-dlp", "-j", url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return data["url"]
    except Exception as e:
        print("[ERROR] Gagal mengambil stream YouTube:", e)
        return None

# ================= YOLO =================
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics belum terinstall.")
    print("Jalankan: pip install ultralytics")
    exit(1)

# ================= CONFIG =================
CLASS_COLORS = {
    "car": (0, 200, 255),
    "motorcycle": (0, 255, 128),
    "truck": (255, 80, 80),
    "bus": (200, 80, 255),
    "bicycle": (255, 220, 0),
}
VEHICLE_CLASSES = {2:"car",3:"motorcycle",5:"bus",7:"truck",1:"bicycle"}

# ================= TRACKER =================
class SpeedTracker:
    def __init__(self, mpp, fps):
        self.mpp = mpp
        self.fps = fps
        self.positions = defaultdict(lambda: deque(maxlen=15))
        self.speeds = defaultdict(lambda: deque(maxlen=5))

    def update(self, tid, cx, cy, frame_no):
        pos = self.positions[tid]
        pos.append((frame_no, cx, cy))

        if len(pos) < 2:
            return 0

        f0,x0,y0 = pos[0]
        f1,x1,y1 = pos[-1]

        d_pixel = math.hypot(x1-x0, y1-y0)
        d_meter = d_pixel * self.mpp
        d_time  = (f1-f0)/self.fps

        if d_time == 0:
            return 0

        speed = (d_meter/d_time)*3.6
        self.speeds[tid].append(speed)
        return np.mean(self.speeds[tid])

# ================= MAIN =================
def detect_speed(video_path, output="output.mp4", scale=0.05, show=False):

    print("[INFO] Load model...")
    model = YOLO("yolov8n.pt")

    # ===== DETECT SOURCE =====
    if "youtube.com" in video_path or "youtu.be" in video_path:
        print("[INFO] Ambil stream YouTube...")
        stream = get_youtube_stream(video_path)
        if stream is None:
            return
        cap = cv2.VideoCapture(stream)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Video tidak bisa dibuka")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(3))
    h = int(cap.get(4))

    out = cv2.VideoWriter(output,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w,h))

    tracker = SpeedTracker(scale, fps)
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results[0].boxes is not None:
            boxes = results[0].boxes
            cls = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else range(len(cls))

            for c, (x1,y1,x2,y2), tid in zip(cls, xyxy, ids):

                if c not in VEHICLE_CLASSES:
                    continue

                label = VEHICLE_CLASSES[c]
                color = CLASS_COLORS[label]

                cx, cy = (x1+x2)//2, (y1+y2)//2
                speed = tracker.update(tid, cx, cy, frame_no)

                # BOX
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"{label} {speed:.1f} km/h",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        out.write(frame)

        if show:
            cv2.imshow("Speed Detection", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("[SELESAI] Output:", output)

# ================= RUN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    detect_speed(args.video, args.output, args.scale, args.show)