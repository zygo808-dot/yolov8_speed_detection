"""
=============================================================
  Deteksi Kecepatan Kendaraan dengan YOLOv8
  Vehicle Speed Detection using YOLOv8 + ByteTrack
=============================================================
  Konfigurasi Kamera (sudah dikalibrasi):
    Ketinggian kamera : 36 meter
    Tilt kamera       : 60 derajat ke bawah (dari vertikal)
    Resolusi          : 1920 x 1080
    FPS               : 60

  Ilustrasi sudut tilt:
         Kamera
          |  <-- vertikal (0 derajat)
          | \
          |  \ <-- tilt 60 derajat
       36m|   \
          |    \
          |     \
         Tanah----> titik pandang (20.78 m dari bawah kamera)

  Perhitungan scale otomatis:
    tilt           = 60 derajat (dari vertikal)
    jarak_pandang  = 36 / tan(60) = 36 / 1.732 = 20.78 meter
    scale          = 20.78 / 1080 = 0.0192 m/piksel

  --tilt memakai sudut dari VERTIKAL (sumbu lurus ke bawah = 0):
    0  derajat = kamera tepat lurus ke bawah
    45 derajat = kamera miring 45 derajat
    60 derajat = kamera miring cukup jauh  <-- kasus ini
    90 derajat = kamera sejajar tanah (tidak valid)

  Cara pakai / Usage:
    python 1_speed.py --video input.mp4 --output hasil.mp4 --show
    python 1_speed.py --video input.mp4 --output hasil.mp4

  Argumen / Arguments:
    --video   : Path file video input
    --output  : Path file video output (default: output.mp4)
    --fps     : Override FPS video (default: baca dari metadata)
    --scale   : Override meter per piksel (default: hitung otomatis)
    --conf    : Confidence threshold (default: 0.4)
    --show    : Tampilkan preview saat proses (flag)
    --height  : Ketinggian kamera dalam meter (default: 36)
    --tilt    : Tilt kamera dari vertikal dalam derajat (default: 60)

  Model YOLO:
    yolov8n   cepat sekali   akurasi rendah
    yolov8s   cepat          akurasi sedang
    yolov8m   ideal          akurasi tinggi   <- dipakai
    yolov8l   lambat         akurasi sangat tinggi
    yolov8x   sangat lambat  akurasi maksimal
=============================================================
"""

import cv2
import numpy as np
import argparse
import time
import math
from collections import defaultdict, deque
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics belum terinstall.")
    print("  Jalankan: pip install ultralytics")
    exit(1)


# ─────────────────────────────────────────────
#  Konstanta Kalibrasi Kamera
# ─────────────────────────────────────────────
CAMERA_HEIGHT_M  = 36.0   # ketinggian kamera dari tanah (meter)
CAMERA_TILT_DEG  = 60.0   # tilt dari vertikal: 0=lurus bawah, 60=miring jauh
DEFAULT_FPS      = 60.0   # FPS default kamera
DEFAULT_RES_W    = 1920   # resolusi lebar (piksel)
DEFAULT_RES_H    = 1080   # resolusi tinggi (piksel)


def hitung_scale(tinggi_m: float, tilt_deg: float,
                 res_h: int = DEFAULT_RES_H) -> float:
    """
    Hitung meter per piksel berdasarkan parameter kamera.

    Parameter:
      tinggi_m : ketinggian kamera dari tanah (meter)
      tilt_deg : tilt dari vertikal (derajat)
                   0  = kamera lurus ke bawah
                   60 = miring 60 derajat dari vertikal  <-- kasus ini
      res_h    : resolusi vertikal video (piksel)

    Rumus:
      jarak_pandang = tinggi / tan(tilt)
      scale         = jarak_pandang / res_h

    Contoh kamera ini:
      jarak = 36 / tan(60) = 36 / 1.732 = 20.78 m
      scale = 20.78 / 1080 = 0.0192 m/piksel
    """
    if not (0 < tilt_deg < 90):
        raise ValueError(f"tilt_deg harus antara 0 dan 90 (eksklusif), dapat: {tilt_deg}")
    tilt_rad      = math.radians(tilt_deg)
    jarak_pandang = tinggi_m / math.tan(tilt_rad)
    return jarak_pandang / res_h


# Scale default hasil kalibrasi: tinggi=36m, tilt=60 derajat
# → 0.0192 m/piksel
DEFAULT_SCALE = hitung_scale(CAMERA_HEIGHT_M, CAMERA_TILT_DEG)


# ─────────────────────────────────────────────
#  Konfigurasi Warna & Label
# ─────────────────────────────────────────────
CLASS_COLORS = {
    "car":        (0,   200, 255),   # kuning-oranye
    "motorcycle": (0,   255, 128),   # hijau terang
    "truck":      (255,  80,  80),   # merah
    "bus":        (200,  80, 255),   # ungu
    "bicycle":    (255, 220,   0),   # kuning
}
DEFAULT_COLOR = (200, 200, 200)

# Kelas kendaraan yang dipantau (COCO class IDs)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}

SPEED_COLORS = [
    (0,  230,   0),   # hijau  : < 40 km/h
    (0,  200, 255),   # kuning : 40-80 km/h
    (0,   80, 255),   # oranye : 80-120 km/h
    (0,    0, 220),   # merah  : > 120 km/h
]


def speed_color(kmh: float):
    if kmh < 40:  return SPEED_COLORS[0]
    if kmh < 80:  return SPEED_COLORS[1]
    if kmh < 120: return SPEED_COLORS[2]
    return SPEED_COLORS[3]


# ─────────────────────────────────────────────
#  Tracker Kecepatan Per Kendaraan
# ─────────────────────────────────────────────
class SpeedTracker:
    def __init__(self, meters_per_pixel: float, fps: float,
                 history: int = 20, smooth: int = 8):
        """
        meters_per_pixel : skala konversi piksel ke meter (hasil kalibrasi)
        fps              : frame per detik video
        history          : jumlah frame posisi yang disimpan
                           dinaikkan ke 20 karena FPS tinggi (60fps)
        smooth           : jumlah sampel kecepatan untuk rata-rata halus
                           dinaikkan ke 8 agar angka tidak loncat-loncat
        """
        self.mpp     = meters_per_pixel
        self.fps     = fps
        self.history = history
        self.smooth  = smooth

        # {track_id: deque[(frame_no, cx, cy)]}
        self.positions: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history)
        )
        # {track_id: deque[speed_kmh]}
        self.speeds: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=smooth)
        )
        self.max_speeds: dict[int, float] = {}

    def update(self, track_id: int, cx: int, cy: int, frame_no: int) -> float:
        """Perbarui posisi dan kembalikan kecepatan saat ini (km/h)."""
        pos = self.positions[track_id]
        pos.append((frame_no, cx, cy))

        if len(pos) < 2:
            return 0.0

        f0, x0, y0 = pos[0]
        f1, x1, y1 = pos[-1]
        d_frames = f1 - f0
        if d_frames == 0:
            return 0.0

        d_pixel = math.hypot(x1 - x0, y1 - y0)
        d_meter = d_pixel * self.mpp
        d_sec   = d_frames / self.fps
        kmh     = (d_meter / d_sec) * 3.6

        self.speeds[track_id].append(kmh)
        avg_kmh = float(np.mean(self.speeds[track_id]))

        if track_id not in self.max_speeds or avg_kmh > self.max_speeds[track_id]:
            self.max_speeds[track_id] = avg_kmh

        return avg_kmh

    def get_trail(self, track_id: int) -> list[tuple[int, int]]:
        """Kembalikan daftar (cx, cy) untuk menggambar jejak."""
        return [(x, y) for _, x, y in self.positions[track_id]]


# ─────────────────────────────────────────────
#  Overlay / HUD
# ─────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, alpha=0.6, radius=8):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_with_bg(img, text, pos, font_scale=0.55, thickness=1,
                     fg=(255, 255, 255), bg=(30, 30, 30), padding=4):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    draw_rounded_rect(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg, alpha=0.7
    )
    cv2.putText(img, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def draw_vehicle_box(img, x1, y1, x2, y2, track_id, label, kmh, color):
    """Kotak bounding box + label kecepatan."""
    spd_col = speed_color(kmh)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Sudut aksen
    L = 12
    for (sx, sy, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(img, (sx, sy), (sx + dx*L, sy), spd_col, 3)
        cv2.line(img, (sx, sy), (sx, sy + dy*L), spd_col, 3)

    # Label ID + tipe
    tag = f"#{track_id} {label}"
    put_text_with_bg(img, tag, (x1, y1 - 6),
                     font_scale=0.45, fg=(220, 220, 220), bg=color)

    # Badge kecepatan
    if kmh > 0:
        spd_txt = f"{kmh:.0f} km/h"
        put_text_with_bg(img, spd_txt, (x1, y2 + 16),
                         font_scale=0.52, fg=(255, 255, 255), bg=spd_col)


def draw_trail(img, trail: list, color):
    if len(trail) < 2:
        return
    pts = np.array(trail, dtype=np.int32)
    for i in range(1, len(pts)):
        alpha_ratio = i / len(pts)
        c = tuple(int(ch * alpha_ratio) for ch in color)
        cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), c, 2, cv2.LINE_AA)


def draw_hud(img, frame_no: int, fps: float, vehicle_count: int,
             total_detected: int, timestamp: str,
             scale: float, cam_height: float, cam_tilt: float,
             h: int, w: int):
    """Panel info di pojok kiri atas — termasuk info kalibrasi kamera."""
    jarak = cam_height / math.tan(math.radians(cam_tilt))
    lines = [
        f" Frame    : {frame_no}",
        f" Detik    : {frame_no/fps:.1f}s",
        f" Waktu    : {timestamp}",
        f" Di layar : {vehicle_count} kendaraan",
        f" Total    : {total_detected} kendaraan",
        f" ----------------------",
        f" Tinggi   : {cam_height:.0f} m",
        f" Tilt     : {cam_tilt:.0f} deg",
        f" Jarak    : {jarak:.1f} m",
        f" Scale    : {scale:.4f} m/px",
    ]
    panel_w, panel_h = 215, len(lines) * 18 + 18
    draw_rounded_rect(img, (8, 8), (8 + panel_w, 8 + panel_h),
                      (15, 15, 15), alpha=0.75)
    for i, line in enumerate(lines):
        warna = (180, 220, 255) if i < 5 else (120, 160, 120)
        cv2.putText(img, line, (12, 26 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    warna, 1, cv2.LINE_AA)


def draw_legend(img, h: int, w: int):
    """Legenda warna kecepatan di pojok kanan bawah."""
    items = [
        ("< 40 km/h",   SPEED_COLORS[0]),
        ("40-80 km/h",  SPEED_COLORS[1]),
        ("80-120 km/h", SPEED_COLORS[2]),
        ("> 120 km/h",  SPEED_COLORS[3]),
    ]
    panel_w, item_h = 145, 20
    panel_h = len(items) * item_h + 24
    x0 = w - panel_w - 8
    y0 = h - panel_h - 8
    draw_rounded_rect(img, (x0, y0), (w - 8, h - 8),
                      (15, 15, 15), alpha=0.75)
    cv2.putText(img, "Kecepatan", (x0 + 8, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    for i, (txt, col) in enumerate(items):
        y = y0 + 28 + i * item_h
        cv2.rectangle(img, (x0 + 8, y - 9), (x0 + 22, y + 3), col, -1)
        cv2.putText(img, txt, (x0 + 28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)


# ─────────────────────────────────────────────
#  Fungsi Utama
# ─────────────────────────────────────────────
def detect_speed(
    video_path:       str,
    output_path:      str   = "output.mp4",
    meters_per_pixel: float = None,   # None = hitung otomatis dari kamera
    fps_override:     float = None,   # None = baca dari metadata video
    conf_threshold:   float = 0.4,
    show_preview:     bool  = False,
    cam_height:       float = CAMERA_HEIGHT_M,
    cam_tilt:         float = CAMERA_TILT_DEG,
):
    # Hitung scale otomatis jika tidak diisi manual
    if meters_per_pixel is None:
        meters_per_pixel = hitung_scale(cam_height, cam_tilt)

    jarak_pandang = cam_height / math.tan(math.radians(cam_tilt))

    print("=" * 60)
    print("  Deteksi Kecepatan Kendaraan - YOLOv8")
    print("=" * 60)
    print(f"  Input         : {video_path}")
    print(f"  Output        : {output_path}")
    print()
    print("  -- Kalibrasi Kamera ---------------------------")
    print(f"  Ketinggian    : {cam_height} m")
    print(f"  Tilt          : {cam_tilt} derajat (dari vertikal)")
    print(f"  Jarak pandang : {jarak_pandang:.2f} m")
    print(f"  Scale         : {meters_per_pixel:.4f} m/piksel")
    print()
    print(f"  Conf threshold: {conf_threshold}")
    print()

    # ── Load Model ──────────────────────────────
    print("[1/3] Memuat model YOLOv8m ...")
    model = YOLO("yolov8m.pt")
    print("      Model siap.\n")

    # ── Buka Video ──────────────────────────────
    print("[2/3] Membuka video ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka video: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    fps     = fps_override if fps_override else src_fps
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sesuaikan scale jika resolusi video berbeda dari asumsi kalibrasi
    if h != DEFAULT_RES_H:
        print(f"  [!] Resolusi video ({w}x{h}) berbeda dari kalibrasi "
              f"({DEFAULT_RES_W}x{DEFAULT_RES_H}).")
        meters_per_pixel = hitung_scale(cam_height, cam_tilt, res_h=h)
        print(f"      Scale disesuaikan: {meters_per_pixel:.4f} m/piksel\n")

    print(f"      Resolusi : {w}x{h}")
    print(f"      FPS      : {fps:.1f}")
    print(f"      Frames   : {total_f}\n")

    # ── Writer ──────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # history=20, smooth=8 disesuaikan untuk FPS tinggi (60fps)
    tracker   = SpeedTracker(meters_per_pixel, fps, history=20, smooth=8)
    frame_no  = 0
    total_ids: set[int] = set()
    t_start   = time.time()

    print("[3/3] Memproses video ...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        ts = datetime.now().strftime("%H:%M:%S")

        # ── Inferensi YOLOv8 dengan ByteTrack ───
        results = model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        vis       = frame.copy()
        on_screen = 0

        if results[0].boxes is not None:
            boxes   = results[0].boxes
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs   = boxes.conf.cpu().numpy()
            xyxy    = boxes.xyxy.cpu().numpy().astype(int)
            ids     = (boxes.id.cpu().numpy().astype(int)
                       if boxes.id is not None else
                       range(len(cls_ids)))

            for cls_id, conf, (x1, y1, x2, y2), tid in \
                    zip(cls_ids, confs, xyxy, ids):

                if cls_id not in VEHICLE_CLASSES:
                    continue

                label  = VEHICLE_CLASSES[cls_id]
                color  = CLASS_COLORS.get(label, DEFAULT_COLOR)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                kmh = tracker.update(int(tid), cx, cy, frame_no)

                trail = tracker.get_trail(int(tid))
                draw_trail(vis, trail, color)

                draw_vehicle_box(vis, x1, y1, x2, y2,
                                 int(tid), label, kmh, color)

                total_ids.add(int(tid))
                on_screen += 1

        # ── HUD ─────────────────────────────────
        draw_hud(vis, frame_no, fps, on_screen, len(total_ids), ts,
                 meters_per_pixel, cam_height, cam_tilt, h, w)
        draw_legend(vis, h, w)

        # Progress bar tipis di bawah frame
        if total_f > 0:
            prog = int(w * frame_no / total_f)
            cv2.rectangle(vis, (0, h - 4), (prog, h), (0, 200, 120), -1)

        writer.write(vis)

        if show_preview:
            cv2.imshow("Vehicle Speed Detector", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  [!] Dihentikan oleh pengguna.")
                break

        # Log progres setiap 100 frame
        if frame_no % 100 == 0:
            elapsed = time.time() - t_start
            pct     = frame_no / total_f * 100 if total_f else 0
            print(f"    Frame {frame_no:>5}/{total_f}"
                  f"  ({pct:5.1f}%)"
                  f"  {elapsed:.1f}s")

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  SELESAI")
    print(f"  Frame diproses  : {frame_no}")
    print(f"  Total kendaraan : {len(total_ids)}")
    print(f"  Waktu proses    : {elapsed:.1f} detik")
    print(f"  Output tersimpan: {output_path}")

    # Kecepatan maksimum per kendaraan (top 10)
    if tracker.max_speeds:
        print()
        print("  Kecepatan Tertinggi per ID:")
        for tid, spd in sorted(tracker.max_speeds.items(),
                               key=lambda x: -x[1])[:10]:
            print(f"    ID #{tid:>4} -> {spd:.0f} km/h")
    print("=" * 60)


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deteksi kecepatan kendaraan menggunakan YOLOv8"
    )
    parser.add_argument("--video",  required=True,
                        help="Path file video input (contoh: rekaman.mp4)")
    parser.add_argument("--output", default="output.mp4",
                        help="Path file video output (default: output.mp4)")
    parser.add_argument("--scale",  type=float, default=None,
                        help="Override meter per piksel "
                             "(default: hitung otomatis dari --height dan --tilt)")
    parser.add_argument("--fps",    type=float, default=None,
                        help="Override FPS video (default: baca dari metadata)")
    parser.add_argument("--conf",   type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--show",   action="store_true",
                        help="Tampilkan preview saat memproses")
    parser.add_argument("--height", type=float, default=CAMERA_HEIGHT_M,
                        help=f"Ketinggian kamera dari tanah dalam meter "
                             f"(default: {CAMERA_HEIGHT_M})")
    parser.add_argument("--tilt",   type=float, default=CAMERA_TILT_DEG,
                        help=f"Tilt kamera dari vertikal dalam derajat "
                             f"(default: {CAMERA_TILT_DEG}) | "
                             f"0=lurus bawah, 60=miring jauh")
    args = parser.parse_args()

    detect_speed(
        video_path       = args.video,
        output_path      = args.output,
        meters_per_pixel = args.scale,
        fps_override     = args.fps,
        conf_threshold   = args.conf,
        show_preview     = args.show,
        cam_height       = args.height,
        cam_tilt         = args.tilt,
    )
