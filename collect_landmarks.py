"""
Thu landmark ban tay (MediaPipe) cho Module 1 - chu cai VSL.

Cai dat:  pip install "mediapipe==0.10.35" numpy pandas scikit-learn matplotlib joblib
Chay:     python collect_landmarks.py --person A
          (moi thanh vien dung --person khac nhau: A, B, C...)

Phim:
  r : bat dau/dung ghi cho nhan hien tai
  n : nhan tiep theo      p : nhan truoc do
  q : thoat
Du lieu luu vao landmarks.csv (ghi noi tiep, chay lai khong mat du lieu cu).
Moi dong: person, label, x0,y0,z0, ..., x20,y20,z20 (toa do tho, chua chuan hoa)
"""

import argparse
import csv
from pathlib import Path

import cv2

from common import load_labels
from hand_tracking import create_hand_landmarker, detect_hand, draw_hand_landmarks, next_timestamp_ms

# Danh sach nhan doc tu labels.txt (moi dong 1 nhan) - sua file do, khong sua o day
LABELS = load_labels()
SAMPLES_PER_LABEL = 250
CSV_PATH = Path(__file__).resolve().with_name("landmarks.csv")
CAMERA_INDEX = 0

parser = argparse.ArgumentParser()
parser.add_argument("--person", required=True, help="ma nguoi thu, vd A, B, C")
args = parser.parse_args()

if not LABELS:
    raise SystemExit("labels.txt chua co nhan. Hay them moi nhan tren mot dong.")

landmarker = create_hand_landmarker()
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Khong mo duoc camera, thu doi CAMERA_INDEX.")

new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
idx, recording, count = 0, False, 0
last_timestamp_ms = 0

try:
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            header = ["person", "label"] + [f"{a}{i}" for i in range(21) for a in "xyz"]
            writer.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # lat guong cho de nhin
            last_timestamp_ms = next_timestamp_ms(last_timestamp_ms)
            landmarks = detect_hand(landmarker, frame, last_timestamp_ms)

            if landmarks:
                draw_hand_landmarks(frame, landmarks)
                if recording:
                    row = [args.person, LABELS[idx]]
                    for point in landmarks:
                        row += [point.x, point.y, point.z]
                    writer.writerow(row)
                    count += 1
                    if count >= SAMPLES_PER_LABEL:
                        recording = False
                        f.flush()

            status = "DANG GHI" if recording else "cho (bam r)"
            cv2.putText(frame, f"Nhan: {LABELS[idx]}  {count}/{SAMPLES_PER_LABEL}  [{status}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if recording else (0, 255, 0), 2)
            cv2.imshow("Collect landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                recording = not recording
            elif key == ord("n") and idx < len(LABELS) - 1:
                idx, recording, count = idx + 1, False, 0
            elif key == ord("p") and idx > 0:
                idx, recording, count = idx - 1, False, 0
            elif key == ord("q"):
                break
finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
