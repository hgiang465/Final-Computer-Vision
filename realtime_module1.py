"""
Demo Module 1: nhan dien chu cai VSL real-time + ghep thanh chuoi van ban.

Chay:  python realtime_module1.py
Phim:  SPACE = them dau cach | b = xoa ky tu cuoi | c = xoa het | q = thoat
Cach ghep: giu on dinh 1 ky hieu ~0.5 giay thi chu duoc them vao; ha tay xuong
(khong thay tay) roi gio lai moi them duoc chu giong het chu vua roi.
"""
import collections
from pathlib import Path

import cv2
import joblib

from common import normalize
from hand_tracking import create_hand_landmarker, detect_hand, draw_hand_landmarks, next_timestamp_ms

CAMERA_INDEX = 0
CONF_THRESHOLD = 0.7   # duoi nguong nay coi nhu "khong chac"
WINDOW = 10            # bo phieu tren 10 frame gan nhat
HOLD_FRAMES = 15       # so frame on dinh de chap nhan 1 chu

MODEL_PATH = Path(__file__).resolve().with_name("model_module1.joblib")
if not MODEL_PATH.exists():
    raise SystemExit("Chua co model_module1.joblib. Hay chay 'python train_module1.py' truoc.")

bundle = joblib.load(MODEL_PATH)
model, classes = bundle["model"], bundle["classes"]
landmarker = create_hand_landmarker()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Khong mo duoc camera, thu doi CAMERA_INDEX.")

recent = collections.deque(maxlen=WINDOW)
text, last_committed, stable_count, stable_label = "", None, 0, None
last_timestamp_ms = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)  # PHAI giong luc thu data
        last_timestamp_ms = next_timestamp_ms(last_timestamp_ms)
        landmarks = detect_hand(landmarker, frame, last_timestamp_ms)

        shown = "-"
        if landmarks:
            draw_hand_landmarks(frame, landmarks)
            raw = [value for point in landmarks for value in (point.x, point.y, point.z)]
            proba = model.predict_proba([normalize(raw)])[0]
            i = int(proba.argmax())
            recent.append(classes[i] if proba[i] >= CONF_THRESHOLD else None)

            voted = collections.Counter(recent).most_common(1)[0][0]
            if voted is not None:
                shown = f"{voted} ({proba[i]:.2f})"
            if voted is not None and voted == stable_label:
                stable_count += 1
            else:
                stable_label, stable_count = voted, 1
            if voted is not None and stable_count >= HOLD_FRAMES and voted != last_committed:
                text += voted
                last_committed = voted
        else:
            recent.clear()
            stable_label, stable_count, last_committed = None, 0, None

        cv2.putText(frame, f"Du doan: {shown}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Van ban: {text}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.imshow("VSL Module 1", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            text += " "
        elif key == ord("b"):
            text = text[:-1]
        elif key == ord("c"):
            text = ""
finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
