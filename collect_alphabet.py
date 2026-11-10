"""Thu dữ liệu MODULE 1 (bảng chữ cái, ký hiệu tĩnh) bằng camera laptop.

Chạy:  python collect_alphabet.py --person A      (mỗi thành viên dùng mã --person khác: A, B, C...)
Phím:  r = bắt đầu/dừng ghi nhãn hiện tại | n = nhãn kế | p = nhãn trước | u = hoàn tác lần ghi vừa rồi | q = thoát
Dữ liệu ghi nối tiếp vào data/alphabet.csv (chạy lại không mất dữ liệu cũ).
Mỗi dòng: person, label, x0,y0,z0,...,x20,y20,z20 (toạ độ thô, chưa chuẩn hóa).
"""
import argparse
import csv
import os
import time
from collections import Counter

import cv2

import config
from features import load_labels, main_hand
from landmark_extractor import LandmarkExtractor, draw_landmarks, open_camera
from ui import draw_texts

parser = argparse.ArgumentParser()
parser.add_argument("--person", required=True, help="mã người thu, ví dụ A, B, C")
args = parser.parse_args()

labels = load_labels(config.LABELS_ALPHABET)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
csv_path = config.ALPHABET_CSV
header = ["person", "label"] + [f"{a}{i}" for i in range(21) for a in "xyz"]
if not csv_path.exists():
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)

# đếm số frame đã có của người này
counts = Counter()
with open(csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["person"] == args.person:
            counts[row["label"]] += 1

extractor = LandmarkExtractor(use_pose=False)
cap = open_camera()
idx, recording, batch, last_batch_start = 0, False, [], None
msg, msg_until = "", 0.0


def commit_batch():
    """Ghi batch hiện tại xuống CSV, nhớ vị trí để có thể hoàn tác."""
    global batch, last_batch_start
    if not batch:
        return
    last_batch_start = os.path.getsize(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(batch)
    counts[labels[idx]] += len(batch)
    batch = []


while True:
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)     # lật gương: PHẢI giống lúc chạy app
    fl = extractor.process(frame)
    hand = main_hand(fl.hand_list)
    draw_landmarks(frame, fl)

    if recording and hand is not None:
        batch.append([args.person, labels[idx]] + [float(v) for v in hand.reshape(-1)])
        if len(batch) >= config.ALPHABET_SAMPLES_PER_LABEL:
            recording = False
            commit_batch()
            msg, msg_until = f"Đủ {config.ALPHABET_SAMPLES_PER_LABEL} frame. Bấm n để sang nhãn kế.", time.time() + 4

    status = "ĐANG GHI" if recording else "chờ (bấm r)"
    color = (0, 0, 255) if recording else (0, 255, 0)
    items = [
        (f"Người: {args.person}   Nhãn {idx + 1}/{len(labels)}:  {labels[idx]}", (15, 10), 30, (255, 255, 255)),
        (f"Frame hiện tại: {len(batch)}/{config.ALPHABET_SAMPLES_PER_LABEL}   "
         f"(đã lưu cho nhãn này: {counts[labels[idx]]})   [{status}]", (15, 48), 24, color),
        ("Không thấy tay!" if hand is None else "", (15, 80), 24, (0, 165, 255)),
        (msg if time.time() < msg_until else "", (15, 110), 24, (0, 255, 255)),
        ("r: ghi/dừng | n: nhãn kế | p: nhãn trước | u: hoàn tác | q: thoát", (15, frame.shape[0] - 35), 22, (255, 255, 255)),
    ]
    draw_texts(frame, items, panel=(0, 0, frame.shape[1], 145))
    cv2.imshow("Thu du lieu - bang chu cai", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        if recording:
            recording = False
            commit_batch()
        else:
            recording, batch = True, []
    elif key in (ord("n"), ord("p")):
        if recording:
            recording = False
            commit_batch()
        idx = min(idx + 1, len(labels) - 1) if key == ord("n") else max(idx - 1, 0)
    elif key == ord("u") and last_batch_start is not None and not recording:
        removed = os.path.getsize(csv_path) - last_batch_start
        os.truncate(csv_path, last_batch_start)
        last_batch_start = None
        # tính lại bộ đếm từ file
        counts.clear()
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["person"] == args.person:
                    counts[row["label"]] += 1
        msg, msg_until = "Đã hoàn tác lần ghi vừa rồi.", time.time() + 3
    elif key == ord("q"):
        break

if recording:
    commit_batch()
extractor.close()
cap.release()
cv2.destroyAllWindows()
print("Đã lưu:", dict(counts))
