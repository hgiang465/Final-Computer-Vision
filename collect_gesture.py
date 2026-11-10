"""Thu dữ liệu MODULE 2 (cử chỉ có chuyển động) bằng camera laptop.

Chạy:  python collect_gesture.py --person A
Phím:  SPACE = quay 1 mẫu (đếm ngược 2 giây rồi ghi 1.5 giây) | n/p = cử chỉ kế/trước
       u = xoá mẫu vừa quay | q = thoát
Mỗi mẫu lưu thành 1 file data/gesture/<nhãn>/<người>_<thời gian>.npz gồm đúng 30 frame
(đã resample theo thời gian nên không phụ thuộc FPS camera).

MẸO: ngồi xa camera đủ để thấy CẢ HAI VAI và 2 bàn tay. Làm cử chỉ trọn vẹn trong ~1.5 giây.
Với nhãn "idle": quay các tình huống KHÔNG ký hiệu (tay nghỉ, gãi đầu, vẫy linh tinh...).
"""
import argparse
import time

import cv2
import numpy as np

import config
from features import load_labels, resample_by_time
from landmark_extractor import LandmarkExtractor, draw_landmarks, open_camera
from ui import draw_texts

parser = argparse.ArgumentParser()
parser.add_argument("--person", required=True, help="mã người thu, ví dụ A, B, C")
args = parser.parse_args()

labels = load_labels(config.LABELS_GESTURE)
config.GESTURE_DIR.mkdir(parents=True, exist_ok=True)


def label_dir(label):
    d = config.GESTURE_DIR / label.replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def count_files(label):
    return len(list(label_dir(label).glob("*.npz")))


def save_sample(label, frames):
    """frames: list of (t, FrameLandmarks). Resample về SEQ_LEN frame rồi lưu."""
    times = [t for t, _ in frames]
    sel = resample_by_time(times, config.SEQ_LEN)
    fls = [frames[i][1] for i in sel]
    path = label_dir(label) / f"{args.person}_{int(time.time() * 1000)}.npz"
    np.savez_compressed(
        path,
        hands=np.stack([f.hands for f in fls]).astype(np.float32),
        hand_present=np.stack([f.hand_present for f in fls]),
        pose=np.stack([f.pose for f in fls]).astype(np.float32),
        pose_present=np.array([f.pose_present for f in fls]),
        person=args.person, label=label)
    return path


extractor = LandmarkExtractor(use_pose=True)
cap = open_camera()
idx = 0
state, state_start = "idle", 0.0           # idle -> countdown -> recording
frames = []
last_saved, msg, msg_until = None, "", 0.0
COUNTDOWN = 2.0

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    now = time.monotonic()
    fl = extractor.process(frame, need_pose=True)
    draw_landmarks(frame, fl, draw_pose=True)

    if state == "countdown" and now - state_start >= COUNTDOWN:
        state, state_start, frames = "recording", now, []
    elif state == "recording":
        frames.append((now, fl))
        if now - state_start >= config.SEQ_SECONDS:
            state = "idle"
            hand_frac = np.mean([f.hand_present.any() for _, f in frames]) if frames else 0
            pose_frac = np.mean([f.pose_present for _, f in frames]) if frames else 0
            lab = labels[idx]
            if len(frames) < 8:
                msg = "Camera quá chậm/ít frame, thử lại."
            elif lab != config.IDLE_LABEL and hand_frac < 0.4:
                msg = f"Chỉ thấy tay {hand_frac:.0%} số frame. Quay lại!"
            elif pose_frac < 0.5:
                msg = f"Chỉ thấy vai/thân {pose_frac:.0%} số frame. Ngồi xa hơn cho thấy cả 2 vai!"
            else:
                last_saved = save_sample(lab, frames)
                msg = f"Đã lưu mẫu #{count_files(lab)} ({len(frames)} frame gốc)."
            msg_until = time.time() + 4

    lab = labels[idx]
    if state == "countdown":
        remain = max(0.0, COUNTDOWN - (now - state_start))
        status, color = f"CHUẨN BỊ... {remain:.1f}s", (0, 255, 255)
    elif state == "recording":
        status, color = "ĐANG GHI - làm cử chỉ ngay!", (0, 0, 255)
    else:
        status, color = "chờ (bấm SPACE để quay)", (0, 255, 0)
    items = [
        (f"Người: {args.person}   Cử chỉ {idx + 1}/{len(labels)}:  {lab}", (15, 10), 30, (255, 255, 255)),
        (f"Đã có {count_files(lab)}/{config.GESTURE_SAMPLES_TARGET} mẫu   [{status}]", (15, 48), 24, color),
        (f"Tay: {'thấy' if fl.hand_present.any() else 'KHÔNG thấy'}   Vai/thân: {'thấy' if fl.pose_present else 'KHÔNG thấy'}",
         (15, 80), 22, (255, 255, 255)),
        (msg if time.time() < msg_until else "", (15, 110), 24, (0, 255, 255)),
        ("SPACE: quay | n/p: đổi cử chỉ | u: xoá mẫu vừa rồi | q: thoát", (15, frame.shape[0] - 35), 22, (255, 255, 255)),
    ]
    draw_texts(frame, items, panel=(0, 0, frame.shape[1], 145))
    cv2.imshow("Thu du lieu - cu chi", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and state == "idle":
        state, state_start = "countdown", now
    elif key in (ord("n"), ord("p")) and state == "idle":
        idx = min(idx + 1, len(labels) - 1) if key == ord("n") else max(idx - 1, 0)
    elif key == ord("u") and last_saved is not None and state == "idle":
        try:
            last_saved.unlink()
            msg = "Đã xoá mẫu vừa quay."
        except OSError:
            msg = "Không xoá được."
        last_saved, msg_until = None, time.time() + 3
    elif key == ord("q"):
        break

extractor.close()
cap.release()
cv2.destroyAllWindows()
print("Tiến độ:", {l: count_files(l) for l in labels})
