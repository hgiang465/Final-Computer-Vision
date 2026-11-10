"""Tự kiểm tra cài đặt (KHÔNG cần camera, KHÔNG đụng vào dữ liệu thật của bạn).

Chạy:  python selftest.py            kiểm tra thư viện + pipeline train/dự đoán bằng dữ liệu giả
       python selftest.py --camera   kiểm tra thêm camera và MediaPipe (đưa bàn tay vào khung hình)
"""
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--camera", action="store_true")
args = ap.parse_args()

tmp = Path(tempfile.mkdtemp(prefix="vsl_selftest_"))
os.environ["VSL_HOME"] = str(tmp)          # phải đặt TRƯỚC khi import config

import numpy as np  # noqa: E402

import config  # noqa: E402
from landmark_extractor import FrameLandmarks  # noqa: E402

rng = np.random.default_rng(0)
ok_all = True


def step(name):
    print(f"\n[{name}]")


step("1/5 Module 1: train bảng chữ cái trên dữ liệu giả")
import train_alphabet  # noqa: E402

(tmp / "data").mkdir(parents=True)
templates = {c: rng.normal(size=(21, 3)) for c in "ABCDEF"}
with open(config.ALPHABET_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["person", "label"] + [f"{a}{i}" for i in range(21) for a in "xyz"])
    for person in ["P1", "P2", "P3"]:
        for lab, tpl in templates.items():
            for _ in range(60):
                pts = tpl * rng.uniform(0.9, 1.1) + rng.normal(0, 0.05, (21, 3)) + rng.normal(0, 0.3, 3)
                w.writerow([person, lab] + [float(v) for v in pts.reshape(-1)])
rep = train_alphabet.run()
acc_a = rep["models"][rep["best_model"]]["leave_one_signer_out"]["accuracy"]
print(f"-> accuracy (dữ liệu giả, leave-one-signer-out): {acc_a:.3f}")
ok_all &= acc_a > 0.8

from predictors import AlphabetPredictor  # noqa: E402
ap_ = AlphabetPredictor(config.ALPHABET_MODEL)
lab, conf = ap_.predict(templates["C"] + rng.normal(0, 0.03, (21, 3)))
print(f"-> dự đoán thử: {lab} ({conf:.2f})  (mong đợi C)")
ok_all &= lab == "C"

step("2/5 Module 2: train cử chỉ (LSTM + GRU) trên dữ liệu giả")
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception as e:  # noqa: BLE001
    HAVE_TORCH = False
    print(f"!! Không import được torch ({e}). Cài: pip install torch  -> bỏ qua kiểm tra module cử chỉ.")
    ok_all = False

if HAVE_TORCH:
    import train_gesture  # noqa: E402
    gestures = ["idle", "xin chào", "cảm ơn", "xin lỗi"]      # có dấu + khoảng trắng để thử tên thư mục
    shape_tpl = {g: rng.normal(size=(21, 3)) for g in gestures}
    T = config.SEQ_LEN

    def make_sample(g, person_off):
        k = gestures.index(g)
        tt = np.linspace(0, 1, T)
        path = np.stack([0.5 + 0.15 * np.cos(2 * np.pi * (k + 1) * tt / 2),
                         0.5 + 0.15 * np.sin(2 * np.pi * (k + 1) * tt / 2)], 1)
        hands = np.zeros((T, 2, 21, 3), np.float32)
        for t in range(T):
            hands[t, 1] = shape_tpl[g] * 0.05 + np.array([path[t, 0], path[t, 1], 0]) + rng.normal(0, 0.003, (21, 3))
        hp = np.zeros((T, 2), bool); hp[:, 1] = g != "idle"
        pose = np.zeros((T, 33, 3), np.float32)
        pose[:, 11] = [0.40 + person_off, 0.55, 0]; pose[:, 12] = [0.60 + person_off, 0.55, 0]
        pose[:, 0] = [0.5 + person_off, 0.3, 0]
        pose += rng.normal(0, 0.003, pose.shape).astype(np.float32)
        return hands, hp, pose, np.ones(T, bool)

    for person_i, person in enumerate(["P1", "P2", "P3"]):
        for g in gestures:
            d = config.GESTURE_DIR / g.replace(" ", "_")
            d.mkdir(parents=True, exist_ok=True)
            for s in range(8):
                h, hp, po, pp = make_sample(g, 0.02 * person_i)
                np.savez_compressed(d / f"{person}_{s}.npz", hands=h, hand_present=hp, pose=po,
                                    pose_present=pp, person=person, label=g)
    rep = train_gesture.run(epochs=10, hidden=32, aug=1)
    acc_g = max(m["accuracy"] for m in rep["models"].values())
    print(f"-> accuracy tốt nhất (dữ liệu giả, leave-one-signer-out): {acc_g:.3f}")
    ok_all &= acc_g > 0.7

    step("3/5 Dự đoán cử chỉ qua GesturePredictor (đúng đường chạy của app)")
    from predictors import GesturePredictor  # noqa: E402
    gp = GesturePredictor(config.GESTURE_MODEL)
    h, hp, po, pp = make_sample("cảm ơn", 0.02)
    frames = []
    for t in range(40):
        fl = FrameLandmarks(hands=h[min(t, T - 1)], hand_present=hp[min(t, T - 1)],
                            pose=po[min(t, T - 1)], pose_present=True)
        frames.append((t / 30.0, fl))
    lab, conf = gp.predict(frames)
    print(f"-> dự đoán thử: {lab} ({conf:.2f})  (mong đợi: cảm ơn)")
    ok_all &= lab == "cảm ơn"

step("4/5 Ghép chữ cái + dấu thành tiếng Việt (Unicode NFC)")
from app import append_token  # noqa: E402
t = ""
for tok in ["T", "I", "E", "DAU_MU", "DAU_SAC", "N", "G"]:
    t = append_token(t, tok)
print(f"-> T I E DAU_MU DAU_SAC N G  =>  {t}  (mong đợi: TIẾNG)")
ok_all &= t == "TIẾNG"

step("5/5 MediaPipe" + (" + camera" if args.camera else " (khởi tạo + 1 ảnh đen)"))
try:
    import cv2
    from landmark_extractor import LandmarkExtractor, draw_landmarks, open_camera
    ex = LandmarkExtractor(use_pose=True)
    fl = ex.process(np.zeros((480, 640, 3), np.uint8), need_pose=True)
    print(f"-> MediaPipe chạy được (backend={ex.backend}); ảnh đen: không thấy tay là đúng ({len(fl.hand_list)} tay)")
    if args.camera:
        cap = open_camera()
        print("Đưa bàn tay vào khung hình, nhấn q để thoát...")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            fl = ex.process(frame, need_pose=True)
            draw_landmarks(frame, fl, draw_pose=True)
            cv2.putText(frame, f"tay: {len(fl.hand_list)}  than: {fl.pose_present}  (q: thoat)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("selftest camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release(); cv2.destroyAllWindows()
    ex.close()
except Exception as e:  # noqa: BLE001
    print(f"!! Lỗi MediaPipe/camera: {e}")
    ok_all = False

print("\n" + ("=== SELFTEST ĐẠT ===" if ok_all else "=== SELFTEST CÓ LỖI (xem các dòng !! phía trên) ==="))
sys.exit(0 if ok_all else 1)
