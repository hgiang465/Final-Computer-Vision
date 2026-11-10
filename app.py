"""ỨNG DỤNG TÍCH HỢP: nhận diện Ngôn ngữ Ký hiệu Việt Nam theo thời gian thực bằng camera laptop.

Chạy:  python app.py
Phím:  1 = chế độ BẢNG CHỮ CÁI | 2 = chế độ CỬ CHỈ | SPACE = thêm dấu cách (chữ cái)
       b = xoá ký tự/từ cuối | c = xoá hết | q = thoát (in thống kê độ trễ, lưu results/latency.json)
Cần đã train: models/alphabet_model.joblib (train_alphabet.py) và/hoặc models/gesture_model.pt (train_gesture.py).
"""
import collections
import json
import time
import unicodedata

import cv2
import numpy as np

import config
from features import main_hand
from landmark_extractor import LandmarkExtractor, draw_landmarks, open_camera
from predictors import AlphabetPredictor
from ui import draw_texts

MARKS = {"DAU_SAC": "\u0301", "DAU_HUYEN": "\u0300", "DAU_HOI": "\u0309", "DAU_NGA": "\u0303",
         "DAU_NANG": "\u0323", "DAU_MU": "\u0302", "DAU_MOC": "\u031B", "DAU_TRANG": "\u0306"}


def append_token(text, token):
    """Ghép chữ cái / dấu vào chuỗi và chuẩn hóa Unicode NFC (A + DAU_MU + DAU_SAC -> Ấ).
    Lưu ý: ký hiệu hình chữ (A, O...) làm TRƯỚC, rồi mới đến dấu mũ/móc/trăng, rồi dấu thanh."""
    if token in MARKS:
        return unicodedata.normalize("NFC", text + MARKS[token]) if text else text
    return unicodedata.normalize("NFC", text + token)


def summarize_latency(lat):
    out = {}
    for mode, v in lat.items():
        if len(v) > 10:
            a = np.array(v)
            out[mode] = {"frames": int(len(a)), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                         "p95_ms": float(np.percentile(a, 95)), "fps_pipeline": float(1000 / a.mean())}
    return out


def main():
    alpha, gest, err = None, None, ""
    if config.ALPHABET_MODEL.exists():
        alpha = AlphabetPredictor()
    if config.GESTURE_MODEL.exists():
        try:
            from predictors import GesturePredictor
            gest = GesturePredictor()
        except ImportError:
            err = "Chưa cài PyTorch nên chưa dùng được chế độ cử chỉ (pip install torch)."
    if alpha is None and gest is None:
        raise SystemExit("Chưa có model nào. Hãy thu dữ liệu và chạy train_alphabet.py / train_gesture.py trước.")
    if err:
        print(err)

    mode = "alphabet" if alpha else "gesture"
    extractor = LandmarkExtractor(use_pose=gest is not None)
    cap = open_camera()

    lat = {"alphabet": [], "gesture": []}
    # trạng thái chế độ chữ cái
    recent = collections.deque(maxlen=config.ALPHABET_VOTE_WINDOW)
    text, last_committed, stable_label, stable_count = "", None, None, 0
    # trạng thái chế độ cử chỉ
    buf = collections.deque()
    preds = collections.deque(maxlen=config.GESTURE_CONFIRM)
    sentence, frame_i, cooldown_until = [], 0, 0.0
    cur, cur_until = None, 0.0
    msg, msg_until = err, time.time() + 6 if err else 0.0
    fps, t_prev = 0.0, time.perf_counter()

    def reset_state():
        nonlocal text, last_committed, stable_label, stable_count, sentence, cur
        recent.clear(); buf.clear(); preds.clear()
        text, last_committed, stable_label, stable_count, sentence, cur = "", None, None, 0, [], None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        now = time.monotonic()
        t0 = time.perf_counter()
        fl = extractor.process(frame, need_pose=(mode == "gesture"))
        shown = None

        if mode == "alphabet":
            hand = main_hand(fl.hand_list)
            if hand is not None:
                label, conf = alpha.predict(hand)
                recent.append(label if conf >= config.ALPHABET_CONF else None)
                voted = collections.Counter(recent).most_common(1)[0][0]
                if voted is not None:
                    shown = f"{voted}  ({conf:.2f})"
                if voted is not None and voted == stable_label:
                    stable_count += 1
                else:
                    stable_label, stable_count = voted, 1
                if voted is not None and stable_count >= config.ALPHABET_HOLD_FRAMES and voted != last_committed:
                    text = append_token(text, voted)
                    last_committed = voted
            else:
                recent.clear()
                stable_label, stable_count, last_committed = None, 0, None
        else:
            buf.append((now, fl))
            while buf and now - buf[0][0] > config.SEQ_SECONDS:
                buf.popleft()
            frame_i += 1
            span = buf[-1][0] - buf[0][0] if len(buf) > 1 else 0.0
            if (now >= cooldown_until and frame_i % config.GESTURE_PREDICT_EVERY == 0 and len(buf) >= 8
                    and span >= 0.85 * config.SEQ_SECONDS):
                hand_frac = float(np.mean([f.hand_present.any() for _, f in buf]))
                if hand_frac >= 0.3:
                    label, conf = gest.predict(list(buf))
                    cur, cur_until = (label, conf), time.time() + 1.0
                    good = label != config.IDLE_LABEL and conf >= config.GESTURE_CONF
                    preds.append(label if good else None)
                    if len(preds) == preds.maxlen and preds[0] is not None and len(set(preds)) == 1:
                        sentence.append(label)
                        cooldown_until = now + config.GESTURE_COOLDOWN
                        buf.clear(); preds.clear()
                else:
                    preds.clear()
            if cur and time.time() < cur_until:
                shown = f"{cur[0]}  ({cur[1]:.2f})" + ("" if cur[0] != config.IDLE_LABEL else "  [không ký hiệu]")
        lat[mode].append((time.perf_counter() - t0) * 1000)

        draw_landmarks(frame, fl, draw_pose=(mode == "gesture"))
        t_now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        mode_name = "BẢNG CHỮ CÁI" if mode == "alphabet" else "CỬ CHỈ"
        line2 = text if mode == "alphabet" else " ".join(sentence[-8:])
        items = [
            (f"Chế độ: {mode_name}      FPS: {fps:4.1f}   xử lý: {np.mean(lat[mode][-30:]):.1f} ms/frame",
             (15, 8), 24, (255, 255, 255)),
            (f"Dự đoán: {shown or '-'}", (15, 40), 34, (0, 255, 0)),
            ("Văn bản: " + line2 if mode == "alphabet" else "Câu: " + line2, (15, 90), 30, (0, 220, 255)),
            (msg if time.time() < msg_until else "", (15, 130), 22, (0, 165, 255)),
            ("1: chữ cái | 2: cử chỉ | SPACE: dấu cách | b: xoá cuối | c: xoá hết | q: thoát",
             (15, frame.shape[0] - 35), 20, (255, 255, 255)),
        ]
        draw_texts(frame, items, panel=(0, 0, frame.shape[1], 160))
        cv2.imshow("VSL - Nhan dien ngon ngu ky hieu Viet Nam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            if alpha: mode = "alphabet"; reset_state()
            else: msg, msg_until = "Chưa có model chữ cái (train_alphabet.py).", time.time() + 4
        elif key == ord("2"):
            if gest: mode = "gesture"; reset_state()
            else: msg, msg_until = "Chưa có model cử chỉ (train_gesture.py) hoặc chưa cài torch.", time.time() + 4
        elif key == ord(" ") and mode == "alphabet":
            text += " "
        elif key == ord("b"):
            if mode == "alphabet": text = unicodedata.normalize("NFC", text[:-1])
            else: sentence = sentence[:-1]
        elif key == ord("c"):
            text, sentence = "", []

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()
    stats = summarize_latency(lat)
    print("\n=== Thống kê độ trễ (trích xuất landmark + dự đoán, không gồm vẽ/hiển thị) ===")
    for m, s in stats.items():
        print(f"{m}: trung bình {s['mean_ms']:.1f} ms | median {s['median_ms']:.1f} | p95 {s['p95_ms']:.1f} "
              f"| ~{s['fps_pipeline']:.0f} FPS ({s['frames']} frame)")
    if stats:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.RESULTS_DIR / "latency.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print("Đã lưu results/latency.json (dùng cho phần Experimental Results của báo cáo).")


if __name__ == "__main__":
    main()
