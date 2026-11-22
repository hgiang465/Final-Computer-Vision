"""APP GHÉP CÂU: một app duy nhất, KHÔNG cần chuyển chế độ.

Bạn ký liên tục:  tôi  tên  là  H  O  [dấu mũ]  [dấu huyền]  N  G   ->   "tôi tên là HỒNG"

Cách app phân biệt (tự động, dựa vào chuyển động của bàn tay):
  * Tay ĐỨNG YÊN ~0,2 giây  -> đang giữ một chữ cái  -> nhận diện bằng model chữ cái (Module 1)
  * Tay ĐANG DI CHUYỂN      -> đang ký một từ hoặc một dấu (sắc/huyền/hỏi/ngã/nặng...) -> model cử chỉ (Module 2)
      - nhãn bắt đầu bằng DAU_  -> ghép vào từ đang đánh vần (đúng chữ cần dấu, đúng vị trí dấu thanh)
      - nhãn khác               -> thêm thành một từ của câu
  * Không thấy tay ~1,5 giây -> kết thúc từ đang đánh vần, sang từ mới

Phím:  SPACE = kết thúc từ đang đánh vần | b = hoàn tác bước cuối | c = xoá hết | q = thoát
       ,  = giảm ngưỡng chuyển động (nhạy hơn)  |  .  = tăng ngưỡng (đỡ nhạy)  -> chỉnh khi app nhầm chữ <-> cử chỉ
Cần có: models/alphabet_model.joblib và models/gesture_model.pt (nếu thiếu một trong hai thì app chỉ dùng phần còn lại).
"""
import collections
import json
import time

import cv2
import numpy as np

import config
import config_sentence as cs
from app import summarize_latency
from features import hand_scale, main_hand
from landmark_extractor import LandmarkExtractor, draw_landmarks, open_camera
from predictors import AlphabetPredictor
from ui import draw_texts
from vietnamese import MARK_NAMES, SentenceBuilder, is_mark

STATE_TEXT = {"none": "không thấy tay", "settling": "chờ tay ổn định",
              "still": "đứng yên (đọc CHỮ CÁI)", "moving": "đang di chuyển (đọc CỬ CHỈ/DẤU)"}


class MotionTracker:
    """Đo mức chuyển động của bàn tay: độ dịch trung bình của 21 điểm giữa 2 frame, chia cho kích thước bàn tay."""

    def __init__(self, hi, lo, still_frames):
        self.hi, self.lo, self.still_frames = hi, lo, still_frames
        self.reset()

    def reset(self):
        self.prev = self.prev_present = None
        self.ema, self.state, self.still_count = 0.0, "none", 0

    def update(self, fl):
        if not fl.hand_present.any():
            self.reset()
            return 0.0, "none"
        vals = []
        if self.prev is not None:
            for s in (0, 1):
                if fl.hand_present[s] and self.prev_present[s]:
                    d = np.linalg.norm(fl.hands[s][:, :2] - self.prev[s][:, :2], axis=1).mean()
                    vals.append(float(d / hand_scale(fl.hands[s])))
        self.prev, self.prev_present = fl.hands.copy(), fl.hand_present.copy()
        v = min(max(vals), 0.25) if vals else 0.0   # chặn các cú nhảy lớn (đổi hình tay) để không kéo dài "đang di chuyển"
        self.ema = 0.6 * self.ema + 0.4 * v
        if self.ema > self.hi:
            self.state, self.still_count = "moving", 0
        elif self.ema < self.lo:
            self.still_count += 1
            if self.still_count >= self.still_frames:
                self.state = "still"
            elif self.state == "none":
                self.state = "settling"
        else:                                  # vùng trễ giữa lo và hi: giữ trạng thái cũ
            if self.state == "none":
                self.state = "settling"
            if self.state != "still":
                self.still_count = 0
        return self.ema, self.state


def main():
    alpha = AlphabetPredictor() if config.ALPHABET_MODEL.exists() else None
    gest = None
    if config.GESTURE_MODEL.exists():
        try:
            from predictors import GesturePredictor
            gest = GesturePredictor()
        except ImportError:
            print("Chưa cài PyTorch nên không dùng được cử chỉ/dấu thanh (pip install torch).")
    if alpha is None and gest is None:
        raise SystemExit("Chưa có model nào. Hãy train_alphabet.py và train_gesture.py trước.")
    if gest is not None and not any(c.startswith("DAU_") for c in gest.classes):
        print("LƯU Ý: model cử chỉ chưa có nhãn DAU_SAC/DAU_HUYEN/... nên chưa ghép được dấu thanh.")

    extractor = LandmarkExtractor(use_pose=gest is not None)
    cap = open_camera()
    tracker = MotionTracker(cs.MOVE_HI, cs.MOVE_LO, cs.STILL_FRAMES)
    builder = SentenceBuilder()

    recent = collections.deque(maxlen=config.ALPHABET_VOTE_WINDOW)
    stable_label, stable_count, last_committed, committed_letter = None, 0, None, None
    buf = collections.deque()
    preds = collections.deque(maxlen=config.GESTURE_CONFIRM)
    frame_i, cooldown_until, need_still, none_since, letters_blocked_until = 0, 0.0, False, None, 0.0
    lat, fps, t_prev = [], 0.0, time.perf_counter()
    note, note_until, cur, cur_until = "", 0.0, None, 0.0

    def say(msg, secs=3.0):
        nonlocal note, note_until
        note, note_until = msg, time.time() + secs

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)                 # lật gương: giống lúc thu dữ liệu
        now = time.monotonic()
        t0 = time.perf_counter()
        fl = extractor.process(frame, need_pose=gest is not None)
        motion, state = tracker.update(fl)

        # ---- ngừng ký một lúc -> kết thúc từ đang đánh vần ----
        if state == "none":
            none_since = none_since or now
            if builder.spelling and now - none_since >= cs.WORD_GAP_SECONDS:
                builder.flush()
                say("Kết thúc từ đánh vần")
        else:
            none_since = None

        # ---- CHỮ CÁI: chỉ đọc khi tay đứng yên ----
        letter_text = None
        if state == "still" and alpha is not None and now >= letters_blocked_until:
            hand = main_hand(fl.hand_list)
            if hand is not None:
                label, conf = alpha.predict(hand)
                good = conf >= config.ALPHABET_CONF and label != config.IDLE_LABEL
                recent.append(label if good else None)
                voted = collections.Counter(recent).most_common(1)[0][0]
                letter_text = f"{voted} ({conf:.2f})" if voted else None
                if voted is not None and voted == stable_label:
                    stable_count += 1
                else:
                    stable_label, stable_count = voted, 1
                if voted is not None and stable_count >= config.ALPHABET_HOLD_FRAMES and voted != last_committed:
                    if is_mark(voted):
                        say(MARK_NAMES[voted] + (" → " + builder.spelling_text() if builder.add_mark(voted)
                                                 else ": chưa có chữ để gắn dấu"))
                    else:
                        builder.add_letter(voted)
                        committed_letter = voted
                        say(f"Chữ: {voted}")
                    last_committed = voted
        else:
            recent.clear()
            stable_label, stable_count = None, 0
            if state in ("moving", "none"):
                last_committed = None              # cho phép ký lại cùng một chữ (vd. "OO") sau khi cử động tay

        # ---- CỬ CHỈ / DẤU: chỉ đọc khi cửa sổ gần nhất có chuyển động ----
        if gest is not None:
            buf.append((now, fl, motion))
            while buf and now - buf[0][0] > config.SEQ_SECONDS:
                buf.popleft()
            frame_i += 1
            if need_still and state in ("still", "none"):
                need_still = False                 # sau khi nhận 1 cử chỉ, chờ tay yên rồi mới nhận cử chỉ tiếp
            span = buf[-1][0] - buf[0][0] if len(buf) > 1 else 0.0
            if (now >= cooldown_until and not need_still and frame_i % config.GESTURE_PREDICT_EVERY == 0
                    and len(buf) >= 8 and span >= 0.85 * config.SEQ_SECONDS):
                moved = max(m for _, _, m in buf) > tracker.hi
                hand_frac = float(np.mean([f.hand_present.any() for _, f, _ in buf]))
                if moved and hand_frac >= 0.3:
                    label, conf = gest.predict([(t, f) for t, f, _ in buf])
                    cur, cur_until = (label, conf), time.time() + 1.0
                    good = label != config.IDLE_LABEL and conf >= config.GESTURE_CONF
                    preds.append(label if good else None)
                    if len(preds) == preds.maxlen and preds[0] is not None and len(set(preds)) == 1:
                        if is_mark(label):
                            if builder.add_mark(label):
                                say(f"{MARK_NAMES[label]} → {builder.spelling_text()}")
                            else:
                                say(f"{MARK_NAMES[label]}: chưa có chữ để gắn dấu")
                        else:
                            builder.add_word(label)
                            say(f"Từ: {label}")
                        cooldown_until = now + config.GESTURE_COOLDOWN
                        need_still = True
                        letters_blocked_until = now + cs.LETTER_BLOCK_AFTER_GESTURE   # tránh đọc nhầm chữ cái ngay sau nét dấu
                        buf.clear(); preds.clear(); recent.clear()
                        stable_label, stable_count, last_committed = None, 0, None
                else:
                    preds.clear()
        gesture_text = f"{cur[0]} ({cur[1]:.2f})" if cur and time.time() < cur_until else None
        lat.append((time.perf_counter() - t0) * 1000)

        # ---- giao diện ----
        draw_landmarks(frame, fl, draw_pose=gest is not None)
        t_now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        sentence = builder.text()
        if len(sentence) > 55:
            sentence = "…" + sentence[-54:]
        items = [
            (f"Tay: {STATE_TEXT[state]}"
             f"{' (' + str(min(tracker.still_count, tracker.still_frames)) + '/' + str(tracker.still_frames) + ')' if state == 'settling' else ''}   "
             f"chuyển động {motion:.3f} (ngưỡng {tracker.lo:.3f}/{tracker.hi:.3f})   "
             f"FPS {fps:4.1f}", (15, 8), 20, (255, 255, 255)),
            (f"Chữ cái: {letter_text or '-'} "
             f"{('|' + ' ' if letter_text else '')}{'giữ ' + str(min(stable_count, config.ALPHABET_HOLD_FRAMES)) + '/' + str(config.ALPHABET_HOLD_FRAMES) if letter_text else ''} "
             f"| vừa nhận: {committed_letter or '-'}      "
             f"Cử chỉ/dấu: {gesture_text or '-'}", (15, 36), 26, (0, 255, 0)),
            (f"Đang đánh vần: {builder.spelling_text() or '-'}", (15, 72), 28, (0, 220, 255)),
            ("Câu: " + sentence, (15, 110), 34, (255, 255, 255)),
            (note if time.time() < note_until else "", (15, 156), 22, (0, 165, 255)),
            ("SPACE: hết từ | b: hoàn tác | c: xoá hết | ,/. : chỉnh độ nhạy chuyển động | q: thoát",
             (15, frame.shape[0] - 32), 20, (255, 255, 255)),
        ]
        draw_texts(frame, items, panel=(0, 0, frame.shape[1], 190))
        cv2.imshow("VSL - Ghep cau", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if builder.flush():
                say("Kết thúc từ đánh vần")
        elif key == ord("b"):
            builder.undo()
        elif key == ord("c"):
            builder.clear()
        elif key in (ord(","), ord(".")):
            f = 0.85 if key == ord(",") else 1.18
            tracker.hi, tracker.lo = tracker.hi * f, tracker.lo * f
            say(f"Ngưỡng chuyển động: {tracker.lo:.3f} / {tracker.hi:.3f}")

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()
    stats = summarize_latency({"sentence": lat})
    for m, s in stats.items():
        print(f"\nĐộ trễ: trung bình {s['mean_ms']:.1f} ms | median {s['median_ms']:.1f} | p95 {s['p95_ms']:.1f} "
              f"| ~{s['fps_pipeline']:.0f} FPS ({s['frames']} frame)")
    if stats:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.RESULTS_DIR / "latency_sentence.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    print(f"Câu cuối cùng: {builder.text()}")
    print(f"(Ngưỡng chuyển động đã chỉnh: MOVE_LO={tracker.lo:.3f}, MOVE_HI={tracker.hi:.3f} -> ghi vào config_sentence.py nếu ổn)")


if __name__ == "__main__":
    main()
