"""Trích xuất landmark bàn tay (và tư thế) bằng MediaPipe.

Tự chọn backend:
  - API cũ  mp.solutions   (mediapipe <= 0.10.21)
  - API mới MediaPipe Tasks (mediapipe >= 0.10.30, đã bỏ mp.solutions) -> tự tải file model .task lần đầu chạy.
"""
import time
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

import config

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]
POSE_DRAW_IDX = [11, 12, 13, 14, 15, 16]
POSE_DRAW_LINES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


@dataclass
class FrameLandmarks:
    hands: np.ndarray = field(default_factory=lambda: np.zeros((2, 21, 3), np.float32))   # slot0=Left, slot1=Right
    hand_present: np.ndarray = field(default_factory=lambda: np.zeros(2, bool))
    hand_list: list = field(default_factory=list)       # các bàn tay theo thứ tự phát hiện, mỗi cái (21,3)
    pose: np.ndarray = field(default_factory=lambda: np.zeros((33, 3), np.float32))
    pose_present: bool = False


def ensure_model(filename, url):
    config.MP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MP_MODELS_DIR / filename
    if path.exists() and path.stat().st_size > 1_000_000:
        return str(path)
    print(f"[MediaPipe] Đang tải model {filename} (chỉ lần đầu)...")
    tmp = str(path) + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        import os
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Không tải được {filename}: {e}\n"
            f"Hãy tải tay từ:\n  {url}\nrồi lưu vào:\n  {path}") from e
    return str(path)


def _assign_slots(hand_arrays, labels):
    hands = np.zeros((2, 21, 3), np.float32)
    present = np.zeros(2, bool)
    for arr, lab in zip(hand_arrays, labels):
        slot = 0 if lab == "Left" else 1
        if present[slot]:
            slot = 1 - slot
        if present[slot]:
            continue
        hands[slot] = arr
        present[slot] = True
    return hands, present


class LandmarkExtractor:
    def __init__(self, use_pose=False, max_hands=2):
        import mediapipe as mp
        self.mp = mp
        self.max_hands = max_hands
        self.use_pose = use_pose
        self._last_ts = 0
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "legacy"
            self._init_legacy()
        else:
            self.backend = "tasks"
            self._init_tasks()
        print(f"[MediaPipe] backend = {self.backend}, version = {getattr(mp, '__version__', '?')}")

    def _init_legacy(self):
        mp = self.mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=self.max_hands,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._pose = None
        if self.use_pose:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=0,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def _init_tasks(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        hand_path = ensure_model("hand_landmarker.task", HAND_MODEL_URL)
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        self._pose = None
        if self.use_pose:
            pose_path = ensure_model("pose_landmarker_lite.task", POSE_MODEL_URL)
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5))

    def process(self, bgr, need_pose=None):
        """bgr: ảnh OpenCV (đã lật gương nếu muốn). Trả về FrameLandmarks."""
        if need_pose is None:
            need_pose = self.use_pose
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.backend == "legacy":
            return self._process_legacy(rgb, need_pose)
        return self._process_tasks(rgb, need_pose)

    def _process_legacy(self, rgb, need_pose):
        out = FrameLandmarks()
        res = self._hands.process(rgb)
        if res.multi_hand_landmarks:
            arrays, labels = [], []
            for i, lm in enumerate(res.multi_hand_landmarks):
                arrays.append(np.array([[p.x, p.y, p.z] for p in lm.landmark], np.float32))
                labels.append(res.multi_handedness[i].classification[0].label
                              if res.multi_handedness else "Right")
            out.hand_list = arrays
            out.hands, out.hand_present = _assign_slots(arrays, labels)
        if need_pose and self._pose is not None:
            pr = self._pose.process(rgb)
            if pr.pose_landmarks:
                out.pose = np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks.landmark], np.float32)
                out.pose_present = True
        return out

    def _process_tasks(self, rgb, need_pose):
        mp = self.mp
        out = FrameLandmarks()
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)   # timestamp phải tăng dần
        self._last_ts = ts
        res = self._hands.detect_for_video(image, ts)
        if res.hand_landmarks:
            arrays = [np.array([[p.x, p.y, p.z] for p in lm], np.float32) for lm in res.hand_landmarks]
            labels = [h[0].category_name if h else "Right" for h in res.handedness]
            out.hand_list = arrays
            out.hands, out.hand_present = _assign_slots(arrays, labels)
        if need_pose and self._pose is not None:
            pr = self._pose.detect_for_video(image, ts)
            if pr.pose_landmarks:
                out.pose = np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks[0]], np.float32)
                out.pose_present = True
        return out

    def close(self):
        for obj in (self._hands, self._pose):
            try:
                if obj is not None:
                    obj.close()
            except Exception:  # noqa: BLE001
                pass


def draw_landmarks(frame, fl, draw_pose=False):
    """Vẽ bàn tay (và vai/khuỷu/cổ tay nếu draw_pose) lên frame."""
    h, w = frame.shape[:2]
    for pts in fl.hand_list:
        px = [(int(p[0] * w), int(p[1] * h)) for p in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, px[a], px[b], (0, 200, 0), 2)
        for p in px:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)
    if draw_pose and fl.pose_present:
        px = {i: (int(fl.pose[i, 0] * w), int(fl.pose[i, 1] * h)) for i in POSE_DRAW_IDX}
        for a, b in POSE_DRAW_LINES:
            cv2.line(frame, px[a], px[b], (255, 160, 0), 2)
        for p in px.values():
            cv2.circle(frame, p, 4, (255, 160, 0), -1)
    return frame


def open_camera(index=None):
    index = config.CAMERA_INDEX if index is None else index
    import os
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera index {index}. Thử đổi CAMERA_INDEX trong config.py (1, 2...).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap
