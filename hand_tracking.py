"""MediaPipe Tasks helpers shared by Module 1 collection and realtime scripts."""
import shutil
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import mediapipe as mp

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def create_hand_landmarker():
    if not MODEL_PATH.exists():
        temp_path = MODEL_PATH.with_suffix(".task.download")
        print("Dang tai mo hinh nhan dien ban tay...")
        try:
            with urlopen(MODEL_URL, timeout=30) as response, temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temp_path.stat().st_size < 1_000_000:
                raise RuntimeError("File mo hinh tai ve khong day du.")
            temp_path.replace(MODEL_PATH)
        except (OSError, URLError, RuntimeError) as exc:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Khong tai duoc mo hinh MediaPipe. Hay kiem tra ket noi Internet "
                f"hoac tai thu cong tu {MODEL_URL} va luu thanh {MODEL_PATH}."
            ) from exc

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hand(landmarker, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    return result.hand_landmarks[0] if result.hand_landmarks else None


def draw_hand_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 0), 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


def next_timestamp_ms(previous_timestamp_ms):
    return max(time.monotonic_ns() // 1_000_000, previous_timestamp_ms + 1)
