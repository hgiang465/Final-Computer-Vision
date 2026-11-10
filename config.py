"""Cấu hình chung cho toàn bộ project."""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
# VSL_HOME chỉ dùng cho selftest (để không đụng vào dữ liệu thật). Bình thường = thư mục project.
ROOT = Path(os.environ.get("VSL_HOME", CODE_DIR))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MP_MODELS_DIR = CODE_DIR / "models" / "mediapipe"   # file .task của MediaPipe (tự tải)

ALPHABET_CSV = DATA_DIR / "alphabet.csv"
GESTURE_DIR = DATA_DIR / "gesture"
ALPHABET_MODEL = MODELS_DIR / "alphabet_model.joblib"
GESTURE_MODEL = MODELS_DIR / "gesture_model.pt"

LABELS_ALPHABET = CODE_DIR / "labels_alphabet.txt"
LABELS_GESTURE = CODE_DIR / "labels_gesture.txt"

CAMERA_INDEX = 0                 # camera laptop thường là 0; nếu sai thử 1, 2
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

ALPHABET_SAMPLES_PER_LABEL = 250  # số frame mỗi người cho mỗi nhãn
ALPHABET_CONF = 0.70              # dưới ngưỡng này coi như "không chắc"
ALPHABET_VOTE_WINDOW = 10         # bỏ phiếu đa số trên 10 frame gần nhất
ALPHABET_HOLD_FRAMES = 15         # giữ ổn định ~15 frame thì nhận 1 chữ

SEQ_LEN = 30                      # mỗi mẫu = 30 frame (sau khi resample)
SEQ_SECONDS = 1.5                 # độ dài thời gian của 1 lần ký hiệu (giây)
GESTURE_SAMPLES_TARGET = 50       # số lần quay mỗi cử chỉ (tổng các người) - chỉ để hiển thị tiến độ
GESTURE_CONF = 0.80               # ngưỡng chấp nhận 1 cử chỉ
GESTURE_CONFIRM = 3               # cần N lần dự đoán liên tiếp giống nhau
GESTURE_PREDICT_EVERY = 3         # chạy dự đoán mỗi 3 frame
GESTURE_COOLDOWN = 0.8            # giây nghỉ sau khi nhận 1 cử chỉ
IDLE_LABEL = "idle"
