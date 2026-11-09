"""Ham dung chung: doc nhan va chuan hoa landmark (thu data, train, realtime phai dung CHUNG)."""
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent


def load_labels(path="labels.txt"):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize(raw63):
    """63 so tho (21 diem x,y,z) -> 63 so chuan hoa:
    tru co tay (diem 0) de bat bien vi tri, chia cho kich thuoc ban tay de bat bien khoang cach."""
    pts = np.asarray(raw63, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale < 1e-6:
        scale = 1.0
    return (pts / scale).reshape(-1)
