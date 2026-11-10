"""Chuẩn hóa landmark và tạo vector đặc trưng. Thu data, train và app đều dùng CHUNG file này."""
import numpy as np

# Các điểm pose dùng cho module cử chỉ: mũi, 2 vai, 2 khuỷu, 2 cổ tay (chỉ số MediaPipe Pose)
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
ALPHABET_DIM = 63                 # 21 điểm x 3
GESTURE_DIM = 2 * 63 + 4 + 2 * len(POSE_IDX)   # 126 hình dạng tay + 4 vị trí cổ tay + 14 tư thế = 144


def load_labels(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def hand_scale(pts):
    """Kích thước bàn tay = khoảng cách xa nhất (theo x,y) từ cổ tay tới các điểm khác. pts: (...,21,3)."""
    rel = pts - pts[..., 0:1, :]
    s = np.linalg.norm(rel[..., :2], axis=-1).max(axis=-1)
    return np.where(s < 1e-6, 1.0, s)


def hand_vector(pts21x3):
    """Module 1: (21,3) landmark thô -> vector 63 chiều, bất biến vị trí và khoảng cách tới camera."""
    pts = np.asarray(pts21x3, dtype=np.float32)
    rel = pts - pts[0:1, :]
    rel = rel / hand_scale(pts)
    return rel.reshape(-1).astype(np.float32)


def main_hand(hand_list):
    """Chọn bàn tay chính (to nhất trong khung hình) khi thấy nhiều tay. Trả về (21,3) hoặc None."""
    if not hand_list:
        return None
    return max(hand_list, key=lambda h: float(hand_scale(h)))


def sequence_features(hands, hand_present, pose, pose_present):
    """Module 2: tạo đặc trưng cho cả chuỗi (vector hóa).
    hands (T,2,21,3), hand_present (T,2) bool, pose (T,33,3), pose_present (T,) bool -> (T,144)
    """
    hands = np.asarray(hands, dtype=np.float32)
    hp = np.asarray(hand_present, dtype=bool)
    pose = np.asarray(pose, dtype=np.float32)
    pp = np.asarray(pose_present, dtype=bool)
    T = hands.shape[0]

    rel = hands - hands[:, :, 0:1, :]
    scale = hand_scale(hands)                                   # (T,2)
    rel = rel / scale[..., None, None]
    rel = rel * hp[..., None, None]
    shape = rel.reshape(T, 126)

    center = (pose[:, 11, :2] + pose[:, 12, :2]) / 2.0          # (T,2)
    width = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=-1)
    width = np.maximum(width, 1e-3)
    pm = pp.astype(np.float32)

    loc = (hands[:, :, 0, :2] - center[:, None, :]) / width[:, None, None]   # (T,2,2)
    loc = loc * hp[..., None] * pm[:, None, None]
    loc = loc.reshape(T, 4)

    body = (pose[:, POSE_IDX, :2] - center[:, None, :]) / width[:, None, None]
    body = (body * pm[:, None, None]).reshape(T, 2 * len(POSE_IDX))

    return np.concatenate([shape, loc, body], axis=1).astype(np.float32)


def resample_by_time(times, n_out):
    """Chọn n_out chỉ số frame gần nhất theo thời gian đều nhau (độc lập với FPS thật của camera)."""
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    if n == 0:
        raise ValueError("chuỗi rỗng")
    if n == 1:
        return np.zeros(n_out, dtype=int)
    targets = np.linspace(times[0], times[-1], n_out)
    idx = np.searchsorted(times, targets)
    idx = np.clip(idx, 1, n - 1)
    left, right = times[idx - 1], times[idx]
    idx = np.where(np.abs(targets - left) <= np.abs(right - targets), idx - 1, idx)
    return idx.astype(int)
