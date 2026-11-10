"""Bộ dự đoán dùng trong app: tải model đã train và dự đoán từ landmark."""
import joblib
import numpy as np

import config
from features import hand_vector, resample_by_time, sequence_features


class AlphabetPredictor:
    def __init__(self, path=None):
        bundle = joblib.load(path or config.ALPHABET_MODEL)
        self.model = bundle["model"]
        self.classes = list(self.model.classes_)
        self.name = bundle.get("name", "?")

    def predict(self, hand21x3):
        """hand21x3: (21,3) landmark thô. Trả về (nhãn, độ tin cậy)."""
        proba = self.model.predict_proba([hand_vector(hand21x3)])[0]
        i = int(np.argmax(proba))
        return self.classes[i], float(proba[i])


class GesturePredictor:
    def __init__(self, path=None):
        import torch                      # import muộn: chỉ cần khi dùng module cử chỉ
        from gesture_model import SeqClassifier
        self.torch = torch
        ck = torch.load(path or config.GESTURE_MODEL, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        self.seq_len = ck["seq_len"]
        self.mean, self.std = ck["mean"], ck["std"]
        self.arch = ck["arch"]
        self.model = SeqClassifier(ck["input_dim"], len(self.classes), ck["arch"], ck["hidden"], ck["layers"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        torch.set_num_threads(2)

    def predict(self, frames):
        """frames: list các (thời_gian, FrameLandmarks) trong cửa sổ gần nhất. Trả về (nhãn, độ tin cậy)."""
        times = [t for t, _ in frames]
        sel = resample_by_time(times, self.seq_len)
        fls = [frames[i][1] for i in sel]
        feats = sequence_features(np.stack([f.hands for f in fls]), np.stack([f.hand_present for f in fls]),
                                  np.stack([f.pose for f in fls]), np.array([f.pose_present for f in fls]))
        x = self.torch.tensor(((feats - self.mean) / self.std)[None].astype(np.float32))
        with self.torch.no_grad():
            proba = self.torch.softmax(self.model(x), dim=1)[0].numpy()
        i = int(np.argmax(proba))
        return self.classes[i], float(proba[i])
