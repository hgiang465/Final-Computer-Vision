import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import config
from features import load_labels, resample_by_time, sequence_features
from gesture_model import SeqClassifier

def load_samples(gesture_dir):
    samples = []
    for p in sorted(gesture_dir.glob("*/*.npz")):
        z = np.load(p, allow_pickle=False)
        samples.append({"hands": z["hands"], "hand_present": z["hand_present"], "pose": z["pose"],
                        "pose_present": z["pose_present"], "person": str(z["person"]), "label": str(z["label"])})
    return samples


def featurize(s):
    return sequence_features(s["hands"], s["hand_present"], s["pose"], s["pose_present"])


def augment(s, rng):
    """Tăng cường dữ liệu trên landmark thô: đổi tốc độ/cắt thời gian, xoay nhẹ, nhiễu, rớt tay ngẫu nhiên."""
    T = s["hands"].shape[0]
    frac = rng.uniform(0.8, 1.0)
    start = rng.uniform(0, (1 - frac) * (T - 1))
    idx = np.clip(np.round(np.linspace(start, start + frac * (T - 1), T)).astype(int), 0, T - 1)
    hands, pose = s["hands"][idx].copy(), s["pose"][idx].copy()
    hp, pp = s["hand_present"][idx].copy(), s["pose_present"][idx].copy()
    ang = np.deg2rad(np.clip(rng.normal(0, 6), -12, 12))
    c, sn = np.cos(ang), np.sin(ang)

    def rot(a):
        x, y = a[..., 0] - 0.5, a[..., 1] - 0.5
        a[..., 0], a[..., 1] = c * x - sn * y + 0.5, sn * x + c * y + 0.5
    rot(hands); rot(pose)
    hands += rng.normal(0, 0.004, hands.shape).astype(np.float32)
    pose += rng.normal(0, 0.004, pose.shape).astype(np.float32)
    drop = rng.random(hp.shape) < 0.03
    hp = hp & ~drop
    return {**s, "hands": hands, "pose": pose, "hand_present": hp, "pose_present": pp}


def build_xy(samples, class_to_idx, aug_factor, rng):
    X, y = [], []
    for s in samples:
        X.append(featurize(s)); y.append(class_to_idx[s["label"]])
        for _ in range(aug_factor):
            X.append(featurize(augment(s, rng))); y.append(class_to_idx[s["label"]])
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------- huấn luyện
def train_model(Xtr, ytr, arch, num_classes, epochs, hidden, layers, device, seed=0, verbose=False):
    torch.manual_seed(seed)
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xn = torch.tensor((Xtr - mean) / std)
    yt = torch.tensor(ytr)
    model = SeqClassifier(Xtr.shape[-1], num_classes, arch, hidden, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    n, bs = len(Xn), 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb, yb = Xn[b].to(device), yt[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs}  loss={tot / n:.4f}")
    return model.eval(), mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def predict(model, mean, std, X, device):
    out = model(torch.tensor((X - mean) / std).to(device))
    return out.argmax(1).cpu().numpy()


def metrics_of(yt, yp):
    return {"accuracy": float(accuracy_score(yt, yp)),
            "macro_precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}


def plot_cm(cm, classes, title, path):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.45)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(gesture_dir=None, models_dir=None, results_dir=None, archs=("lstm", "gru"), epochs=40,
        hidden=96, layers=2, aug=3, final_arch=None, seed=0):
    gesture_dir = gesture_dir or config.GESTURE_DIR
    models_dir = models_dir or config.MODELS_DIR
    results_dir = results_dir or config.RESULTS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = load_samples(gesture_dir)
    if not samples:
        raise SystemExit(f"Không có dữ liệu trong {gesture_dir}. Hãy chạy collect_gesture.py trước.")
    present = {s["label"] for s in samples}
    classes = [l for l in load_labels(config.LABELS_GESTURE) if l in present]
    classes += sorted(present - set(classes))
    c2i = {c: i for i, c in enumerate(classes)}
    people = sorted({s["person"] for s in samples})
    persons = np.array([s["person"] for s in samples])
    ylab = np.array([c2i[s["label"]] for s in samples])
    print(f"{len(samples)} mẫu | {len(classes)} cử chỉ | người thu: {people} | thiết bị: {device}")
    if len(people) < 3:
        print("CẢNH BÁO: nên có ít nhất 3 người để đánh giá signer-independent có ý nghĩa.")
    rng = np.random.default_rng(seed)
    report = {"n_samples": len(samples), "classes": classes, "signers": people, "models": {},
              "config": {"epochs": epochs, "hidden": hidden, "layers": layers, "aug_factor": aug}}

    best_arch, best_f1 = None, -1
    for arch in archs:
        print(f"\n===== {arch.upper()} =====")
        t0 = time.time()
        yt_all, yp_all, per = [], [], {}
        if len(people) >= 2:
            folds = [(np.where(persons != p)[0], np.where(persons == p)[0], p) for p in people]
        else:
            idx = np.arange(len(samples))
            strat = ylab if min(np.bincount(ylab)) >= 2 else None
            tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
            folds = [(tr, te, "random_split")]
        for tr, te, name in folds:
            Xtr, ytr = build_xy([samples[i] for i in tr], c2i, aug, rng)
            Xte, yte = build_xy([samples[i] for i in te], c2i, 0, rng)
            model, mean, std = train_model(Xtr, ytr, arch, len(classes), epochs, hidden, layers, device, seed)
            yp = predict(model, mean, std, Xte, device)
            per[name] = float(accuracy_score(yte, yp))
            yt_all += list(yte); yp_all += list(yp)
            print(f"  giữ lại {name}: acc={per[name]:.3f}")
        m = metrics_of(yt_all, yp_all)
        # độ trễ suy luận 1 chuỗi (CPU)
        model_cpu = model.to("cpu")
        x1 = torch.tensor((Xte[:1] - mean) / std)
        with torch.no_grad():
            for _ in range(5):
                model_cpu(x1)
            t = time.perf_counter()
            for _ in range(50):
                model_cpu(x1)
        lat = (time.perf_counter() - t) / 50 * 1000
        n_params = sum(p.numel() for p in model.parameters())
        report["models"][arch] = {**m, "per_signer_accuracy": per, "evaluation":
                                  "leave_one_signer_out" if len(people) >= 2 else "random_split",
                                  "latency_ms_per_sequence_cpu": float(lat), "parameters": int(n_params),
                                  "train_eval_seconds": float(time.time() - t0)}
        print(f"  {arch.upper()}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"latency={lat:.2f}ms params={n_params}")
        cm = confusion_matrix(yt_all, yp_all, labels=list(range(len(classes))))
        plot_cm(cm, classes, f"Confusion matrix - {arch.upper()}", results_dir / f"gesture_confusion_{arch}.png")
        if m["macro_f1"] > best_f1:
            best_arch, best_f1 = arch, m["macro_f1"]

    final_arch = final_arch or best_arch
    print(f"\nHuấn luyện lại {final_arch.upper()} trên toàn bộ dữ liệu để dùng cho app...")
    Xall, yall = build_xy(samples, c2i, aug, rng)
    model, mean, std = train_model(Xall, yall, final_arch, len(classes), epochs, hidden, layers, device, seed, True)
    torch.save({"state_dict": model.to("cpu").state_dict(), "arch": final_arch, "classes": classes,
                "input_dim": int(Xall.shape[-1]), "hidden": hidden, "layers": layers, "mean": mean, "std": std,
                "seq_len": config.SEQ_LEN, "seq_seconds": config.SEQ_SECONDS},
               models_dir / "gesture_model.pt")
    report["best_model"] = final_arch
    with open(results_dir / "gesture_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Đã lưu models/gesture_model.pt và results/gesture_metrics.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="chạy thử nhanh (ít epoch)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--aug", type=int, default=3, help="số bản tăng cường dữ liệu cho mỗi mẫu")
    ap.add_argument("--archs", nargs="+", default=["lstm", "gru"], choices=["lstm", "gru"])
    ap.add_argument("--final", choices=["lstm", "gru"], default=None, help="ép chọn kiến trúc lưu cho app")
    a = ap.parse_args()
    run(epochs=8 if a.quick else a.epochs, hidden=a.hidden, layers=a.layers, aug=1 if a.quick else a.aug,
        archs=tuple(a.archs), final_arch=a.final)
