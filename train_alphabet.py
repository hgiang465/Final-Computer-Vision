"""Huấn luyện + đánh giá MODULE 1 (bảng chữ cái) từ data/alphabet.csv.

Chạy:  python train_alphabet.py
So sánh Random Forest và MLP bằng leave-one-signer-out (lần lượt giữ 1 người làm tập test) và,
để tham khảo, bằng chia ngẫu nhiên 80/20 (thường cao ảo vì các frame của cùng một người rất giống nhau).
Kết quả: models/alphabet_model.joblib, results/alphabet_metrics.json, results/alphabet_confusion.png
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from features import hand_vector, load_labels


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=0)),
    }


def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    raw = df.drop(columns=["person", "label"]).to_numpy(dtype=np.float32)
    X = np.stack([hand_vector(r.reshape(21, 3)) for r in raw])
    return X, df["label"].astype(str).to_numpy(), df["person"].astype(str).to_numpy()


def metrics_of(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def run(csv_path=None, models_dir=None, results_dir=None):
    csv_path = csv_path or config.ALPHABET_CSV
    models_dir = models_dir or config.MODELS_DIR
    results_dir = results_dir or config.RESULTS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y, persons = load_data(csv_path)
    order = [l for l in load_labels(config.LABELS_ALPHABET) if l in set(y)]
    classes = order + sorted(set(y) - set(order))
    people = sorted(set(persons))
    print(f"{len(y)} frame | {len(classes)} nhãn | người thu: {people}")
    if len(people) < 3:
        print("CẢNH BÁO: nên có ít nhất 3 người để đánh giá signer-independent có ý nghĩa.")

    report = {"n_frames": int(len(y)), "classes": classes, "signers": people, "models": {}}
    best_name, best_f1, best_pred = None, -1, None
    for name in make_models():
        entry = {}
        if len(people) >= 2:
            yt, yp, per = [], [], {}
            for p in people:
                te = persons == p
                if len(set(y[~te])) < 2:
                    continue
                m = make_models()[name].fit(X[~te], y[~te])
                pr = m.predict(X[te])
                per[p] = float(accuracy_score(y[te], pr))
                yt += list(y[te]); yp += list(pr)
            entry["leave_one_signer_out"] = {**metrics_of(yt, yp), "per_signer_accuracy": per}
            score, pred_pair = entry["leave_one_signer_out"]["macro_f1"], (yt, yp)
            print(f"{name}: LOSO acc={entry['leave_one_signer_out']['accuracy']:.3f} "
                  f"macroF1={score:.3f} | từng người: {per}")
        else:
            pred_pair, score = None, -1
        strat = y if min(np.unique(y, return_counts=True)[1]) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=strat)
        pr = make_models()[name].fit(Xtr, ytr).predict(Xte)
        entry["random_split"] = metrics_of(yte, pr)
        print(f"{name}: chia ngẫu nhiên acc={entry['random_split']['accuracy']:.3f} (chỉ tham khảo)")
        if pred_pair is None:
            score, pred_pair = entry["random_split"]["macro_f1"], (list(yte), list(pr))
        m_full = make_models()[name].fit(X, y)
        t = time.perf_counter()
        for i in range(200):
            m_full.predict_proba(X[i % len(X)][None])
        entry["latency_ms_per_prediction"] = float((time.perf_counter() - t) / 200 * 1000)
        report["models"][name] = entry
        if score > best_f1:
            best_name, best_f1, best_pred = name, score, pred_pair

    print(f"\nMô hình tốt nhất: {best_name} (macro-F1 {best_f1:.3f})")
    yt, yp = best_pred
    cm = confusion_matrix(yt, yp, labels=classes)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.45)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(results_dir / "alphabet_confusion.png", dpi=150); plt.close(fig)

    final = make_models()[best_name].fit(X, y)
    joblib.dump({"model": final, "classes": list(final.classes_ if hasattr(final, "classes_")
                                                 else final[-1].classes_), "name": best_name},
                models_dir / "alphabet_model.joblib")
    report["best_model"] = best_name
    with open(results_dir / "alphabet_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Đã lưu models/alphabet_model.joblib, results/alphabet_metrics.json, results/alphabet_confusion.png")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
