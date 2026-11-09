"""
Huan luyen + danh gia Module 1 tu landmarks.csv.

Chay:  python train_module1.py
Can:   pip install "mediapipe==0.10.35" numpy pandas scikit-learn matplotlib joblib

Danh gia "leave-one-person-out": lan luot lay 1 nguoi ra lam tap test (model chua tung thay nguoi do),
train bang cac nguoi con lai -> do duoc kha nang tong quat sang nguoi la (signer-independent).
Neu chi co 1 nguoi, se chia ngau nhien (so se cao ao, khong dung de bao cao).
Ket qua: model_module1.joblib, confusion_matrix.png, in accuracy ra man hinh.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import PROJECT_DIR, normalize

CSV_PATH = PROJECT_DIR / "landmarks.csv"
MODEL_PATH = PROJECT_DIR / "model_module1.joblib"
CONFUSION_MATRIX_PATH = PROJECT_DIR / "confusion_matrix.png"
LANDMARK_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]

if not CSV_PATH.exists():
    raise SystemExit(
        "Chua co landmarks.csv. Hay chay "
        "'python collect_landmarks.py --person A' va thu landmark truoc."
    )

df = pd.read_csv(CSV_PATH)
if df.empty:
    raise SystemExit("landmarks.csv chua co mau. Hay thu du lieu truoc khi huan luyen.")

required_columns = ["person", "label", *LANDMARK_COLUMNS]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise SystemExit(
        "landmarks.csv sai dinh dang hoac thieu cot: " + ", ".join(missing_columns)
    )
if df[["person", "label"]].isna().any().any():
    raise SystemExit("landmarks.csv co dong thieu person/label. Hay kiem tra file du lieu.")

try:
    raw = df[LANDMARK_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
except (TypeError, ValueError) as exc:
    raise SystemExit("Toa do landmark trong landmarks.csv phai la so.") from exc
if not np.isfinite(raw).all():
    raise SystemExit("landmarks.csv co toa do khong hop le (NaN/inf).")

X = np.stack([normalize(row) for row in raw])
y = df["label"].to_numpy()
persons = df["person"].to_numpy()
classes = sorted(set(y))
people = sorted(set(persons))
print(f"{len(df)} frame, {len(classes)} nhan, nguoi thu: {people}")

if len(classes) < 2:
    raise SystemExit("Can thu it nhat 2 nhan khac nhau de huan luyen bo phan loai.")
if len(people) == 1:
    class_counts = pd.Series(y).value_counts()
    test_count = int(np.ceil(0.2 * len(df)))
    if class_counts.min() < 2 or test_count < len(classes):
        raise SystemExit(
            "Du lieu qua it de chia train/validation. Hay thu them mau cho moi nhan "
            "(it nhat 2 mau/nhan va tong so validation du bao phu du cac nhan)."
        )


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=0)),
    }


results, preds_all = {}, {}
for name in make_models():
    y_true, y_pred, accs = [], [], []
    if len(people) >= 2:
        for person in people:
            test_mask = persons == person
            model = make_models()[name].fit(X[~test_mask], y[~test_mask])
            predictions = model.predict(X[test_mask])
            accs.append(accuracy_score(y[test_mask], predictions))
            y_true += list(y[test_mask])
            y_pred += list(predictions)
        print(f"{name}: accuracy tung nguoi bi giau = "
              f"{[f'{person}:{accuracy:.3f}' for person, accuracy in zip(people, accs)]}  "
              f"| trung binh = {np.mean(accs):.3f}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        predictions = make_models()[name].fit(X_train, y_train).predict(X_test)
        y_true, y_pred = list(y_test), list(predictions)
        accs = [accuracy_score(y_test, predictions)]
        print(f"{name}: accuracy (chia ngau nhien, KHONG dung de bao cao) = {accs[0]:.3f}")
    results[name] = float(np.mean(accs))
    preds_all[name] = (y_true, y_pred)

best = max(results, key=results.get)
print(f"\nModel tot nhat: {best} ({results[best]:.3f})")
yt, yp = preds_all[best]
print(classification_report(yt, yp, zero_division=0))

cm = confusion_matrix(yt, yp, labels=classes)
fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5),) * 2)
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
ax.set_xlabel("Du doan"); ax.set_ylabel("Thuc te"); ax.set_title(f"Confusion matrix - {best}")
fig.tight_layout(); fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)

final = make_models()[best].fit(X, y)
joblib.dump({"model": final, "classes": classes}, MODEL_PATH)
print(f"Da luu {MODEL_PATH.name} va {CONFUSION_MATRIX_PATH.name}")
