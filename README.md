# Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL) theo thời gian thực

Project môn Computer Vision (Topic 4). Chạy hoàn toàn trên camera laptop, không cần GPU.

- **Module 1 – Bảng chữ cái:** MediaPipe Hands (21 điểm) → chuẩn hóa → Random Forest / MLP.
- **Module 2 – Cử chỉ:** MediaPipe Hands + Pose → chuỗi 30 frame → LSTM / GRU (PyTorch).
- **App tích hợp:** chuyển chế độ bằng phím 1/2, ghép chữ có dấu (Unicode NFC), đo độ trễ/FPS.

> **Trung thực về dữ liệu:** project KHÔNG kèm dữ liệu VSL thật. Bạn phải tự thu (script thu data đã có sẵn).
> Các danh sách nhãn trong `labels_alphabet.txt` / `labels_gesture.txt` chỉ là **danh sách khởi đầu**,
> hãy đối chiếu với video/từ điển VSL của nhóm và người am hiểu VSL rồi sửa lại.

## 1. Cài đặt

```bash
python --version        # khuyên dùng Python 3.10 - 3.12 (bản mới hơn vẫn chạy được)
python -m venv venv
venv\Scripts\activate           # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python selftest.py              # tự kiểm tra, KHÔNG cần camera (lần đầu sẽ tự tải 2 file model MediaPipe ~13MB)
python selftest.py --camera     # kiểm tra camera + MediaPipe: đưa tay vào khung hình, bấm q
```

Lỗi cũ `module 'mediapipe' has no attribute 'solutions'` đã được xử lý: code tự dùng API mới (MediaPipe Tasks)
khi bản mediapipe không còn `mp.solutions`. Nếu máy không tải được model, tải tay 2 file này vào `models/mediapipe/`:

- hand: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- pose: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

## 2. Quy trình làm việc

| Bước | Lệnh                                                | Kết quả                                                |
| ------ | ---------------------------------------------------- | -------------------------------------------------------- |
| 0      | sửa `labels_alphabet.txt`, `labels_gesture.txt` | danh sách lớp thống nhất của nhóm                  |
| 1      | `python collect_alphabet.py --person A`            | `data/alphabet.csv`                                    |
| 2      | `python collect_gesture.py --person A`             | `data/gesture/<nhãn>/*.npz`                           |
| 3      | gom dữ liệu của các thành viên (xem mục 4)    | dữ liệu đủ nhiều người                            |
| 4      | `python train_alphabet.py`                         | `models/alphabet_model.joblib`, `results/alphabet_*` |
| 5      | `python train_gesture.py`                          | `models/gesture_model.pt`, `results/gesture_*`       |
| 6      | `python app.py`                                    | demo thời gian thực + `results/latency.json`         |

### Thu dữ liệu Module 1 (chữ cái)

- Phím: `r` ghi/dừng · `n`/`p` đổi nhãn · `u` hoàn tác lần ghi vừa rồi · `q` thoát.
- Mỗi nhãn ~250 frame (~8 giây) mỗi người. **Giữ đúng ký hiệu**, nhưng đổi nhẹ khoảng cách, góc cổ tay, vị trí trong khung hình.
- Ký hiệu hình chữ làm trước, sau đó mới tới dấu mũ/móc/trăng, rồi dấu thanh (app ghép theo thứ tự đó).

### Thu dữ liệu Module 2 (cử chỉ)

- Phím: `SPACE` quay 1 mẫu (đếm ngược 2 giây rồi ghi 1,5 giây) · `n`/`p` đổi cử chỉ · `u` xoá mẫu vừa quay.
- Ngồi xa camera đủ để thấy **cả hai vai và hai bàn tay**. Làm trọn cử chỉ trong ~1,5 giây.
- Mỗi cử chỉ ~40–60 mẫu (cộng tất cả các người), thu từ ít nhất 3 người.
- Nhãn `idle` (không ký hiệu) rất quan trọng: quay tay nghỉ, tay thả xuống, cử động ngẫu nhiên.

## 3. Đánh giá cho báo cáo (Section V)

- `results/alphabet_metrics.json`, `results/gesture_metrics.json`: accuracy, macro precision/recall/F1,
  accuracy từng người bị giữ lại (leave-one-signer-out), độ trễ dự đoán, số tham số. → điền **Bảng III**.
- `results/alphabet_confusion.png`, `results/gesture_confusion_lstm.png`, `..._gru.png`: ma trận nhầm lẫn → mục Discussion.
- `results/latency.json`: độ trễ/FPS của pipeline thật, tạo ra khi thoát `app.py`.
- JSON còn có điểm "chia ngẫu nhiên" để so sánh mức giảm khi đánh giá trên người lạ (câu hỏi thứ 3 ở mục Discussion).

## 4. Gom dữ liệu của nhiều thành viên

- Chữ cái: mỗi người gửi file `data/alphabet.csv`; nối lại thành một file (giữ **một** dòng tiêu đề).
- Cử chỉ: copy các thư mục `data/gesture/<nhãn>/` của mọi người vào chung một chỗ (tên file đã có mã người).
- Mã `--person` phải **khác nhau** giữa các thành viên, vì đánh giá leave-one-signer-out dựa vào mã này.

## 5. Lưu ý & giới hạn

- Camera bị lật gương khi thu và khi chạy app (giống nhau). Người thuận tay trái ký hiệu đối xứng sẽ cho kết quả kém hơn.
- Hệ thống chỉ nhận diện các lớp đã train, không dịch toàn bộ VSL, không xử lý biểu cảm khuôn mặt.
- Dấu có chuyển động tay nên đưa vào `labels_gesture.txt`, không đặt trong `labels_alphabet.txt`.
- Chưa có dữ liệu thật thì chưa có con số nào: đừng ghi accuracy vào báo cáo trước khi chạy `train_*.py` trên dữ liệu của nhóm.
- Muốn train nhanh/thử: `python train_gesture.py --quick`. Có GPU NVIDIA thì PyTorch tự dùng.
