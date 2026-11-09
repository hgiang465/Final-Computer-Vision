# Quy trình làm project VSL Real-time Detection

Áp dụng cho Topic 4 (Real-time sign language / gesture recognition), phiên bản tiếng Việt,
dùng object detection (YOLOv8) thay vì DETR-from-scratch để vừa thời gian một kỳ học.

## Bước 0 — Chuẩn bị

```bash
pip install ultralytics opencv-python
```

Cả nhóm thống nhất trước:
- Danh sách 15–20 ký hiệu VSL sẽ làm (nên chọn ký hiệu tĩnh, dễ phân biệt hình dạng tay,
  ví dụ lấy từ giáo trình VSL của ĐH Sư phạm Hà Nội hoặc bộ VOYA_VSL/VSignChat để tham khảo tên gọi).
- Mỗi thành viên sẽ tự thu ảnh bằng camera laptop của mình cho TẤT CẢ các nhãn
  (để có sự đa dạng người/ánh sáng/góc quay — quan trọng cho phần đánh giá signer-independent).

## Bước 1 — Thu thập ảnh

Chạy `1_collect_images.py`. Mỗi thành viên chạy trên máy mình, đổi `LABELS` cho khớp danh sách chung.
Gộp toàn bộ ảnh của cả nhóm vào một thư mục `data/images/` trước khi gán nhãn.

## Bước 2 — Gán nhãn bounding box (làm thủ công qua giao diện, không có script)

Khuyên dùng **Roboflow** (miễn phí, web-based, dễ hơn label-studio cho nhóm mới bắt đầu):

1. Tạo project mới trên roboflow.com, loại "Object Detection".
2. Upload toàn bộ ảnh trong `data/images/`.
3. Vẽ bounding box quanh bàn tay cho từng ảnh, gán đúng nhãn ký hiệu.
4. Dùng tính năng "Generate" → chia train/val (khuyên 80/20) → có thể bật augmentation
   (flip, brightness) để tăng dữ liệu nếu ảnh gốc ít.
5. Export dataset ở định dạng **YOLOv8**, tải về, giải nén sao cho khớp cấu trúc:

```
data/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Thay thế cho `data/` mà `1_collect_images.py` tạo ra.

Nếu nhóm muốn dùng LabelImg (chạy local, không cần internet) cũng được — export cùng định dạng YOLO (.txt).

## Bước 3 — Sửa `data.yaml`

Cập nhật đúng thứ tự và tên các nhãn (`names:`) khớp với danh sách `LABELS` đã dùng ở bước 1.

## Bước 4 — Huấn luyện

```bash
python 2_train.py
```

Theo dõi `runs/detect/vsl_yolov8n/results.png` để xem đường cong loss/mAP.
Nếu mAP thấp hoặc quá overfit, quay lại tăng số ảnh/nhãn hoặc chỉnh augmentation ở Roboflow.

## Bước 5 — Demo real-time

```bash
python 3_realtime.py
```
(nhớ sửa `CHECKPOINT_PATH` trỏ đúng tới `best.pt` vừa train ra)

## Gợi ý cho phần "Extensions" trong đề bài

- **Signer-independent evaluation:** chia người quay thành 2 nhóm — nhóm A dùng để train,
  nhóm B chỉ dùng để test — để đo model có generalize sang người lạ không.
- **Confusion analysis:** dùng `model.val()` để lấy confusion matrix (`runs/detect/.../confusion_matrix.png`),
  chỉ ra các ký hiệu hay bị nhầm với nhau.
- **Latency:** script `3_realtime.py` đã in sẵn thời gian inference (ms) mỗi frame để bạn so sánh
  giữa `yolov8n` (nhanh, nhẹ) và `yolov8s` (chính xác hơn, chậm hơn) — đúng hướng
  "latency-vs-accuracy" mà đề bài nhắc tới.

## Module 1 — Nhận diện chữ cái bằng landmark

Module 1 dùng các nhãn trong `labels.txt`, thu tọa độ bàn tay từ webcam rồi huấn luyện
classifier. Chạy các lệnh sau trong PowerShell tại thư mục project:

```powershell
python -m pip install "mediapipe==0.10.35" numpy pandas scikit-learn matplotlib joblib
python collect_landmarks.py --person A
python train_module1.py
python realtime_module1.py
```

Lần chạy đầu, script tự tải model bàn tay của MediaPipe (cần Internet). Trong cửa sổ thu dữ liệu,
nhấn `r` để bắt đầu/dừng ghi nhãn, `n`/`p` để chuyển nhãn và `q` để thoát. Giữ mỗi ký hiệu
trước camera cho đến khi thu đủ mẫu. Có thể chạy collector nhiều lần với mã người khác, ví dụ
`--person B`, để đánh giá trên người chưa có trong dữ liệu. Cần có ít nhất hai nhãn và đủ mẫu
cho mỗi nhãn trước khi huấn luyện. Khi realtime chạy, giữ ký hiệu ổn định để thêm chữ; nhấn `q`
để thoát.
