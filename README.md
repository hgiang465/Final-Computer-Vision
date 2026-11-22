# Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)

Dự án môn Thị giác máy tính nhận diện ký hiệu qua webcam và ghép kết quả thành câu.

## Tính năng

- **Bảng chữ cái:** MediaPipe Hands trích xuất 21 điểm bàn tay; mô hình Random Forest hoặc MLP phân loại ký hiệu tĩnh.
- **Cử chỉ và dấu:** MediaPipe Hands + Pose trích xuất landmark trên từng khung hình; ứng dụng lấy mẫu lại thành chuỗi 30 khung hình để mô hình LSTM hoặc GRU phân loại.
- **Ghép câu:** `sentence_app.py` kết hợp chữ cái, dấu và từ/cử chỉ thành câu tiếng Việt (Unicode NFC).

> **Dữ liệu:** Repository không chứa bộ dữ liệu VSL thực tế. Hãy tự thu thập dữ liệu của nhóm. Các nhãn trong
> `config/labels_alphabet.txt` và `config/labels_gesture.txt` chỉ là danh sách khởi đầu; cần thống nhất nhãn và cách ký hiệu
> với người am hiểu VSL trước khi thu dữ liệu.

## Cấu trúc dự án

```text
.
├── config/       # Danh sách nhãn
├── data/         # Dữ liệu tự thu thập
├── models/       # Model đã huấn luyện và model MediaPipe
├── results/      # Metrics, biểu đồ và kết quả chạy
├── scripts/      # Mã thu thập dữ liệu, huấn luyện, ứng dụng và tự kiểm tra
├── vsl/          # Module lõi: cấu hình, đặc trưng, dự đoán, giao diện
├── *.py          # Lệnh khởi chạy tương thích từ thư mục gốc
├── README.md
└── requirements.txt
```

Chạy các lệnh từ thư mục gốc của dự án; các file `.py` ở gốc chuyển tiếp tới script tương ứng
trong `scripts/`, nên cách gọi lệnh hiện tại không thay đổi.

## Yêu cầu

- Python 3.10–3.12 được khuyến nghị.
- Webcam.
- PyTorch để huấn luyện và nhận diện cử chỉ; ứng dụng chữ cái có thể chạy độc lập nếu chỉ có model chữ cái.

## Cài đặt

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra cài đặt (không cần mở camera):

```powershell
python selftest.py
```

Nếu PowerShell chặn kích hoạt môi trường, có thể chạy trực tiếp bằng Python trong venv:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

MediaPipe thường tự tải các model cần thiết khi chạy lần đầu. Nếu tải tự động thất bại, tải các tệp dưới đây
vào thư mục `models/mediapipe/`:

- [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- [pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task)

## Quy trình dữ liệu và huấn luyện

1. Thống nhất nhãn trong `config/labels_alphabet.txt` và `config/labels_gesture.txt`.
2. Thu dữ liệu chữ cái và cử chỉ theo hướng dẫn bên dưới.
3. Huấn luyện từng mô hình:

   ```bash
   python train_alphabet.py
   python train_gesture.py
   ```
4. Chạy ứng dụng:

   ```bash
   python sentence_app.py
   ```

Các mô hình được lưu ở `models/alphabet_model.joblib` và `models/gesture_model.pt`.
Kết quả huấn luyện và biểu đồ được lưu trong `results/`.

## Thu dữ liệu

### Bảng chữ cái

```bash
python collect_alphabet.py --person A
```

- `r`: bắt đầu/dừng ghi nhãn hiện tại.
- `n` / `p`: chuyển sang nhãn kế/trước.
- `u`: hoàn tác lần ghi gần nhất.
- `q`: thoát.

Thu khoảng 250 khung hình cho mỗi nhãn, mỗi người. Giữ đúng ký hiệu nhưng thay đổi nhẹ vị trí,
khoảng cách và góc cổ tay. Nên thu từ nhiều người để mô hình hoạt động tốt hơn với người mới.

### Cử chỉ

```bash
python collect_gesture.py --person A
```

- `SPACE`: quay một mẫu (đếm ngược 2 giây, sau đó ghi khoảng 1,5 giây).
- `n` / `p`: chuyển cử chỉ.
- `u`: xóa mẫu vừa quay.
- `q`: thoát.

Ngồi đủ xa để camera thấy hai vai và hai bàn tay; thực hiện trọn cử chỉ trong khoảng 1,5 giây.
Hướng tới 40–60 mẫu cho mỗi cử chỉ, tổng hợp từ ít nhất 3 người. Nhãn `idle` cũng quan trọng: hãy thu các tình huống không ký hiệu như tay nghỉ hoặc cử động ngẫu nhiên.

## Sử dụng ứng dụng ghép câu

Trong `sentence_app.py`, tay đứng yên được dùng để nhận diện chữ cái; chuyển động tay được dùng để nhận diện
cử chỉ hoặc dấu. Giao diện hiển thị trạng thái tay, chữ đang được dự đoán, tiến trình giữ ổn định và câu hiện tại.

- `SPACE`: kết thúc từ đang đánh vần.
- `b`: hoàn tác bước cuối.
- `c`: xóa câu.
- `,` / `.`: giảm/tăng độ nhạy phát hiện chuyển động.
- `q`: thoát.

Nếu chỉ có một trong hai mô hình, ứng dụng dùng phần tương ứng. Để ghép câu đầy đủ, hãy huấn luyện cả hai mô hình.
Khi báo cáo độ chính xác, chỉ sử dụng kết quả đo sau khi huấn luyện trên dữ liệu thực của nhóm.

## Huấn luyện nhanh

Để chạy thử huấn luyện cử chỉ với ít epoch hơn:

```bash
python train_gesture.py --quick
```
