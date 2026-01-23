# 🎭 ODIN Sentiment Analysis (Phân Tích Cảm Xúc)

Dự án **ODIN Sentiment Analysis** là một ứng dụng web sử dụng trí tuệ nhân tạo để phân tích cảm xúc của văn bản tiếng Việt. Hệ thống có khả năng nhận diện và phân loại ý kiến người dùng thành 3 nhóm: **Tích cực**, **Tiêu cực**, hoặc **Trung lập**.

Đặc biệt, ứng dụng tích hợp **LIME** để giải thích lý do tại sao mô hình đưa ra dự đoán đó, giúp tăng tính minh bạch và độ tin cậy.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)

---

## 🚀 Công Nghệ Sử Dụng (Tech Stack)

Dự án được xây dựng trên các nền tảng và thư viện mã nguồn mở hàng đầu:

*   **Core AI/ML:**
    *   **Google Colab / PyTorch:** Huấn luyện mô hình Deep Learning.
    *   **PhoBERT (`vinai/phobert-base`):** Mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Model) tối ưu cho tiếng Việt.
    *   **Transformers (Hugging Face):** Thư viện nòng cốt để làm việc với BERT/PhoBERT.
    *   **LIME:** Kỹ thuật giải thích mô hình (Model Explainability).

*   **Xử lý dữ liệu (NLP & Data):**
    *   **PyVi:** Tách từ tiếng Việt (Vietnamese Tokenizer).
    *   **Pandas & NumPy:** Xử lý và thao tác dữ liệu bảng.
    *   **Unicode:** Chuẩn hóa bảng mã tiếng Việt.

*   **Web Application:**
    *   **Flask:** Framework Python nhẹ nhàng để xây dựng API và Web Server.
    *   **HTML/CSS/JS:** Giao diện người dùng thân thiện (Frontend).

---

## 📂 Cấu Trúc Dự Án

ODIN_sentiment_analysis/
├── app.py                 # File chính chạy Web App (Flask)
├── train.py               # Script huấn luyện (Fine-tune) mô hình
├── preprocess.py          # Script tiền xử lý dữ liệu (sạch hóa, tách từ)
├── visualize.py           # Script trực quan hóa dữ liệu
├── requirements.txt       # Danh sách các thư viện cần cài đặt
├── data/                  # Thư mục chứa dữ liệu
│   ├── train.csv          # Dữ liệu huấn luyện
│   └── test.csv           # Dữ liệu kiểm thử
├── templates/             # Thư mục chứa giao diện HTML
│   └── index.html         # Giao diện trang chủ
├── model_save/            # (Tự tạo) Thư mục chứa Model sau khi train
└── README.md              # Tài liệu hướng dẫn này
```

---

## 🛠️ Hướng Dẫn Cài Đặt (Installation)

### 1. Yêu cầu hệ thống

*   Python 3.8 trở lên.
*   Khuyến khích sử dụng máy có GPU (NVIDIA) để huấn luyện nhanh hơn, nhưng CPU vẫn chạy tốt việc dự đoán.

### 2. Cài đặt

Bước 1: Clone hoặc tải dự án về máy.
Bước 2: Mở terminal tại thư mục dự án.
Bước 3: Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

*Lưu ý: Nếu gặp lỗi cài đặt PyTorch, hãy truy cập [trang chủ PyTorch](https://pytorch.org/) để lấy câu lệnh cài đặt chính xác cho máy của bạn.*

---

## 📖 Hướng Dẫn Sử Dụng (Usage)

### Bước 1: Chuẩn bị dữ liệu
Nếu bạn có dữ liệu thô mới, hãy chạy file này để làm sạch và chuẩn hóa:

```bash
python preprocess.py
```

### Bước 2: Huấn luyện mô hình (Training)
Nếu bạn cần huấn luyện lại model từ đầu:

```bash
python train.py
```
*   Model sau khi train sẽ được lưu vào thư mục `model_save`.
*   *Lưu ý: Quá trình này có thể mất thời gian tùy thuộc vào phần cứng máy tính.*

### Bước 3: Khởi chạy ứng dụng Web
Để sử dụng giao diện phân tích cảm xúc:

```bash
python app.py
```
Đối với app.py có thể chỉnh tham số num_samples để đưa ra kết quả nhanh hơn, do để số lượng mẫu lấy càng lớn thì độ chính xác càng lớn, tuy nhiên thời gian chạy truy xấut sẽ càng lâu hơn
```
Sau đó truy cập trình duyệt tại địa chỉ: `http://localhost:5000`

---

## 🔥 Tính Năng Nổi Bật

1.  **Phân Tích Đa Chiều:** Đánh giá độ tin cậy (Confidence Score) cho từng dự đoán.
2.  **Giải Thích Minh Bạch:** Hiển thị các từ khóa quan trọng (Key Factors) ảnh hưởng đến kết quả (ví dụ: từ "tệ" làm giảm điểm, từ "tốt" làm tăng điểm).
3.  **Xử Lý Teencode:** Hệ thống tự động nhận diện và chuẩn hóa các từ viết tắt phổ biến (vd: "ko" -> "không", "iu" -> "yêu").

---

## 🤝 Đóng Góp

Mọi ý kiến đóng góp xin vui lòng gửi Pull Request hoặc tạo Issue trên Github.
