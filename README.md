# vn-pdf-to-docx

Công cụ chuyển đổi văn bản hành chính Việt Nam dạng **PDF/A scan** (có dấu mộc đỏ, bảng biểu phức tạp, trang xoay ngang 90°/270°) sang **Microsoft Word (`.docx`)** chuẩn thể thức **Nghị định 30/2020/NĐ-CP** — chạy **100% Offline trên máy** (không cần API Key, không dùng `.env`, không gọi AI đám mây).

---

## Tính năng chính

- **Khử dấu mộc đỏ**: Tách kênh Red + khử nhiễu + tăng tương phản `CLAHE` giúp xóa con dấu đỏ đè lên chữ mà vẫn giữ nét chữ đen.
- **Tự động xoay trang ngang (`90° / 270°`)**: Phát hiện trang scan bị xoay dọc hoặc khổ ngang và tự động chuyển Section Word sang **Landscape** (rồi quay lại **Portrait** ở trang dọc tiếp theo).
- **Nhận diện bảng biểu chính xác**: Bắt khung bảng và ô `(r, c)` bằng OpenCV Morphological Grid (hỗ trợ ô gộp ngang `colspan` và fallback theo chiếu cột khi mất viền).
- **OCR Tiếng Việt Offline tối ưu tốc độ**: Dùng `RapidOCR` phát hiện vùng chữ + `VietOCR` (`vgg_transformer`) nhận diện theo lô (`batch_size=64`), tự động tận dụng **GPU NVIDIA (CUDA)** nếu có hoặc chạy trên **CPU**.
- **Xuất Word chuẩn Nghị định 30/2020/NĐ-CP**: Khổ A4, font **Times New Roman 13pt**, tự gom bảng 2 cột không viền (Quốc hiệu – Tiêu ngữ, Nơi nhận – Chữ ký), tự động lưu sang tên mới (`... (1).docx`) nếu tệp đích đang mở trong Word.

---

## Cài đặt nhanh (Windows)

### Yêu cầu
- **Python**: `3.10` trở lên (khuyên dùng `3.11`).
- **Phần cứng**: Chạy được trên **CPU** thông thường; nếu máy có card **NVIDIA (CUDA)** sẽ tự động tăng tốc `VietOCR`.

### Các bước cài đặt

1. **Tải mã nguồn và mở thư mục dự án:**
   ```bash
   git clone https://github.com/anhdungkudo/vn-pdf-to-docx.git
   cd vn-pdf-to-docx
   ```

2. **(Khuyên dùng) Tạo môi trường ảo Python:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. *(Tùy chọn)* **Nếu máy có GPU NVIDIA muốn chạy nhanh hơn bằng CUDA:**
   Cài đè `torch` và `torchvision` bản CUDA (ví dụ CUDA 13.0 / 12.x tương thích driver máy bạn):
   ```bash
   pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
   *(Ứng dụng sẽ tự động nhận diện GPU CUDA khi chạy, không cần cấu hình thêm.)*

---

## Cách sử dụng

### Cách 1: Giao diện đồ họa (GUI) — Khuyên dùng

- Nhấp đúp **`Chay_Giao_Dien.bat`** (hoặc `Chay_Giao_Dien_An_Console.vbs`), **hoặc** chạy trong terminal:
  ```bash
  python main.py
  ```
- Nếu đã đóng gói `.exe`, mở trực tiếp:
  ```text
  dist\ChuyenDoiVanBanPDF\ChuyenDoiVanBanPDF.exe
  ```

**Các tùy chọn trên giao diện:**
- **Phạm vi trang**: Chuyển toàn bộ hoặc nhập trang cụ thể (ví dụ `1`, `1-5`, `1,3,5`).
- **Độ nét (DPI)**: `72–600 DPI` (mặc định **`200 DPI`** — cân bằng tốt nhất giữa tốc độ và độ nét; chỉ nên tăng `250–300` với trang scan mờ hoặc chữ rất nhỏ).
- **Căn lề Nơi nhận**: `Căn giữa` (mặc định) hoặc `Căn trái (Nghị định 30)`.
- **Khử dấu đỏ con dấu tròn**: Bật/tắt tách kênh Red + NLM Denoise (mặc định bật).
- **Tự động mở file Word khi hoàn tất**.

### Cách 2: Dòng lệnh (CLI)

```bash
# Chuyển đổi toàn bộ file PDF (mặc định 200 DPI)
python main.py van_ban_scan.pdf -o van_ban_sach.docx -v

# Chỉ chuyển đổi các trang cụ thể
python main.py van_ban_scan.pdf -o van_ban_sach.docx --pages 1-5 -v

# Hoặc gọi trực tiếp module CLI
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx --dpi 200
```

**Các tham số hỗ trợ:**
- `-o`, `--output`: Đường dẫn file `.docx` đầu ra (mặc định trùng tên file PDF đầu vào).
- `--dpi`: Độ phân giải render PDF (mặc định `200`).
- `--pages`: Phạm vi trang cần xử lý (ví dụ `1`, `1-5`, `1,3,5`).
- `--device {auto,cpu,cuda}`: Chọn thiết bị chạy `VietOCR` (`auto` mặc định tự dùng GPU CUDA nếu có).
- `--batch-size`: Số vùng chữ mỗi batch khi chạy `VietOCR` (mặc định `64`).
- `--footer-left-align {center,left}`: Căn lề cột trái phần `Nơi nhận` (mặc định `center`).
- `-v`, `--verbose`: Hiện log chi tiết trong quá trình chạy.

---

## Quy trình xử lý 3 bước

1. **Tiền xử lý ảnh (`vn_pdf_to_docx.preprocessor`)**: Render PDF ở `200 DPI` bằng `PyMuPDF` (`fitz`), lọc dấu đỏ bằng kênh Red, khử nhiễu (`fastNlMeansDenoising`) và tăng tương phản `CLAHE`.
2. **Bố cục, Bảng & OCR (`vn_pdf_to_docx.ocr_layout`, `vn_pdf_to_docx.table_detector`)**: Tự động xoay trang ngang `90°/270°`, dò khung bảng OpenCV, phát hiện vùng chữ bằng `RapidOCR`, gom dòng/căn lề và đọc chữ tiếng Việt theo batch (`FP32`, `batch_size=64`) bằng `VietOCR`.
3. **Dựng file Word (`vn_pdf_to_docx.docx_builder`)**: Xuất `.docx` chuẩn A4 (tự động chuyển `Portrait` / `Landscape`), giữ nguyên bảng 2 cột không viền, bảng dữ liệu `Table Grid` và xử lý an toàn khi tệp đích đang mở.

---

## Đóng gói sang `.EXE` (Windows) & Kiểm thử

### Đóng gói `.EXE`
Cài thêm `pyinstaller` (`pip install pyinstaller`), sau đó chạy **`Build_EXE.bat`** hoặc:

```bash
# Đóng gói dạng thư mục (khởi động nhanh, ổn định nhất - khuyên dùng)
python build_app.py --clean

# Đóng gói thành 1 file .exe duy nhất
python build_app.py --onefile --clean
```

### Chạy bộ kiểm thử (Unit Tests)
```bash
python -m unittest discover -s tests
```

---

## Cấu trúc dự án

```text
vn-pdf-to-docx/
├── vn_pdf_to_docx/          # Mã nguồn chính
│   ├── preprocessor.py      # Render PDF, lọc dấu đỏ kênh Red, khử nhiễu & CLAHE
│   ├── table_detector.py    # Nhận diện khung bảng & ô bằng OpenCV
│   ├── ocr_models.py        # Cấu hình & mô hình dữ liệu OCR / Layout
│   ├── ocr_layout.py        # Auto-Rotation, RapidOCR, VietOCR (GPU/CPU batch), phân tích căn lề
│   ├── docx_builder.py      # Dựng Word (.docx) chuẩn A4 / Landscape & Nghị định 30
│   ├── schemas.py           # Pydantic schemas cho cấu trúc văn bản
│   ├── pipeline.py          # Luồng xử lý tuần tự 3 bước
│   ├── page_utils.py        # Xử lý chuỗi chọn trang (--pages)
│   ├── exceptions.py        # Các lớp ngoại lệ tùy biến
│   ├── cli.py               # Giao diện dòng lệnh
│   └── gui.py               # Giao diện đồ họa PyQt5
├── tests/                   # Bộ unit test (unittest)
├── main.py                  # Điểm khởi chạy chung (không tham số -> GUI, có tham số -> CLI)
├── run_gui.py               # Script mở nhanh GUI
├── build_app.py             # Script đóng gói PyInstaller sang .exe
├── Build_EXE.bat            # Chạy nhanh đóng gói .exe trên Windows
├── Chay_Giao_Dien.bat       # Chạy nhanh giao diện GUI trên Windows
├── requirements.txt         # Danh sách thư viện cần thiết
└── pyproject.toml           # Cấu hình package Python
```
