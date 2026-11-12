# vn-pdf-to-docx (100% Offline Engine)

Công cụ chuyển đổi văn bản hành chính Việt Nam dạng PDF/A scan (có dấu mộc đỏ, bảng biểu phức tạp, trang xoay ngang 90°/270°) sang Microsoft Word (`.docx`) chuẩn thể thức **Nghị định 30/2020/NĐ-CP** — hoạt động **100% Offline trên máy** (không cần API Key, không dùng `.env`, không gọi AI đám mây).

---

## Kiến trúc xử lý tuần tự 3 bước

1. **Tiền xử lý ảnh & Lọc dấu đỏ (`vn_pdf_to_docx.preprocessor`)**:
   - Kết xuất từng trang PDF sang ảnh RGB bằng `PyMuPDF` (`fitz`) ở độ phân giải mặc định **200 DPI** (hỗ trợ `72–600 DPI`).
   - Tách **Red Channel** (`extract_red_channel_and_suppress_seal`) để triệt tiêu con dấu mộc đỏ nhưng vẫn giữ nguyên nét chữ đen bên dưới.
   - Khử nhiễu (`cv2.fastNlMeansDenoising`) và tăng độ tương phản cục bộ bằng `CLAHE` (`denoise_and_enhance_contrast`).

2. **Tự động xoay trang ngang, Nhận diện bảng & OCR Tiếng Việt Offline (`vn_pdf_to_docx.ocr_layout`, `vn_pdf_to_docx.table_detector`)**:
   - **Auto-Rotation (90° / 270°)**: Kiểm tra kích thước trang (`width > height`) hoặc tỷ lệ khung chữ dựng đứng (`box_height > box_width * 2.0` vượt quá `60%`) qua `detect_and_fix_orientation`, tự xoay ảnh ngang về chiều đọc chuẩn và đánh dấu `is_landscape = True`.
   - **Phát hiện khung bảng & ô `(r, c)` bằng OpenCV**: Dùng Morphological Grid Line Detection (`cv2.adaptiveThreshold` kết hợp kernel `(40, 1)` ngang và `(1, 40)` dọc trong `detect_table_grids`), hỗ trợ ảnh scan hơi nghiêng, hàng tiêu đề mất viền trên và gộp ô ngang (`colspan`). Khi đường viền bảng bị đứt hoặc quá mờ, tự động chuyển sang chế độ **Column Alignment Projection Fallback** (`extract_tables_via_column_projection`).
   - **Phát hiện vùng chữ & phân loại bố cục (`rapidocr_onnxruntime` - CPU)**:
     - Phát hiện bounding boxes `[x_min, y_min, x_max, y_max]`.
     - Ánh xạ chính xác từng box trong khung bảng vào đúng ô `(r, c)` và nối các dòng xuống hàng trong cùng một ô.
     - Gom các box ngoài bảng có chênh lệch tọa độ Y không quá `10px` thành cùng một dòng (`group_boxes_into_lines`), sau đó phân loại (`classify_page_lines`): bảng 2 cột không viền (`two_column_row` cho Quốc hiệu - Tiêu ngữ / Nơi nhận - Chữ ký), căn giữa, căn đều và thụt lề đầu dòng.
   - **Nhận diện chữ Tiếng Việt có dấu (`vietocr`)**: Cắt vùng chữ và nhận diện theo lô (`batch_size=64`) qua mô hình `vgg_transformer` trên **GPU CUDA (FP32 thuần GPU)** (tự động dùng GPU khi PyTorch CUDA khả dụng, fallback sang CPU nếu không có GPU).

3. **Xuất file Word chuẩn với Section Dọc/Ngang động (`vn_pdf_to_docx.docx_builder`)**:
   - **Trang dọc (`Portrait`)**: Khổ A4 (`21.0 × 29.7 cm`), lề chuẩn Nghị định 30/2020/NĐ-CP (Trái `3.0 cm`, Phải `1.5 cm`, Trên `2.0 cm`, Dưới `2.0 cm`, vùng nội dung `16.5 cm`), font **Times New Roman 13pt**.
   - **Trang ngang tự động (`Landscape`)**: Khi trang có `is_landscape == True`, tự động chèn Section mới (`WD_SECTION.NEW_PAGE`, `WD_ORIENT.LANDSCAPE`, `29.7 × 21.0 cm`, lề Trên `3.0 cm`, Dưới `1.5 cm`, Trái `2.0 cm`, Phải `2.0 cm`, vùng nội dung `25.7 cm`) và tự động chuyển lại `Portrait` ở trang dọc kế tiếp.
   - **Bảng biểu**: Gom cụm `two_column_row` thành bảng 2 cột ẩn viền (`two_column_table`); dựng bảng dữ liệu theo chuẩn `Table Grid` (viền 4 cạnh, hỗ trợ `colspan`, tự chia tỷ lệ độ rộng cột theo pixel trên trang ngang).
   - **Chống lỗi khóa file trên Windows**: Nếu file `.docx` đích đang mở trong Word gây `PermissionError`, công cụ tự động lưu sang tên mới kế tiếp như `ten_file (1).docx`, `ten_file (2).docx`.

---

## Yêu cầu & Cài đặt

- **Hệ điều hành**: Windows (hỗ trợ Python `>= 3.10`).
- **Phần cứng**: Tự động dùng **GPU NVIDIA (CUDA)** cho `VietOCR` nếu đã cài PyTorch CUDA, hoặc chạy **CPU** bình thường (không yêu cầu cấu hình `.env` hay API Key).

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Hoặc cài đặt như một package Python:

```bash
pip install -e .
```

> **Lưu ý về thư viện**: Ứng dụng đã loại bỏ hoàn toàn AI đám mây (`google-genai`, `python-dotenv`), chỉ giữ lại các thư viện xử lý PDF/ảnh/Word/GUI (`pymupdf`, `opencv-python`, `numpy`, `Pillow`, `pydantic`, `python-docx`, `PyQt5`) và mô hình OCR chạy cục bộ (`rapidocr-onnxruntime`, `vietocr`, `torch`, `torchvision`).

---

## Cách sử dụng

### 1. Chạy file thực thi đóng gói sẵn (`.exe`)

Nếu bạn đã đóng gói bằng PyInstaller, chỉ cần mở trực tiếp:

```text
dist\ChuyenDoiVanBanPDF\ChuyenDoiVanBanPDF.exe
```

### 2. Giao diện đồ họa (GUI) từ mã nguồn

Nhấp đúp vào `Chay_Giao_Dien.bat` (hoặc `Chay_Giao_Dien_An_Console.vbs`), hoặc chạy qua terminal:

```bash
python run_gui.py
# Hoặc chạy main.py không kèm tham số:
python main.py
```

Các tùy chọn chính trên giao diện:

- **Phạm vi trang**: Tất cả hoặc chọn trang cụ thể (ví dụ `1`, `1-5`, `1,3,5`).
- **Độ nét (DPI)**: `72–600 DPI` (mặc định `200 DPI`).
- **Căn lề Nơi nhận**: Căn giữa (`center`, mặc định) hoặc căn trái theo Nghị định 30 (`left`).
- **Khử dấu đỏ con dấu tròn**: Bật/tắt lọc kênh Red + NLM Denoise (mặc định bật).
- **Tự động mở file Word khi hoàn tất**.

### 3. Dòng lệnh (CLI)

```bash
# Chuyển đổi toàn bộ file PDF (mặc định 200 DPI, xuất ra <ten_file>.docx)
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx -v

# Chuyển đổi các trang cụ thể
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx --pages 1-5 -v

# Hoặc gọi qua main.py kèm tham số
python main.py van_ban_scan.pdf -o van_ban_sach.docx --dpi 200 --footer-left-align center
```

Các tham số hỗ trợ:

- `-o`, `--output`: Đường dẫn file `.docx` đầu ra (mặc định trùng tên file PDF đầu vào).
- `--dpi`: Độ phân giải render PDF (mặc định `200`).
- `--pages`: Danh sách hoặc khoảng trang cần xử lý (ví dụ `1`, `1-5`, `1,3,5`).
- `--device {auto,cpu,cuda}`: Thiết bị chạy `VietOCR` (`auto` mặc định tự dùng GPU CUDA nếu có).
- `--batch-size`: Số vùng chữ tối đa mỗi batch khi chạy `VietOCR` (mặc định `64`).
- `--footer-left-align {center,left}`: Căn lề cột trái phần `Nơi nhận` trong bảng 2 cột cuối trang (mặc định `center`).
- `-v`, `--verbose`: In log chi tiết quá trình xử lý.

---

## Đóng gói sang file `.EXE` (Windows)

Yêu cầu cài thêm `pyinstaller` (`pip install pyinstaller`), sau đó chạy `Build_EXE.bat` hoặc dùng lệnh:

```bash
# Đóng gói dạng thư mục (khởi động nhanh, ổn định nhất - khuyên dùng)
python build_app.py --clean

# Đóng gói thành 1 file .exe duy nhất
python build_app.py --onefile --clean
```

Kết quả đầu ra nằm trong thư mục `dist\`.

---

## Kiểm thử (Tests)

Chạy toàn bộ bộ unit test bằng lệnh:

```bash
python -m unittest discover -s tests
```

---

## Cấu trúc dự án

```text
OCRVIET/
├── vn_pdf_to_docx/          # Mã nguồn lõi
│   ├── preprocessor.py      # Render PDF 200 DPI, tách kênh Red khử dấu đỏ, CLAHE
│   ├── table_detector.py    # Nhận diện khung bảng & cột bằng OpenCV Morphological Grid
│   ├── ocr_models.py        # Mô hình cấu hình và kết quả OCR / Layout
│   ├── ocr_layout.py        # Auto-Rotation 90°/270°, RapidOCR, VietOCR, phân tích căn lề
│   ├── docx_builder.py      # Dựng Word (.docx) chuẩn A4/Landscape & Nghị định 30
│   ├── schemas.py           # Pydantic schemas cho cấu trúc văn bản đầu ra
│   ├── pipeline.py          # Luồng điều phối tuần tự 3 bước
│   ├── page_utils.py        # Phân tích chuỗi khoảng trang (--pages)
│   ├── exceptions.py        # Các lớp lỗi tùy biến
│   ├── cli.py               # Giao diện dòng lệnh
│   └── gui.py               # Giao diện đồ họa PyQt5
├── tests/                   # Bộ kiểm thử đơn vị (unittest)
├── main.py                  # Điểm vào chung (tự mở GUI nếu không truyền tham số CLI)
├── run_gui.py               # Script khởi chạy GUI
├── build_app.py             # Script đóng gói PyInstaller sang .exe
├── Build_EXE.bat            # Script Windows build nhanh .exe
├── Chay_Giao_Dien.bat       # Script Windows mở nhanh GUI
├── requirements.txt         # Danh sách thư viện chạy tối thiểu
└── pyproject.toml           # Cấu hình package Python
```
