
# vn-pdf-to-docx (100% Offline Engine)

Module Python chuyển đổi file PDF/A dạng scan (văn bản hành chính Việt Nam có dấu mộc đỏ) sang Microsoft Word (`.docx`) chuẩn Nghị định 30/2020/NĐ-CP hoàn toàn **Offline trên máy** — không cần API Key, không gọi AI đám mây, không lỗi `429 Quota` hay `504 Timeout`.


## Kiến trúc xử lý tuần tự 3 bước (Không đa luồng)

1. **Tiền xử lý ảnh & Lọc dấu đỏ (`vn_pdf_to_docx.preprocessor`)**:
   - Render từng trang PDF sang mảng numpy RGB ở độ phân giải **200 DPI** bằng `fitz` (PyMuPDF).
   - Tách **Red Channel** (`extract_red_channel_and_suppress_seal`) làm biến mất con dấu mộc đỏ mà vẫn giữ nguyên nét chữ đen bên dưới.
   - Khử nhiễu (`cv2.fastNlMeansDenoising`) và tăng tương phản bằng `CLAHE` (`denoise_and_enhance_contrast`).

3. **Tự động xoay trang ngang (`Auto-Rotation`), phát hiện bảng & Đọc chữ tiếng Việt Offline (`vn_pdf_to_docx.ocr_layout`, `vn_pdf_to_docx.table_detector`)**:
   - **Auto-Rotation (90° / 270°)**: Kiểm tra kích thước trang (`width > height`) hoặc tỷ lệ bounding box dựng đứng (`box_height > box_width * 2.0` vượt quá `60%`) trước khi trích xuất bảng (`detect_and_fix_orientation`). Tự động đánh dấu `is_landscape = True` và thử xoay `90° CW` / `90° CCW` bằng OpenCV kết hợp kiểm tra tiêu đề đầu trang để xoay thẳng chuẩn chiều đọc từ trái sang phải.
   - Phát hiện khung bảng và tọa độ ô `(r, c)` bằng **OpenCV Morphological Grid Line Detection** (`cv2.adaptiveThreshold` + kernel `(40, 1)` ngang / `(1, 40)` dọc, hỗ trợ ảnh scan hơi nghiêng, hàng tiêu đề mất viền trên và gộp ô ngang `colspan`) trước khi gom dòng (`detect_table_grids`, `split_boxes_by_grid_dividers`, `partition_boxes_by_tables`). Khi viền bảng bị mờ/đứt nét, tự động kích hoạt **Column Alignment Projection Fallback** theo từ khóa tiêu đề `"STT"` + `"Danh mục"`/`"Mức giá"` (`extract_tables_via_column_projection`).
   - Sử dụng `rapidocr_onnxruntime` để phát hiện các vùng chứa chữ `[x_min, y_min, x_max, y_max]`, ánh xạ các box trong khung bảng đúng vào ô `(r, c)` (không nhận nhầm sang `two_column_row` và nối các dòng xuống hàng vào chung một ô).
   - Gom các box ngoài bảng có tọa độ Y chênh lệch không quá `10px` thành cùng một dòng (`group_boxes_into_lines`) và phân tích tọa độ X (`classify_page_lines`) cho bảng 2 cột ẩn viền, căn giữa, căn đều và thụt lề đầu dòng.
   - Cắt từng vùng chữ chạy tuần tự qua `vietocr` (`vgg_transformer`, CPU) để đọc văn bản tiếng Việt.

6. **Xuất file Word chuẩn với Section Ngang/Dọc động (`vn_pdf_to_docx.docx_builder`)**:
   - Trang dọc (Portrait): Khổ A4 (`21.0 x 29.7 cm`), lề chuẩn Nghị định 30/2020/NĐ-CP (Trái `3.0 cm`, Phải `1.5 cm`, Trên `2.0 cm`, Dưới `2.0 cm`), chiều rộng khả dụng `16.5 cm`, font **Times New Roman 13pt**.
   - Trang ngang tự động (`is_landscape == True`): Tự động chèn Section mới (`WD_SECTION.NEW_PAGE`, `WD_ORIENT.LANDSCAPE`, `29.7 x 21.0 cm`), lề Trên `3.0 cm`, Dưới `1.5 cm`, Trái `2.0 cm`, Phải `2.0 cm` (khả dụng `25.7 cm`), và tự động chuyển lại về `WD_ORIENT.PORTRAIT` khi sang trang dọc tiếp theo.
   - Các dòng `two_column_row` được gom lại thành bảng 2 cột ẩn viền (`two_column_table`), căn giữa từng cột.
   - Bảng dữ liệu (`table`) được dựng bằng `Table Grid` (viền đen 4 cạnh, hỗ trợ gộp ô ngang `colspan`): trên trang dọc chuẩn `[1.2 cm, 11.5 cm, 3.0 cm]`, trên trang ngang tự động chia tỷ lệ từng cột theo pixel thực tế nhân `25.7 cm`; căn giữa tiêu đề & cột STT/Mã, căn phải cột Mức giá/Đơn giá, căn trái cột Tên dịch vụ & Ghi chú.
   - Tự động lưu sang tên kế tiếp (`... (1).docx`) nếu tệp Word đích đang mở.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Giao diện đồ họa (GUI)

- Mở thẳng `dist\ChuyenDoiVanBanPDF\ChuyenDoiVanBanPDF.exe`, nhấp đúp `Chay_Giao_Dien.bat` hoặc chạy:
  ```bash
  python run_gui.py
  ```

### 2. Dòng lệnh (CLI)

```bash
# Chuyển đổi toàn bộ tài liệu (mặc định 200 DPI, 100% Offline)
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx -v

# Chuyển đổi các trang cụ thể
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx --pages 1-3 -v
```
