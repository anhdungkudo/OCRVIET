# vn-pdf-to-docx (Kiến trúc Hybrid)

Module Python chuyển đổi file PDF/A dạng scan (văn bản hành chính Việt Nam có dấu mộc đỏ) sang Microsoft Word (`.docx`) chuẩn Nghị định 30/2020/NĐ-CP theo **kiến trúc Hybrid (Offline OCR & Layout + 1-Shot LLM Text Correction)** — xử lý tuần tự, không gửi ảnh lên Gemini, giải quyết triệt để lỗi `429 Quota` và `504 Timeout`.

## Kiến trúc Hybrid 4 bước (Tuần tự, không đa luồng)

1. **Tiền xử lý ảnh & Lọc dấu đỏ (`vn_pdf_to_docx.preprocessor`)**:
   - Render từng trang PDF sang mảng numpy RGB ở độ phân giải **200 DPI** bằng `fitz` (PyMuPDF).
   - Tách **Red Channel** (`extract_red_channel_and_suppress_seal`) làm biến mất con dấu mộc đỏ mà vẫn giữ nguyên nét chữ đen bên dưới.
   - Khử nhiễu (`cv2.fastNlMeansDenoising`) và tăng tương phản bằng `CLAHE` (`denoise_and_enhance_contrast`).

2. **Phát hiện tọa độ căn lề & Đọc chữ tiếng Việt Offline (`vn_pdf_to_docx.ocr_layout`)**:
   - Sử dụng `rapidocr_onnxruntime` để phát hiện các vùng chứa chữ `[x_min, y_min, x_max, y_max]`.
   - Gom các box có tọa độ Y chênh lệch không quá `10px` thành cùng một dòng (`group_boxes_into_lines`).
   - Phân tích tọa độ X (`classify_page_lines`):
     - 2 cụm tách biệt trái/phải qua mốc 45% chiều rộng trang -> đánh dấu `two_column_row` (Quốc hiệu - Tiêu ngữ, Nơi nhận - Chữ ký).
     - Dòng đơn có tâm X nằm trong ±8% giữa trang và không chạm lề -> `alignment = "center"`.
     - Dòng chạm sát hai lề văn bản -> `alignment = "justify"`; nếu lề trái thụt vào `30-80px` (tới `110px` cho 1.27cm ở 200 DPI) -> `first_line_indent = True`.
   - Cắt từng vùng chữ chạy tuần tự qua `vietocr` (`vgg_transformer`, CPU) để đọc văn bản tiếng Việt.

3. **Sửa lỗi chính tả toàn bộ tài liệu trong 1 Request duy nhất (`vn_pdf_to_docx.extractor`)**:
   - Tuyệt đối **không** gửi hình ảnh lên Gemini.
   - Gom toàn bộ text thô của tất cả các trang vào 1 JSON Dictionary phẳng `{"id_1": "...", "id_2": "...", ...}` (chia tối đa `5000` từ/lần gọi nếu tài liệu rất dài).
   - Gọi `google.genai` đúng model chính được chọn (mặc định `gemini-3.8-flash`). Nếu gặp lỗi `429 RESOURCE_EXHAUSTED` hoặc mất mạng, tự động chuyển sang cơ chế Fallback an toàn để xuất ngay văn bản gốc từ VietOCR ra Word (hoặc tích chọn chế độ Offline để bỏ qua bước gọi Gemini).

4. **Xuất file Word chuẩn (`vn_pdf_to_docx.docx_builder`)**:
   - Khổ A4 (`21.0 x 29.7 cm`), lề chuẩn Nghị định 30/2020/NĐ-CP (Trái `3.0 cm`, Phải `1.5 cm`, Trên `2.0 cm`, Dưới `2.0 cm`), font **Times New Roman 13pt**.
   - Các dòng `two_column_row` được gom lại thành bảng 2 cột ẩn viền (`two_column_table`), căn giữa từng cột.
   - Các đoạn văn thường áp dụng đúng `alignment` và `first_line_indent` với nội dung đã sửa sạch lỗi chính tả.

## Cài đặt

```bash
pip install -r requirements.txt
```

Thiết lập API key trong `.env` hoặc biến môi trường:

```bash
set GEMINI_API_KEY=your_api_key_here
```

## Sử dụng

### 1. Giao diện đồ họa (GUI)

- Nhấp đúp `Chay_Giao_Dien.bat` hoặc `Chay_Giao_Dien_An_Console.vbs`, hoặc chạy:
  ```bash
  python run_gui.py
  ```

### 2. Dòng lệnh (CLI)

```bash
# Chuyển đổi toàn bộ tài liệu (mặc định 200 DPI, Hybrid 1-Shot Text)
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx -v

# Chuyển đổi các trang cụ thể
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx --pages 1-3 -v

# Chế độ hoàn toàn Offline (không gọi Gemini)
python -m vn_pdf_to_docx.cli van_ban_scan.pdf -o van_ban_sach.docx --skip-llm -v
```
