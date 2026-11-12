"""End-to-end 100% Offline pipeline: PyMuPDF+OpenCV -> Auto-Rotation/Table Grid + RapidOCR+VietOCR -> DOCX."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from .docx_builder import DocxRenderConfig, build_docx
from .exceptions import PdfToDocxError
from .ocr_layout import (
    LocalOcrLayoutAnalyzer,
    OcrLayoutConfig,
    PageLayoutResult,
    build_administrative_document_from_layouts,
    collect_document_text_batch,
    describe_vietocr_device,
    resolve_vietocr_device,
)
from .preprocessor import PreprocessedPage, PreprocessConfig, preprocess_pdf
from .schemas import AdministrativeDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversionResult:
    """Artifacts returned from an end-to-end offline PDF-to-DOCX conversion."""

    output_docx: Path
    document: AdministrativeDocument
    preprocessed_pages: list[PreprocessedPage]
    page_layouts: list[PageLayoutResult] = field(default_factory=list)
    raw_text_batch: dict[str, str] = field(default_factory=dict)


def convert_pdf_to_docx(
    pdf_path: Union[str, Path],
    output_docx_path: Union[str, Path],
    *,
    preprocess_config: Optional[PreprocessConfig] = None,
    ocr_config: Optional[OcrLayoutConfig] = None,
    docx_config: Optional[DocxRenderConfig] = None,
    page_numbers: Optional[Iterable[int]] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    detailed_progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> ConversionResult:
    """Convert a scanned Vietnamese administrative PDF/A to DOCX 100% offline.

    Sequential Steps (no multithreading, no cloud API calls):
        1. Rasterize PDF pages at 200 DPI via PyMuPDF (`fitz`), suppress red seals via the
           Red channel, and sharpen text via `cv2.fastNlMeansDenoising` + `CLAHE`.
        2. Auto-orient rotated 90°/270° landscape scans, detect OpenCV table grids &
           `two_column_row` / alignment rules with `rapidocr_onnxruntime`, and recognize
           Vietnamese text sequentially with `vietocr` (`vgg_transformer`, CPU).
        3. Reconstruct an A4 Word (`.docx`) file with Nghị định 30/2020/NĐ-CP margins,
           dynamic Portrait/Landscape sections, centered 2-column borderless tables, and
           `Table Grid` data tables.
    """
    src_path = Path(pdf_path)
    out_docx = Path(output_docx_path)
    prep_cfg = preprocess_config or PreprocessConfig()
    ocr_cfg = ocr_config or OcrLayoutConfig(dpi=prep_cfg.dpi)
    doc_cfg = docx_config or DocxRenderConfig()

    logger.info("Step 1/3: Rendering & preprocessing '%s' at %d DPI...", src_path, prep_cfg.dpi)
    if detailed_progress_callback is not None:
        detailed_progress_callback(
            "render", 0, 0, f"Đang nạp PDF & chuẩn bị tiền xử lý ({prep_cfg.dpi} DPI)..."
        )

    is_suppressing = prep_cfg.red_suppression_strength > 0.0 or prep_cfg.nlm_h > 0.0

    def _on_prep_page(cur: int, tot: int) -> None:
        if detailed_progress_callback is not None:
            stage_msg = (
                f"Đang khử dấu đỏ & tăng tương phản CLAHE trang {cur}/{tot}..."
                if is_suppressing
                else f"Đang kết xuất ảnh {prep_cfg.dpi} DPI trang {cur}/{tot}..."
            )
            detailed_progress_callback("preprocess", cur, tot, stage_msg)

    preprocessed_pages = preprocess_pdf(
        src_path,
        config=prep_cfg,
        page_numbers=page_numbers,
        page_callback=_on_prep_page,
    )
    total_pages = len(preprocessed_pages)

    if progress_callback is not None:
        progress_callback("preprocessed", total_pages, total_pages)

    # Step 2: Offline RapidOCR + X/Y Layout + VietOCR (GPU CUDA / CPU, batched)
    resolved_device = resolve_vietocr_device(ocr_cfg.vietocr_device)
    device_label = describe_vietocr_device(resolved_device)
    logger.info(
        "Step 2/3: Running offline RapidOCR + layout + VietOCR (%s, device=%s, batch=%d) across %d page(s)...",
        ocr_cfg.vietocr_config_name,
        device_label,
        ocr_cfg.vietocr_batch_size,
        total_pages,
    )

    def _on_ocr_page(cur: int, tot: int) -> None:
        if detailed_progress_callback is not None:
            detailed_progress_callback(
                "ocr",
                cur,
                tot,
                f"Đang quét bố cục & đọc chữ VietOCR [{device_label}, batch={ocr_cfg.vietocr_batch_size}] trang {cur}/{tot}...",
            )

    analyzer = LocalOcrLayoutAnalyzer(config=ocr_cfg)
    page_layouts = analyzer.process_document(
        preprocessed_pages,
        page_callback=_on_ocr_page,
    )

    if progress_callback is not None:
        progress_callback("ocr", total_pages, total_pages)

    raw_text_batch, _ = collect_document_text_batch(page_layouts)
    extracted_doc = build_administrative_document_from_layouts(
        page_layouts,
        source_pdf=str(src_path),
    )

    if progress_callback is not None:
        progress_callback("extracted", total_pages, total_pages)

    # Step 3: Build Word (.docx)
    logger.info("Step 3/3: Building formatted Word document '%s'...", out_docx)
    if detailed_progress_callback is not None:
        detailed_progress_callback(
            "docx",
            total_pages,
            total_pages,
            "Đang tạo file Word (.docx) chuẩn thể thức Nghị định 30/2020/NĐ-CP...",
        )

    resolved_docx = build_docx(extracted_doc, out_docx, config=doc_cfg)
    if resolved_docx != out_docx.resolve():
        lock_warn = (
            f"[CẢNH BÁO] Tệp '{out_docx.name}' đang được mở trong Word hoặc bị khóa ghi — "
            f"đã tự động lưu kết quả sang tệp mới: '{resolved_docx.name}'"
        )
        logger.warning("%s", lock_warn)
        if detailed_progress_callback is not None:
            detailed_progress_callback("docx_warning", total_pages, total_pages, lock_warn)

    if progress_callback is not None:
        progress_callback("completed", total_pages, total_pages)

    if detailed_progress_callback is not None:
        detailed_progress_callback(
            "completed",
            total_pages,
            total_pages,
            f"Chuyển đổi hoàn tất thành công! Đã lưu: {resolved_docx.name}",
        )

    return ConversionResult(
        output_docx=resolved_docx,
        document=extracted_doc,
        preprocessed_pages=preprocessed_pages,
        page_layouts=page_layouts,
        raw_text_batch=raw_text_batch,
    )
