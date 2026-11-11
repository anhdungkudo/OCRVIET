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
)
from .preprocessor import PreprocessedPage, PreprocessConfig, encode_image_bytes, preprocess_pdf
from .schemas import AdministrativeDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversionResult:
    """Artifacts returned from an end-to-end offline PDF-to-DOCX conversion."""

    output_docx: Path
    document: AdministrativeDocument
    preprocessed_pages: list[PreprocessedPage]
    json_path: Optional[Path] = None
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
    save_debug_images_dir: Optional[Union[str, Path]] = None,
    save_json_path: Optional[Union[str, Path]] = None,
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

    if save_debug_images_dir is not None:
        debug_dir = Path(save_debug_images_dir)
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise PdfToDocxError(
                f"Failed to create debug images directory '{debug_dir}': {exc}"
            ) from exc
        for page in preprocessed_pages:
            out_img = debug_dir / f"page_{page.page_number:03d}_preprocessed.png"
            try:
                if page.mime_type == "image/png":
                    out_img.write_bytes(page.encoded_bytes)
                else:
                    png_bytes, _ = encode_image_bytes(page.image, image_format="png")
                    out_img.write_bytes(png_bytes)
            except PdfToDocxError:
                raise
            except Exception as exc:
                raise PdfToDocxError(f"Failed to write debug image '{out_img}': {exc}") from exc

    if progress_callback is not None:
        progress_callback("preprocessed", total_pages, total_pages)

    # Step 2: Offline RapidOCR + X/Y Layout + VietOCR (sequential CPU)
    logger.info(
        "Step 2/3: Running offline RapidOCR + layout + VietOCR (%s) across %d page(s)...",
        ocr_cfg.vietocr_config_name,
        total_pages,
    )

    def _on_ocr_page(cur: int, tot: int) -> None:
        if detailed_progress_callback is not None:
            detailed_progress_callback(
                "ocr",
                cur,
                tot,
                f"Đang quét bố cục RapidOCR & đọc chữ VietOCR trang {cur}/{tot}...",
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

    written_json_path: Optional[Path] = None
    if save_json_path is not None:
        written_json_path = Path(save_json_path)
        try:
            if written_json_path.parent:
                written_json_path.parent.mkdir(parents=True, exist_ok=True)
            written_json_path.write_text(
                extracted_doc.model_dump_json(indent=2),
                encoding="utf-8",
            )
            written_json_path = written_json_path.resolve()
        except Exception as exc:
            raise PdfToDocxError(
                f"Failed to save structured JSON '{written_json_path}': {exc}"
            ) from exc

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
        json_path=written_json_path,
        page_layouts=page_layouts,
        raw_text_batch=raw_text_batch,
    )
