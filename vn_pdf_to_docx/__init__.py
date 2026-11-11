"""Vietnamese scanned administrative PDF/A to DOCX converter (100% Offline Engine)."""

import os
import sys
from pathlib import Path


def _preload_win_msvc_runtime() -> None:
    """Load up-to-date Windows MSVC runtime before PyQt5 prepends older Qt5\\bin DLLs."""
    if sys.platform != "win32":
        return
    import ctypes

    search_dirs: list[Path] = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        search_dirs.append(Path(sys._MEIPASS))
    search_dirs.append(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")

    for dll_name in (
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "vcruntime140_threads.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
    ):
        for directory in search_dirs:
            dll_path = directory / dll_name
            if dll_path.is_file():
                try:
                    ctypes.WinDLL(str(dll_path))
                    break
                except OSError:
                    continue


_preload_win_msvc_runtime()

from .docx_builder import DocxRenderConfig, build_docx
from .exceptions import (
    DocxBuildError,
    ImagePreprocessError,
    LocalOcrError,
    PdfRenderError,
    PdfToDocxError,
)
from .ocr_layout import (
    DetectedLine,
    LayoutSegment,
    LocalOcrLayoutAnalyzer,
    OcrLayoutConfig,
    PageLayoutResult,
    RapidOcrDetector,
    TextBox,
    VietOcrRecognizer,
    build_administrative_document_from_layouts,
    classify_page_lines,
    collect_document_text_batch,
    detect_and_fix_orientation,
    detect_vertical_text_ratio,
    group_boxes_into_lines,
    group_lines_into_segments,
    rotate_boxes_90,
)
from .table_detector import (
    DetectedTableGrid,
    build_table_data_from_grid,
    detect_table_grids,
    extract_tables_via_column_projection,
    filter_clipped_edge_boxes,
    partition_boxes_by_tables,
    split_boxes_by_grid_dividers,
)
from .pipeline import ConversionResult, convert_pdf_to_docx
from .preprocessor import (
    PreprocessConfig,
    PreprocessedPage,
    denoise_and_enhance_contrast,
    encode_image_bytes,
    extract_red_channel_and_suppress_seal,
    preprocess_page_image,
    preprocess_pdf,
    render_pdf_to_images,
)
from .schemas import (
    AdministrativeDocument,
    BodyBlock,
    FooterTable,
    HeaderTable,
    PageStructure,
    TitleBlock,
    merge_page_structures,
)

__all__ = [
    "AdministrativeDocument",
    "BodyBlock",
    "ConversionResult",
    "DetectedLine",
    "DocxBuildError",
    "DocxRenderConfig",
    "FooterTable",
    "HeaderTable",
    "ImagePreprocessError",
    "LayoutSegment",
    "LocalOcrError",
    "LocalOcrLayoutAnalyzer",
    "OcrLayoutConfig",
    "PageLayoutResult",
    "PageStructure",
    "PdfRenderError",
    "PdfToDocxError",
    "PreprocessConfig",
    "PreprocessedPage",
    "RapidOcrDetector",
    "TextBox",
    "TitleBlock",
    "VietOcrRecognizer",
    "build_administrative_document_from_layouts",
    "build_docx",
    "classify_page_lines",
    "collect_document_text_batch",
    "convert_pdf_to_docx",
    "denoise_and_enhance_contrast",
    "encode_image_bytes",
    "extract_red_channel_and_suppress_seal",
    "group_boxes_into_lines",
    "group_lines_into_segments",
    "merge_page_structures",
    "preprocess_page_image",
    "preprocess_pdf",
    "render_pdf_to_images",
    "DetectedTableGrid",
    "build_table_data_from_grid",
    "detect_and_fix_orientation",
    "detect_table_grids",
    "detect_vertical_text_ratio",
    "extract_tables_via_column_projection",
    "filter_clipped_edge_boxes",
    "partition_boxes_by_tables",
    "rotate_boxes_90",
    "split_boxes_by_grid_dividers",
]
