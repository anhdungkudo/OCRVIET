"""Vietnamese scanned administrative PDF/A to DOCX converter (Hybrid Architecture)."""

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

try:
    from dotenv import load_dotenv
    if getattr(sys, "frozen", False):
        exe_env = Path(sys.executable).resolve().parent / ".env"
        if exe_env.exists():
            load_dotenv(exe_env, override=True)
        else:
            load_dotenv(override=True)
    else:
        load_dotenv(override=True)
except ImportError:  # pragma: no cover
    pass

from .docx_builder import DocxRenderConfig, build_docx
from .exceptions import (
    DocxBuildError,
    GeminiExtractionError,
    ImagePreprocessError,
    LocalOcrError,
    PdfRenderError,
    PdfToDocxError,
)
from .extractor import (
    DEFAULT_MODEL_ROTATION,
    GeminiExtractorConfig,
    GeminiTextCorrector,
    build_model_rotation,
    split_text_dict_into_batches,
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
    group_boxes_into_lines,
    group_lines_into_segments,
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
    GeminiBodyBlock,
    GeminiFooterTable,
    GeminiHeaderTable,
    GeminiPageExtraction,
    GeminiTitleBlock,
    HeaderTable,
    PageStructure,
    TitleBlock,
    gemini_extraction_to_page_structure,
    merge_page_structures,
)

__all__ = [
    "AdministrativeDocument",
    "BodyBlock",
    "ConversionResult",
    "DEFAULT_MODEL_ROTATION",
    "DetectedLine",
    "DocxBuildError",
    "DocxRenderConfig",
    "FooterTable",
    "GeminiBodyBlock",
    "GeminiExtractionError",
    "GeminiExtractorConfig",
    "GeminiFooterTable",
    "GeminiHeaderTable",
    "GeminiPageExtraction",
    "GeminiTextCorrector",
    "GeminiTitleBlock",
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
    "build_model_rotation",
    "classify_page_lines",
    "collect_document_text_batch",
    "convert_pdf_to_docx",
    "denoise_and_enhance_contrast",
    "encode_image_bytes",
    "extract_red_channel_and_suppress_seal",
    "gemini_extraction_to_page_structure",
    "group_boxes_into_lines",
    "group_lines_into_segments",
    "merge_page_structures",
    "preprocess_page_image",
    "preprocess_pdf",
    "render_pdf_to_images",
    "split_text_dict_into_batches",
]
