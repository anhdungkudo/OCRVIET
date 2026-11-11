"""Custom exception hierarchy for Vietnamese PDF-to-DOCX conversion pipeline."""


class PdfToDocxError(Exception):
    """Base exception for all errors in the vn_pdf_to_docx pipeline."""


class PdfRenderError(PdfToDocxError):
    """Raised when a PDF file cannot be opened or rendered to page images."""


class ImagePreprocessError(PdfToDocxError):
    """Raised when OpenCV image preprocessing or encoding fails."""


class LocalOcrError(PdfToDocxError):
    """Raised when offline RapidOCR / VietOCR detection, recognition, or layout fails."""


class DocxBuildError(PdfToDocxError):
    """Raised when constructing or saving the target Microsoft Word (.docx) fails."""
