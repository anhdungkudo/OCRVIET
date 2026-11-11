"""Command-line interface for vn_pdf_to_docx."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .docx_builder import DocxRenderConfig
from .exceptions import PdfToDocxError
from .ocr_layout import OcrLayoutConfig
from .page_utils import parse_page_ranges
from .pipeline import convert_pdf_to_docx
from .preprocessor import PreprocessConfig


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vn-pdf-to-docx",
        description=(
            "100% Offline converter for scanned Vietnamese administrative PDF/A documents: "
            "OpenCV red-seal removal -> Auto-Rotation -> RapidOCR + VietOCR -> DOCX."
        ),
    )
    parser.add_argument("input_pdf", type=Path, help="Path to the input scanned PDF/A file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output .docx file (defaults to <input_stem>.docx).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rasterization DPI for PyMuPDF rendering (default: 200).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the extracted structured JSON payload.",
    )
    parser.add_argument(
        "--debug-images-dir",
        type=Path,
        default=None,
        help="Optional directory to dump preprocessed 200-DPI grayscale PNG pages.",
    )
    parser.add_argument(
        "--footer-left-align",
        choices=("center", "left"),
        default=None,
        help="Horizontal alignment for the left 'Nơi nhận' column in 2-column footer (default: center).",
    )
    parser.add_argument(
        "--center-footer-left",
        action="store_true",
        help="Explicitly center-align the left 'Nơi nhận' column in 2-column footer (default behavior).",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Specific page(s) to convert, e.g. '1', '1-5', '1,3,5' (default: all pages).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose INFO logging.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_docx = args.output or args.input_pdf.with_suffix(".docx")

    if args.footer_left_align is not None:
        footer_left_align = args.footer_left_align
    elif args.center_footer_left:
        footer_left_align = "center"
    else:
        footer_left_align = "center"

    try:
        page_numbers = parse_page_ranges(args.pages) if args.pages else None
        result = convert_pdf_to_docx(
            pdf_path=args.input_pdf,
            output_docx_path=output_docx,
            preprocess_config=PreprocessConfig(dpi=args.dpi),
            ocr_config=OcrLayoutConfig(dpi=args.dpi),
            docx_config=DocxRenderConfig(footer_left_alignment=footer_left_align),
            page_numbers=page_numbers,
            save_debug_images_dir=args.debug_images_dir,
            save_json_path=args.save_json,
        )
        print(f"Converted successfully: {result.output_docx}")
        return 0
    except (PdfToDocxError, ValidationError, ValueError) as exc:
        print(f"Conversion error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
