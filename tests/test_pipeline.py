"""Tests for end-to-end Hybrid pipeline: PyMuPDF+OpenCV -> RapidOCR+VietOCR -> 1-Shot Text -> DOCX."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

import vn_pdf_to_docx.pipeline as pipeline_mod
from vn_pdf_to_docx.exceptions import PdfToDocxError
from vn_pdf_to_docx.ocr_layout import LayoutSegment, PageLayoutResult
from vn_pdf_to_docx.pipeline import convert_pdf_to_docx
from vn_pdf_to_docx.preprocessor import PreprocessConfig


def _write_two_page_pdf(path: Path) -> None:
    doc = fitz.open()
    for i in range(1, 3):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 120), f"TRANG {i} - UBND THANH PHO HA NOI", fontsize=14, color=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def _sample_page_layouts() -> list[PageLayoutResult]:
    return [
        PageLayoutResult(
            page_number=1,
            page_width=1653,
            page_height=2339,
            body_left_margin=236.0,
            body_right_margin=1520.0,
            segments=[
                LayoutSegment(
                    kind="two_column_table",
                    page_number=1,
                    left_lines=["SO NOI VU"],
                    right_lines=["CONG HOA XA HOI CHU NGHIA VIET NAM"],
                ),
                LayoutSegment(
                    kind="paragraph",
                    page_number=1,
                    alignment="center",
                    first_line_indent=False,
                    text="QUYET DINH",
                ),
            ],
        ),
        PageLayoutResult(
            page_number=2,
            page_width=1653,
            page_height=2339,
            body_left_margin=236.0,
            body_right_margin=1520.0,
            segments=[
                LayoutSegment(
                    kind="paragraph",
                    page_number=2,
                    alignment="justify",
                    first_line_indent=True,
                    text="Điểu 1. Quy định chung ngày O5/O3/2O2O.",
                ),
                LayoutSegment(
                    kind="two_column_table",
                    page_number=2,
                    left_lines=["Noi nhan:", "- Luu: VT."],
                    right_lines=["GIAM DOC", "", "Nguyen Van A"],
                ),
            ],
        ),
    ]


class TestPipeline(unittest.TestCase):
    """Verify 3-step offline pipeline execution, error wrapping, and artifacts."""

    def test_save_debug_images_dir_existing_file_raises_pdf_to_docx_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            blocking_file = tmp_dir / "not_a_directory"
            blocking_file.write_text("occupied", encoding="utf-8")

            with self.assertRaises(PdfToDocxError) as ctx:
                convert_pdf_to_docx(
                    pdf_path,
                    tmp_dir / "out.docx",
                    preprocess_config=PreprocessConfig(dpi=100),
                    save_debug_images_dir=blocking_file,
                )
            self.assertIn("debug images directory", str(ctx.exception))

    def test_save_debug_images_dir_permission_error_raises_pdf_to_docx_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            with patch.object(Path, "mkdir", side_effect=PermissionError("access denied")):
                with self.assertRaises(PdfToDocxError) as ctx:
                    convert_pdf_to_docx(
                        pdf_path,
                        tmp_dir / "out.docx",
                        preprocess_config=PreprocessConfig(dpi=100),
                        save_debug_images_dir=tmp_dir / "debug_out",
                    )
            self.assertIn("debug images directory", str(ctx.exception))

    def test_offline_pipeline_processes_all_pages_and_builds_real_docx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            debug_dir = tmp_dir / "debug_imgs"
            json_path = tmp_dir / "nested" / "extracted.json"
            out_docx = tmp_dir / "result.docx"

            fake_analyzer = MagicMock()
            fake_analyzer.process_document.return_value = _sample_page_layouts()

            events: list[tuple[str, int, int]] = []

            with patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer):
                result = convert_pdf_to_docx(
                    pdf_path,
                    out_docx,
                    preprocess_config=PreprocessConfig(dpi=100, image_format="jpeg"),
                    save_debug_images_dir=debug_dir,
                    save_json_path=json_path,
                    progress_callback=lambda stage, cur, tot: events.append((stage, cur, tot)),
                )

            # Debug PNGs written for both pages
            self.assertTrue((debug_dir / "page_001_preprocessed.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((debug_dir / "page_002_preprocessed.png").exists())

            # JSON persisted with VietOCR texts directly
            self.assertTrue(json_path.exists())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["total_pages"], 2)
            self.assertEqual(loaded["body_blocks"][2]["content"], "Điểu 1. Quy định chung ngày O5/O3/2O2O.")

            # Real DOCX file built with invisible 2-col tables and aligned paragraphs
            self.assertTrue(result.output_docx.exists())
            doc = Document(str(result.output_docx))
            self.assertEqual(len(doc.tables), 2)
            for tbl in doc.tables:
                borders = tbl._tbl.tblPr.find(qn("w:tblBorders"))
                self.assertEqual(borders.find(qn("w:top")).get(qn("w:val")), "none")
                self.assertEqual(tbl.cell(0, 0).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)
                self.assertEqual(tbl.cell(0, 1).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)

            self.assertEqual(
                events,
                [("preprocessed", 2, 2), ("ocr", 2, 2), ("extracted", 2, 2), ("completed", 2, 2)],
            )

    def test_save_json_path_error_raises_pdf_to_docx_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            occupied_dir = tmp_dir / "occupied_dir"
            occupied_dir.mkdir()

            fake_analyzer = MagicMock()
            fake_analyzer.process_document.return_value = _sample_page_layouts()

            with patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer):
                with self.assertRaises(PdfToDocxError) as ctx:
                    convert_pdf_to_docx(
                        pdf_path,
                        tmp_dir / "out.docx",
                        preprocess_config=PreprocessConfig(dpi=100),
                        save_json_path=occupied_dir,
                    )
            self.assertIn("structured JSON", str(ctx.exception))

    def test_pipeline_emits_warning_when_docx_saved_to_alternate_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            orig_out = tmp_dir / "Phan_07_XV_den_XVI.docx"
            alt_out = (tmp_dir / "Phan_07_XV_den_XVI (1).docx").resolve()

            fake_analyzer = MagicMock()
            fake_analyzer.process_document.return_value = _sample_page_layouts()
            detailed_events: list[tuple[str, int, int, str]] = []

            with (
                patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer),
                patch.object(pipeline_mod, "build_docx", return_value=alt_out),
            ):
                result = convert_pdf_to_docx(
                    pdf_path,
                    orig_out,
                    preprocess_config=PreprocessConfig(dpi=100),
                    detailed_progress_callback=lambda st, c, t, m: detailed_events.append((st, c, t, m)),
                )

            self.assertEqual(result.output_docx, alt_out)
            self.assertTrue(
                any(
                    st == "docx_warning" and "Phan_07_XV_den_XVI (1).docx" in m
                    for st, _, _, m in detailed_events
                )
            )


if __name__ == "__main__":
    unittest.main()
