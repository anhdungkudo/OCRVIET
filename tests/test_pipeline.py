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
    """Verify Hybrid pipeline execution, 1-shot batching, error wrapping, and artifacts."""

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

    def test_hybrid_pipeline_runs_1_shot_for_all_pages_and_builds_real_docx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            debug_dir = tmp_dir / "debug_imgs"
            json_path = tmp_dir / "nested" / "extracted.json"
            out_docx = tmp_dir / "result.docx"

            fake_analyzer = MagicMock()
            fake_analyzer.process_document.return_value = _sample_page_layouts()

            fake_corrector = MagicMock()
            fake_corrector.correct_document_texts.return_value = {
                "id_1": "SỞ NỘI VỤ",
                "id_2": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
                "id_3": "QUYẾT ĐỊNH",
                "id_4": "Điều 1. Quy định chung ngày 05/03/2020.",
                "id_5": "Nơi nhận:",
                "id_6": "- Lưu: VT.",
                "id_7": "GIÁM ĐỐC",
                "id_8": "Nguyễn Văn A",
            }

            events: list[tuple[str, int, int]] = []

            with (
                patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer),
                patch.object(pipeline_mod, "GeminiTextCorrector", return_value=fake_corrector),
            ):
                result = convert_pdf_to_docx(
                    pdf_path,
                    out_docx,
                    api_key="dummy-key",
                    preprocess_config=PreprocessConfig(dpi=100, image_format="jpeg"),
                    save_debug_images_dir=debug_dir,
                    save_json_path=json_path,
                    progress_callback=lambda stage, cur, tot: events.append((stage, cur, tot)),
                )

            # Debug PNGs written for both pages
            self.assertTrue((debug_dir / "page_001_preprocessed.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((debug_dir / "page_002_preprocessed.png").exists())

            # GeminiTextCorrector.correct_document_texts called EXACTLY ONCE for all 2 pages!
            fake_corrector.correct_document_texts.assert_called_once()
            sent_dict = fake_corrector.correct_document_texts.call_args[0][0]
            self.assertEqual(len(sent_dict), 8)
            self.assertEqual(sent_dict["id_4"], "Điểu 1. Quy định chung ngày O5/O3/2O2O.")

            # JSON persisted with corrected Vietnamese texts
            self.assertTrue(json_path.exists())
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["total_pages"], 2)
            self.assertEqual(loaded["body_blocks"][2]["content"], "Điều 1. Quy định chung ngày 05/03/2020.")

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

    def test_pipeline_skip_llm_correction_makes_zero_gemini_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            fake_analyzer = MagicMock()
            fake_analyzer.process_document.return_value = _sample_page_layouts()

            with (
                patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer),
                patch.object(pipeline_mod, "GeminiTextCorrector") as corrector_cls,
            ):
                result = convert_pdf_to_docx(
                    pdf_path,
                    tmp_dir / "offline.docx",
                    preprocess_config=PreprocessConfig(dpi=100),
                    skip_llm_correction=True,
                )

            corrector_cls.assert_not_called()
            self.assertTrue(result.output_docx.exists())
            self.assertEqual(result.corrected_text_batch, result.raw_text_batch)

    def test_pipeline_graceful_degradation_on_429_or_network_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "sample.pdf"
            _write_two_page_pdf(pdf_path)

            fake_analyzer = MagicMock()
            fake_analyzer.process_document.return_value = _sample_page_layouts()

            fake_corrector = MagicMock()
            fake_corrector.correct_document_texts.side_effect = RuntimeError(
                "429 RESOURCE_EXHAUSTED: Quota exceeded"
            )

            detailed_events: list[tuple[str, int, int, str]] = []
            with (
                patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer),
                patch.object(pipeline_mod, "GeminiTextCorrector", return_value=fake_corrector),
                self.assertLogs("vn_pdf_to_docx.pipeline", level="WARNING") as log_ctx,
            ):
                result = convert_pdf_to_docx(
                    pdf_path,
                    tmp_dir / "fallback.docx",
                    api_key="dummy-key",
                    preprocess_config=PreprocessConfig(dpi=100),
                    detailed_progress_callback=lambda st, c, t, m: detailed_events.append((st, c, t, m)),
                )

            self.assertTrue(result.output_docx.exists())
            self.assertEqual(result.corrected_text_batch, result.raw_text_batch)

            expected_warn = (
                "[CẢNH BÁO] Không thể gọi Gemini sửa chính tả, "
                "tự động sử dụng văn bản gốc từ VietOCR để xuất file Word"
            )
            self.assertTrue(any(expected_warn in entry for entry in log_ctx.output))
            self.assertTrue(
                any(st == "warning" and m == expected_warn for st, _, _, m in detailed_events)
            )

            doc = Document(str(result.output_docx))
            body_text = "\n".join(p.text for p in doc.paragraphs)
            self.assertIn("Điểu 1. Quy định chung ngày O5/O3/2O2O.", body_text)

            # User cancellation (InterruptedError) must still propagate immediately
            fake_corrector.correct_document_texts.side_effect = InterruptedError("Cancelled by user")
            with (
                patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer),
                patch.object(pipeline_mod, "GeminiTextCorrector", return_value=fake_corrector),
            ):
                with self.assertRaises(InterruptedError):
                    convert_pdf_to_docx(
                        pdf_path,
                        tmp_dir / "cancelled.docx",
                        api_key="dummy-key",
                        preprocess_config=PreprocessConfig(dpi=100),
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
            fake_corrector = MagicMock()
            fake_corrector.correct_document_texts.return_value = {"id_1": "OK"}

            with (
                patch.object(pipeline_mod, "LocalOcrLayoutAnalyzer", return_value=fake_analyzer),
                patch.object(pipeline_mod, "GeminiTextCorrector", return_value=fake_corrector),
            ):
                with self.assertRaises(PdfToDocxError) as ctx:
                    convert_pdf_to_docx(
                        pdf_path,
                        tmp_dir / "out.docx",
                        api_key="dummy-key",
                        preprocess_config=PreprocessConfig(dpi=100),
                        save_json_path=occupied_dir,
                    )
            self.assertIn("structured JSON", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
