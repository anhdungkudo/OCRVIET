"""Unit and integration tests for PyQt5 GUI application (100% Offline)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Force Qt offscreen platform before creating any widgets
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt5.QtWidgets import QApplication, QMessageBox

from vn_pdf_to_docx.gui import ConversionWorker, MainWindow

# Shared QApplication instance
_app = QApplication.instance()
if _app is None:
    _app = QApplication(["--platform", "offscreen"])


def _create_minimal_pdf(file_path: Path) -> None:
    """Create a minimal 2-page PDF file for testing."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((50, 100), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 100), "Nơi nhận: Văn phòng Chính phủ")
    doc.save(str(file_path))
    doc.close()


class TestMainWindow(unittest.TestCase):
    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.worker = None
        self.window.close()

    def test_window_initial_state(self) -> None:
        self.assertIn("Nghị định 30/2020/NĐ-CP", self.window.windowTitle())
        self.assertTrue(self.window.rad_all_pages.isChecked())
        self.assertFalse(self.window.txt_custom_pages.isEnabled())
        self.assertTrue(self.window.btn_start.isEnabled())
        self.assertFalse(self.window.btn_cancel.isEnabled())
        self.assertFalse(self.window.btn_open_docx.isEnabled())
        self.assertEqual(self.window.spn_dpi.value(), 200)
        self.assertFalse(hasattr(self.window, "txt_api_key"))
        self.assertFalse(hasattr(self.window, "cmb_model"))
        self.assertFalse(hasattr(self.window, "chk_skip_llm"))
        self.assertFalse(hasattr(self.window, "chk_save_json"))
        self.assertFalse(hasattr(self.window, "chk_save_debug"))

    def test_page_mode_toggle(self) -> None:
        self.window.rad_custom_pages.setChecked(True)
        self.assertTrue(self.window.txt_custom_pages.isEnabled())
        self.window.rad_all_pages.setChecked(True)
        self.assertFalse(self.window.txt_custom_pages.isEnabled())

    def test_pdf_path_changed_updates_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "sample_test.pdf"
            _create_minimal_pdf(pdf_path)

            self.window._on_pdf_path_changed(str(pdf_path))
            self.assertEqual(self.window.current_pdf_total_pages, 2)
            self.assertIn("2 trang", self.window.lbl_pdf_info.text())
            self.assertEqual(
                self.window.txt_docx.text(),
                str(pdf_path.with_suffix(".docx")),
            )

    def test_start_conversion_validation_missing_pdf(self) -> None:
        self.window.txt_pdf.setText("")
        with (
            patch.object(self.window, "_append_log"),
            patch("PyQt5.QtWidgets.QMessageBox.warning") as mock_box,
        ):
            self.window._start_conversion()
            mock_box.assert_called_once()
            self.assertIn("Thiếu tệp đầu vào", mock_box.call_args[0][1])

    def test_start_conversion_validation_invalid_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "sample.pdf"
            _create_minimal_pdf(pdf_path)
            self.window.txt_pdf.setText(str(pdf_path))
            self.window.rad_custom_pages.setChecked(True)
            self.window.txt_custom_pages.setText("99-100")  # Document only has 2 pages

            with patch("PyQt5.QtWidgets.QMessageBox.critical") as mock_box:
                self.window._start_conversion()
                mock_box.assert_called_once()
                self.assertIn("vượt quá tổng số trang", str(mock_box.call_args[0][2]))

    def test_footer_alignment_setting(self) -> None:
        self.assertEqual(self.window.cmb_footer_align.count(), 2)
        self.assertEqual(self.window.cmb_footer_align.currentData(), "center")
        self.window.cmb_footer_align.setCurrentIndex(1)
        self.assertEqual(self.window.cmb_footer_align.currentData(), "left")

    def test_start_conversion_ensures_docx_extension_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "doc.pdf"
            _create_minimal_pdf(pdf_path)
            self.window.txt_pdf.setText(str(pdf_path))
            self.window.txt_docx.setText(str(Path(tmp_dir) / "output_without_ext"))

            with patch("vn_pdf_to_docx.gui.ConversionWorker.start") as mock_start:
                self.window._start_conversion()
                mock_start.assert_called_once()
                self.assertTrue(self.window.txt_docx.text().endswith(".docx"))
                self.assertEqual(self.window.worker.output_docx.suffix, ".docx")

    def test_start_conversion_guards_against_concurrent_run(self) -> None:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        self.window.worker = mock_worker

        with patch("PyQt5.QtWidgets.QMessageBox.warning") as mock_box:
            self.window._start_conversion()
            mock_box.assert_called_once()
            self.assertIn("Một tiến trình chuyển đổi đang chạy", mock_box.call_args[0][2])

    def test_close_event_with_running_worker(self) -> None:
        mock_worker = MagicMock()
        mock_worker.isRunning.return_value = True
        self.window.worker = mock_worker

        mock_event = MagicMock()
        with patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.No):
            self.window.closeEvent(mock_event)
            mock_event.ignore.assert_called_once()
            mock_worker.cancel.assert_not_called()

        mock_event2 = MagicMock()
        with patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.Yes):
            self.window.closeEvent(mock_event2)
            mock_worker.cancel.assert_called_once()
            mock_event2.accept.assert_called_once()
        self.window.worker = None

    def test_on_worker_finished_success_and_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "res.docx"
            out_file.write_text("dummy docx")

            self.window.chk_open_word.setChecked(True)
            with (
                patch.object(self.window, "_open_output_docx") as mock_open,
                patch("PyQt5.QtWidgets.QMessageBox.information") as mock_info,
            ):
                self.window._on_worker_finished(True, str(out_file))
                mock_open.assert_called_once()
                mock_info.assert_called_once()
                self.assertTrue(self.window.btn_open_docx.isEnabled())

            with (
                patch.object(self.window, "_append_log"),
                patch("PyQt5.QtWidgets.QMessageBox.critical") as mock_crit,
            ):
                self.window._on_worker_finished(False, "Tiến trình đã bị người dùng hủy.")
                mock_crit.assert_not_called()
                self.assertIn("Đã hủy", self.window.lbl_status.text())

    def test_open_output_docx_and_dir_missing(self) -> None:
        self.window.last_generated_docx = Path("nonexistent_missing_file_xyz.docx")
        with patch("PyQt5.QtWidgets.QMessageBox.warning") as mock_warn:
            self.window._open_output_docx()
            mock_warn.assert_called_once()
            self.assertIn("Chưa tìm thấy tệp", mock_warn.call_args[0][2])

    def test_start_conversion_prompts_rename_when_output_docx_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "locked_demo.pdf"
            docx_path = Path(tmp_dir) / "locked_demo.docx"
            _create_minimal_pdf(pdf_path)
            docx_path.write_bytes(b"locked")

            self.window.txt_pdf.setText(str(pdf_path))
            self.window.txt_docx.setText(str(docx_path))
            try:
                with (
                    patch("vn_pdf_to_docx.gui.is_file_write_locked", return_value=True),
                    patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.No),
                    patch("vn_pdf_to_docx.gui.ConversionWorker.start") as mock_start,
                ):
                    self.window._start_conversion()
                    mock_start.assert_not_called()

                with (
                    patch("vn_pdf_to_docx.gui.is_file_write_locked", return_value=True),
                    patch("PyQt5.QtWidgets.QMessageBox.question", return_value=QMessageBox.Yes),
                    patch("vn_pdf_to_docx.gui.ConversionWorker.start") as mock_start,
                ):
                    self.window._start_conversion()
                    mock_start.assert_called_once()
                    expected_alt = Path(tmp_dir) / "locked_demo (1).docx"
                    self.assertEqual(Path(self.window.txt_docx.text()), expected_alt)
                    self.assertEqual(self.window.worker.output_docx, expected_alt)
            finally:
                self.window.worker = None


class TestConversionWorker(unittest.TestCase):
    def test_worker_run_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            docx_path = Path(tmp_dir) / "test.docx"
            _create_minimal_pdf(pdf_path)

            mock_res = MagicMock()
            mock_res.output_docx = docx_path

            worker = ConversionWorker(
                pdf_path=pdf_path,
                output_docx=docx_path,
                dpi=100,
                footer_align="center",
                suppress_red=True,
                page_numbers=[1],
            )

            progress_events: list[tuple[int, str]] = []
            finish_event: list[tuple[bool, str]] = []
            worker.sig_progress.connect(lambda pct, msg: progress_events.append((pct, msg)))
            worker.sig_finished.connect(lambda ok, msg: finish_event.append((ok, msg)))

            with patch("vn_pdf_to_docx.gui.convert_pdf_to_docx", return_value=mock_res) as mock_conv:
                worker.run()
                mock_conv.assert_called_once()

            self.assertTrue(len(progress_events) >= 2)
            self.assertEqual(finish_event, [(True, str(docx_path))])

    def test_worker_run_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            docx_path = Path(tmp_dir) / "test.docx"

            worker = ConversionWorker(
                pdf_path=pdf_path,
                output_docx=docx_path,
                dpi=100,
                footer_align="center",
                suppress_red=True,
                page_numbers=None,
            )

            finish_event: list[tuple[bool, str]] = []
            worker.sig_finished.connect(lambda ok, msg: finish_event.append((ok, msg)))

            with patch(
                "vn_pdf_to_docx.gui.convert_pdf_to_docx",
                side_effect=RuntimeError("Simulated OCR failure"),
            ):
                worker.run()

            self.assertEqual(len(finish_event), 1)
            self.assertFalse(finish_event[0][0])
            self.assertIn("Simulated OCR failure", finish_event[0][1])

    def test_worker_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            docx_path = Path(tmp_dir) / "test.docx"

            worker = ConversionWorker(
                pdf_path=pdf_path,
                output_docx=docx_path,
                dpi=100,
                footer_align="center",
                suppress_red=True,
                page_numbers=None,
            )

            def fake_convert(*args, **kwargs):
                cb = kwargs.get("detailed_progress_callback")
                if cb:
                    cb("preprocess", 1, 5, "Processing page 1")
                return MagicMock()

            worker.cancel()
            finish_event: list[tuple[bool, str]] = []
            worker.sig_finished.connect(lambda ok, msg: finish_event.append((ok, msg)))

            with patch("vn_pdf_to_docx.gui.convert_pdf_to_docx", side_effect=fake_convert):
                worker.run()

            self.assertEqual(len(finish_event), 1)
            self.assertFalse(finish_event[0][0])
            self.assertIn("hủy", finish_event[0][1])


if __name__ == "__main__":
    unittest.main()
