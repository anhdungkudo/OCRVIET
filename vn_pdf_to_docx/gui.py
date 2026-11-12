"""Graphical User Interface (GUI) for Vietnamese PDF to Word Converter (Nghị định 30/2020/NĐ-CP)."""

from __future__ import annotations

import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def _preload_win_msvc_runtime() -> None:
    """Load up-to-date MSVC runtime before PyQt5 prepends older Qt5\\bin DLLs to PATH."""
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

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QIcon, QTextCursor
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .docx_builder import (
    DocxRenderConfig,
    find_next_available_docx_path,
    is_file_write_locked,
)
from .exceptions import PdfToDocxError
from .ocr_layout import OcrLayoutConfig, describe_vietocr_device
from .page_utils import parse_page_ranges
from .pipeline import ConversionResult, convert_pdf_to_docx
from .preprocessor import PreprocessConfig

logger = logging.getLogger(__name__)

APP_TITLE = "Chuyển Đổi Văn Bản PDF Sang Word (Nghị định 30/2020/NĐ-CP)"

MODERN_STYLE = """
QWidget {
    font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
    font-size: 13px;
    color: #212121;
}

QMainWindow, QDialog {
    background-color: #f7f9fa;
}

QGroupBox {
    font-weight: bold;
    font-size: 13px;
    border: 1px solid #dcdfe6;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #1a73e8;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #cfd4dc;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 2px solid #1a73e8;
    background-color: #ffffff;
}

QPushButton {
    background-color: #f1f3f4;
    color: #3c4043;
    border: 1px solid #dadce0;
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #e8eaed;
    border-color: #dadce0;
}

QPushButton:pressed {
    background-color: #dadce0;
}

QPushButton#btnPrimary {
    background-color: #1a73e8;
    color: #ffffff;
    font-size: 14px;
    font-weight: bold;
    border: none;
    padding: 10px 24px;
    border-radius: 6px;
}

QPushButton#btnPrimary:hover {
    background-color: #1765cc;
}

QPushButton#btnPrimary:disabled {
    background-color: #bdc1c6;
    color: #ffffff;
}

QPushButton#btnCancel {
    background-color: #d93025;
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
}

QPushButton#btnCancel:hover {
    background-color: #c5221f;
}

QPushButton#btnCancel:disabled {
    background-color: #f1f3f4;
    color: #bdc1c6;
    border: 1px solid #dadce0;
}

QPushButton#btnSuccess {
    background-color: #1e8e3e;
    color: #ffffff;
    font-weight: bold;
    border: none;
    padding: 8px 18px;
    border-radius: 6px;
}

QPushButton#btnSuccess:hover {
    background-color: #187733;
}

QPushButton#btnSuccess:disabled {
    background-color: #f1f3f4;
    color: #bdc1c6;
    border: 1px solid #dadce0;
}

QProgressBar {
    border: 1px solid #dadce0;
    border-radius: 6px;
    text-align: center;
    background-color: #ffffff;
    height: 22px;
    font-weight: bold;
    color: #202124;
}

QProgressBar::chunk {
    background-color: #1a73e8;
    border-radius: 5px;
}

QPlainTextEdit#logView {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 8px;
}

QScrollBar:vertical {
    border: none;
    background: #f1f1f1;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #c1c1c1;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #a8a8a8;
}
"""


class ConversionWorker(QThread):
    """Background worker thread for converting PDF to DOCX without freezing GUI."""

    sig_progress = pyqtSignal(int, str)  # (percent, step_label)
    sig_log = pyqtSignal(str, str)  # (message, level)
    sig_finished = pyqtSignal(bool, str)  # (success, docx_path_or_err)

    def __init__(
        self,
        pdf_path: Path,
        output_docx: Path,
        dpi: int,
        footer_align: str,
        suppress_red: bool,
        page_numbers: Optional[list[int]],
        vietocr_device: str = "auto",
        vietocr_batch_size: int = 64,
        vietocr_use_fp16: bool = False,
    ) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.output_docx = output_docx
        self.dpi = dpi
        self.footer_align = footer_align
        self.suppress_red = suppress_red
        self.page_numbers = page_numbers
        self.vietocr_device = vietocr_device
        self.vietocr_batch_size = vietocr_batch_size
        self.vietocr_use_fp16 = vietocr_use_fp16
        self._is_cancelled = False

    def cancel(self) -> None:
        """Signal thread to cancel at next opportunity."""
        self._is_cancelled = True

    def run(self) -> None:
        """Run 100% offline conversion pipeline with progress reporting."""
        try:
            self.sig_log.emit(f"Bắt đầu chuyển đổi Offline tệp: {self.pdf_path.name}", "INFO")
            self.sig_progress.emit(5, "Đang khởi tạo cấu hình xử lý Offline...")

            prep_cfg = PreprocessConfig(
                dpi=self.dpi,
                red_suppression_strength=1.3 if self.suppress_red else 0.0,
                nlm_h=10.0 if self.suppress_red else 0.0,
            )
            ocr_cfg = OcrLayoutConfig(
                dpi=self.dpi,
                vietocr_device=self.vietocr_device,
                vietocr_batch_size=self.vietocr_batch_size,
                vietocr_use_fp16=self.vietocr_use_fp16,
            )
            doc_cfg = DocxRenderConfig(footer_left_alignment=self.footer_align)

            def detailed_cb(stage: str, cur: int, tot: int, msg: str) -> None:
                if self._is_cancelled:
                    raise InterruptedError("Tiến trình đã bị người dùng hủy.")

                pct = 10
                log_level = "INFO"
                if stage == "preprocess" and tot > 0:
                    pct = int(10 + (cur / tot) * 25)  # 10% -> 35%
                elif stage == "ocr" and tot > 0:
                    pct = int(35 + (cur / tot) * 55)  # 35% -> 90%
                elif stage == "warning":
                    pct = 90
                    log_level = "WARNING"
                elif stage == "docx":
                    pct = 95
                elif stage == "docx_warning":
                    pct = 98
                    log_level = "WARNING"
                elif stage == "completed":
                    pct = 100

                self.sig_progress.emit(pct, msg)
                self.sig_log.emit(msg, log_level)

            result = convert_pdf_to_docx(
                pdf_path=self.pdf_path,
                output_docx_path=self.output_docx,
                preprocess_config=prep_cfg,
                ocr_config=ocr_cfg,
                docx_config=doc_cfg,
                page_numbers=self.page_numbers,
                detailed_progress_callback=detailed_cb,
            )

            self.sig_progress.emit(100, "Hoàn tất thành công!")
            self.sig_log.emit(
                f"Đã xuất tệp Word chuẩn Nghị định 30: {result.output_docx}", "SUCCESS"
            )
            self.sig_finished.emit(True, str(result.output_docx))

        except InterruptedError:
            self.sig_log.emit("Người dùng đã hủy quá trình chuyển đổi.", "WARNING")
            self.sig_finished.emit(False, "Quá trình đã bị hủy bởi người dùng.")
        except Exception as exc:
            self.sig_log.emit(f"Lỗi: {exc}", "ERROR")
            self.sig_finished.emit(False, str(exc))


class MainWindow(QMainWindow):
    """Main window for the PDF to DOCX application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(880, 720)
        self.setStyleSheet(MODERN_STYLE)
        self.setAcceptDrops(True)

        self.worker: Optional[ConversionWorker] = None
        self.current_pdf_total_pages = 0
        self.last_generated_docx: Optional[Path] = None
        self._previous_suggested_docx = ""

        self._build_ui()

    def _build_ui(self) -> None:
        """Create and layout all widgets."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # 1. Header Banner
        header_frame = QFrame()
        header_frame.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a73e8, stop:1 #0d47a1);"
            "border-radius: 8px; padding: 12px 18px;"
        )
        h_layout = QVBoxLayout(header_frame)
        h_layout.setContentsMargins(4, 4, 4, 4)
        h_layout.setSpacing(4)

        title_label = QLabel("CHUYỂN ĐỔI VĂN BẢN HÀNH CHÍNH SANG WORD (.DOCX)")
        title_label.setFont(QFont("Segoe UI", 15, QFont.Bold))
        title_label.setStyleSheet("color: #ffffff; background: transparent;")

        sub_label = QLabel(
            "Khử dấu đỏ OpenCV • Tự xoay trang ngang • RapidOCR + VietOCR Offline 100% • Chuẩn Nghị định 30/2020/NĐ-CP"
        )
        sub_label.setFont(QFont("Segoe UI", 10))
        sub_label.setStyleSheet("color: #e3f2fd; background: transparent;")

        h_layout.addWidget(title_label)
        h_layout.addWidget(sub_label)
        main_layout.addWidget(header_frame)

        # 2. File Selection Group
        file_group = QGroupBox("1. Chọn Tệp PDF Đầu Vào & Nơi Lưu DOCX")
        file_layout = QGridLayout(file_group)
        file_layout.setSpacing(8)

        file_layout.addWidget(QLabel("Tệp PDF:"), 0, 0)
        self.txt_pdf = QLineEdit()
        self.txt_pdf.setPlaceholderText("Chọn hoặc kéo thả tệp PDF văn bản quét vào đây...")
        self.txt_pdf.textChanged.connect(self._on_pdf_path_changed)
        file_layout.addWidget(self.txt_pdf, 0, 1)

        self.btn_browse_pdf = QPushButton("📂 Chọn tệp PDF...")
        self.btn_browse_pdf.clicked.connect(self._browse_pdf)
        file_layout.addWidget(self.btn_browse_pdf, 0, 2)

        # Document info label
        self.lbl_pdf_info = QLabel("Chưa chọn tệp PDF nào.")
        self.lbl_pdf_info.setStyleSheet("color: #5f6368; font-style: italic;")
        file_layout.addWidget(self.lbl_pdf_info, 1, 1, 1, 2)

        file_layout.addWidget(QLabel("Lưu tệp Word:"), 2, 0)
        self.txt_docx = QLineEdit()
        self.txt_docx.setPlaceholderText("Đường dẫn tệp .docx xuất ra...")
        file_layout.addWidget(self.txt_docx, 2, 1)

        self.btn_browse_docx = QPushButton("💾 Chọn nơi lưu...")
        self.btn_browse_docx.clicked.connect(self._browse_docx)
        file_layout.addWidget(self.btn_browse_docx, 2, 2)

        main_layout.addWidget(file_group)

        # 3. Page Selection Group
        pages_group = QGroupBox("2. Chọn Trang Cần Chuyển Đổi")
        pages_layout = QHBoxLayout(pages_group)
        pages_layout.setSpacing(16)

        self.rad_all_pages = QRadioButton("Toàn bộ tài liệu (Tất cả các trang)")
        self.rad_all_pages.setChecked(True)
        self.rad_all_pages.toggled.connect(self._on_page_mode_toggled)

        self.rad_custom_pages = QRadioButton("Chỉ chuyển đổi trang cụ thể:")
        self.rad_custom_pages.toggled.connect(self._on_page_mode_toggled)

        self.txt_custom_pages = QLineEdit()
        self.txt_custom_pages.setPlaceholderText("Ví dụ: 1 hoặc 1-3 hoặc 1, 3, 5-8")
        self.txt_custom_pages.setEnabled(False)
        self.txt_custom_pages.setFixedWidth(240)

        pages_layout.addWidget(self.rad_all_pages)
        pages_layout.addWidget(self.rad_custom_pages)
        pages_layout.addWidget(self.txt_custom_pages)
        pages_layout.addStretch()

        main_layout.addWidget(pages_group)

        # 4. Settings Group (Collapsible / Compact)
        settings_group = QGroupBox("3. Cấu Hình Xử Lý & Định Dạng Word")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(8)

        # Row 0: DPI & Footer alignment
        settings_layout.addWidget(QLabel("Độ nét (DPI):"), 0, 0)
        self.spn_dpi = QSpinBox()
        self.spn_dpi.setRange(72, 600)
        self.spn_dpi.setValue(200)
        self.spn_dpi.setSuffix(" DPI")
        self.spn_dpi.setMinimumWidth(95)
        settings_layout.addWidget(self.spn_dpi, 0, 1)

        align_box = QHBoxLayout()
        align_box.addWidget(QLabel("Căn lề Nơi nhận:"))
        self.cmb_footer_align = QComboBox()
        self.cmb_footer_align.addItem("Căn giữa (Chuẩn thông dụng)", "center")
        self.cmb_footer_align.addItem("Căn trái (Nghị định 30)", "left")
        align_box.addWidget(self.cmb_footer_align, stretch=1)
        settings_layout.addLayout(align_box, 0, 2)

        # Row 1: Processing options
        settings_layout.addWidget(QLabel("Tùy chọn xử lý:"), 1, 0)
        opt_box = QHBoxLayout()
        self.chk_suppress_red = QCheckBox("Khử dấu đỏ con dấu tròn (Kênh Red + NLM Denoise)")
        self.chk_suppress_red.setChecked(True)
        opt_box.addWidget(self.chk_suppress_red)

        self.chk_open_word = QCheckBox("Tự động mở file Word khi hoàn tất")
        self.chk_open_word.setChecked(True)
        opt_box.addWidget(self.chk_open_word)

        settings_layout.addLayout(opt_box, 1, 1, 1, 2)

        main_layout.addWidget(settings_group)

        # 5. Execution Controls & Progress
        exec_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ BẮT ĐẦU CHUYỂN ĐỔI")
        self.btn_start.setObjectName("btnPrimary")
        self.btn_start.clicked.connect(self._start_conversion)
        exec_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("⏹ Dừng lại")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_conversion)
        exec_layout.addWidget(self.btn_cancel)

        self.btn_open_docx = QPushButton("📄 Mở File Word")
        self.btn_open_docx.setObjectName("btnSuccess")
        self.btn_open_docx.setEnabled(False)
        self.btn_open_docx.clicked.connect(self._open_output_docx)
        exec_layout.addWidget(self.btn_open_docx)

        self.btn_open_dir = QPushButton("📁 Mở Thư Mục")
        self.btn_open_dir.setEnabled(False)
        self.btn_open_dir.clicked.connect(self._open_output_dir)
        exec_layout.addWidget(self.btn_open_dir)

        main_layout.addLayout(exec_layout)

        # Progress bar & status label
        self.lbl_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_status.setStyleSheet("font-weight: 500; color: #202124;")
        main_layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        # 6. Log View
        log_header_box = QHBoxLayout()
        log_label = QLabel("Nhật ký xử lý (Log):")
        log_label.setStyleSheet("font-weight: bold; color: #5f6368;")
        log_header_box.addWidget(log_label)
        log_header_box.addStretch()

        self.btn_clear_log = QPushButton("Xóa nhật ký")
        self.btn_clear_log.setFixedHeight(26)
        self.btn_clear_log.clicked.connect(self._clear_log)
        log_header_box.addWidget(self.btn_clear_log)
        main_layout.addLayout(log_header_box)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setObjectName("logView")
        self.txt_log.setReadOnly(True)
        main_layout.addWidget(self.txt_log, stretch=1)

        self._append_log("Ứng dụng đã sẵn sàng. Vui lòng chọn tệp PDF cần chuyển đổi.", "INFO")

    def _on_page_mode_toggled(self) -> None:
        """Enable or disable custom page range input based on radio selection."""
        is_custom = self.rad_custom_pages.isChecked()
        self.txt_custom_pages.setEnabled(is_custom)
        if is_custom:
            self.txt_custom_pages.setFocus()

    def _browse_pdf(self) -> None:
        """Open file dialog to pick an input PDF."""
        downloads_173811 = Path.home() / "Downloads" / "173811"
        if downloads_173811.exists() and downloads_173811.is_dir():
            initial_dir = str(downloads_173811)
        elif (Path.home() / "Downloads").exists():
            initial_dir = str(Path.home() / "Downloads")
        else:
            initial_dir = str(Path.cwd())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn tệp văn bản PDF scan",
            initial_dir,
            "Tệp PDF (*.pdf);;Tất cả tệp (*.*)",
        )
        if file_path:
            self.txt_pdf.setText(file_path)

    def _browse_docx(self) -> None:
        """Open file dialog to choose output docx location."""
        current_docx = self.txt_docx.text().strip()
        initial = current_docx or str(Path.home() / "Documents" / "van_ban.docx")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Chọn nơi lưu tệp Word",
            initial,
            "Microsoft Word (*.docx);;Tất cả tệp (*.*)",
        )
        if file_path:
            if not file_path.lower().endswith(".docx"):
                file_path += ".docx"
            self.txt_docx.setText(file_path)

    def _on_pdf_path_changed(self, text: str) -> None:
        """Inspect PDF when path changes."""
        path_str = text.strip().strip('"').strip("'")
        if not path_str:
            self.lbl_pdf_info.setText("Chưa chọn tệp PDF nào.")
            self.current_pdf_total_pages = 0
            return

        pdf_path = Path(path_str)
        if not pdf_path.exists() or not pdf_path.is_file():
            self.lbl_pdf_info.setText("⚠️ Đường dẫn không tồn tại hoặc không phải là tệp.")
            self.current_pdf_total_pages = 0
            return

        try:
            with fitz.open(str(pdf_path)) as doc:
                self.current_pdf_total_pages = len(doc)
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            info_text = f"📄 {pdf_path.name} — {self.current_pdf_total_pages} trang ({size_mb:.1f} MB)"
            self.lbl_pdf_info.setText(info_text)
            self.lbl_pdf_info.setStyleSheet("color: #1e8e3e; font-weight: bold;")

            self.rad_all_pages.setText(f"Toàn bộ tài liệu ({self.current_pdf_total_pages} trang)")

            # Auto set output docx if empty or matches previous auto-suggested docx
            suggested_docx = str(pdf_path.with_suffix(".docx"))
            curr_docx = self.txt_docx.text().strip()
            if not curr_docx or curr_docx == self._previous_suggested_docx:
                self.txt_docx.setText(suggested_docx)
                self._previous_suggested_docx = suggested_docx

            if self.current_pdf_total_pages > 10:
                self._append_log(
                    f"Tài liệu có {self.current_pdf_total_pages} trang. Mẹo: Bạn có thể chọn "
                    f"'Chỉ chuyển đổi trang cụ thể' (ví dụ: '1' hoặc '1-3') để thử nghiệm nhanh.",
                    "WARNING",
                )
        except Exception as exc:
            self.lbl_pdf_info.setText(f"⚠️ Lỗi đọc tệp PDF: {exc}")
            self.lbl_pdf_info.setStyleSheet("color: #d93025; font-weight: bold;")
            self.current_pdf_total_pages = 0

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept dragged PDF files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(u.toLocalFile().lower().endswith(".pdf") for u in urls):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped PDF files."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(".pdf"):
                self.txt_pdf.setText(file_path)
                break

    def _append_log(self, text: str, level: str = "INFO") -> None:
        """Add a timestamped line to the log display."""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = f"[{now}]"
        body_text = text
        if level == "SUCCESS":
            color = "#4caf50"
            tag = "[THÀNH CÔNG]"
        elif level == "WARNING":
            color = "#ff9800"
            if body_text.startswith("[CẢNH BÁO]"):
                tag = "[CẢNH BÁO]"
                body_text = body_text[len("[CẢNH BÁO]"):].lstrip()
            else:
                tag = "[LƯU Ý]"
        elif level == "ERROR":
            color = "#f44336"
            tag = "[LỖI]"
        else:
            color = "#90caf9"
            tag = "[THÔNG BÁO]"

        html_line = f"<span style='color: #888888;'>{prefix}</span> <span style='color: {color}; font-weight: bold;'>{tag}</span> {body_text}"
        self.txt_log.appendHtml(html_line)
        self.txt_log.moveCursor(QTextCursor.End)

    def _clear_log(self) -> None:
        """Clear log viewer."""
        self.txt_log.clear()

    def _start_conversion(self) -> None:
        """Validate inputs and launch the worker thread."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Đang xử lý",
                "Một tiến trình chuyển đổi đang chạy. Vui lòng đợi hoặc bấm 'Dừng lại' trước khi bắt đầu lượt mới.",
            )
            return

        pdf_str = self.txt_pdf.text().strip().strip('"').strip("'")
        if not pdf_str:
            QMessageBox.warning(self, "Thiếu tệp đầu vào", "Vui lòng chọn tệp PDF cần chuyển đổi.")
            return

        pdf_path = Path(pdf_str)
        if not pdf_path.exists() or not pdf_path.is_file():
            QMessageBox.critical(self, "Tệp không tồn tại", f"Không tìm thấy tệp: {pdf_path}")
            return

        docx_str = self.txt_docx.text().strip().strip('"').strip("'")
        if not docx_str:
            docx_str = str(pdf_path.with_suffix(".docx"))
        if not docx_str.lower().endswith(".docx"):
            docx_str += ".docx"
        self.txt_docx.setText(docx_str)
        output_docx = Path(docx_str)

        if is_file_write_locked(output_docx):
            alt_docx = find_next_available_docx_path(output_docx)
            reply = QMessageBox.question(
                self,
                "Tệp Word Đang Mở",
                (
                    f"Tệp '{output_docx.name}' hiện đang được mở trong Microsoft Word (hoặc đang bị khóa ghi).\n\n"
                    f"Bấm 'Yes' để tự động lưu sang tệp mới '{alt_docx.name}' và chạy ngay,\n"
                    f"hoặc bấm 'No' để đóng Microsoft Word rồi thử lại."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            output_docx = alt_docx
            self.txt_docx.setText(str(output_docx))

        dpi = self.spn_dpi.value()
        suppress_red = self.chk_suppress_red.isChecked()
        footer_align = self.cmb_footer_align.currentData() or "center"

        # Parse page selection
        page_numbers: Optional[list[int]] = None
        if self.rad_custom_pages.isChecked():
            raw_range = self.txt_custom_pages.text().strip()
            if not raw_range:
                QMessageBox.warning(
                    self,
                    "Chưa nhập số trang",
                    "Bạn đã chọn chuyển đổi trang cụ thể nhưng chưa nhập số trang (ví dụ: '1' hoặc '1-3').",
                )
                return
            try:
                page_numbers = parse_page_ranges(
                    raw_range,
                    max_pages=self.current_pdf_total_pages if self.current_pdf_total_pages > 0 else None,
                )
            except ValueError as exc:
                QMessageBox.critical(self, "Lỗi khoảng trang", str(exc))
                return

        # Update UI state for running
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_open_docx.setEnabled(False)
        self.btn_open_dir.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Trạng thái: Đang xử lý...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #1a73e8;")

        pages_desc = f"{len(page_numbers)} trang ({page_numbers})" if page_numbers else "Toàn bộ trang"
        device_label = describe_vietocr_device("auto")
        self._append_log(
            f"Bắt đầu: {pdf_path.name} | VietOCR [{device_label}, batch=64] | DPI: {dpi} | Trang: {pages_desc} | Căn lề Nơi nhận: {footer_align}",
            "INFO",
        )

        self.worker = ConversionWorker(
            pdf_path=pdf_path,
            output_docx=output_docx,
            dpi=dpi,
            footer_align=footer_align,
            suppress_red=suppress_red,
            page_numbers=page_numbers,
        )
        self.worker.sig_progress.connect(self._on_worker_progress)
        self.worker.sig_log.connect(self._on_worker_log)
        self.worker.sig_finished.connect(self._on_worker_finished)
        self.worker.start()

    def _cancel_conversion(self) -> None:
        """Request worker thread cancellation."""
        if self.worker and self.worker.isRunning():
            self.lbl_status.setText("Trạng thái: Đang hủy...")
            self.lbl_status.setStyleSheet("color: #d93025; font-weight: bold;")
            self.btn_cancel.setEnabled(False)
            self.worker.cancel()

    def _on_worker_progress(self, percent: int, msg: str) -> None:
        """Update progress bar and status."""
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(f"Trạng thái: {msg}")

    def _on_worker_log(self, msg: str, level: str) -> None:
        """Route worker log messages to GUI log view."""
        self._append_log(msg, level)

    def _on_worker_finished(self, success: bool, result_msg: str) -> None:
        """Handle completion of the worker thread."""
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if success:
            self.last_generated_docx = Path(result_msg)
            self.lbl_status.setText("Trạng thái: Chuyển đổi thành công!")
            self.lbl_status.setStyleSheet("color: #1e8e3e; font-weight: bold;")
            self.btn_open_docx.setEnabled(True)
            self.btn_open_dir.setEnabled(True)

            if self.chk_open_word.isChecked():
                self._open_output_docx()
                QMessageBox.information(
                    self,
                    "Hoàn Tất Thành Công!",
                    f"Đã chuyển đổi văn bản sang Word thành công!\n\nTệp đã được tạo và tự động mở:\n{result_msg}",
                )
            else:
                reply = QMessageBox.question(
                    self,
                    "Hoàn Tất Thành Công!",
                    f"Đã chuyển đổi văn bản sang Word thành công!\n\nTệp lưu tại:\n{result_msg}\n\nBạn có muốn mở file Word ngay bây giờ không?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self._open_output_docx()
        else:
            if "hủy" in result_msg.lower() or "cancel" in result_msg.lower():
                self.lbl_status.setText("Trạng thái: Đã hủy theo yêu cầu.")
                self.lbl_status.setStyleSheet("color: #e65100; font-weight: bold;")
                self._append_log("Tiến trình đã được dừng an toàn.", "WARNING")
            else:
                self.lbl_status.setText("Trạng thái: Thất bại.")
                self.lbl_status.setStyleSheet("color: #d93025; font-weight: bold;")
                QMessageBox.critical(
                    self,
                    "Không Thể Hoàn Tất",
                    f"Quá trình chuyển đổi gặp sự cố:\n\n{result_msg}",
                )

    def _open_output_docx(self) -> None:
        """Open the generated Word document in Microsoft Word or default app."""
        target = self.last_generated_docx or Path(self.txt_docx.text().strip())
        if target and target.exists():
            try:
                if hasattr(os, "startfile"):
                    os.startfile(str(target.resolve()))
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", str(target.resolve())], check=False)
                else:
                    import subprocess
                    subprocess.run(["xdg-open", str(target.resolve())], check=False)
                self._append_log(f"Đang mở tệp Word: {target.name}", "INFO")
            except Exception as exc:
                QMessageBox.warning(self, "Không thể mở tệp", f"Lỗi khi mở file: {exc}")
        else:
            QMessageBox.warning(self, "Tệp không tồn tại", f"Chưa tìm thấy tệp {target}")

    def _open_output_dir(self) -> None:
        """Open containing folder in Windows Explorer."""
        target = self.last_generated_docx or Path(self.txt_docx.text().strip())
        folder = target.parent if target else Path.cwd()
        if folder.exists():
            try:
                if hasattr(os, "startfile"):
                    os.startfile(str(folder.resolve()))
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", str(folder.resolve())], check=False)
                else:
                    import subprocess
                    subprocess.run(["xdg-open", str(folder.resolve())], check=False)
            except Exception as exc:
                QMessageBox.warning(self, "Lỗi mở thư mục", f"Không thể mở thư mục: {exc}")

    def closeEvent(self, event) -> None:
        """Handle window close event gracefully, preventing crashes while thread is active."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Xác nhận thoát",
                "Tiến trình chuyển đổi đang chạy ngầm. Bạn có chắc chắn muốn dừng lại và thoát không?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main() -> int:
    """Run GUI application entrypoint."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
