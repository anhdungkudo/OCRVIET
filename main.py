#!/usr/bin/env python
"""Điểm khởi chạy chính (Main Entry Point) cho công cụ chuyển đổi PDF sang Word.

- Chạy không tham số (`python main.py`): Mở giao diện đồ họa (GUI).
- Chạy kèm tham số (`python main.py file.pdf -o file.docx`): Chạy chế độ dòng lệnh (CLI).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc nằm trong sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def main() -> int:
    """Điều hướng khởi chạy: GUI khi không có tham số, CLI khi có tham số."""
    if len(sys.argv) > 1:
        from vn_pdf_to_docx.cli import main as cli_main

        return cli_main()
    else:
        from vn_pdf_to_docx.gui import main as gui_main

        return gui_main()


if __name__ == "__main__":
    sys.exit(main())

