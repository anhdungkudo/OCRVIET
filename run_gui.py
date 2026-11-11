#!/usr/bin/env python
"""Runner script for Vietnamese PDF to Word GUI Application."""

import sys
from pathlib import Path

# Ensure root workspace is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vn_pdf_to_docx.gui import main

if __name__ == "__main__":
    sys.exit(main())
