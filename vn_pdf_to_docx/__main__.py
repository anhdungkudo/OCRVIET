"""Module execution entrypoint: `python -m vn_pdf_to_docx`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
