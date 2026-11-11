"""Tests for vn_pdf_to_docx command-line interface."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vn_pdf_to_docx.cli import build_arg_parser, main


class TestCLI(unittest.TestCase):
    def test_parser_options(self) -> None:
        parser = build_arg_parser()
        args_default = parser.parse_args(["sample.pdf"])
        self.assertEqual(args_default.dpi, 200)

        args = parser.parse_args(["sample.pdf", "--pages", "1-3", "--dpi", "200"])
        self.assertEqual(args.input_pdf, Path("sample.pdf"))
        self.assertEqual(args.pages, "1-3")
        self.assertEqual(args.dpi, 200)

    def test_main_with_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_pdf = Path(tmp_dir) / "sample.pdf"
            sample_pdf.write_bytes(b"%PDF-1.4 dummy")
            out_docx = Path(tmp_dir) / "out.docx"

            mock_res = MagicMock()
            mock_res.output_docx = out_docx

            with patch("vn_pdf_to_docx.cli.convert_pdf_to_docx", return_value=mock_res) as mock_conv:
                code = main([str(sample_pdf), "-o", str(out_docx), "--pages", "1-2"])
                self.assertEqual(code, 0)
                mock_conv.assert_called_once()
                self.assertEqual(mock_conv.call_args[1]["page_numbers"], [1, 2])

    def test_main_invalid_pages_returns_error_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_pdf = Path(tmp_dir) / "sample.pdf"
            sample_pdf.write_bytes(b"%PDF-1.4 dummy")

            code = main([str(sample_pdf), "--pages", "invalid-range"])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
