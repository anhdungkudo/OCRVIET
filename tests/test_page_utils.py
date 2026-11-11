"""Unit tests for page range parsing and validation."""

import unittest
from vn_pdf_to_docx.page_utils import parse_page_ranges


class TestPageRanges(unittest.TestCase):
    def test_empty_or_none(self) -> None:
        self.assertIsNone(parse_page_ranges(None))
        self.assertIsNone(parse_page_ranges(""))
        self.assertIsNone(parse_page_ranges("   "))
        self.assertIsNone(parse_page_ranges("all"))
        self.assertIsNone(parse_page_ranges("ALL"))
        self.assertIsNone(parse_page_ranges("*"))
        self.assertIsNone(parse_page_ranges("tất cả"))

    def test_single_page(self) -> None:
        self.assertEqual(parse_page_ranges("1"), [1])
        self.assertEqual(parse_page_ranges("  5  "), [5])

    def test_range(self) -> None:
        self.assertEqual(parse_page_ranges("1-4"), [1, 2, 3, 4])
        self.assertEqual(parse_page_ranges("2-2"), [2])

    def test_comma_separated_and_mixed(self) -> None:
        self.assertEqual(parse_page_ranges("1, 3, 5"), [1, 3, 5])
        self.assertEqual(parse_page_ranges("1, 3-5, 8"), [1, 3, 4, 5, 8])
        self.assertEqual(parse_page_ranges("5-7, 2, 1-3"), [1, 2, 3, 5, 6, 7])

    def test_deduplication(self) -> None:
        self.assertEqual(parse_page_ranges("1, 1, 2-3, 2"), [1, 2, 3])

    def test_max_pages_constraint(self) -> None:
        self.assertEqual(parse_page_ranges("1-3", max_pages=5), [1, 2, 3])
        with self.assertRaises(ValueError) as ctx:
            parse_page_ranges("1-6", max_pages=5)
        self.assertIn("vượt quá tổng số trang", str(ctx.exception))

    def test_invalid_syntax_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_page_ranges("0")
        with self.assertRaises(ValueError):
            parse_page_ranges("-1")
        with self.assertRaises(ValueError):
            parse_page_ranges("5-2")
        with self.assertRaises(ValueError):
            parse_page_ranges("1-abc")
        with self.assertRaises(ValueError):
            parse_page_ranges("abc")
        with self.assertRaises(ValueError):
            parse_page_ranges("1-")
        with self.assertRaises(ValueError):
            parse_page_ranges("-5")


if __name__ == "__main__":
    unittest.main()
