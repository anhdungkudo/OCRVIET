"""Utility functions for page range parsing and validation."""

from __future__ import annotations

import re
from typing import Optional


def parse_page_ranges(
    range_str: Optional[str],
    max_pages: Optional[int] = None,
) -> Optional[list[int]]:
    """Parse a page range expression into a sorted list of unique 1-based page numbers.

    Supported syntax:
      - Empty, None, 'all', '*' -> returns None (meaning all pages)
      - Single page: '1' -> [1]
      - Hyphen range: '1-5' -> [1, 2, 3, 4, 5]
      - Mixed comma-separated: '1, 3, 5-7, 10' -> [1, 3, 5, 6, 7, 10]

    Args:
        range_str: Page expression string from user input or CLI.
        max_pages: Optional upper bound on valid page numbers.

    Returns:
        Sorted list of unique positive integer page numbers, or None for all pages.

    Raises:
        ValueError: If the expression format is invalid or references out-of-bound pages.
    """
    if range_str is None:
        return None

    cleaned = range_str.strip()
    if not cleaned or cleaned.lower() in ("all", "*", "tat ca", "tất cả"):
        return None

    pages: set[int] = set()
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]

    if not parts:
        return None

    for part in parts:
        if "-" in part:
            sub = [s.strip() for s in part.split("-")]
            if len(sub) != 2 or not sub[0] or not sub[1]:
                raise ValueError(
                    f"Khoảng trang không hợp lệ: '{part}'. Định dạng đúng ví dụ: '1-5'."
                )
            try:
                start = int(sub[0])
                end = int(sub[1])
            except ValueError as exc:
                raise ValueError(
                    f"Khoảng trang chứa ký tự không hợp lệ: '{part}'."
                ) from exc

            if start <= 0 or end <= 0:
                raise ValueError(
                    f"Số trang phải lớn hơn 0, nhận được: '{part}'."
                )
            if start > end:
                raise ValueError(
                    f"Khoảng trang không hợp lệ: '{part}' (trang bắt đầu {start} lớn hơn trang kết thúc {end})."
                )
            for p in range(start, end + 1):
                pages.add(p)
        else:
            try:
                p = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"Số trang không hợp lệ: '{part}'. Vui lòng chỉ nhập số và dấu gạch nối (ví dụ: 1, 2-5)."
                ) from exc
            if p <= 0:
                raise ValueError(f"Số trang phải lớn hơn 0, nhận được: {p}.")
            pages.add(p)

    sorted_pages = sorted(pages)

    if max_pages is not None:
        if max_pages <= 0:
            raise ValueError(f"Tổng số trang tài liệu phải > 0, nhận được: {max_pages}.")
        for p in sorted_pages:
            if p > max_pages:
                raise ValueError(
                    f"Trang {p} vượt quá tổng số trang của tài liệu ({max_pages} trang)."
                )

    return sorted_pages
