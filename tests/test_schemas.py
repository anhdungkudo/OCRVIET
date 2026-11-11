"""Tests for Pydantic v2 schemas, Vietnamese NFC normalization, and multi-page merging."""

from __future__ import annotations

import unicodedata
import unittest

from pydantic import ValidationError

from vn_pdf_to_docx.schemas import (
    AdministrativeDocument,
    BodyBlock,
    FooterTable,
    HeaderTable,
    PageStructure,
    TitleBlock,
    merge_page_structures,
)


class TestSchemas(unittest.TestCase):
    """Validate structured output schemas, converters, and edge cases."""

    def test_vietnamese_nfc_normalization(self) -> None:
        # Decomposed NFD string -> NFC
        expected = "Cộng hòa xã hội chủ nghĩa Việt Nam"
        nfd_text = unicodedata.normalize("NFD", expected)
        self.assertNotEqual(nfd_text, expected)
        header = HeaderTable(quoc_hieu=nfd_text)
        self.assertEqual(header.quoc_hieu, expected)

    def test_body_block_validation_and_table_normalization(self) -> None:
        p = BodyBlock(
            type="paragraph",
            alignment="justify",
            first_line_indent=True,
            bold=False,
            italic=False,
            content="  Điều 1. Quy định chung.  ",
        )
        self.assertEqual(p.content, "Điều 1. Quy định chung.")

        t = BodyBlock(
            type="table",
            alignment="center",
            content=[["STT", "Nội dung"], ["1", "Hạng mục A"]],
        )
        self.assertEqual(t.content, [["STT", "Nội dung"], ["1", "Hạng mục A"]])

        with self.assertRaises(ValidationError):
            BodyBlock(type="table", content="not a 2d list")

        # 2D list coerced into paragraph joins rows cleanly; 1D list violates schema
        p_from_2d = BodyBlock(type="paragraph", content=[["Dòng 1"], ["Dòng 2"]])
        self.assertEqual(p_from_2d.content, "Dòng 1\nDòng 2")
        with self.assertRaises(ValidationError):
            BodyBlock(type="paragraph", content=["Dòng 1", "Dòng 2"])  # type: ignore[arg-type]

    def test_footer_table_accepts_list_or_str(self) -> None:
        footer = FooterTable(
            noi_nhan=["Nơi nhận:", "- Như trên;", "- Lưu: VT."],
            chuc_vu_nguoi_ky="CHỦ TỊCH",
            ho_ten_nguoi_ky="Nguyễn Văn A",
        )
        self.assertEqual(footer.noi_nhan, "Nơi nhận:\n- Như trên;\n- Lưu: VT.")
        self.assertFalse(footer.is_empty())

    def test_none_cell_and_content_coercion_never_produces_none_literal(self) -> None:
        tbl_block = BodyBlock(
            type="table",
            content=[["STT", None], [None, "Giá trị"]],  # type: ignore[list-item]
        )
        self.assertEqual(tbl_block.content, [["STT", ""], ["", "Giá trị"]])

        para_from_2d = BodyBlock(
            type="paragraph",
            content=[["Dòng 1", None], [None, "Dòng 2"]],  # type: ignore[list-item]
        )
        self.assertEqual(para_from_2d.content, "Dòng 1\nDòng 2")

        para_none = BodyBlock(type="paragraph", content=None)  # type: ignore[arg-type]
        self.assertEqual(para_none.content, "")

        with self.assertRaises(ValidationError):
            BodyBlock(type="table", content=None)  # type: ignore[arg-type]

        # Direct validator invocation via model_construct must also never emit "None"
        constructed_para = BodyBlock.model_construct(
            type="paragraph",
            alignment="justify",
            first_line_indent=False,
            bold=False,
            italic=False,
            content=None,
        )._validate_and_normalize_content()
        self.assertEqual(constructed_para.content, "")

        constructed_tbl = BodyBlock.model_construct(
            type="table",
            alignment="center",
            first_line_indent=False,
            bold=False,
            italic=False,
            content=[["A", None]],
        )._validate_and_normalize_content()
        self.assertEqual(constructed_tbl.content, [["A", ""]])

        footer = FooterTable(noi_nhan=["Nơi nhận:", None, "- Lưu: VT."])  # type: ignore[list-item]
        self.assertEqual(footer.noi_nhan, "Nơi nhận:\n- Lưu: VT.")

        header_none = HeaderTable(
            co_quan_chu_quan=None,
            co_quan_ban_hanh=None,  # type: ignore[arg-type]
            so_ky_hieu=None,  # type: ignore[arg-type]
            quoc_hieu=None,  # type: ignore[arg-type]
            tieu_ngu=None,  # type: ignore[arg-type]
            dia_danh_ngay_thang=None,  # type: ignore[arg-type]
        )
        self.assertIsNone(header_none.co_quan_chu_quan)
        self.assertEqual(header_none.co_quan_ban_hanh, "")
        self.assertTrue(header_none.is_empty())

        title_none = TitleBlock(ten_loai_van_ban=None, trich_yeu=None)  # type: ignore[arg-type]
        self.assertIsNone(title_none.ten_loai_van_ban)
        self.assertEqual(title_none.trich_yeu, "")
        self.assertTrue(title_none.is_empty())

    def test_merge_page_structures_preserves_secondary_title_blocks(self) -> None:
        p1 = PageStructure(
            page_number=1,
            header_table=HeaderTable(co_quan_ban_hanh="SỞ NỘI VỤ"),
            title_block=TitleBlock(ten_loai_van_ban="QUYẾT ĐỊNH", trich_yeu="Về việc A"),
            body_blocks=[BodyBlock(type="paragraph", content="Điều 1. Nội dung chính.")],
        )
        p2 = PageStructure(
            page_number=2,
            title_block=TitleBlock(ten_loai_van_ban="PHỤ LỤC", trich_yeu="Danh mục kèm theo"),
            body_blocks=[BodyBlock(type="table", content=[["STT", "Mục"], ["1", "A"]])],
            footer_table=FooterTable(ho_ten_nguoi_ky="Nguyễn Văn A"),
        )
        merged = merge_page_structures([p2, p1], source_pdf="sample.pdf")
        self.assertIsInstance(merged, AdministrativeDocument)
        self.assertEqual(merged.total_pages, 2)
        self.assertEqual(merged.header_table.co_quan_ban_hanh, "SỞ NỘI VỤ")
        self.assertEqual(merged.title_block.ten_loai_van_ban, "QUYẾT ĐỊNH")
        # Secondary title block from page 2 must be preserved inside body_blocks
        texts = [b.content for b in merged.body_blocks if b.type == "paragraph"]
        self.assertIn("PHỤ LỤC", texts)
        self.assertIn("Danh mục kèm theo", texts)
        self.assertEqual(merged.footer_table.ho_ten_nguoi_ky, "Nguyễn Văn A")

        with self.assertRaises(ValueError):
            merge_page_structures([])

    def test_two_column_table_body_block_normalization(self) -> None:
        block = BodyBlock(
            type="two_column_table",
            alignment="center",
            content=[
                ["  SỞ NỘI VỤ  ", None],  # type: ignore[list-item]
                ["Chỉ 1 cột"],
                ["Cột 1", "Phần A", "Phần B"],
            ],
        )
        self.assertEqual(
            block.content,
            [
                ["SỞ NỘI VỤ", ""],
                ["Chỉ 1 cột", ""],
                ["Cột 1", "Phần A Phần B"],
            ],
        )
        with self.assertRaises(ValidationError):
            BodyBlock(type="two_column_table", content="not a 2d list")


if __name__ == "__main__":
    unittest.main()
