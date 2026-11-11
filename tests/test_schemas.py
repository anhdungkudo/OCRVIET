"""Tests for Pydantic v2 schemas, Vietnamese NFC normalization, and multi-page merging."""

from __future__ import annotations

import unicodedata
import unittest
from typing import Any

from pydantic import TypeAdapter, ValidationError

from vn_pdf_to_docx.schemas import (
    AdministrativeDocument,
    BodyBlock,
    FooterTable,
    GeminiBodyBlock,
    GeminiFooterTable,
    GeminiHeaderTable,
    GeminiPageExtraction,
    GeminiTitleBlock,
    HeaderTable,
    PageStructure,
    TitleBlock,
    gemini_extraction_to_page_structure,
    merge_page_structures,
)


def _find_schema_key(node: Any, target_key: str, path: str = "") -> list[str]:
    """Recursively locate occurrences of a JSON Schema keyword key anywhere in a schema tree."""
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            current_path = f"{path}.{key}" if path else key
            if key == target_key:
                hits.append(current_path)
            hits.extend(_find_schema_key(value, target_key, current_path))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            hits.extend(_find_schema_key(item, target_key, f"{path}[{idx}]"))
    return hits


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

    def test_gemini_schema_has_no_non_null_union_on_body_content(self) -> None:
        schema = GeminiPageExtraction.model_json_schema()
        self.assertEqual(_find_schema_key(schema, "default"), [])
        self.assertEqual(
            _find_schema_key(GeminiPageExtraction.model_json_schema(mode="serialization"), "default"),
            [],
        )
        self.assertEqual(_find_schema_key(TypeAdapter(GeminiPageExtraction).json_schema(), "default"), [])
        self.assertEqual(
            _find_schema_key(
                TypeAdapter(GeminiPageExtraction).json_schema(mode="serialization"),
                "default",
            ),
            [],
        )
        for gemini_cls in (
            GeminiHeaderTable,
            GeminiTitleBlock,
            GeminiFooterTable,
            GeminiBodyBlock,
        ):
            self.assertEqual(_find_schema_key(gemini_cls.model_json_schema(), "default"), [])

        body_block_def = schema["$defs"]["GeminiBodyBlock"]["properties"]
        self.assertIn("text_content", body_block_def)
        self.assertIn("table_rows", body_block_def)
        # Neither field should have a multi-type non-null anyOf
        for field_name in ("text_content", "table_rows"):
            if "anyOf" in body_block_def[field_name]:
                types = [opt.get("type") for opt in body_block_def[field_name]["anyOf"]]
                self.assertIn("null", types)
                self.assertEqual(len(types), 2)

        # Gemini transport header/title/footer submodels must have explicit `required`
        # lists and zero non-None string defaults (prevents Gemini schema rejection and
        # prevents empty page-2 header objects from inventing default Quốc hiệu).
        for submodel_name in ("GeminiHeaderTable", "GeminiTitleBlock", "GeminiFooterTable"):
            sub_def = schema["$defs"][submodel_name]
            self.assertIn("required", sub_def)
            self.assertEqual(set(sub_def["required"]), set(sub_def["properties"].keys()))
            for prop in sub_def["properties"].values():
                self.assertNotIn("default", prop)

        # Convenient Python instantiation with omitted optional fields must still work
        minimal_page = GeminiPageExtraction(
            body_blocks=[
                GeminiBodyBlock(
                    type="paragraph",
                    alignment="justify",
                    first_line_indent=True,
                    bold=False,
                    italic=False,
                    text_content="Nội dung ngắn.",
                )
            ]
        )
        self.assertIsNone(minimal_page.header_table)
        self.assertIsNone(minimal_page.title_block)
        self.assertIsNone(minimal_page.footer_table)
        self.assertIsNone(minimal_page.body_blocks[0].table_rows)

        # Empty/null header fields on page 2 must collapse cleanly to None
        empty_header_page = GeminiPageExtraction(
            header_table=GeminiHeaderTable(
                co_quan_chu_quan=None,
                co_quan_ban_hanh="",
                so_ky_hieu=None,
                quoc_hieu=None,
                tieu_ngu="",
                dia_danh_ngay_thang=None,
            ),
            body_blocks=[],
        )
        converted = gemini_extraction_to_page_structure(empty_header_page, page_number=2)
        self.assertIsNone(converted.header_table)

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

        # Transport payload with null table cells (both validated & model_construct)
        raw_page = GeminiPageExtraction(
            body_blocks=[
                GeminiBodyBlock(
                    type="table",
                    alignment="center",
                    first_line_indent=False,
                    bold=False,
                    italic=False,
                    table_rows=[["Cột 1", None], [None, None]],  # type: ignore[list-item]
                ),
            ]
        )
        page_struct = gemini_extraction_to_page_structure(raw_page, page_number=1)
        self.assertEqual(page_struct.body_blocks[0].content, [["Cột 1", ""], ["", ""]])

        constructed_raw_page = GeminiPageExtraction.model_construct(
            header_table=None,
            title_block=None,
            footer_table=None,
            body_blocks=[
                GeminiBodyBlock.model_construct(
                    type="paragraph",
                    alignment="justify",
                    first_line_indent=False,
                    bold=False,
                    italic=False,
                    text_content=None,
                    table_rows=[["Dòng", None]],
                ),
                GeminiBodyBlock.model_construct(
                    type="table",
                    alignment="center",
                    first_line_indent=False,
                    bold=False,
                    italic=False,
                    text_content=None,
                    table_rows=[["X", None]],
                ),
            ],
        )
        converted = gemini_extraction_to_page_structure(constructed_raw_page, page_number=1)
        self.assertEqual(converted.body_blocks[0].content, "Dòng")
        self.assertEqual(converted.body_blocks[1].content, [["X", ""]])

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
        merged = merge_page_structures([p1, p2])
        self.assertEqual(merged.title_block.ten_loai_van_ban, "QUYẾT ĐỊNH")
        # Secondary title block from page 2 must be preserved inside body_blocks
        texts = [b.content for b in merged.body_blocks if b.type == "paragraph"]
        self.assertIn("PHỤ LỤC", texts)
        self.assertIn("Danh mục kèm theo", texts)

    def test_gemini_extraction_conversion_and_multipage_merge(self) -> None:
        page1_raw = GeminiPageExtraction(
            header_table=HeaderTable(
                co_quan_chu_quan="ỦY BAN NHÂN DÂN TỈNH X",
                co_quan_ban_hanh="SỞ NỘI VỤ",
                so_ky_hieu="Số: 12/QĐ-SNV",
                dia_danh_ngay_thang="Hà Nội, ngày 10 tháng 11 năm 2026",
            ),
            title_block=TitleBlock(
                ten_loai_van_ban="QUYẾT ĐỊNH",
                trich_yeu="Về việc ban hành quy chế làm việc",
            ),
            body_blocks=[
                GeminiBodyBlock(
                    type="paragraph",
                    alignment="justify",
                    first_line_indent=True,
                    bold=False,
                    italic=True,
                    text_content="Căn cứ Nghị định số 30/2020/NĐ-CP;",
                ),
                GeminiBodyBlock(
                    type="paragraph",
                    alignment="justify",
                    first_line_indent=True,
                    bold=False,
                    italic=False,
                    text_content="   ",  # Empty whitespace block should be skipped
                ),
            ],
            footer_table=None,
        )

        # Simulate page 2 returning empty-string header/title objects instead of null
        page2_raw = GeminiPageExtraction(
            header_table=HeaderTable(
                co_quan_chu_quan="",
                co_quan_ban_hanh="",
                so_ky_hieu="",
                quoc_hieu="",
                tieu_ngu="",
                dia_danh_ngay_thang="",
            ),
            title_block=TitleBlock(ten_loai_van_ban="", trich_yeu=""),
            body_blocks=[
                GeminiBodyBlock(
                    type="table",
                    alignment="center",
                    first_line_indent=True,
                    bold=False,
                    italic=False,
                    table_rows=[["STT", "Tên đơn vị"], ["1", "Phòng Tổ chức"]],
                )
            ],
            footer_table=FooterTable(
                noi_nhan="Nơi nhận:\n- Như Điều 3;\n- Lưu: VT.",
                chuc_vu_nguoi_ky="GIÁM ĐỐC",
                ho_ten_nguoi_ky="Trần Thị B",
            ),
        )

        p2 = gemini_extraction_to_page_structure(page2_raw, page_number=2)
        p1 = gemini_extraction_to_page_structure(page1_raw, page_number=1)

        self.assertIsNone(p2.header_table)
        self.assertIsNone(p2.title_block)
        self.assertEqual(len(p1.body_blocks), 1)
        self.assertEqual(p2.body_blocks[0].type, "table")
        self.assertFalse(p2.body_blocks[0].first_line_indent)

        # Deliberately pass pages out of order [p2, p1]
        merged = merge_page_structures([p2, p1], source_pdf="sample.pdf")
        self.assertIsInstance(merged, AdministrativeDocument)
        self.assertEqual(merged.total_pages, 2)
        self.assertEqual(merged.header_table.co_quan_ban_hanh, "SỞ NỘI VỤ")
        self.assertEqual(merged.title_block.ten_loai_van_ban, "QUYẾT ĐỊNH")
        self.assertEqual(len(merged.body_blocks), 2)
        self.assertEqual(merged.body_blocks[0].type, "paragraph")
        self.assertEqual(merged.body_blocks[1].type, "table")
        self.assertEqual(merged.footer_table.ho_ten_nguoi_ky, "Trần Thị B")

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
