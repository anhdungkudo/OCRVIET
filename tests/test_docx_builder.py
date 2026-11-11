"""Tests for Word (.docx) builder and CLI validation."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import vn_pdf_to_docx.docx_builder as docx_builder
from vn_pdf_to_docx.cli import build_arg_parser, main as cli_main
from vn_pdf_to_docx.docx_builder import (
    DocxRenderConfig,
    _build_two_col_cell_entries,
    _compute_two_column_widths,
    _has_mid_document_footer,
    _set_element_rfonts,
    add_body_blocks,
    add_footer_table,
    add_header_table,
    add_title_block,
    build_docx,
)
from vn_pdf_to_docx.exceptions import DocxBuildError
from vn_pdf_to_docx.schemas import (
    AdministrativeDocument,
    BodyBlock,
    FooterTable,
    HeaderTable,
    PageStructure,
    TitleBlock,
)

HAS_DOCX = importlib.util.find_spec("docx") is not None


def _sample_document() -> AdministrativeDocument:
    return AdministrativeDocument(
        total_pages=1,
        header_table=HeaderTable(
            co_quan_chu_quan="ỦY BAN NHÂN DÂN THÀNH PHỐ HÀ NỘI",
            co_quan_ban_hanh="SỞ THÔNG TIN VÀ TRUYỀN THÔNG",
            so_ky_hieu="Số: 108/QĐ-STTTT",
            dia_danh_ngay_thang="Hà Nội, ngày 10 tháng 11 năm 2026",
        ),
        title_block=TitleBlock(
            ten_loai_van_ban="QUYẾT ĐỊNH",
            trich_yeu="Về việc ban hành quy trình số hóa tài liệu hành chính",
        ),
        body_blocks=[
            BodyBlock(
                type="paragraph",
                alignment="justify",
                first_line_indent=True,
                bold=False,
                italic=True,
                content="Căn cứ Nghị định số 30/2020/NĐ-CP ngày 05 tháng 3 năm 2020 của Chính phủ về công tác văn thư;",
            ),
            BodyBlock(
                type="paragraph",
                alignment="center",
                first_line_indent=False,
                bold=True,
                italic=False,
                content="QUYẾT ĐỊNH:",
            ),
            BodyBlock(
                type="paragraph",
                alignment="justify",
                first_line_indent=True,
                bold=False,
                italic=False,
                content="Điều 1. Ban hành kèm theo Quyết định này bảng phân công nhiệm vụ cụ thể.",
            ),
            BodyBlock(
                type="table",
                alignment="center",
                first_line_indent=False,
                bold=False,
                italic=False,
                content=[
                    ["STT", "Đơn vị", "Ghi chú"],
                    ["1", "Văn phòng Sở"],  # Ragged row to test safe padding
                    ["2", "Phòng Công nghệ", "Chủ trì"],
                ],
            ),
        ],
        footer_table=FooterTable(
            noi_nhan="Nơi nhận:\n- Như Điều 3;\n- Lưu: VT, VP.",
            chuc_vu_nguoi_ky="KT. GIÁM ĐỐC\nPHÓ GIÁM ĐỐC",
            ho_ten_nguoi_ky="Nguyễn Văn An",
        ),
    )


class TestDocxBuilderAndCli(unittest.TestCase):
    """Test DOCX configuration validation, CLI errors, and real DOCX output when installed."""

    def test_docx_render_config_margins_and_usable_width(self) -> None:
        cfg = DocxRenderConfig()
        self.assertEqual(cfg.page_width_cm, 21.0)
        self.assertEqual(cfg.page_height_cm, 29.7)
        self.assertEqual(cfg.margin_left_cm, 3.0)
        self.assertEqual(cfg.margin_right_cm, 1.5)
        self.assertEqual(cfg.margin_top_cm, 2.0)
        self.assertEqual(cfg.margin_bottom_cm, 2.0)
        self.assertEqual(cfg.font_name, "Times New Roman")
        self.assertEqual(cfg.font_size_pt, 13.0)
        self.assertEqual(cfg.footer_left_alignment, "center")
        self.assertAlmostEqual(cfg.usable_width_cm, 16.5, places=4)

        left_w, right_w = _compute_two_column_widths(16.5, 7.2)
        self.assertAlmostEqual(left_w, 7.2, places=4)
        self.assertAlmostEqual(right_w, 9.3, places=4)

        narrow_l, narrow_r = _compute_two_column_widths(1.5, 7.2)
        self.assertGreater(narrow_l, 0.0)
        self.assertGreater(narrow_r, 0.0)
        self.assertAlmostEqual(narrow_l + narrow_r, 1.5, places=4)

        bad_cfg = DocxRenderConfig(margin_left_cm=12.0, margin_right_cm=10.0)
        with self.assertRaises(DocxBuildError):
            _ = bad_cfg.usable_width_cm

    def test_mid_document_footer_detection(self) -> None:
        doc_normal = AdministrativeDocument(
            total_pages=2,
            pages=[
                PageStructure(page_number=1, header_table=HeaderTable(co_quan_ban_hanh="SỞ A")),
                PageStructure(page_number=2, footer_table=FooterTable(ho_ten_nguoi_ky="Nguyễn B")),
            ],
        )
        self.assertFalse(_has_mid_document_footer(doc_normal))

        doc_with_annex = AdministrativeDocument(
            total_pages=2,
            pages=[
                PageStructure(page_number=1, footer_table=FooterTable(ho_ten_nguoi_ky="Nguyễn B")),
                PageStructure(page_number=2, body_blocks=[BodyBlock(type="paragraph", content="Phụ lục")]),
            ],
        )
        self.assertTrue(_has_mid_document_footer(doc_with_annex))

    def test_build_docx_raises_clear_error_when_python_docx_missing(self) -> None:
        if HAS_DOCX:
            self.skipTest("python-docx is installed in this environment.")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DocxBuildError) as ctx:
                build_docx(_sample_document(), Path(tmp) / "out.docx")
            self.assertIn("python-docx", str(ctx.exception))

    @unittest.skipUnless(HAS_DOCX, "python-docx is not installed in this environment.")
    def test_build_docx_real_layout_and_styles(self) -> None:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "output" / "van_ban.docx"
            saved = build_docx(_sample_document(), out_path)
            self.assertTrue(saved.exists())

            doc = Document(str(saved))
            section = doc.sections[0]
            self.assertAlmostEqual(section.page_width.cm, 21.0, places=1)
            self.assertAlmostEqual(section.page_height.cm, 29.7, places=1)
            self.assertAlmostEqual(section.left_margin.cm, 3.0, places=1)
            self.assertAlmostEqual(section.right_margin.cm, 1.5, places=1)
            self.assertAlmostEqual(section.top_margin.cm, 2.0, places=1)
            self.assertAlmostEqual(section.bottom_margin.cm, 2.0, places=1)

            normal = doc.styles["Normal"]
            self.assertEqual(normal.font.name, "Times New Roman")
            self.assertEqual(normal.font.size, Pt(13))

            # Expect 3 tables: header_table, body table, footer_table
            self.assertEqual(len(doc.tables), 3)
            header_tbl, body_tbl, footer_tbl = doc.tables

            # Verify invisible borders and strict OOXML child element order on header & footer tables
            for tbl in (header_tbl, footer_tbl):
                tbl_pr = tbl._tbl.tblPr
                borders = tbl_pr.find(qn("w:tblBorders"))
                self.assertIsNotNone(borders)
                for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                    self.assertEqual(borders.find(qn(f"w:{edge}")).get(qn("w:val")), "none")
                tags = [el.tag for el in tbl_pr]
                self.assertLess(tags.index(qn("w:tblW")), tags.index(qn("w:tblBorders")))
                if qn("w:tblLayout") in tags:
                    self.assertLess(tags.index(qn("w:tblBorders")), tags.index(qn("w:tblLayout")))
                tc_tags = [el.tag for el in tbl.cell(0, 0)._tc.tcPr]
                self.assertLess(tc_tags.index(qn("w:tcBorders")), tc_tags.index(qn("w:tcMar")))
                self.assertLess(tc_tags.index(qn("w:tcMar")), tc_tags.index(qn("w:vAlign")))

            self.assertEqual(header_tbl.cell(0, 0).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)
            self.assertEqual(header_tbl.cell(0, 1).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)
            self.assertEqual(footer_tbl.cell(0, 0).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)
            self.assertEqual(footer_tbl.cell(0, 1).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)

            # Verify body table style and padded ragged cell
            self.assertEqual(body_tbl.style.name, "Table Grid")
            self.assertEqual(len(body_tbl.rows), 3)
            self.assertEqual(len(body_tbl.columns), 3)
            self.assertEqual(body_tbl.cell(1, 2).text, "")

            # Verify justified indented body paragraph
            justified = [p for p in doc.paragraphs if "Căn cứ Nghị định" in p.text][0]
            self.assertEqual(justified.alignment, WD_ALIGN_PARAGRAPH.JUSTIFY)
            self.assertAlmostEqual(justified.paragraph_format.first_line_indent.cm, 1.27, places=2)
            self.assertEqual(Cm(1.27).twips, justified.paragraph_format.first_line_indent.twips)

    @unittest.skipUnless(HAS_DOCX, "python-docx is not installed in this environment.")
    def test_build_docx_separates_body_table_after_header_when_title_missing(self) -> None:
        from docx import Document
        from docx.oxml.ns import qn

        doc_data = AdministrativeDocument(
            total_pages=1,
            header_table=HeaderTable(co_quan_ban_hanh="SỞ NỘI VỤ", so_ky_hieu="Số: 01/BC"),
            title_block=None,
            body_blocks=[
                BodyBlock(type="table", alignment="center", content=[["STT", "Tên"], ["1", "A"]]),
                BodyBlock(type="paragraph", alignment="justify", content="   "),
                BodyBlock(type="table", alignment="center", content=[["Cột", "Giá trị"], ["X", "1"]]),
            ],
            footer_table=FooterTable(ho_ten_nguoi_ky="Nguyễn Văn A"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "separated_tables.docx"
            build_docx(doc_data, out_path)
            doc = Document(str(out_path))
            self.assertEqual(len(doc.tables), 4)
            tags = [
                el.tag
                for el in doc._body._body
                if el.tag != qn("w:sectPr")
            ]
            for prev_tag, cur_tag in zip(tags, tags[1:]):
                self.assertFalse(
                    prev_tag == qn("w:tbl") and cur_tag == qn("w:tbl"),
                    f"Adjacent <w:tbl> elements found in body: {tags}",
                )

    @unittest.skipUnless(HAS_DOCX, "python-docx is not installed in this environment.")
    def test_build_docx_renders_two_column_table_invisible_and_centered(self) -> None:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt

        doc_data = AdministrativeDocument(
            total_pages=1,
            body_blocks=[
                BodyBlock(
                    type="two_column_table",
                    alignment="center",
                    content=[[
                        "ỦY BAN NHÂN DÂN\nSỞ NỘI VỤ\nSố: 10/QĐ-SNV",
                        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\nHà Nội, ngày 10 tháng 11 năm 2026",
                    ]],
                ),
                BodyBlock(
                    type="paragraph",
                    alignment="center",
                    first_line_indent=False,
                    content="QUYẾT ĐỊNH",
                ),
                BodyBlock(
                    type="paragraph",
                    alignment="justify",
                    first_line_indent=True,
                    content="Điều 1. Quy định chung về định dạng văn bản hành chính.",
                ),
                BodyBlock(
                    type="two_column_table",
                    alignment="center",
                    content=[[
                        "Nơi nhận:\n- Như Điều 1;\n- Lưu: VT.",
                        "GIÁM ĐỐC\n\nNguyễn Văn A",
                    ]],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "hybrid.docx"
            saved = build_docx(doc_data, out_path)
            doc = Document(str(saved))

            section = doc.sections[0]
            self.assertAlmostEqual(section.page_width.cm, 21.0, places=1)
            self.assertAlmostEqual(section.page_height.cm, 29.7, places=1)
            self.assertAlmostEqual(section.left_margin.cm, 3.0, places=1)
            self.assertAlmostEqual(section.right_margin.cm, 1.5, places=1)
            self.assertAlmostEqual(section.top_margin.cm, 2.0, places=1)
            self.assertAlmostEqual(section.bottom_margin.cm, 2.0, places=1)
            self.assertEqual(doc.styles["Normal"].font.name, "Times New Roman")
            self.assertEqual(doc.styles["Normal"].font.size, Pt(13))

            self.assertEqual(len(doc.tables), 2)
            header_tbl, footer_tbl = doc.tables
            for tbl in (header_tbl, footer_tbl):
                borders = tbl._tbl.tblPr.find(qn("w:tblBorders"))
                for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                    self.assertEqual(borders.find(qn(f"w:{edge}")).get(qn("w:val")), "none")
                self.assertEqual(tbl.cell(0, 0).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)
                self.assertEqual(tbl.cell(0, 1).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)

            # Footer right cell must preserve signature vertical space before signer name
            signer_p = [p for p in footer_tbl.cell(0, 1).paragraphs if "Nguyễn Văn A" in p.text][0]
            self.assertEqual(signer_p.paragraph_format.space_before, Pt(48))

            center_p = [p for p in doc.paragraphs if p.text == "QUYẾT ĐỊNH"][0]
            self.assertEqual(center_p.alignment, WD_ALIGN_PARAGRAPH.CENTER)
            self.assertEqual(center_p.paragraph_format.first_line_indent, Cm(0))

            justify_p = [p for p in doc.paragraphs if "Điều 1." in p.text][0]
            self.assertEqual(justify_p.alignment, WD_ALIGN_PARAGRAPH.JUSTIFY)
            self.assertEqual(justify_p.paragraph_format.first_line_indent.twips, Cm(1.27).twips)

    def test_table_separation_after_header_and_rfonts_cstheme_guard(self) -> None:
        w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

        def fake_qn(tag: str) -> str:
            prefix, local = tag.split(":", 1)
            if prefix == "w":
                return f"{{{w_ns}}}{local}"
            return tag

        class FakeXmlElement(ET.Element):
            def get_or_add_rPr(self) -> "FakeXmlElement":
                el = self.find(fake_qn("w:rPr"))
                if el is None:
                    el = FakeXmlElement(fake_qn("w:rPr"))
                    self.append(el)
                return el  # type: ignore[return-value]

            def get_or_add_rFonts(self) -> "FakeXmlElement":
                el = self.find(fake_qn("w:rFonts"))
                if el is None:
                    el = FakeXmlElement(fake_qn("w:rFonts"))
                    self.append(el)
                return el  # type: ignore[return-value]

            def get_or_add_tcPr(self) -> "FakeXmlElement":
                el = self.find(fake_qn("w:tcPr"))
                if el is None:
                    el = FakeXmlElement(fake_qn("w:tcPr"))
                    self.append(el)
                return el  # type: ignore[return-value]

            @property
            def tblPr(self) -> "FakeXmlElement":
                el = self.find(fake_qn("w:tblPr"))
                if el is None:
                    el = FakeXmlElement(fake_qn("w:tblPr"))
                    self.append(el)
                return el  # type: ignore[return-value]

        class FakeRun:
            def __init__(self, text: str) -> None:
                self.text = text
                self.font = SimpleNamespace(name=None, size=None)
                self.bold = False
                self.italic = False
                self._element = FakeXmlElement(fake_qn("w:r"))

        class FakeParagraph:
            def __init__(self) -> None:
                self.alignment = None
                self.paragraph_format = SimpleNamespace(
                    first_line_indent=None,
                    space_before=None,
                    space_after=None,
                    line_spacing=None,
                )
                self.runs: list[FakeRun] = []

            def add_run(self, text: str) -> FakeRun:
                run = FakeRun(text)
                self.runs.append(run)
                return run

        class FakeCell:
            def __init__(self) -> None:
                self._tc = FakeXmlElement(fake_qn("w:tc"))
                self.vertical_alignment = None
                self.width = None
                self.paragraphs = [FakeParagraph()]

            def add_paragraph(self) -> FakeParagraph:
                p = FakeParagraph()
                self.paragraphs.append(p)
                return p

        class FakeTable:
            def __init__(self, rows: int, cols: int) -> None:
                self._tbl = FakeXmlElement(fake_qn("w:tbl"))
                self.autofit = True
                self.alignment = None
                self.style = "Normal"
                self.columns = [SimpleNamespace(width=None) for _ in range(cols)]
                self.rows = [
                    SimpleNamespace(cells=[FakeCell() for _ in range(cols)])
                    for _ in range(rows)
                ]

            def cell(self, row_idx: int, col_idx: int) -> FakeCell:
                return self.rows[row_idx].cells[col_idx]

        class FakeDoc:
            def __init__(self) -> None:
                body_el = FakeXmlElement(fake_qn("w:body"))
                body_el.append(FakeXmlElement(fake_qn("w:sectPr")))
                self._body = SimpleNamespace(_body=body_el)
                self.sections = [
                    SimpleNamespace(
                        page_width=None,
                        page_height=None,
                        left_margin=None,
                        right_margin=None,
                        top_margin=None,
                        bottom_margin=None,
                    )
                ]
                self.styles = {
                    "Normal": SimpleNamespace(
                        font=SimpleNamespace(name=None, size=None),
                        element=FakeXmlElement(fake_qn("w:style")),
                    )
                }
                self.tables: list[FakeTable] = []
                self.paragraphs: list[FakeParagraph] = []
                self.saved_paths: list[str] = []

            def _insert_before_sect_pr(self, el: FakeXmlElement) -> None:
                body_el = self._body._body
                body_el.insert(len(body_el) - 1, el)

            def add_paragraph(self) -> FakeParagraph:
                self._insert_before_sect_pr(FakeXmlElement(fake_qn("w:p")))
                p = FakeParagraph()
                self.paragraphs.append(p)
                return p

            def add_page_break(self) -> FakeParagraph:
                return self.add_paragraph()

            def add_table(self, rows: int, cols: int) -> FakeTable:
                tbl = FakeTable(rows, cols)
                self._insert_before_sect_pr(tbl._tbl)
                self.tables.append(tbl)
                return tbl

            def save(self, path: str) -> None:
                self.saved_paths.append(path)
                Path(path).write_bytes(b"fake-docx-bytes")

        class FakeCm:
            def __init__(self, val: float) -> None:
                self.cm = val
                self.twips = int(round(val * 567))

        fake_align_p = SimpleNamespace(LEFT=0, CENTER=1, RIGHT=2, JUSTIFY=3)
        fake_align_v = SimpleNamespace(TOP=0, CENTER=1)
        fake_tbl_align = SimpleNamespace(CENTER=1)

        with (
            patch.object(docx_builder, "qn", side_effect=fake_qn),
            patch.object(docx_builder, "OxmlElement", side_effect=lambda tag: FakeXmlElement(fake_qn(tag))),
            patch.object(docx_builder, "Cm", side_effect=FakeCm),
            patch.object(docx_builder, "Pt", side_effect=lambda v: v),
            patch.object(docx_builder, "WD_ALIGN_PARAGRAPH", fake_align_p),
            patch.object(docx_builder, "WD_ALIGN_VERTICAL", fake_align_v),
            patch.object(docx_builder, "WD_TABLE_ALIGNMENT", fake_tbl_align),
        ):
            cfg = DocxRenderConfig()
            fake_doc = FakeDoc()
            add_header_table(
                fake_doc,
                HeaderTable(co_quan_ban_hanh="SỞ NỘI VỤ", so_ky_hieu="Số: 01/BC"),
                cfg,
            )
            add_title_block(fake_doc, None, cfg)
            add_body_blocks(
                fake_doc,
                [
                    BodyBlock(type="table", alignment="center", content=[["STT", "Tên"], ["1", "A"]]),
                    BodyBlock(type="paragraph", alignment="justify", content="   "),
                    BodyBlock(type="table", alignment="center", content=[["Cột", "Giá trị"], ["X", "1"]]),
                ],
                cfg,
            )
            add_footer_table(
                fake_doc,
                FooterTable(ho_ten_nguoi_ky="Nguyễn Văn A"),
                cfg,
            )

            tags = [el.tag for el in fake_doc._body._body if el.tag != fake_qn("w:sectPr")]
            self.assertEqual(
                tags,
                [
                    fake_qn("w:tbl"),  # header_table
                    fake_qn("w:p"),    # separating spacer before first body table
                    fake_qn("w:tbl"),  # body table 1
                    fake_qn("w:p"),    # separating spacer before body table 2 (after skipped paragraph)
                    fake_qn("w:tbl"),  # body table 2
                    fake_qn("w:p"),    # trailing spacer (reused before footer_table without double spacer)
                    fake_qn("w:tbl"),  # footer_table
                ],
            )

            # Verify _set_element_rfonts deletes both w:cstheme and w:csTheme
            el = FakeXmlElement(fake_qn("w:r"))
            r_fonts = el.get_or_add_rPr().get_or_add_rFonts()
            for theme_attr in (
                "w:asciiTheme",
                "w:hAnsiTheme",
                "w:eastAsiaTheme",
                "w:cstheme",
                "w:csTheme",
            ):
                r_fonts.set(fake_qn(theme_attr), "minorHAnsi")
            _set_element_rfonts(el, "Times New Roman")
            for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                self.assertEqual(r_fonts.attrib.get(fake_qn(attr)), "Times New Roman")
            for theme_attr in (
                "w:asciiTheme",
                "w:hAnsiTheme",
                "w:eastAsiaTheme",
                "w:cstheme",
                "w:csTheme",
            ):
                self.assertNotIn(fake_qn(theme_attr), r_fonts.attrib)

            # Verify build_docx end-to-end (empty TitleBlock + body table after header + mid-doc footer)
            created_docs: list[FakeDoc] = []

            def fake_document_factory() -> FakeDoc:
                d = FakeDoc()
                created_docs.append(d)
                return d

            with (
                patch.object(docx_builder, "_DOCX_IMPORT_ERROR", None),
                patch.object(docx_builder, "Document", side_effect=fake_document_factory),
                tempfile.TemporaryDirectory() as tmp,
            ):
                doc_single = AdministrativeDocument(
                    total_pages=1,
                    header_table=HeaderTable(co_quan_ban_hanh="SỞ NỘI VỤ", so_ky_hieu="Số: 01/BC"),
                    title_block=TitleBlock(ten_loai_van_ban=None, trich_yeu=""),
                    body_blocks=[
                        BodyBlock(type="table", alignment="center", content=[["STT", "Tên"], ["1", "A"]]),
                    ],
                    footer_table=FooterTable(ho_ten_nguoi_ky="Nguyễn Văn A"),
                )
                out_single = build_docx(doc_single, Path(tmp) / "sub" / "single.docx")
                self.assertTrue(out_single.exists())
                single_tags = [
                    el.tag for el in created_docs[-1]._body._body if el.tag != fake_qn("w:sectPr")
                ]
                self.assertEqual(
                    single_tags,
                    [
                        fake_qn("w:tbl"),
                        fake_qn("w:p"),
                        fake_qn("w:tbl"),
                        fake_qn("w:p"),
                        fake_qn("w:tbl"),
                    ],
                )
                self.assertAlmostEqual(created_docs[-1].sections[0].left_margin.cm, 3.0)
                self.assertAlmostEqual(created_docs[-1].sections[0].right_margin.cm, 1.5)
                self.assertEqual(created_docs[-1].styles["Normal"].font.name, "Times New Roman")
                self.assertEqual(created_docs[-1].styles["Normal"].font.size, 13.0)

                doc_multipage_annex = AdministrativeDocument(
                    total_pages=2,
                    pages=[
                        PageStructure(
                            page_number=1,
                            header_table=HeaderTable(co_quan_ban_hanh="SỞ A"),
                            body_blocks=[BodyBlock(type="table", content=[["A", "B"]])],
                            footer_table=FooterTable(ho_ten_nguoi_ky="Nguyễn B"),
                        ),
                        PageStructure(
                            page_number=2,
                            body_blocks=[BodyBlock(type="table", content=[["C", "D"]])],
                        ),
                    ],
                )
                out_multi = build_docx(doc_multipage_annex, Path(tmp) / "multi.docx")
                self.assertTrue(out_multi.exists())
                multi_tags = [
                    el.tag for el in created_docs[-1]._body._body if el.tag != fake_qn("w:sectPr")
                ]
                for prev_tag, cur_tag in zip(multi_tags, multi_tags[1:]):
                    self.assertFalse(
                        prev_tag == fake_qn("w:tbl") and cur_tag == fake_qn("w:tbl"),
                        f"Adjacent <w:tbl> found in multipage build_docx: {multi_tags}",
                    )

    def test_cli_parser_and_missing_input_exit_code(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["sample.pdf", "--dpi", "300", "--footer-left-align", "left"])
        self.assertEqual(args.input_pdf, Path("sample.pdf"))
        self.assertEqual(args.dpi, 300)
        self.assertEqual(args.footer_left_align, "left")

        compat_args = parser.parse_args(["sample.pdf", "--center-footer-left"])
        self.assertTrue(compat_args.center_footer_left)

        code = cli_main(["nonexistent_file_12345.pdf"])
        self.assertEqual(code, 1)

        with tempfile.TemporaryDirectory() as tmp:
            dummy = Path(tmp) / "dummy.pdf"
            dummy.write_bytes(b"%PDF-1.4")
            bad_dpi_code = cli_main([str(dummy), "--dpi", "10"])
            self.assertEqual(bad_dpi_code, 1)

    def test_two_col_cell_entries_handles_old_tone_and_endash(self) -> None:
        entries = _build_two_col_cell_entries(
            "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập – Tự do – Hạnh phúc",
            bold=False,
            italic=False,
            signature_space_pt=48.0,
        )
        self.assertEqual(len(entries), 2)
        self.assertTrue(entries[0]["bold"])
        self.assertTrue(entries[1]["bold"])


if __name__ == "__main__":
    unittest.main()
