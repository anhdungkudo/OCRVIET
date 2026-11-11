"""Unit and integration tests for OpenCV morphological table grid & projection fallback."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from vn_pdf_to_docx import (
    AdministrativeDocument,
    BodyBlock,
    DetectedTableGrid,
    LocalOcrLayoutAnalyzer,
    OcrLayoutConfig,
    TextBox,
    build_administrative_document_from_layouts,
    build_docx,
    build_table_data_from_grid,
    classify_page_lines,
    collect_document_text_batch,
    detect_table_grids,
    extract_tables_via_column_projection,
    group_boxes_into_lines,
    partition_boxes_by_tables,
)

HAS_DOCX = importlib.util.find_spec("docx") is not None


def _draw_synthetic_bordered_table_image(
    *,
    width: int = 1654,
    height: int = 2339,
    merge_middle_row: bool = True,
    tilt_deg: float = 0.0,
) -> np.ndarray:
    """Create a white RGB scan page containing a 4x3 bordered table."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    col_xs = [150, 290, 1220, 1490]
    row_ys = [520, 610, 700, 810, 960]

    # Horizontal grid borders
    for y in row_ys:
        cv2.line(img, (col_xs[0], y), (col_xs[-1], y), (0, 0, 0), thickness=3)

    # Outer vertical borders span all rows
    cv2.line(img, (col_xs[0], row_ys[0]), (col_xs[0], row_ys[-1]), (0, 0, 0), thickness=3)
    cv2.line(img, (col_xs[-1], row_ys[0]), (col_xs[-1], row_ys[-1]), (0, 0, 0), thickness=3)

    # Inner vertical dividers; skip row 1 if merged horizontally across all 3 columns
    for vx in col_xs[1:-1]:
        if merge_middle_row:
            cv2.line(img, (vx, row_ys[0]), (vx, row_ys[1]), (0, 0, 0), thickness=3)
            cv2.line(img, (vx, row_ys[2]), (vx, row_ys[-1]), (0, 0, 0), thickness=3)
        else:
            cv2.line(img, (vx, row_ys[0]), (vx, row_ys[-1]), (0, 0, 0), thickness=3)

    # Add synthetic dark text strokes inside cells & header above table so morphology ignores text
    cv2.putText(img, "Don vi: dong", (1260, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "STT", (190, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Danh muc dich vu", (340, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Muc gia", (1275, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "I. Danh muc dich vu kham benh", (210, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "1", (210, 765), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Gia Kham benh", (340, 765), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "39.800", (1290, 765), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    if abs(tilt_deg) > 1e-6:
        mat = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), tilt_deg, 1.0)
        img = cv2.warpAffine(
            img,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    return img


class TestTableDetector(unittest.TestCase):
    """Verify OpenCV morphological grid detection, box mapping, fallback, and DOCX export."""

    def test_detect_table_grids_morphological_and_merged_group_row(self) -> None:
        img = _draw_synthetic_bordered_table_image(merge_middle_row=True, tilt_deg=0.0)
        grids = detect_table_grids(img, dpi=200, color_order="RGB")

        self.assertEqual(len(grids), 1)
        grid = grids[0]
        self.assertEqual(len(grid.col_xs), 4)
        self.assertEqual(len(grid.row_ys), 5)
        self.assertEqual(grid.row_colspans[0], [1, 1, 1])
        self.assertEqual(grid.row_colspans[1], [3])
        self.assertEqual(grid.row_colspans[2], [1, 1, 1])
        self.assertEqual(grid.row_colspans[3], [1, 1, 1])

    def test_detect_table_grids_handles_tilted_scan_lines(self) -> None:
        img = _draw_synthetic_bordered_table_image(merge_middle_row=False, tilt_deg=1.1)
        grids = detect_table_grids(img, dpi=200, color_order="RGB")

        self.assertEqual(len(grids), 1)
        grid = grids[0]
        self.assertEqual(len(grid.col_xs), 4)
        self.assertEqual(len(grid.row_ys), 5)

    def test_partition_boxes_keeps_above_header_outside_and_joins_multiline_cells(self) -> None:
        grid = DetectedTableGrid(
            x_min=150.0,
            y_min=520.0,
            x_max=1490.0,
            y_max=990.0,
            col_xs=[150, 290, 1220, 1490],
            row_ys=[520, 600, 670, 760, 990],
            row_colspans=[[1, 1, 1], [1, 1, 1], [3], [1, 1, 1]],
        )
        # Box sits right above table top; its bottom slightly touches y=520, center_y=508 < 520.
        above_note = TextBox(x_min=1260, y_min=492, x_max=1485, y_max=524, text="Đơn vị: đồng")
        table_boxes = [
            TextBox(x_min=185, y_min=545, x_max=255, y_max=575, text="STT"),
            TextBox(x_min=540, y_min=545, x_max=820, y_max=575, text="Danh mục dịch vụ"),
            TextBox(x_min=1285, y_min=545, x_max=1425, y_max=575, text="Mức giá"),
            TextBox(x_min=210, y_min=620, x_max=230, y_max=648, text="1"),
            TextBox(x_min=670, y_min=620, x_max=690, y_max=648, text="2"),
            TextBox(x_min=1345, y_min=620, x_max=1365, y_max=648, text="3"),
            TextBox(
                x_min=200,
                y_min=700,
                x_max=980,
                y_max=732,
                text="I. Danh mục dịch vụ khám bệnh",
            ),
            TextBox(x_min=210, y_min=830, x_max=230, y_max=860, text="1"),
            TextBox(
                x_min=310,
                y_min=795,
                x_max=1160,
                y_max=825,
                text="Khám cấp giấy chứng thương, giám định y khoa",
            ),
            TextBox(
                x_min=310,
                y_min=845,
                x_max=920,
                y_max=875,
                text="(không kể xét nghiệm, X-quang)",
            ),
            TextBox(x_min=1315, y_min=830, x_max=1445, y_max=860, text="160.000"),
        ]

        inside_groups, outside_boxes = partition_boxes_by_tables(
            [above_note, *table_boxes], [grid], dpi=200
        )
        self.assertEqual([b.text for b in outside_boxes], ["Đơn vị: đồng"])
        self.assertEqual(len(inside_groups), 1)

        rows, colspans, widths, alignments, header_rows = build_table_data_from_grid(
            grid, inside_groups[0][1], dpi=200
        )
        self.assertEqual(header_rows, 2)
        self.assertEqual(widths, [1.2, 11.5, 3.0])
        self.assertEqual(alignments, ["center", "left", "right"])
        self.assertEqual(colspans[2], [3])
        self.assertEqual(rows[0], ["STT", "Danh mục dịch vụ", "Mức giá"])
        self.assertEqual(rows[1], ["1", "2", "3"])
        self.assertEqual(rows[2][0], "I. Danh mục dịch vụ khám bệnh")
        self.assertEqual(
            rows[3][1],
            "Khám cấp giấy chứng thương, giám định y khoa (không kể xét nghiệm, X-quang)",
        )
        self.assertEqual(rows[3][2], "160.000")

    def test_process_page_isolates_bordered_table_from_two_col_and_paragraph(self) -> None:
        img = _draw_synthetic_bordered_table_image(merge_middle_row=True, tilt_deg=0.0)

        box_texts = [
            (TextBox(x_min=150, y_min=180, x_max=480, y_max=210, text=""), "UBND TỈNH BẮC NINH"),
            (TextBox(x_min=1120, y_min=180, x_max=1480, y_max=210, text=""), "Phụ lục I"),
            (TextBox(x_min=1260, y_min=492, x_max=1485, y_max=516, text=""), "Đơn vị: đồng"),
            # Inside table rows
            (TextBox(x_min=185, y_min=550, x_max=255, y_max=580, text=""), "STT"),
            (TextBox(x_min=540, y_min=550, x_max=820, y_max=580, text=""), "Danh mục dịch vụ"),
            (TextBox(x_min=1285, y_min=550, x_max=1425, y_max=580, text=""), "Mức giá"),
            (TextBox(x_min=200, y_min=640, x_max=860, y_max=670, text=""), "I. Danh mục dịch vụ khám bệnh"),
            (TextBox(x_min=210, y_min=740, x_max=230, y_max=770, text=""), "1"),
            (TextBox(x_min=315, y_min=740, x_max=590, y_max=770, text=""), "Giá Khám bệnh"),
            (TextBox(x_min=1315, y_min=740, x_max=1445, y_max=770, text=""), "39.800"),
            (TextBox(x_min=210, y_min=860, x_max=230, y_max=890, text=""), "2"),
            (
                TextBox(x_min=315, y_min=840, x_max=1140, y_max=870, text=""),
                "Khám cấp giấy chứng thương, giám định y khoa",
            ),
            (TextBox(x_min=1315, y_min=860, x_max=1445, y_max=890, text=""), "160.000"),
            (
                TextBox(x_min=315, y_min=890, x_max=920, y_max=920, text=""),
                "(không kể xét nghiệm, X-quang)",
            ),
            # Bottom paragraph
            (
                TextBox(x_min=230, y_min=1120, x_max=1480, y_max=1150, text=""),
                "Ghi chú: Giá dịch vụ áp dụng từ ngày ban hành Quyết định.",
            ),
        ]
        scripted_boxes = [b for b, _ in box_texts]
        text_by_coord = {(int(b.x_min), int(b.y_min)): txt for b, txt in box_texts}

        detector = MagicMock()
        detector.detect_boxes.return_value = scripted_boxes
        recognizer = MagicMock()
        recognizer.recognize_box.side_effect = (
            lambda _img, box, **_kwargs: text_by_coord[(int(box.x_min), int(box.y_min))]
        )

        analyzer = LocalOcrLayoutAnalyzer(
            config=OcrLayoutConfig(dpi=200),
            detector=detector,
            recognizer=recognizer,
        )
        layout = analyzer.process_page(img, page_number=1, color_order="RGB")

        kinds = [seg.kind for seg in layout.segments]
        self.assertEqual(kinds, ["two_column_table", "paragraph", "table", "paragraph"])
        self.assertEqual(layout.segments[1].text, "Đơn vị: đồng")
        self.assertEqual(layout.segments[1].alignment, "right")

        table_seg = layout.segments[2]
        self.assertEqual(len(table_seg.table_rows), 4)
        self.assertEqual(table_seg.cell_colspans[1], [3])
        self.assertEqual(table_seg.table_rows[2], ["1", "Giá Khám bệnh", "39.800"])
        self.assertEqual(
            table_seg.table_rows[3],
            [
                "2",
                "Khám cấp giấy chứng thương, giám định y khoa (không kể xét nghiệm, X-quang)",
                "160.000",
            ],
        )

        # Collect text batch -> skip pure digit cells, preserve table metadata through assembly
        raw_batch, bindings = collect_document_text_batch([layout])
        doc = build_administrative_document_from_layouts([layout], raw_batch, bindings)
        table_blocks = [b for b in doc.body_blocks if b.type == "table"]
        self.assertEqual(len(table_blocks), 1)
        self.assertEqual(table_blocks[0].col_widths_cm, [1.2, 11.5, 3.0])
        self.assertEqual(table_blocks[0].col_alignments, ["center", "left", "right"])
        self.assertEqual(table_blocks[0].cell_colspans[1], [3])
        self.assertNotIn("39.800", raw_batch.values())

    def test_column_projection_fallback_borderless_table_and_blank_price_row(self) -> None:
        boxes = [
            TextBox(x_min=200, y_min=160, x_max=1450, y_max=190, text="Bảng giá áp dụng như sau:"),
            # Borderless table header
            TextBox(x_min=165, y_min=260, x_max=225, y_max=288, text="STT"),
            TextBox(x_min=300, y_min=260, x_max=650, y_max=288, text="Danh mục dịch vụ"),
            TextBox(x_min=1310, y_min=260, x_max=1445, y_max=288, text="Mức giá"),
            # Column index subheader
            TextBox(x_min=185, y_min=305, x_max=205, y_max=332, text="1"),
            TextBox(x_min=675, y_min=305, x_max=695, y_max=332, text="2"),
            TextBox(x_min=1365, y_min=305, x_max=1385, y_max=332, text="3"),
            # Group heading row
            TextBox(
                x_min=175,
                y_min=355,
                x_max=1150,
                y_max=385,
                text="Danh mục dịch vụ khám bệnh, chữa bệnh không thuộc quỹ BHYT",
            ),
            # Data row 1
            TextBox(x_min=185, y_min=405, x_max=205, y_max=433, text="1"),
            TextBox(x_min=290, y_min=405, x_max=560, y_max=433, text="Giá Khám bệnh"),
            TextBox(x_min=1325, y_min=405, x_max=1440, y_max=433, text="39.800"),
            # Data row 2 with wrapped continuation line inside content column
            TextBox(x_min=185, y_min=455, x_max=205, y_max=483, text="2"),
            TextBox(
                x_min=290,
                y_min=455,
                x_max=1160,
                y_max=483,
                text="Khám cấp giấy chứng thương, giám định y khoa",
            ),
            TextBox(x_min=1315, y_min=455, x_max=1440, y_max=483, text="160.000"),
            TextBox(
                x_min=290,
                y_min=495,
                x_max=880,
                y_max=523,
                text="(không kể xét nghiệm, X-quang)",
            ),
            # Sub-item 3.1 with empty price cell: MUST NOT collapse to colspan=[1, 2]
            TextBox(x_min=178, y_min=550, x_max=220, y_max=578, text="3.1"),
            TextBox(x_min=290, y_min=550, x_max=980, y_max=578, text="Dịch vụ phẫu thuật đặc biệt ngoài giờ hành chính"),
        ]

        grouped_lines = group_boxes_into_lines(boxes, y_tolerance_px=10.0)
        cls_lines, _, _ = classify_page_lines(
            grouped_lines, page_width=1654, config=OcrLayoutConfig(dpi=200)
        )
        remaining_lines, proj_tables = extract_tables_via_column_projection(
            cls_lines, page_width=1654, dpi=200
        )

        self.assertEqual([ln.text for ln in remaining_lines], ["Bảng giá áp dụng như sau:"])
        self.assertEqual(len(proj_tables), 1)

        tbl = proj_tables[0]
        self.assertEqual(tbl.header_rows, 2)
        self.assertEqual(tbl.col_widths_cm, [1.2, 11.5, 3.0])
        self.assertEqual(tbl.col_alignments, ["center", "left", "right"])
        self.assertEqual(tbl.rows[0], ["STT", "Danh mục dịch vụ", "Mức giá"])
        self.assertEqual(tbl.rows[1], ["1", "2", "3"])
        self.assertEqual(tbl.row_colspans[2], [3])
        self.assertEqual(tbl.rows[3], ["1", "Giá Khám bệnh", "39.800"])
        self.assertEqual(
            tbl.rows[4],
            [
                "2",
                "Khám cấp giấy chứng thương, giám định y khoa (không kể xét nghiệm, X-quang)",
                "160.000",
            ],
        )
        self.assertEqual(tbl.rows[5], ["3.1", "Dịch vụ phẫu thuật đặc biệt ngoài giờ hành chính", ""])
        self.assertEqual(tbl.row_colspans[5], [1, 1, 1])

    @unittest.skipUnless(HAS_DOCX, "python-docx is not installed in this environment.")
    def test_build_docx_renders_table_grid_widths_alignments_and_merges(self) -> None:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn

        doc_data = AdministrativeDocument(
            total_pages=1,
            body_blocks=[
                BodyBlock(
                    type="table",
                    alignment="center",
                    content=[
                        ["STT", "Danh mục dịch vụ", "Mức giá"],
                        ["1", "2", "3"],
                        ["I. Danh mục dịch vụ khám bệnh", "", ""],
                        ["1", "Giá Khám bệnh", "39.800"],
                    ],
                    cell_colspans=[
                        [1, 1, 1],
                        [1, 1, 1],
                        [3],
                        [1, 1, 1],
                    ],
                    col_widths_cm=[1.2, 11.5, 3.0],
                    col_alignments=["center", "left", "right"],
                    header_rows=2,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "table_grid.docx"
            build_docx(doc_data, out_path)
            doc = Document(str(out_path))

            self.assertEqual(len(doc.tables), 1)
            tbl = doc.tables[0]
            self.assertEqual(tbl.style.name, "Table Grid")

            borders = tbl._tbl.tblPr.find(qn("w:tblBorders"))
            self.assertIsNotNone(borders)
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                edge_el = borders.find(qn(f"w:{edge}"))
                self.assertIsNotNone(edge_el)
                self.assertEqual(edge_el.get(qn("w:val")), "single")
                self.assertEqual(edge_el.get(qn("w:color")), "000000")
                self.assertEqual(edge_el.get(qn("w:sz")), "4")

            self.assertAlmostEqual(tbl.columns[0].width.cm, 1.2, delta=0.1)
            self.assertAlmostEqual(tbl.columns[1].width.cm, 11.5, delta=0.1)
            self.assertAlmostEqual(tbl.columns[2].width.cm, 3.0, delta=0.1)

            # Header rows centered across all columns
            for r in (0, 1):
                for c in (0, 1, 2):
                    self.assertEqual(
                        tbl.cell(r, c).paragraphs[0].alignment,
                        WD_ALIGN_PARAGRAPH.CENTER,
                    )

            # Merged group heading row has XML gridSpan == 3
            merged_tc_pr = tbl.cell(2, 0)._tc.tcPr
            grid_span = merged_tc_pr.find(qn("w:gridSpan"))
            self.assertIsNotNone(grid_span)
            self.assertEqual(grid_span.get(qn("w:val")), "3")

            # Data row column alignments: STT center, Content left, Price right
            self.assertEqual(tbl.cell(3, 0).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)
            self.assertEqual(tbl.cell(3, 1).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.LEFT)
            self.assertEqual(tbl.cell(3, 2).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.RIGHT)


if __name__ == "__main__":
    unittest.main()
