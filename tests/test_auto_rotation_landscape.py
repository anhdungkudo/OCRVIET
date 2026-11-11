"""Tests for Auto-Rotation (90°/270°) & Dynamic Landscape Section / Table Reconstruction."""

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
    LocalOcrLayoutAnalyzer,
    OcrLayoutConfig,
    PageStructure,
    TextBox,
    build_docx,
    build_table_data_from_grid,
    detect_and_fix_orientation,
    detect_table_grids,
    detect_vertical_text_ratio,
    filter_clipped_edge_boxes,
    partition_boxes_by_tables,
    split_boxes_by_grid_dividers,
)

HAS_DOCX = importlib.util.find_spec("docx") is not None


def _draw_rotated_vertical_table_scan(
    *,
    width: int = 1654,
    height: int = 2339,
    header_on_right: bool = True,
) -> np.ndarray:
    """Create a Portrait scan image whose 5-col table was rotated 90° onto its side."""
    # Draw upright 2339x1654 landscape 5-col bordered table, then rotate CW or CCW.
    lw, lh = height, width
    up = np.full((lh, lw, 3), 255, dtype=np.uint8)
    col_xs = [190, 400, 965, 1525, 1760, 2240]
    row_ys = [40, 160, 260, 360]

    for y in row_ys:
        cv2.line(up, (col_xs[0], y), (col_xs[-1], y), (0, 0, 0), thickness=3)
    for x in col_xs:
        cv2.line(up, (x, row_ys[0]), (x, row_ys[-1]), (0, 0, 0), thickness=3)

    if header_on_right:
        # Rotating CW places TOP (row_ys[0]) on the RIGHT -> requires CCW to fix
        return cv2.rotate(up, cv2.ROTATE_90_CLOCKWISE)
    # Rotating CCW places TOP (row_ys[0]) on the LEFT -> requires CW to fix
    return cv2.rotate(up, cv2.ROTATE_90_COUNTERCLOCKWISE)


class TestAutoRotationAndLandscape(unittest.TestCase):
    """Verify >60% vertical-box detection, 90°/270° rotation, open-top 5-col grids, and Word sections."""

    def test_native_wide_page_and_portrait_vertical_ratio(self) -> None:
        wide_img = np.full((1654, 2339, 3), 255, dtype=np.uint8)
        detector = MagicMock()
        wide_boxes = [TextBox(x_min=200, y_min=100, x_max=500, y_max=130, text="")]
        detector.detect_boxes.return_value = wide_boxes
        recognizer = MagicMock()

        img_out, is_land, angle, boxes_out = detect_and_fix_orientation(
            wide_img,
            detector=detector,
            recognizer=recognizer,
            config=OcrLayoutConfig(dpi=200),
        )
        self.assertTrue(is_land)
        self.assertEqual(angle, 0)
        self.assertEqual(img_out.shape[:2], (1654, 2339))
        self.assertEqual(boxes_out, wide_boxes)
        recognizer.recognize_box.assert_not_called()

        # Portrait boxes (all horizontal) -> ratio 0.0 -> not rotated
        ratio, count = detect_vertical_text_ratio(
            [
                TextBox(x_min=100, y_min=100, x_max=500, y_max=130),
                TextBox(x_min=100, y_min=150, x_max=550, y_max=180),
                TextBox(x_min=100, y_min=200, x_max=480, y_max=230),
            ],
            min_aspect_ratio=2.0,
        )
        self.assertEqual(count, 3)
        self.assertEqual(ratio, 0.0)

    def test_auto_rotate_picks_ccw_and_cw_from_header_orientation(self) -> None:
        # Case A: Header lies on the RIGHT edge of portrait scan -> ROTATE_90_COUNTERCLOCKWISE (270)
        img_right_hdr = _draw_rotated_vertical_table_scan(header_on_right=True)
        # Tall vertical boxes (h > 2*w) in original portrait coords (H=2339, W=1654)
        vertical_boxes_right = [
            # Header boxes on RIGHT side (x ≈ 1490..1590 -> after CCW, top y ≈ 64..164)
            TextBox(x_min=1490, y_min=200, x_max=1590, y_max=410),
            TextBox(x_min=1490, y_min=450, x_max=1590, y_max=920),
            TextBox(x_min=1490, y_min=980, x_max=1590, y_max=1450),
            TextBox(x_min=1490, y_min=1520, x_max=1590, y_max=1720),
            # Data row boxes (near LEFT/middle)
            TextBox(x_min=1280, y_min=450, x_max=1320, y_max=900),
            TextBox(x_min=1280, y_min=1520, x_max=1320, y_max=1700),
        ]

        det_a = MagicMock()
        det_a.detect_boxes.return_value = vertical_boxes_right

        def rec_side_effect_a(_img: np.ndarray, box: TextBox, **_kw: object) -> str:
            # In upright CCW, top header boxes sit around x=200..1720, y < 200
            if box.center_y < 200:
                if box.x_min < 430:
                    return "Mã tương đương"
                if box.x_min < 960:
                    return "Tên dịch vụ kỹ thuật"
                if box.x_min < 1500:
                    return "Tên dịch vụ phê duyệt giá"
                return "Mức giá"
            # In upside-down CW, bottom rows become top y < 500
            return "uos[98N-14917 000'87"

        rec_a = MagicMock()
        rec_a.recognize_box.side_effect = rec_side_effect_a

        out_img_a, is_land_a, angle_a, _ = detect_and_fix_orientation(
            img_right_hdr,
            detector=det_a,
            recognizer=rec_a,
            config=OcrLayoutConfig(dpi=200),
        )
        self.assertTrue(is_land_a)
        self.assertEqual(angle_a, 270)
        self.assertEqual(out_img_a.shape[:2], (1654, 2339))

        # Case B: Header lies on the LEFT edge of portrait scan -> ROTATE_90_CLOCKWISE (90)
        img_left_hdr = _draw_rotated_vertical_table_scan(header_on_right=False)
        vertical_boxes_left = [
            TextBox(x_min=50, y_min=200, x_max=150, y_max=410),
            TextBox(x_min=50, y_min=450, x_max=150, y_max=920),
            TextBox(x_min=50, y_min=980, x_max=150, y_max=1450),
            TextBox(x_min=50, y_min=1520, x_max=150, y_max=1720),
            TextBox(x_min=280, y_min=450, x_max=320, y_max=900),
            TextBox(x_min=280, y_min=1520, x_max=320, y_max=1700),
        ]
        det_b = MagicMock()
        det_b.detect_boxes.return_value = vertical_boxes_left

        rec_b = MagicMock()
        rec_b.recognize_box.side_effect = rec_side_effect_a

        out_img_b, is_land_b, angle_b, _ = detect_and_fix_orientation(
            img_left_hdr,
            detector=det_b,
            recognizer=rec_b,
            config=OcrLayoutConfig(dpi=200),
        )
        self.assertTrue(is_land_b)
        self.assertEqual(angle_b, 90)
        self.assertEqual(out_img_b.shape[:2], (1654, 2339))

    def test_open_top_5col_landscape_grid_and_proportional_25_7cm_widths(self) -> None:
        # Landscape 2339x1654 table whose very top border of Row 0 was cropped off at scan edge
        lw, lh = 2339, 1654
        up = np.full((lh, lw, 3), 255, dtype=np.uint8)
        col_xs = [190, 400, 965, 1525, 1760, 2240]
        # Draw horizontal borders ONLY for bottom of header (y=150) and data rows (y=250, 350)
        for y in (150, 250, 350):
            cv2.line(up, (col_xs[0], y), (col_xs[-1], y), (0, 0, 0), thickness=3)
        # Vertical dividers start at y=18 (above y=150!) framing the open-top header row
        for x in col_xs:
            cv2.line(up, (x, 18), (x, 350), (0, 0, 0), thickness=3)

        grids = detect_table_grids(up, dpi=200, color_order="RGB")
        self.assertEqual(len(grids), 1)
        g = grids[0]
        self.assertEqual(g.num_cols, 5)
        # Must prepend top row border (~18) so 3 rows exist (Header + 2 data rows)
        self.assertEqual(g.num_rows, 3)
        self.assertLess(g.row_ys[0], 50.0)

        boxes = [
            # Clipped STT gutter stubs immediately to left of col_xs[0]=190 -> must be filtered
            TextBox(x_min=150, y_min=60, x_max=180, y_max=95, text="T"),
            TextBox(x_min=152, y_min=185, x_max=182, y_max=220, text="5"),
            # Genuine Row 0 Header cells inside open-top row [18..150]
            TextBox(x_min=210, y_min=60, x_max=370, y_max=115, text="Mã tương đương"),
            TextBox(x_min=440, y_min=60, x_max=920, y_max=115, text="Tên dịch vụ kỹ thuật theo Thông tư 23/2024/TT-BYT"),
            TextBox(x_min=1010, y_min=60, x_max=1480, y_max=115, text="Tên dịch vụ phê duyệt giá"),
            TextBox(x_min=1580, y_min=60, x_max=1710, y_max=115, text="Mức giá"),
            TextBox(x_min=1920, y_min=60, x_max=2080, y_max=115, text="Ghi chú"),
            # Row 1 (with blank Ghi chú!)
            TextBox(x_min=205, y_min=185, x_max=385, y_max=218, text="23.0042.1482"),
            TextBox(x_min=415, y_min=185, x_max=890, y_max=218, text="Đo hoạt độ CK [Máu]"),
            TextBox(x_min=980, y_min=185, x_max=1450, y_max=218, text="Đo hoạt độ CK [Máu]"),
            TextBox(x_min=1635, y_min=185, x_max=1745, y_max=218, text="28.000"),
            # Row 2
            TextBox(x_min=205, y_min=285, x_max=385, y_max=318, text="23.0051.1494"),
            TextBox(x_min=415, y_min=285, x_max=890, y_max=318, text="Định lượng Creatinin"),
            TextBox(x_min=980, y_min=285, x_max=1450, y_max=318, text="Định lượng Creatinin"),
            TextBox(x_min=1635, y_min=285, x_max=1745, y_max=318, text="22.400"),
            TextBox(x_min=1775, y_min=285, x_max=1920, y_max=318, text="Mỗi chất"),
            # Bottom page number below table
            TextBox(x_min=1140, y_min=1580, x_max=1195, y_max=1610, text="32"),
        ]

        populated, outside = partition_boxes_by_tables(boxes, grids, dpi=200)
        outside_clean = filter_clipped_edge_boxes(outside, grids, page_width=lw, dpi=200)
        self.assertEqual([b.text for b in outside_clean], ["32"])

        rows, colspans, widths, alignments, header_rows = build_table_data_from_grid(
            populated[0][0],
            populated[0][1],
            dpi=200,
            is_landscape=True,
            usable_width_cm=25.7,
        )
        self.assertEqual(header_rows, 1)
        self.assertEqual(
            rows[0],
            [
                "Mã tương đương",
                "Tên dịch vụ kỹ thuật theo Thông tư 23/2024/TT-BYT",
                "Tên dịch vụ phê duyệt giá",
                "Mức giá",
                "Ghi chú",
            ],
        )
        # Blank Ghi chú in row 1 must stay 5 separate cells [1,1,1,1,1]
        self.assertEqual(colspans[1], [1, 1, 1, 1, 1])
        self.assertEqual(alignments, ["center", "left", "left", "right", "left"])
        self.assertAlmostEqual(sum(widths), 25.7, delta=0.05)
        # Columns 1 & 2 (565px & 560px) must be roughly twice Column 0 (210px)
        self.assertGreater(widths[1], widths[0] * 2.2)

        # Wide glommed box crossing solid vertical dividers div_x=[400, 965] splits into 3 clean cells
        glommed = [TextBox(x_min=205, y_min=185, x_max=1450, y_max=218)]
        sliced = split_boxes_by_grid_dividers(glommed, grids, dpi=200)
        self.assertEqual(len(sliced), 3)
        self.assertLess(sliced[0].x_max, 400.0)
        self.assertGreater(sliced[1].x_min, 400.0)
        self.assertLess(sliced[1].x_max, 965.0)
        self.assertGreater(sliced[2].x_min, 965.0)

    def test_process_page_rotates_and_marks_page_layout_landscape(self) -> None:
        img = _draw_rotated_vertical_table_scan(header_on_right=True)
        # Place vertical boxes inside grid rows/cols after CCW rotation
        vertical_boxes = [
            # Row 0 headers (x=1490..1580 -> CCW y=64..154)
            TextBox(x_min=1490, y_min=200, x_max=1580, y_max=380),
            TextBox(x_min=1490, y_min=430, x_max=1580, y_max=930),
            TextBox(x_min=1490, y_min=1000, x_max=1580, y_max=1490),
            TextBox(x_min=1490, y_min=1560, x_max=1580, y_max=1730),
            TextBox(x_min=1490, y_min=1820, x_max=1580, y_max=2180),
            # Row 1 data boxes (x=1405..1445 -> CCW y=199..239)
            TextBox(x_min=1405, y_min=200, x_max=1445, y_max=380),
            TextBox(x_min=1405, y_min=430, x_max=1445, y_max=880),
            TextBox(x_min=1405, y_min=1000, x_max=1445, y_max=1450),
            TextBox(x_min=1405, y_min=1580, x_max=1445, y_max=1740),
        ]

        detector = MagicMock()
        detector.detect_boxes.return_value = vertical_boxes

        def rec_fn(_img: np.ndarray, box: TextBox, **_kw: object) -> str:
            if box.center_y < 160:
                if box.x_min < 400:
                    return "Mã tương đương"
                if box.x_min < 965:
                    return "Tên dịch vụ kỹ thuật"
                if box.x_min < 1525:
                    return "Tên dịch vụ phê duyệt giá"
                if box.x_min < 1760:
                    return "Mức giá"
                return "Ghi chú"
            if box.x_min < 400:
                return "23.0042.1482"
            if box.x_min < 965:
                return "Đo hoạt độ CK"
            if box.x_min < 1525:
                return "Đo hoạt độ CK"
            if box.x_min < 1760:
                return "28.000"
            return "TE"

        recognizer = MagicMock()
        recognizer.recognize_box.side_effect = rec_fn

        analyzer = LocalOcrLayoutAnalyzer(
            config=OcrLayoutConfig(dpi=200),
            detector=detector,
            recognizer=recognizer,
        )
        layout = analyzer.process_page(img, page_number=36, color_order="RGB")
        self.assertTrue(layout.is_landscape)
        self.assertEqual(layout.rotation_angle, 270)
        self.assertEqual((layout.page_width, layout.page_height), (2339, 1654))
        tbl_segs = [s for s in layout.segments if s.kind == "table"]
        self.assertEqual(len(tbl_segs), 1)
        self.assertAlmostEqual(sum(tbl_segs[0].col_widths_cm or []), 25.7, delta=0.05)

    @unittest.skipUnless(HAS_DOCX, "python-docx is not installed in this environment.")
    def test_build_docx_switches_portrait_landscape_portrait_sections(self) -> None:
        from docx import Document
        from docx.enum.section import WD_ORIENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc_data = AdministrativeDocument(
            total_pages=3,
            pages=[
                PageStructure(
                    page_number=1,
                    is_landscape=False,
                    body_blocks=[
                        BodyBlock(
                            type="paragraph",
                            alignment="justify",
                            first_line_indent=True,
                            content="Trang 1 khổ dọc bình thường.",
                        )
                    ],
                ),
                PageStructure(
                    page_number=2,
                    is_landscape=True,
                    body_blocks=[
                        BodyBlock(
                            type="table",
                            alignment="center",
                            content=[
                                [
                                    "Mã tương đương",
                                    "Tên dịch vụ kỹ thuật theo Thông tư 23/2024/TT-BYT",
                                    "Tên dịch vụ phê duyệt giá",
                                    "Mức giá",
                                    "Ghi chú",
                                ],
                                [
                                    "23.0042.1482",
                                    "Đo hoạt độ CK (Creatine kinase) [Máu]",
                                    "Đo hoạt độ CK (Creatine kinase) [Máu]",
                                    "28.000",
                                    "",
                                ],
                            ],
                            col_widths_cm=[2.64, 7.09, 7.02, 2.99, 5.96],
                            col_alignments=["center", "left", "left", "right", "left"],
                            header_rows=1,
                        )
                    ],
                ),
                PageStructure(
                    page_number=3,
                    is_landscape=False,
                    body_blocks=[
                        BodyBlock(
                            type="paragraph",
                            alignment="justify",
                            first_line_indent=True,
                            content="Trang 3 quay lại khổ dọc.",
                        )
                    ],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "dynamic_sections.docx"
            build_docx(doc_data, out_path)
            doc = Document(str(out_path))

            self.assertEqual(len(doc.sections), 3)

            # Section 0: Portrait
            s0 = doc.sections[0]
            self.assertEqual(s0.orientation, WD_ORIENT.PORTRAIT)
            self.assertAlmostEqual(s0.page_width.cm, 21.0, places=1)
            self.assertAlmostEqual(s0.page_height.cm, 29.7, places=1)
            self.assertAlmostEqual(s0.left_margin.cm, 3.0, places=1)
            self.assertAlmostEqual(s0.right_margin.cm, 1.5, places=1)

            # Section 1: Landscape (29.7 x 21.0, margins T=3.0, B=1.5, L=2.0, R=2.0 -> 25.7cm usable)
            s1 = doc.sections[1]
            self.assertEqual(s1.orientation, WD_ORIENT.LANDSCAPE)
            self.assertAlmostEqual(s1.page_width.cm, 29.7, places=1)
            self.assertAlmostEqual(s1.page_height.cm, 21.0, places=1)
            self.assertAlmostEqual(s1.top_margin.cm, 3.0, places=1)
            self.assertAlmostEqual(s1.bottom_margin.cm, 1.5, places=1)
            self.assertAlmostEqual(s1.left_margin.cm, 2.0, places=1)
            self.assertAlmostEqual(s1.right_margin.cm, 2.0, places=1)

            # Section 2: Restored to Portrait
            s2 = doc.sections[2]
            self.assertEqual(s2.orientation, WD_ORIENT.PORTRAIT)
            self.assertAlmostEqual(s2.page_width.cm, 21.0, places=1)
            self.assertAlmostEqual(s2.page_height.cm, 29.7, places=1)
            self.assertAlmostEqual(s2.left_margin.cm, 3.0, places=1)
            self.assertAlmostEqual(s2.right_margin.cm, 1.5, places=1)

            # Verify 5-column landscape table uses Table Grid, 25.7 cm total width & right-aligned Mức giá
            self.assertEqual(len(doc.tables), 1)
            tbl = doc.tables[0]
            self.assertEqual(tbl.style.name, "Table Grid")
            widths = [c.width.cm for c in tbl.columns]
            self.assertAlmostEqual(sum(widths), 25.7, delta=0.1)
            self.assertEqual(tbl.cell(1, 0).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.CENTER)
            self.assertEqual(tbl.cell(1, 1).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.LEFT)
            self.assertEqual(tbl.cell(1, 2).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.LEFT)
            self.assertEqual(tbl.cell(1, 3).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.RIGHT)
            self.assertEqual(tbl.cell(1, 4).paragraphs[0].alignment, WD_ALIGN_PARAGRAPH.LEFT)


if __name__ == "__main__":
    unittest.main()
