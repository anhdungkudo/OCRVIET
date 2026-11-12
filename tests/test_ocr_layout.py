"""Unit and integration tests for Step 2 offline RapidOCR + X/Y layout + VietOCR."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from vn_pdf_to_docx.exceptions import LocalOcrError
from vn_pdf_to_docx.ocr_layout import (
    LocalOcrLayoutAnalyzer,
    OcrLayoutConfig,
    RapidOcrDetector,
    TextBox,
    VietOcrRecognizer,
    _compat_torch_load_for_vietocr,
    _crop_to_pil_rgb,
    _ensure_pillow_antialias,
    _ensure_pkg_resources_shim,
    _load_vietocr_config,
    _resolve_local_vietocr_weights,
    build_administrative_document_from_layouts,
    classify_page_lines,
    cluster_line_boxes,
    collect_document_text_batch,
    describe_vietocr_device,
    group_boxes_into_lines,
    group_lines_into_segments,
    parse_rapidocr_boxes,
    resolve_vietocr_device,
)
from vn_pdf_to_docx.preprocessor import PreprocessedPage


class TestOcrLayout(unittest.TestCase):
    """Test RapidOCR box parsing, Y-line grouping, X-alignment rules, and sequential VietOCR."""

    def test_parse_rapidocr_boxes_supports_multiple_output_formats(self) -> None:
        # 1. Tuple (list_of_4pt_polygons, elapse)
        poly1 = [[10, 20], [110, 20], [110, 45], [10, 45]]
        poly2 = [[-5, -5], [50, 0], [50, 30], [0, 30]]
        tiny = [[1, 1], [2, 1], [2, 2], [1, 2]]
        boxes = parse_rapidocr_boxes(([poly1, poly2, tiny], 0.12), img_width=1000, img_height=1400)
        self.assertEqual(len(boxes), 2)
        self.assertEqual((boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max), (10.0, 20.0, 110.0, 45.0))
        # Clamped to 0
        self.assertEqual((boxes[1].x_min, boxes[1].y_min), (0.0, 0.0))

        # 2. TextDetOutput-like object with .boxes ndarray
        det_obj = SimpleNamespace(
            boxes=np.array([[[100, 200], [300, 200], [300, 230], [100, 230]]], dtype=np.float32)
        )
        obj_boxes = parse_rapidocr_boxes(det_obj, img_width=1000, img_height=1400)
        self.assertEqual(len(obj_boxes), 1)
        self.assertEqual(obj_boxes[0].x_max, 300.0)

        # 3. Full [poly, text, score] tuple format
        full_fmt = ([([ [50, 60], [200, 60], [200, 90], [50, 90] ], "Văn bản", 0.99)], [0.1])
        full_boxes = parse_rapidocr_boxes(full_fmt, img_width=1000, img_height=1400)
        self.assertEqual(len(full_boxes), 1)
        self.assertEqual(full_boxes[0].text, "Văn bản")

        # 4. Empty / None output
        self.assertEqual(parse_rapidocr_boxes((None, 0.0), img_width=1000, img_height=1400), [])
        self.assertEqual(parse_rapidocr_boxes(None, img_width=1000, img_height=1400), [])

    def test_group_boxes_into_lines_within_10px(self) -> None:
        boxes = [
            TextBox(x_min=600, y_min=104, x_max=900, y_max=134, text="Phải"),   # center_y = 119
            TextBox(x_min=100, y_min=100, x_max=400, y_max=124, text="Trái"),   # center_y = 112 (diff 7px <= 10)
            TextBox(x_min=100, y_min=150, x_max=500, y_max=176, text="Dòng 2"), # center_y = 163 (diff > 10)
            TextBox(x_min=100, y_min=50, x_max=300, y_max=74, text="Dòng 0"),   # center_y = 62
        ]
        lines = group_boxes_into_lines(boxes, y_tolerance_px=10.0)
        self.assertEqual(len(lines), 3)
        self.assertEqual([b.text for b in lines[0]], ["Dòng 0"])
        self.assertEqual([b.text for b in lines[1]], ["Trái", "Phải"])
        self.assertEqual([b.text for b in lines[2]], ["Dòng 2"])

    def test_classify_two_column_center_justify_indent_and_right(self) -> None:
        page_w = 1654  # A4 at 200 DPI
        cfg = OcrLayoutConfig(dpi=200)

        lines_boxes = [
            # Line 0: two_column_row (left < 45% W=744, right > 45% W)
            [
                TextBox(x_min=236, y_min=100, x_max=600, y_max=126, text="SỞ NỘI VỤ"),
                TextBox(x_min=860, y_min=100, x_max=1500, y_max=126, text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"),
            ],
            # Line 1: centered title (center ~ 827, doesn't touch left/right)
            [
                TextBox(x_min=700, y_min=220, x_max=954, y_max=250, text="QUYẾT ĐỊNH"),
            ],
            # Line 2: wide centered subtitle (symmetric inset, should still be center)
            [
                TextBox(x_min=360, y_min=270, x_max=1390, y_max=298, text="Về việc ban hành quy trình số hóa văn bản"),
            ],
            # Line 3: justified + indented (left offset = 290 - 236 = 54px in 30..80px, touches right 1520)
            [
                TextBox(x_min=290, y_min=350, x_max=1520, y_max=378, text="Căn cứ Nghị định số 30/2020/NĐ-CP của"),
            ],
            # Line 4: justified wrapped continuation (touches left=236 and right=1520, no indent)
            [
                TextBox(x_min=236, y_min=385, x_max=1520, y_max=413, text="Chính phủ về công tác văn thư."),
            ],
            # Line 5: left-aligned short line (touches left=236, does not touch right)
            [
                TextBox(x_min=236, y_min=440, x_max=620, y_max=468, text="Kính gửi: Các đơn vị."),
            ],
            # Line 6: right-aligned line
            [
                TextBox(x_min=1100, y_min=500, x_max=1520, y_max=528, text="Hà Nội, ngày 10 tháng 11 năm 2026"),
            ],
            # Line 7: 4-column tabular row (>= 3 clusters -> NOT two_column_row)
            [
                TextBox(x_min=236, y_min=560, x_max=280, y_max=585, text="STT"),
                TextBox(x_min=450, y_min=560, x_max=620, y_max=585, text="Đơn vị"),
                TextBox(x_min=850, y_min=560, x_max=1020, y_max=585, text="Chỉ tiêu"),
                TextBox(x_min=1250, y_min=560, x_max=1480, y_max=585, text="Ghi chú"),
            ],
        ]

        classified, body_left, body_right = classify_page_lines(
            lines_boxes, page_width=page_w, config=cfg
        )
        self.assertAlmostEqual(body_left, 236.0, delta=1.0)
        self.assertAlmostEqual(body_right, 1520.0, delta=1.0)

        self.assertEqual(classified[0].line_type, "two_column_row")
        self.assertEqual(classified[0].left_text, "SỞ NỘI VỤ")
        self.assertEqual(classified[0].right_text, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")

        self.assertEqual(classified[1].alignment, "center")
        self.assertFalse(classified[1].first_line_indent)

        self.assertEqual(classified[2].alignment, "center")
        self.assertFalse(classified[2].first_line_indent)

        self.assertEqual(classified[3].alignment, "justify")
        self.assertTrue(classified[3].first_line_indent)

        self.assertEqual(classified[4].alignment, "justify")
        self.assertFalse(classified[4].first_line_indent)

        self.assertEqual(classified[5].alignment, "left")
        self.assertFalse(classified[5].first_line_indent)

        self.assertEqual(classified[6].alignment, "right")
        self.assertFalse(classified[6].first_line_indent)

        self.assertEqual(classified[7].line_type, "single")
        self.assertTrue(classified[7].is_tabular_row)

    def test_group_lines_into_segments_merges_two_col_blocks_and_wrapped_prose(self) -> None:
        page_w, page_h = 1654, 2339
        cfg = OcrLayoutConfig(dpi=200)
        lines_boxes = [
            # Header two_column_row #1
            [
                TextBox(x_min=236, y_min=120, x_max=580, y_max=146, text="ỦY BAN NHÂN DÂN"),
                TextBox(x_min=860, y_min=120, x_max=1500, y_max=146, text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"),
            ],
            # Header sandwiched left-only line
            [
                TextBox(x_min=240, y_min=154, x_max=610, y_max=180, text="SỞ THÔNG TIN VÀ TRUYỀN THÔNG"),
            ],
            # Header two_column_row #2
            [
                TextBox(x_min=260, y_min=190, x_max=520, y_max=216, text="Số: 125/QĐ-STTTT"),
                TextBox(x_min=900, y_min=190, x_max=1480, y_max=216, text="Hà Nội, ngày 10 tháng 11 năm 2026"),
            ],
            # Centered title
            [
                TextBox(x_min=710, y_min=300, x_max=940, y_max=328, text="QUYẾT ĐỊNH"),
            ],
            # Wrapped 3-line paragraph (line 1 indented & wraps, line 2 wraps, line 3 short ending)
            [
                TextBox(x_min=290, y_min=400, x_max=1520, y_max=426, text="Căn cứ Nghị định số 30/2020/NĐ-CP ngày 05"),
            ],
            [
                TextBox(x_min=236, y_min=434, x_max=1520, y_max=460, text="tháng 3 năm 2020 của Chính phủ về công tác"),
            ],
            [
                TextBox(x_min=236, y_min=468, x_max=500, y_max=494, text="văn thư lưu trữ;"),
            ],
            # Footer two_column_row #1
            [
                TextBox(x_min=236, y_min=1400, x_max=400, y_max=1426, text="Nơi nhận:"),
                TextBox(x_min=1040, y_min=1400, x_max=1360, y_max=1426, text="KT. GIÁM ĐỐC"),
            ],
            # Footer two_column_row #2
            [
                TextBox(x_min=236, y_min=1434, x_max=450, y_max=1460, text="- Như Điều 3;"),
                TextBox(x_min=1030, y_min=1434, x_max=1370, y_max=1460, text="PHÓ GIÁM ĐỐC"),
            ],
            # Footer left continuation lines
            [
                TextBox(x_min=236, y_min=1468, x_max=520, y_max=1494, text="- Lưu: VT, VP."),
            ],
            # Footer right signer name after signature gap (~150px below PHÓ GIÁM ĐỐC)
            [
                TextBox(x_min=1050, y_min=1610, x_max=1350, y_max=1638, text="Nguyễn Văn Tuấn"),
            ],
        ]

        classified, _, body_right = classify_page_lines(lines_boxes, page_width=page_w, config=cfg)
        segments = group_lines_into_segments(
            classified,
            page_number=1,
            page_width=page_w,
            page_height=page_h,
            body_right_margin=body_right,
            config=cfg,
        )

        self.assertEqual(len(segments), 4)
        # 1. Header two_column_table
        self.assertEqual(segments[0].kind, "two_column_table")
        self.assertEqual(
            segments[0].left_lines,
            ["ỦY BAN NHÂN DÂN", "SỞ THÔNG TIN VÀ TRUYỀN THÔNG", "Số: 125/QĐ-STTTT"],
        )
        self.assertEqual(
            segments[0].right_lines,
            ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Hà Nội, ngày 10 tháng 11 năm 2026"],
        )

        # 2. Centered title
        self.assertEqual(segments[1].kind, "paragraph")
        self.assertEqual(segments[1].alignment, "center")
        self.assertEqual(segments[1].text, "QUYẾT ĐỊNH")

        # 3. Merged 3-line justified indented paragraph
        self.assertEqual(segments[2].kind, "paragraph")
        self.assertEqual(segments[2].alignment, "justify")
        self.assertTrue(segments[2].first_line_indent)
        self.assertEqual(
            segments[2].text,
            "Căn cứ Nghị định số 30/2020/NĐ-CP ngày 05 tháng 3 năm 2020 của Chính phủ về công tác văn thư lưu trữ;",
        )

        # 4. Footer two_column_table including Nơi nhận continuation and signature gap
        self.assertEqual(segments[3].kind, "two_column_table")
        self.assertEqual(segments[3].left_lines, ["Nơi nhận:", "- Như Điều 3;", "- Lưu: VT, VP."])
        self.assertEqual(
            segments[3].right_lines,
            ["KT. GIÁM ĐỐC", "PHÓ GIÁM ĐỐC", "", "Nguyễn Văn Tuấn"],
        )

    def test_crop_pillow_patch_and_end_to_end_local_analyzer_with_mocks(self) -> None:
        _ensure_pillow_antialias()
        self.assertTrue(hasattr(Image, "ANTIALIAS"))

        gray_img = np.full((600, 1000), 255, dtype=np.uint8)
        crop = _crop_to_pil_rgb(
            gray_img,
            TextBox(x_min=10, y_min=20, x_max=100, y_max=50),
            padding_px=3,
        )
        self.assertEqual(crop.mode, "RGB")
        self.assertEqual(crop.size, (96, 36))

        # Mock RapidOCR engine and VietOCR predictor running sequentially
        fake_engine = lambda img, **kw: (
            [
                [[100, 50], [350, 50], [350, 80], [100, 80]],
                [[550, 50], [900, 50], [900, 80], [550, 80]],
                [[145, 200], [920, 200], [920, 230], [145, 230]],
                [[100, 240], [920, 240], [920, 270], [100, 270]],
            ],
            [0.05],
        )
        predict_outputs = iter(["SO NOI VU", "CONG HOA XHCN", "Can cu Nghi dinh", "so 3O/2O2O/ND-CP."])
        fake_predictor = SimpleNamespace(predict=lambda pil_img: next(predict_outputs))

        analyzer = LocalOcrLayoutAnalyzer(
            config=OcrLayoutConfig(dpi=200),
            detector=RapidOcrDetector(engine=fake_engine),
            recognizer=VietOcrRecognizer(predictor=fake_predictor),
        )
        pages = [
            PreprocessedPage(
                page_number=1,
                image=gray_img,
                encoded_bytes=b"png",
                mime_type="image/png",
            )
        ]
        layouts = analyzer.process_document(pages)
        self.assertEqual(len(layouts), 1)

        flat_dict, bindings = collect_document_text_batch(layouts)
        self.assertEqual(
            flat_dict,
            {
                "id_1": "SO NOI VU",
                "id_2": "CONG HOA XHCN",
                "id_3": "Can cu Nghi dinh so 3O/2O2O/ND-CP.",
            },
        )

        corrected = {
            "id_1": "SỞ NỘI VỤ",
            "id_2": "CỘNG HÒA XHCN",
            "id_3": "Căn cứ Nghị định số 30/2020/NĐ-CP.",
        }
        doc = build_administrative_document_from_layouts(
            layouts, corrected, bindings, source_pdf="sample.pdf"
        )
        self.assertEqual(doc.body_blocks[0].type, "two_column_table")
        self.assertEqual(doc.body_blocks[0].content, [["SỞ NỘI VỤ", "CỘNG HÒA XHCN"]])
        self.assertEqual(doc.body_blocks[1].type, "paragraph")
        self.assertEqual(doc.body_blocks[1].alignment, "justify")
        self.assertTrue(doc.body_blocks[1].first_line_indent)
        self.assertEqual(doc.body_blocks[1].content, "Căn cứ Nghị định số 30/2020/NĐ-CP.")

        with self.assertRaises(LocalOcrError):
            analyzer.process_document([])

    def test_staggered_top_header_and_empty_page_handling(self) -> None:
        page_w, page_h = 1654, 2339
        cfg = OcrLayoutConfig(dpi=200)
        lines_boxes = [
            # Line 0: staggered left-only line at top of header
            [TextBox(x_min=240, y_min=90, x_max=520, y_max=114, text="BỘ NỘI VỤ")],
            # Line 1: two_column_row just below
            [
                TextBox(x_min=236, y_min=124, x_max=580, y_max=150, text="CỤC VĂN THƯ"),
                TextBox(x_min=860, y_min=124, x_max=1500, y_max=150, text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"),
            ],
        ]
        classified, _, body_right = classify_page_lines(lines_boxes, page_width=page_w, config=cfg)
        segments = group_lines_into_segments(
            classified,
            page_number=1,
            page_width=page_w,
            page_height=page_h,
            body_right_margin=body_right,
            config=cfg,
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].kind, "two_column_table")
        self.assertEqual(segments[0].left_lines, ["BỘ NỘI VỤ", "CỤC VĂN THƯ"])
        self.assertEqual(segments[0].right_lines, ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"])

        # Empty page handling
        empty_classified, _, empty_r = classify_page_lines([], page_width=page_w, config=cfg)
        self.assertEqual(empty_classified, [])
        self.assertEqual(
            group_lines_into_segments(
                empty_classified,
                page_number=1,
                page_width=page_w,
                page_height=page_h,
                body_right_margin=empty_r,
                config=cfg,
            ),
            [],
        )

    def test_parse_rapidocr_boxes_handles_numpy_scalars_in_python_lists(self) -> None:
        # 1. List of 4-pt polygons whose coordinates are np.float32 scalars
        poly_np_list = [
            [np.float32(10.0), np.float32(20.0)],
            [np.float32(210.0), np.float32(20.0)],
            [np.float32(210.0), np.float32(50.0)],
            [np.float32(10.0), np.float32(50.0)],
        ]
        # 2. Flat 4-element bbox with np.int64 scalars
        flat_np_bbox = [np.int64(300), np.int64(20), np.int64(600), np.int64(50)]
        # 3. Wrapped [poly, None, None] when use_rec=False
        wrapped_none = [
            [
                [np.float32(50.0), np.float32(100.0)],
                [np.float32(250.0), np.float32(100.0)],
                [np.float32(250.0), np.float32(130.0)],
                [np.float32(50.0), np.float32(130.0)],
            ],
            None,
            None,
        ]
        boxes = parse_rapidocr_boxes(
            ([poly_np_list, flat_np_bbox, wrapped_none], [0.05]),
            img_width=1654,
            img_height=2339,
        )
        self.assertEqual(len(boxes), 3)
        self.assertEqual((boxes[0].x_min, boxes[0].x_max), (10.0, 210.0))
        self.assertEqual((boxes[1].x_min, boxes[1].x_max), (300.0, 600.0))
        self.assertEqual((boxes[2].x_min, boxes[2].x_max), (50.0, 250.0))
        self.assertEqual(boxes[2].text, "")

    def test_header_footer_outliers_and_sample_pdf_geometry_never_flip_to_right_or_center(self) -> None:
        # Exact 200-DPI layout from sample_van_ban_scan.pdf where header line SỞ THÔNG TIN sits at x=97.2
        page_w, page_h = 1652, 2338
        cfg = OcrLayoutConfig(dpi=200)
        raw_boxes = [
            TextBox(x_min=138.9, y_min=134.8, x_max=412.1, y_max=177.1, text="ỦY BAN NHÂN DÂN"),
            TextBox(x_min=861.1, y_min=134.8, x_max=1434.9, y_max=177.1, text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"),
            TextBox(x_min=111.1, y_min=176.4, x_max=383.6, y_max=218.8, text="THÀNH PHỐ HÀ NỘI"),
            TextBox(x_min=972.2, y_min=176.4, x_max=1298.2, y_max=218.8, text="Độc lập - Tự do - Hạnh phúc"),
            TextBox(x_min=97.2, y_min=221.0, x_max=556.4, y_max=259.5, text="SỞ THÔNG TIN VÀ TRUYỀN THÔNG"),
            TextBox(x_min=180.6, y_min=273.4, x_max=407.2, y_max=314.1, text="Số: 125/QĐ-STTTT"),
            TextBox(x_min=916.7, y_min=276.5, x_max=1347.8, y_max=313.8, text="Hà Nội, ngày 10 tháng 11 năm 2026"),
            TextBox(x_min=680.6, y_min=390.0, x_max=890.1, y_max=443.8, text="QUYẾT ĐỊNH"),
            TextBox(x_min=375.0, y_min=451.3, x_max=1301.2, y_max=497.5, text="Về việc ban hành quy chế tiếp nhận và số hóa"),
            TextBox(x_min=236.1, y_min=551.6, x_max=1138.8, y_max=592.3, text="Căn cứ Luật Tổ chức chính quyền;"),
            TextBox(x_min=236.1, y_min=607.2, x_max=1486.1, y_max=647.9, text="Căn cứ Nghị định số 30/2020/NĐ-CP ngày 05 tháng 3 năm 2020 của Chính phủ;"),
            TextBox(x_min=694.4, y_min=743.0, x_max=885.2, y_max=789.1, text="QUYẾT ĐỊNH:"),
            TextBox(x_min=236.1, y_min=812.1, x_max=1445.3, y_max=856.6, text="Điều 1. Ban hành kèm theo Quyết định này Quy chế số hóa."),
            TextBox(x_min=236.1, y_min=1242.7, x_max=1536.9, y_max=1287.1, text="Điều 3. Chánh Văn phòng, Trưởng các phòng chịu trách nhiệm thi hành."),
            TextBox(x_min=138.9, y_min=1414.7, x_max=253.5, y_max=1454.3, text="Nơi nhận:"),
            TextBox(x_min=1055.6, y_min=1409.6, x_max=1263.9, y_max=1455.8, text="KT. GIÁM ĐỐC"),
            TextBox(x_min=138.9, y_min=1456.9, x_max=267.0, y_max=1493.9, text="- Như Điều 3;"),
            TextBox(x_min=1041.7, y_min=1459.6, x_max=1265.7, y_max=1505.8, text="PHÓ GIÁM ĐỐC"),
            TextBox(x_min=138.9, y_min=1498.5, x_max=441.3, y_max=1535.6, text="- Lưu: VT."),
            TextBox(x_min=1055.6, y_min=1743.0, x_max=1293.6, y_max=1789.1, text="Nguyễn Văn Tuấn"),
        ]
        lines = group_boxes_into_lines(raw_boxes, y_tolerance_px=10.0)
        classified, body_left, body_right = classify_page_lines(lines, page_width=page_w, config=cfg)
        self.assertAlmostEqual(body_left, 138.9, delta=1.0)
        self.assertGreater(body_right, 1480.0)

        by_text = {ln.text: ln for ln in classified}
        self.assertEqual(by_text["QUYẾT ĐỊNH"].alignment, "center")
        self.assertEqual(by_text["Về việc ban hành quy chế tiếp nhận và số hóa"].alignment, "center")
        self.assertEqual(by_text["QUYẾT ĐỊNH:"].alignment, "center")

        for full_line in (
            "Căn cứ Nghị định số 30/2020/NĐ-CP ngày 05 tháng 3 năm 2020 của Chính phủ;",
            "Điều 1. Ban hành kèm theo Quyết định này Quy chế số hóa.",
            "Điều 3. Chánh Văn phòng, Trưởng các phòng chịu trách nhiệm thi hành.",
        ):
            self.assertEqual(by_text[full_line].alignment, "justify")
            self.assertTrue(by_text[full_line].first_line_indent)

        segs = group_lines_into_segments(
            classified,
            page_number=1,
            page_width=page_w,
            page_height=page_h,
            body_right_margin=body_right,
            config=cfg,
        )
        self.assertEqual(segs[0].kind, "two_column_table")
        self.assertEqual(segs[-1].kind, "two_column_table")
        self.assertIn("Nguyễn Văn Tuấn", segs[-1].right_lines)

    def test_indented_line_near_center_deep_indent_short_right_and_tabular_2cell_guard(self) -> None:
        page_w = 1654
        cfg = OcrLayoutConfig(dpi=200)
        lines_boxes = [
            # Standard left & right anchors
            [TextBox(x_min=236, y_min=200, x_max=1520, y_max=226, text="Dòng chuẩn lề trái và chạm lề phải đầy đủ chiều rộng trang.")],
            # 1-line indented paragraph whose midpoint (326+1326)/2=826 falls right on page center (827),
            # must stay left + first_line_indent=True (NOT center!)
            [TextBox(x_min=326, y_min=250, x_max=1326, y_max=276, text="Điều 4. Quyết định này có hiệu lực kể từ ngày ký.")],
            # Deep-indented full-width line (x_min=370 < 0.35*W) touching right margin: must NOT become right!
            [TextBox(x_min=370, y_min=300, x_max=1520, y_max=326, text="a) Nội dung tiểu mục thụt lề sâu chạm sát lề phải văn bản.")],
            # Majority of short lines (> 0.65*W) must not drag body_right down from 1520
            [TextBox(x_min=236, y_min=350, x_max=1100, y_max=376, text="Mục ngắn thứ nhất kết thúc sớm.")],
            [TextBox(x_min=236, y_min=390, x_max=1110, y_max=416, text="Mục ngắn thứ hai kết thúc sớm.")],
            [TextBox(x_min=236, y_min=430, x_max=1240, y_max=456, text="Mục ngắn thứ ba kết thúc trước lề phải khá xa.")],
            # 4-col table with a 2-cell row sandwiched in the middle -> must NOT split into two_column_row
            [
                TextBox(x_min=236, y_min=500, x_max=290, y_max=525, text="STT"),
                TextBox(x_min=450, y_min=500, x_max=650, y_max=525, text="Đơn vị"),
                TextBox(x_min=900, y_min=500, x_max=1050, y_max=525, text="Chỉ tiêu"),
                TextBox(x_min=1300, y_min=500, x_max=1480, y_max=525, text="Ghi chú"),
            ],
            [
                TextBox(x_min=236, y_min=535, x_max=420, y_max=560, text="Tổng cộng"),
                TextBox(x_min=900, y_min=535, x_max=1020, y_max=560, text="100%"),
            ],
            [
                TextBox(x_min=236, y_min=570, x_max=270, y_max=595, text="1"),
                TextBox(x_min=450, y_min=570, x_max=620, y_max=595, text="Văn phòng"),
                TextBox(x_min=900, y_min=570, x_max=1020, y_max=595, text="100%"),
                TextBox(x_min=1300, y_min=570, x_max=1450, y_max=595, text="Đạt"),
            ],
        ]
        classified, body_left, body_right = classify_page_lines(lines_boxes, page_width=page_w, config=cfg)
        self.assertAlmostEqual(body_left, 236.0, delta=1.0)
        self.assertAlmostEqual(body_right, 1520.0, delta=1.0)

        self.assertEqual(classified[1].alignment, "left")
        self.assertTrue(classified[1].first_line_indent)

        self.assertEqual(classified[2].alignment, "justify")
        self.assertNotEqual(classified[2].alignment, "right")

        self.assertEqual(classified[5].alignment, "left")

        # Sandwiched 2-cell table row must stay single tabular row
        self.assertEqual(classified[7].line_type, "single")
        self.assertTrue(classified[7].is_tabular_row)

    def test_staggered_header_and_footer_beyond_10px_and_tall_signature_gap(self) -> None:
        page_w, page_h = 1654, 2339
        cfg = OcrLayoutConfig(dpi=200)
        # Left and right columns offset vertically by 18px (> 10px tolerance -> 0 two_column_row lines!)
        raw_boxes = [
            TextBox(x_min=140, y_min=120, x_max=520, y_max=146, text="ỦY BAN NHÂN DÂN TỈNH X"),
            TextBox(x_min=860, y_min=138, x_max=1480, y_max=164, text="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"),
            TextBox(x_min=180, y_min=168, x_max=480, y_max=194, text="Số: 12/QĐ-UBND"),
            TextBox(x_min=960, y_min=186, x_max=1360, y_max=212, text="Độc lập - Tự do - Hạnh phúc"),
            # Body line
            TextBox(x_min=236, y_min=360, x_max=1520, y_max=388, text="Điều 1. Nội dung văn bản hành chính chuẩn."),
            # Staggered footer with tall 350px signature gap
            TextBox(x_min=140, y_min=1380, x_max=280, y_max=1408, text="Nơi nhận:"),
            TextBox(x_min=1020, y_min=1398, x_max=1320, y_max=1426, text="KT. CHỦ TỊCH"),
            TextBox(x_min=140, y_min=1424, x_max=380, y_max=1450, text="- Như Điều 1;"),
            TextBox(x_min=1010, y_min=1442, x_max=1330, y_max=1470, text="PHÓ CHỦ TỊCH"),
            TextBox(x_min=140, y_min=1466, x_max=340, y_max=1492, text="- Lưu: VT."),
            TextBox(x_min=1030, y_min=1820, x_max=1310, y_max=1850, text="Trần Văn Bình"),
        ]
        lines = group_boxes_into_lines(raw_boxes, y_tolerance_px=10.0)
        classified, _, body_right = classify_page_lines(lines, page_width=page_w, config=cfg)
        self.assertTrue(all(ln.line_type == "single" for ln in classified))

        segments = group_lines_into_segments(
            classified,
            page_number=1,
            page_width=page_w,
            page_height=page_h,
            body_right_margin=body_right,
            config=cfg,
        )
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].kind, "two_column_table")
        self.assertEqual(segments[0].left_lines, ["ỦY BAN NHÂN DÂN TỈNH X", "Số: 12/QĐ-UBND"])
        self.assertEqual(segments[0].right_lines, ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc"])
        self.assertEqual(segments[1].kind, "paragraph")
        self.assertEqual(segments[2].kind, "two_column_table")
        self.assertEqual(segments[2].left_lines, ["Nơi nhận:", "- Như Điều 1;", "- Lưu: VT."])
        self.assertEqual(segments[2].right_lines, ["KT. CHỦ TỊCH", "PHÓ CHỦ TỊCH", "", "Trần Văn Bình"])

    def test_vietocr_pytorch_weights_only_compat_context(self) -> None:
        observed_kwargs: list[dict] = []

        def fake_load(path, weights_only=True):
            observed_kwargs.append({"path": path, "weights_only": weights_only})
            return {"state": 1}

        fake_torch = SimpleNamespace(load=fake_load)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with _compat_torch_load_for_vietocr():
                res = fake_torch.load("model.pth")
                self.assertEqual(res, {"state": 1})
            self.assertIs(fake_torch.load, fake_load)
        self.assertEqual(observed_kwargs, [{"path": "model.pth", "weights_only": False}])

    def test_vietocr_pkg_resources_shim_offline_config_and_cache(self) -> None:
        # 1. _ensure_pkg_resources_shim works when pkg_resources is absent (setuptools>=82)
        with patch.dict(sys.modules, {"pkg_resources": None}):
            with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module named 'pkg_resources'")):
                _ensure_pkg_resources_shim()
            shim = sys.modules.get("pkg_resources")
            self.assertIsNotNone(shim)
            dist = shim.get_distribution("nonexistent-pkg-xyz-123")
            self.assertEqual(dist.version, "0.0.0")
            dist_torch = shim.get_distribution("pydantic")
            self.assertTrue(bool(dist_torch.version))

        # 2. _load_vietocr_config loads vgg_transformer completely offline & writes local YAML cache
        class FakeCfg(dict):
            load_config_from_name = MagicMock(side_effect=AssertionError("Must not call online fetch"))
            load_config_from_file = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cfg = _load_vietocr_config("vgg_transformer", FakeCfg, cache_dir=cache_dir)
            FakeCfg.load_config_from_name.assert_not_called()
            self.assertEqual(cfg["backbone"], "vgg19_bn")
            self.assertEqual(len(cfg["vocab"]), 229)
            self.assertTrue((cache_dir / "base.yml").is_file())
            self.assertTrue((cache_dir / "vgg_transformer.yml").is_file())

            # 3. Non-builtin config is fetched once and served from local YAML cache offline thereafter
            online_mock = MagicMock(return_value={"backbone": "resnet50", "vocab": "abc"})

            class CustomFakeCfg(dict):
                load_config_from_name = online_mock
                load_config_from_file = MagicMock()

            custom_cfg_1 = _load_vietocr_config("resnet_transformer", CustomFakeCfg, cache_dir=cache_dir)
            self.assertEqual(custom_cfg_1["backbone"], "resnet50")
            self.assertEqual(online_mock.call_count, 1)

            custom_cfg_2 = _load_vietocr_config("resnet_transformer", CustomFakeCfg, cache_dir=cache_dir)
            self.assertEqual(custom_cfg_2["backbone"], "resnet50")
            self.assertEqual(online_mock.call_count, 1)  # served from cache!

            # 4. _resolve_local_vietocr_weights finds local cached .pth in tempdir
            fake_pth = Path(tmp) / "unit_test_vgg_transformer.pth"
            fake_pth.write_bytes(b"weights")
            with patch("tempfile.gettempdir", return_value=tmp):
                resolved = _resolve_local_vietocr_weights(
                    "https://vocr.vn/data/vietocr/unit_test_vgg_transformer.pth"
                )
            self.assertEqual(resolved, str(fake_pth))

    def test_three_column_table_continuation_rows_never_become_two_column_tables(self) -> None:
        page_w, page_h = 1654, 2339
        cfg = OcrLayoutConfig(dpi=200)

        # Page 1: Genuine 2-col header + title + 3-column table with narrow STT gutter (~45px)
        # and continuation rows (including 2-cluster rows like `3 | 450.000`)
        page1_boxes = [
            # Genuine top-of-page 2-col header
            TextBox(x_min=150, y_min=110, x_max=340, y_max=138, text="DANH MỤC"),
            TextBox(x_min=1150, y_min=110, x_max=1450, y_max=138, text="Phụ lục XV-1A"),
            # Centered titles
            TextBox(x_min=520, y_min=180, x_max=1134, y_max=208, text="GIÁ DỊCH VỤ KHÁM BỆNH, HỘI CHẨN"),
            TextBox(x_min=360, y_min=220, x_max=1294, y_max=248, text="Đơn vị: Khối khám chữa bệnh thuộc Trung tâm y tế"),
            TextBox(x_min=290, y_min=260, x_max=1364, y_max=288, text="(Ban hành kèm theo Nghị quyết số 120/2024/NQ-HĐND ngày 11 tháng 12 năm 2024"),
            TextBox(x_min=560, y_min=296, x_max=1094, y_max=324, text="của Hội đồng nhân dân tỉnh Cao Bằng)"),
            # Right-aligned unit caption above table
            TextBox(x_min=1260, y_min=350, x_max=1480, y_max=376, text="Đơn vị: đồng"),
            # 3-column table header with 45px gutter between STT and Danh mục dịch vụ
            TextBox(x_min=150, y_min=395, x_max=215, y_max=422, text="STT"),
            TextBox(x_min=260, y_min=395, x_max=560, y_max=422, text="Danh mục dịch vụ"),
            TextBox(x_min=1290, y_min=395, x_max=1450, y_max=422, text="Mức giá"),
            # Column numbering row 1 | 2 | 3
            TextBox(x_min=175, y_min=434, x_max=195, y_max=458, text="1"),
            TextBox(x_min=720, y_min=434, x_max=740, y_max=458, text="2"),
            TextBox(x_min=1360, y_min=434, x_max=1380, y_max=458, text="3"),
            # Section row (col 0 + col 1 only)
            TextBox(x_min=175, y_min=472, x_max=195, y_max=498, text="1"),
            TextBox(x_min=250, y_min=472, x_max=680, y_max=498, text="Danh mục dịch vụ khám bệnh"),
            # Row 1 (col 0 + col 1 merged by tight gap or separate + col 2)
            TextBox(x_min=175, y_min=510, x_max=460, y_max=538, text="1 Giá Khám bệnh"),
            TextBox(x_min=1310, y_min=510, x_max=1430, y_max=538, text="39.800"),
            # Long wrapped col 1 lines
            TextBox(x_min=250, y_min=550, x_max=1220, y_max=576, text="Danh mục dịch vụ khám bệnh, chữa bệnh không thuộc danh mục do quỹ bảo hiểm y tế"),
            TextBox(x_min=250, y_min=586, x_max=1140, y_max=612, text="thanh toán mà không phải là dịch vụ khám bệnh, chữa bệnh theo yêu cầu"),
            TextBox(x_min=250, y_min=624, x_max=1210, y_max=650, text="Khám cấp giấy chứng thương, giám định y khoa (không kể xét"),
            # Staggered STT "1" + 2-cluster continuation line "nghiệm, X-quang) | 160.000"
            TextBox(x_min=175, y_min=642, x_max=195, y_max=668, text="1"),
            TextBox(x_min=250, y_min=660, x_max=540, y_max=686, text="nghiệm, X-quang)"),
            TextBox(x_min=1305, y_min=660, x_max=1435, y_max=686, text="160.000"),
            # Next wrapped row with 2-cluster row "2 | (không kể xét nghiệm, X-quang) | 160.000"
            TextBox(x_min=250, y_min=700, x_max=1210, y_max=726, text="Khám sức khỏe toàn diện lao động, lái xe, khám sức khỏe định kỳ"),
            TextBox(x_min=175, y_min=736, x_max=195, y_max=762, text="2"),
            TextBox(x_min=250, y_min=736, x_max=670, y_max=762, text="(không kể xét nghiệm, X-quang)"),
            TextBox(x_min=1305, y_min=736, x_max=1435, y_max=762, text="160.000"),
            # Row 3 with blank col 2 on that Y-line: "3 | 450.000"
            TextBox(x_min=250, y_min=774, x_max=1210, y_max=800, text="Khám sức khỏe toàn diện cho người đi xuất khẩu lao động (không"),
            TextBox(x_min=175, y_min=810, x_max=195, y_max=836, text="3"),
            TextBox(x_min=1305, y_min=810, x_max=1435, y_max=836, text="450.000"),
            TextBox(x_min=250, y_min=846, x_max=560, y_max=872, text="kể xét nghiệm, X-quang)"),
        ]

        # Narrow column gutter in cluster_line_boxes must keep STT and Danh mục separate
        hdr_clusters = cluster_line_boxes(page1_boxes[7:10], page_w)
        self.assertEqual(len(hdr_clusters), 3)

        p1_lines = group_boxes_into_lines(page1_boxes, y_tolerance_px=10.0)
        p1_cls, _, p1_r = classify_page_lines(p1_lines, page_width=page_w, config=cfg)
        p1_segs = group_lines_into_segments(
            p1_cls, page_number=1, page_width=page_w, page_height=page_h, body_right_margin=p1_r, config=cfg
        )
        two_col_segs_p1 = [s for s in p1_segs if s.kind == "two_column_table"]
        self.assertEqual(len(two_col_segs_p1), 1)
        self.assertEqual(two_col_segs_p1[0].left_lines, ["DANH MỤC"])
        self.assertEqual(two_col_segs_p1[0].right_lines, ["Phụ lục XV-1A"])
        # No table row may have false first_line_indent=True
        for seg in p1_segs[6:]:
            self.assertEqual(seg.kind, "paragraph")
            self.assertFalse(seg.first_line_indent)

        # Page 2: Fused header fallback ("STT Danh mục dịch vụ | Mức giá") + "Đơn vị: đồng" above
        page2_boxes = [
            TextBox(x_min=1180, y_min=120, x_max=1460, y_max=148, text="Phụ lục XV-1B"),
            TextBox(x_min=520, y_min=170, x_max=1134, y_max=198, text="GIÁ DỊCH VỤ KHÁM BỆNH, HỘI CHẨN"),
            TextBox(x_min=1260, y_min=230, x_max=1480, y_max=256, text="Đơn vị: đồng"),
            TextBox(x_min=150, y_min=270, x_max=560, y_max=298, text="STT Danh mục dịch vụ"),
            TextBox(x_min=1290, y_min=270, x_max=1450, y_max=298, text="Mức giá"),
            TextBox(x_min=250, y_min=312, x_max=510, y_max=340, text="Giá Khám bệnh"),
            TextBox(x_min=1310, y_min=312, x_max=1430, y_max=340, text="36.500"),
        ]
        p2_lines = group_boxes_into_lines(page2_boxes, y_tolerance_px=10.0)
        p2_cls, _, p2_r = classify_page_lines(p2_lines, page_width=page_w, config=cfg)
        p2_segs = group_lines_into_segments(
            p2_cls, page_number=2, page_width=page_w, page_height=page_h, body_right_margin=p2_r, config=cfg
        )
        self.assertEqual([s for s in p2_segs if s.kind == "two_column_table"], [])
        self.assertTrue(any(s.text == "Đơn vị: đồng" and s.alignment == "right" for s in p2_segs))

        # Page 3: 3-column table followed directly by genuine 2-col footer (`Nơi nhận: | KT. GIÁM ĐỐC`)
        page3_boxes = [
            TextBox(x_min=150, y_min=300, x_max=215, y_max=328, text="STT"),
            TextBox(x_min=260, y_min=300, x_max=560, y_max=328, text="Các loại dịch vụ"),
            TextBox(x_min=1290, y_min=300, x_max=1450, y_max=328, text="Mức giá"),
            TextBox(x_min=175, y_min=345, x_max=195, y_max=372, text="1"),
            TextBox(x_min=260, y_min=345, x_max=780, y_max=372, text="Ngày giường bệnh Hồi sức cấp cứu"),
            TextBox(x_min=1305, y_min=345, x_max=1435, y_max=372, text="364.400"),
            TextBox(x_min=175, y_min=385, x_max=225, y_max=412, text="2.2"),
            TextBox(x_min=1305, y_min=385, x_max=1435, y_max=412, text="211.000"),
            # Genuine administrative footer directly below the table
            TextBox(x_min=150, y_min=1400, x_max=280, y_max=1428, text="Nơi nhận:"),
            TextBox(x_min=1050, y_min=1400, x_max=1280, y_max=1428, text="KT. GIÁM ĐỐC"),
            TextBox(x_min=150, y_min=1438, x_max=360, y_max=1466, text="- Lưu: VT."),
            TextBox(x_min=1040, y_min=1438, x_max=1290, y_max=1466, text="PHÓ GIÁM ĐỐC"),
            TextBox(x_min=1060, y_min=1600, x_max=1280, y_max=1628, text="Nguyễn Văn A"),
        ]
        p3_lines = group_boxes_into_lines(page3_boxes, y_tolerance_px=10.0)
        p3_cls, _, p3_r = classify_page_lines(p3_lines, page_width=page_w, config=cfg)
        p3_segs = group_lines_into_segments(
            p3_cls, page_number=3, page_width=page_w, page_height=page_h, body_right_margin=p3_r, config=cfg
        )
        two_col_segs_p3 = [s for s in p3_segs if s.kind == "two_column_table"]
        self.assertEqual(len(two_col_segs_p3), 1)
        self.assertEqual(two_col_segs_p3[0].left_lines, ["Nơi nhận:", "- Lưu: VT."])
        self.assertEqual(two_col_segs_p3[0].right_lines, ["KT. GIÁM ĐỐC", "PHÓ GIÁM ĐỐC", "", "Nguyễn Văn A"])

    def test_staggered_stt_price_wrapped_col_same_page_multi_table_and_staggered_footer(self) -> None:
        page_w, page_h = 1654, 2339
        cfg = OcrLayoutConfig(dpi=200)

        boxes = [
            TextBox(x_min=150, y_min=110, x_max=340, y_max=138, text="DANH MỤC"),
            TextBox(x_min=1150, y_min=110, x_max=1450, y_max=138, text="Phụ lục XV-2A"),
            TextBox(x_min=520, y_min=160, x_max=1134, y_max=188, text="GIÁ DỊCH VỤ NGÀY GIƯỜNG BỆNH"),
            TextBox(x_min=1260, y_min=220, x_max=1480, y_max=246, text="Đơn vị: đồng"),
            # Staggered STT above Các loại dịch vụ | Mức giá (no 3-cluster row at all)
            TextBox(x_min=150, y_min=255, x_max=215, y_max=280, text="STT"),
            TextBox(x_min=260, y_min=275, x_max=560, y_max=302, text="Các loại dịch vụ"),
            TextBox(x_min=1290, y_min=275, x_max=1450, y_max=302, text="Mức giá"),
            TextBox(x_min=175, y_min=320, x_max=780, y_max=348, text="1 Ngày giường bệnh Hồi sức cấp cứu"),
            TextBox(x_min=1305, y_min=320, x_max=1435, y_max=348, text="364.400"),
            TextBox(x_min=250, y_min=365, x_max=1180, y_max=392, text="Loại 1: Các khoa: Truyền nhiễm, Hô hấp, Huyết học, Ung"),
            TextBox(x_min=175, y_min=405, x_max=225, y_max=432, text="2.1"),
            TextBox(x_min=1305, y_min=425, x_max=1435, y_max=452, text="245.000"),
            # Midpoint of col-1 wrapped line (250+1200)/2 = 725 falls within ±8% of page center; must NOT break table
            TextBox(x_min=250, y_min=460, x_max=1200, y_max=488, text="Thận học; nội tiết; Dị ứng (đối với bệnh nhân dị ứng thuốc"),
            TextBox(x_min=175, y_min=500, x_max=225, y_max=528, text="2.2"),
            TextBox(x_min=1305, y_min=500, x_max=1435, y_max=528, text="211.000"),
            # Second annex block on the SAME page
            TextBox(x_min=150, y_min=700, x_max=340, y_max=728, text="DANH MỤC"),
            TextBox(x_min=1150, y_min=700, x_max=1450, y_max=728, text="Phụ lục XV-2B"),
            TextBox(x_min=560, y_min=750, x_max=1094, y_max=778, text="DỊCH VỤ PHẪU THUẬT"),
            # Second table without STT or 3-cluster rows (2-cluster price table)
            TextBox(x_min=160, y_min=820, x_max=540, y_max=848, text="Tên phẫu thuật"),
            TextBox(x_min=1290, y_min=820, x_max=1450, y_max=848, text="Đơn giá"),
            TextBox(x_min=160, y_min=865, x_max=680, y_max=893, text="Phẫu thuật loại đặc biệt"),
            TextBox(x_min=1285, y_min=865, x_max=1445, y_max=893, text="2.500.000"),
            TextBox(x_min=160, y_min=908, x_max=560, y_max=936, text="Phẫu thuật loại I"),
            TextBox(x_min=1285, y_min=908, x_max=1445, y_max=936, text="1.800.000"),
            # Staggered footer where KT. GIÁM ĐỐC sits 18px above Nơi nhận:
            TextBox(x_min=1050, y_min=1400, x_max=1280, y_max=1428, text="KT. GIÁM ĐỐC"),
            TextBox(x_min=150, y_min=1418, x_max=280, y_max=1446, text="Nơi nhận:"),
            TextBox(x_min=150, y_min=1456, x_max=360, y_max=1484, text="- Lưu: VT."),
            TextBox(x_min=1060, y_min=1600, x_max=1280, y_max=1628, text="Nguyễn Văn B"),
        ]
        lines = group_boxes_into_lines(boxes, y_tolerance_px=10.0)
        cls, _, br = classify_page_lines(lines, page_width=page_w, config=cfg)
        segs = group_lines_into_segments(
            cls, page_number=1, page_width=page_w, page_height=page_h, body_right_margin=br, config=cfg
        )
        two_cols = [s for s in segs if s.kind == "two_column_table"]
        self.assertEqual(len(two_cols), 3)
        self.assertEqual((two_cols[0].left_lines, two_cols[0].right_lines), (["DANH MỤC"], ["Phụ lục XV-2A"]))
        self.assertEqual((two_cols[1].left_lines, two_cols[1].right_lines), (["DANH MỤC"], ["Phụ lục XV-2B"]))
        self.assertEqual(two_cols[2].left_lines, ["Nơi nhận:", "- Lưu: VT."])
        self.assertEqual(two_cols[2].right_lines, ["KT. GIÁM ĐỐC", "", "Nguyễn Văn B"])

        # Table rows must stay paragraphs with first_line_indent=False
        table_para_texts = [s.text for s in segs if s.kind == "paragraph" and not s.first_line_indent]
        self.assertIn("STT", table_para_texts)
        self.assertIn("Các loại dịch vụ Mức giá", table_para_texts)
        self.assertIn("2.2 211.000", table_para_texts)
        self.assertIn("Phẫu thuật loại đặc biệt 2.500.000", table_para_texts)
        for ln in cls:
            if "Thận học" in ln.text or "Loại 1" in ln.text:
                self.assertTrue(ln.is_tabular_row)
                self.assertFalse(ln.first_line_indent)

    def test_real_offline_rapidocr_vietocr_after_pyqt5_import(self) -> None:
        import importlib.util
        from vn_pdf_to_docx.pipeline import convert_pdf_to_docx
        from vn_pdf_to_docx.preprocessor import PreprocessConfig

        sample_pdf = Path(__file__).resolve().parent.parent / "sample_van_ban_scan.pdf"
        weights_pth = Path(tempfile.gettempdir()) / "vgg_transformer.pth"
        if (
            not sample_pdf.is_file()
            or not weights_pth.is_file()
            or importlib.util.find_spec("rapidocr_onnxruntime") is None
            or importlib.util.find_spec("vietocr") is None
            or importlib.util.find_spec("torch") is None
        ):
            self.skipTest("Offline RapidOCR/VietOCR dependencies or sample PDF not present.")

        with tempfile.TemporaryDirectory() as tmp:
            out_docx = Path(tmp) / "real_offline.docx"
            result = convert_pdf_to_docx(
                sample_pdf,
                out_docx,
                preprocess_config=PreprocessConfig(dpi=200),
            )
            self.assertTrue(result.output_docx.is_file())
            self.assertGreater(result.output_docx.stat().st_size, 10_000)
            self.assertEqual(len(result.page_layouts), 1)
            self.assertGreaterEqual(len(result.raw_text_batch), 20)
            blocks = result.document.body_blocks
            self.assertEqual(blocks[0].type, "two_column_table")
            self.assertEqual(blocks[-1].type, "two_column_table")
            self.assertTrue(any(b.type == "paragraph" and b.first_line_indent for b in blocks))

    def test_resolve_vietocr_device_auto_and_explicit(self) -> None:
        fake_torch_gpu = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda idx=0: "NVIDIA GeForce GTX 1650",
            )
        )
        fake_torch_cpu = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )

        with patch.dict(sys.modules, {"torch": fake_torch_gpu}):
            self.assertEqual(resolve_vietocr_device("auto"), "cuda:0")
            self.assertEqual(resolve_vietocr_device("cuda"), "cuda:0")
            self.assertEqual(resolve_vietocr_device("cpu"), "cpu")
            self.assertEqual(describe_vietocr_device("auto"), "GPU CUDA (NVIDIA GeForce GTX 1650)")

        with patch.dict(sys.modules, {"torch": fake_torch_cpu}):
            self.assertEqual(resolve_vietocr_device("auto"), "cpu")
            self.assertEqual(describe_vietocr_device("auto"), "CPU")

    def test_vietocr_recognize_boxes_batches_and_falls_back_on_error(self) -> None:
        page_img = np.full((120, 400, 3), 255, dtype=np.uint8)
        boxes = [
            TextBox(x_min=10, y_min=10, x_max=80, y_max=35),
            TextBox(x_min=90, y_min=10, x_max=160, y_max=35),
            TextBox(x_min=10, y_min=50, x_max=80, y_max=75),
        ]

        # 1. Batched predict_batch chunked by batch_size=2
        batch_calls: list[int] = []

        class BatchPredictor:
            def predict_batch(self, imgs):
                batch_calls.append(len(imgs))
                # Include decomposed Unicode to ensure NFC normalization runs
                return ["Co\u0323ng" if i == 0 else f"dong_{i}" for i in range(len(imgs))]

            def predict(self, img):
                raise AssertionError("Single predict should not run when predict_batch succeeds")

        rec = VietOcrRecognizer(batch_size=2, predictor=BatchPredictor())
        texts = rec.recognize_boxes(page_img, boxes, padding_px=2)
        self.assertEqual(batch_calls, [2, 1])
        self.assertEqual(texts, ["Cọng", "dong_1", "Cọng"])

        # 2. If predict_batch raises (e.g. simulated CUDA OOM), falls back to single predict per chunk
        single_calls = 0

        class FallbackPredictor:
            def predict_batch(self, imgs):
                raise RuntimeError("CUDA out of memory")

            def predict(self, img):
                nonlocal single_calls
                single_calls += 1
                return f"fallback_{single_calls}"

        rec_fb = VietOcrRecognizer(batch_size=2, predictor=FallbackPredictor())
        fb_texts = rec_fb.recognize_boxes(page_img, boxes, padding_px=2)
        self.assertEqual(single_calls, 3)
        self.assertEqual(fb_texts, ["fallback_1", "fallback_2", "fallback_3"])


if __name__ == "__main__":
    unittest.main()
