"""OpenCV Morphological Grid Table Detection, Cell Mapping, and Column Projection Fallback."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import cv2
import numpy as np

from .ocr_models import DetectedLine, TextBox

_PRICE_VALUE_RE = re.compile(
    r"^\s*\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\s*(?:%|đ|đồng)?\s*$",
    re.IGNORECASE,
)
_STT_VALUE_RE = re.compile(r"^\s*(?:[IVX]+|\d+(?:\.\d+)*)\s*[.)]?\s*$", re.IGNORECASE)
_STT_HEADER_RE = re.compile(r"(?:^|\b)(?:stt|số\s*tt|tt)(?:\b|$)", re.IGNORECASE)
_TABLE_COMPANION_KEYWORDS = (
    "danh mục",
    "đơn giá",
    "mức giá",
    "thành tiền",
    "nội dung",
    "tên đơn vị",
    "chỉ tiêu",
    "loại dịch vụ",
    "dịch vụ",
    "mã tương",
    "tên dịch vụ",
    "phê duyệt giá",
    "ghi chú",
)
_GROUP_HEADING_KEYWORDS = (
    "danh mục dịch vụ",
    "không thuộc danh mục",
    "theo yêu cầu",
    "nhóm dịch vụ",
)
_FOOTER_STOP_PREFIXES = (
    "nơi nhận",
    "kt.",
    "tm.",
    "tl.",
    "chủ tịch",
    "phó chủ tịch",
    "giám đốc",
    "phó giám đốc",
    "thủ trưởng",
    "người ký",
    "điều ",
)


@dataclass
class DetectedTableGrid:
    """Grid geometry discovered on a page image by OpenCV morphological line filtering."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    col_xs: list[float]
    row_ys: list[float]
    row_colspans: list[list[int]] = field(default_factory=list)

    @property
    def num_cols(self) -> int:
        return max(0, len(self.col_xs) - 1)

    @property
    def num_rows(self) -> int:
        return max(0, len(self.row_ys) - 1)

    def contains_center(
        self,
        cx: float,
        cy: float,
        *,
        x_pad: float = 6.0,
        top_pad: float = 0.0,
        bottom_pad: float = 4.0,
    ) -> bool:
        """Check whether `(cx, cy)` falls inside the detected grid bounds."""
        left_edge = self.col_xs[0] if self.col_xs else self.x_min
        right_edge = self.col_xs[-1] if self.col_xs else self.x_max
        top_edge = self.row_ys[0] if self.row_ys else self.y_min
        bottom_edge = self.row_ys[-1] if self.row_ys else self.y_max
        return (
            (left_edge - x_pad) <= cx <= (right_edge + x_pad)
            and (top_edge - top_pad) <= cy <= (bottom_edge + bottom_pad)
        )


def _to_grayscale(image: np.ndarray, *, color_order: Literal["RGB", "BGR"] = "RGB") -> np.ndarray:
    """Convert 2D/3D/4D page image to 2D uint8 grayscale for OpenCV thresholding."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if arr.shape[2] == 3:
            code = cv2.COLOR_RGB2GRAY if color_order == "RGB" else cv2.COLOR_BGR2GRAY
            return cv2.cvtColor(arr, code)
        if arr.shape[2] == 4:
            code = cv2.COLOR_RGBA2GRAY if color_order == "RGB" else cv2.COLOR_BGRA2GRAY
            return cv2.cvtColor(arr, code)
    raise ValueError(f"Unsupported image shape for table detection: {arr.shape}")


def _cluster_1d_positions(positions: Sequence[float], min_gap: float) -> list[float]:
    """Merge nearby 1D coordinates within `min_gap` px into sorted cluster centers."""
    if not positions:
        return []
    ordered = sorted(float(p) for p in positions)
    clusters: list[list[float]] = [[ordered[0]]]
    for val in ordered[1:]:
        if (val - clusters[-1][-1]) <= min_gap:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    return [float(sum(grp) / len(grp)) for grp in clusters]


def _extract_axis_lines_from_mask(
    roi_mask: np.ndarray,
    *,
    offset: int,
    axis: Literal["horizontal", "vertical"],
    min_span_px: float,
    merge_gap_px: float,
) -> list[float]:
    """Extract sorted row Y or column X line positions tolerant of mild scan skew."""
    if roi_mask.size == 0:
        return []

    if axis == "horizontal":
        thick = cv2.dilate(roi_mask, np.ones((7, 3), np.uint8))
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thick)
        candidates: list[float] = []
        for lbl in range(1, num_labels):
            span = float(stats[lbl, cv2.CC_STAT_WIDTH])
            if span >= min_span_px:
                candidates.append(float(offset) + float(centroids[lbl][1]))
        return _cluster_1d_positions(candidates, min_gap=merge_gap_px)

    thick = cv2.dilate(roi_mask, np.ones((3, 7), np.uint8))
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thick)
    candidates = []
    for lbl in range(1, num_labels):
        span = float(stats[lbl, cv2.CC_STAT_HEIGHT])
        if span >= min_span_px:
            candidates.append(float(offset) + float(centroids[lbl][0]))
    return _cluster_1d_positions(candidates, min_gap=merge_gap_px)


def _maybe_prepend_open_top_header_row(
    v_roi: np.ndarray,
    *,
    roi_x: int,
    roi_y: int,
    col_xs: Sequence[float],
    row_ys: list[float],
    dpi: int,
) -> list[float]:
    """If vertical column dividers extend >= 40px above `row_ys[0]`, restore clipped top edge."""
    if not row_ys or len(col_xs) < 2 or v_roi.size == 0:
        return row_ys

    scale = dpi / 200.0
    roi_h, roi_w = v_roi.shape[:2]
    row0_local = int(round(row_ys[0] - roi_y))
    min_open_h = max(40.0 * scale, 28.0)
    if row0_local < min_open_h:
        return row_ys

    x_half_band = max(12, int(round(20 * scale)))
    v_ys = np.where(v_roi > 0)[0]
    if v_ys.size == 0:
        return row_ys

    top_local = int(v_ys.min())
    if (row0_local - top_local) < min_open_h:
        return row_ys

    band_slice = v_roi[top_local : max(top_local + 1, row0_local - max(3, int(round(4 * scale)))), :]
    if band_slice.shape[0] < 12:
        return row_ys

    supporting_cols = 0
    for cx in col_xs:
        cx_local = int(round(cx - roi_x))
        bx1 = max(0, cx_local - x_half_band)
        bx2 = min(roi_w, cx_local + x_half_band + 1)
        col_band = band_slice[:, bx1:bx2]
        if col_band.size > 0:
            cov = float((col_band > 0).any(axis=1).sum()) / float(band_slice.shape[0])
            if cov >= 0.35:
                supporting_cols += 1

    if supporting_cols >= 2:
        return [float(roi_y + top_local), *row_ys]
    return row_ys


def _compute_row_colspans_from_vertical_lines(
    v_roi: np.ndarray,
    *,
    roi_x: int,
    roi_y: int,
    col_xs: Sequence[float],
    row_ys: Sequence[float],
    dpi: int,
) -> list[list[int]]:
    """Inspect vertical line ink coverage inside each row to detect horizontal merged cells."""
    num_cols = len(col_xs) - 1
    num_rows = len(row_ys) - 1
    if num_cols <= 0 or num_rows <= 0:
        return []

    roi_h, roi_w = v_roi.shape[:2]
    scale = dpi / 200.0
    v_pad = max(3, int(round(4 * scale)))
    x_half_band = max(14, int(round(28 * scale)))
    all_row_spans: list[list[int]] = []

    for r in range(num_rows):
        ry1 = int(round(row_ys[r] - roi_y)) + v_pad
        ry2 = int(round(row_ys[r + 1] - roi_y)) - v_pad
        ry1 = max(0, min(roi_h, ry1))
        ry2 = max(0, min(roi_h, ry2))
        if (ry2 - ry1) < 6 or num_cols == 1:
            all_row_spans.append([1] * num_cols)
            continue

        interior = v_roi[ry1:ry2, :]
        has_divider: list[bool] = []
        for div_x in col_xs[1:-1]:
            cx_local = int(round(div_x - roi_x))
            bx1 = max(0, cx_local - x_half_band)
            bx2 = min(roi_w, cx_local + x_half_band + 1)
            band = interior[:, bx1:bx2]
            if band.size == 0:
                has_divider.append(False)
            else:
                coverage = float((band > 0).any(axis=1).sum()) / float(max(1, interior.shape[0]))
                has_divider.append(coverage >= 0.25)

        spans: list[int] = [1]
        for present in has_divider:
            if present:
                spans.append(1)
            else:
                spans[-1] += 1
        all_row_spans.append(spans)

    return all_row_spans


def detect_table_grids(
    image: np.ndarray,
    *,
    dpi: int = 200,
    color_order: Literal["RGB", "BGR"] = "RGB",
) -> list[DetectedTableGrid]:
    """Detect bordered tables via OpenCV adaptiveThreshold + (40,1)/(1,40) morphological lines."""
    gray = _to_grayscale(image, color_order=color_order)
    img_h, img_w = gray.shape[:2]
    if img_h < 60 or img_w < 120:
        return []

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    scale = dpi / 200.0
    k_len = max(20, int(round(40 * scale)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))

    horizontal_lines = cv2.dilate(cv2.erode(binary, h_kernel), h_kernel)
    vertical_lines = cv2.dilate(cv2.erode(binary, v_kernel), v_kernel)

    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    join_kernel = np.ones((max(3, int(round(5 * scale))), max(3, int(round(5 * scale)))), np.uint8)
    grid_joined = cv2.dilate(grid_mask, join_kernel)

    contours, _ = cv2.findContours(grid_joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_table_w = max(120.0 * scale, 0.25 * float(img_w))
    min_table_h = max(36.0 * scale, 20.0)

    detected: list[DetectedTableGrid] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if float(w) < min_table_w or float(h) < min_table_h:
            continue

        h_roi = horizontal_lines[y : y + h, x : x + w]
        v_roi = vertical_lines[y : y + h, x : x + w]

        # Must contain genuine line intersections (filters isolated rules or divider bars)
        inter_mask = cv2.bitwise_and(h_roi, v_roi)
        num_inter = cv2.connectedComponents(inter_mask)[0] - 1
        if num_inter < 4:
            continue

        row_ys = _extract_axis_lines_from_mask(
            h_roi,
            offset=y,
            axis="horizontal",
            min_span_px=max(60.0 * scale, 0.18 * float(w)),
            merge_gap_px=max(10.0 * scale, 8.0),
        )
        if len(row_ys) < 2:
            continue

        row_heights = [row_ys[i + 1] - row_ys[i] for i in range(len(row_ys) - 1)]
        min_row_h = float(min(row_heights)) if row_heights else 25.0 * scale
        col_xs = _extract_axis_lines_from_mask(
            v_roi,
            offset=x,
            axis="vertical",
            min_span_px=max(28.0 * scale, min(0.15 * float(h), 0.75 * min_row_h)),
            merge_gap_px=max(12.0 * scale, 10.0),
        )
        # Filter degenerate tiny gaps (< 18px wide at 200 DPI) caused by double outer lines
        col_xs = _cluster_1d_positions(col_xs, min_gap=max(18.0 * scale, 14.0))
        row_ys = _cluster_1d_positions(row_ys, min_gap=max(14.0 * scale, 10.0))

        if len(col_xs) < 3 or len(row_ys) < 2:
            continue

        row_ys = _maybe_prepend_open_top_header_row(
            v_roi,
            roi_x=x,
            roi_y=y,
            col_xs=col_xs,
            row_ys=row_ys,
            dpi=dpi,
        )

        row_colspans = _compute_row_colspans_from_vertical_lines(
            v_roi,
            roi_x=x,
            roi_y=y,
            col_xs=col_xs,
            row_ys=row_ys,
            dpi=dpi,
        )
        detected.append(
            DetectedTableGrid(
                x_min=float(col_xs[0]),
                y_min=float(row_ys[0]),
                x_max=float(col_xs[-1]),
                y_max=float(row_ys[-1]),
                col_xs=col_xs,
                row_ys=row_ys,
                row_colspans=row_colspans,
            )
        )

    detected.sort(key=lambda g: (g.y_min, g.x_min))
    return detected


def partition_boxes_by_tables(
    boxes: Sequence[TextBox],
    grids: Sequence[DetectedTableGrid],
    *,
    dpi: int = 200,
) -> tuple[list[tuple[DetectedTableGrid, list[TextBox]]], list[TextBox]]:
    """Assign each box whose `(center_x, center_y)` lies inside a table grid to that table."""
    if not grids:
        return [], list(boxes)

    scale = dpi / 200.0
    x_pad = max(6.0, 6.0 * scale)
    bottom_pad = max(4.0, 4.0 * scale)

    grid_buckets: list[list[TextBox]] = [[] for _ in grids]
    outside_boxes: list[TextBox] = []

    for box in boxes:
        cx = box.center_x
        cy = box.center_y
        matched_idx: Optional[int] = None
        for idx, grid in enumerate(grids):
            if grid.contains_center(cx, cy, x_pad=x_pad, top_pad=0.0, bottom_pad=bottom_pad):
                matched_idx = idx
                break
        if matched_idx is not None:
            grid_buckets[matched_idx].append(box)
        else:
            outside_boxes.append(box)

    populated_tables: list[tuple[DetectedTableGrid, list[TextBox]]] = []
    for grid, bucket in zip(grids, grid_buckets):
        if bucket:
            populated_tables.append((grid, bucket))

    return populated_tables, outside_boxes


def filter_clipped_edge_boxes(
    boxes: Sequence[TextBox],
    grids: Sequence[DetectedTableGrid],
    *,
    page_width: int,
    dpi: int = 200,
) -> list[TextBox]:
    """Drop narrow clipped edge stubs lying immediately left of a full-width multi-col table."""
    if not boxes or not grids or page_width <= 0:
        return list(boxes)

    scale = dpi / 200.0
    max_gutter_offset = 80.0 * scale
    max_stub_width = 75.0 * scale
    edge_grids = [
        g for g in grids if g.num_cols >= 4 and g.x_min <= 0.18 * float(page_width)
    ]
    if not edge_grids:
        return list(boxes)

    kept: list[TextBox] = []
    for box in boxes:
        cx, cy = box.center_x, box.center_y
        stripped = box.text.strip()
        is_clipped_stub = False
        for g in edge_grids:
            if (
                g.y_min <= cy <= g.y_max
                and (g.x_min - max_gutter_offset) <= cx < g.x_min
                and box.width <= max_stub_width
                and len(stripped) <= 5
            ):
                is_clipped_stub = True
                break
        if not is_clipped_stub:
            kept.append(box)
    return kept


def split_boxes_by_grid_dividers(
    boxes: Sequence[TextBox],
    grids: Sequence[DetectedTableGrid],
    *,
    dpi: int = 200,
) -> list[TextBox]:
    """Split wide RapidOCR boxes along confirmed OpenCV vertical grid dividers inside their row."""
    if not boxes or not grids:
        return list(boxes)

    scale = dpi / 200.0
    x_pad = max(6.0, 6.0 * scale)
    bottom_pad = max(4.0, 4.0 * scale)
    cross_margin = max(12.0 * scale, 8.0)
    trim = max(2.0, 2.0 * scale)
    min_piece_w = max(8.0, 8.0 * scale)

    result: list[TextBox] = []
    for box in boxes:
        cx, cy = box.center_x, box.center_y
        matched_grid: Optional[DetectedTableGrid] = None
        for g in grids:
            if g.num_cols >= 2 and g.contains_center(cx, cy, x_pad=x_pad, top_pad=0.0, bottom_pad=bottom_pad):
                matched_grid = g
                break
        if matched_grid is None:
            result.append(box)
            continue

        r = _locate_axis_index(cy, matched_grid.row_ys)
        base_spans = (
            matched_grid.row_colspans[r]
            if r < len(matched_grid.row_colspans)
            and sum(matched_grid.row_colspans[r]) == matched_grid.num_cols
            else [1] * matched_grid.num_cols
        )
        divider_flags = _spans_to_divider_flags(base_spans)
        cut_xs = [
            matched_grid.col_xs[div_idx + 1]
            for div_idx, has_div in enumerate(divider_flags)
            if has_div
            and box.x_min < (matched_grid.col_xs[div_idx + 1] - cross_margin)
            and box.x_max > (matched_grid.col_xs[div_idx + 1] + cross_margin)
        ]
        if not cut_xs:
            result.append(box)
            continue

        cur_x = box.x_min
        pieces_added = 0
        for cut_x in sorted(cut_xs):
            piece_right = min(box.x_max, cut_x - trim)
            if (piece_right - cur_x) >= min_piece_w:
                result.append(
                    box.model_copy(update={"x_min": float(cur_x), "x_max": float(piece_right)})
                )
                pieces_added += 1
            cur_x = max(cur_x, cut_x + trim)
        if (box.x_max - cur_x) >= min_piece_w:
            result.append(
                box.model_copy(update={"x_min": float(cur_x), "x_max": float(box.x_max)})
            )
            pieces_added += 1
        if pieces_added == 0:
            result.append(box)

    return result


def _join_cell_boxes(
    boxes: Sequence[TextBox],
    *,
    y_tolerance_px: float = 10.0,
) -> str:
    """Sort boxes inside one table cell by reading order (top->bottom, left->right) and join text."""
    if not boxes:
        return ""
    ordered = sorted(boxes, key=lambda b: (b.center_y, b.x_min))
    lines: list[list[TextBox]] = [[ordered[0]]]
    anchor_y = ordered[0].center_y
    for box in ordered[1:]:
        if abs(box.center_y - anchor_y) <= y_tolerance_px:
            lines[-1].append(box)
        else:
            lines.append([box])
            anchor_y = box.center_y
    line_texts = [
        " ".join(b.text.strip() for b in sorted(ln, key=lambda x: x.x_min) if b.text.strip())
        for ln in lines
    ]
    return " ".join(part for part in line_texts if part).strip()


def _locate_axis_index(coord: float, edges: Sequence[float]) -> int:
    """Locate interval index `i` in `edges[i] <= coord < edges[i+1]` (clamped to valid range)."""
    count = len(edges) - 1
    if count <= 1:
        return 0
    for idx in range(count):
        if coord < edges[idx + 1]:
            return idx
    return count - 1


def _compute_standard_col_widths_from_geometry(
    col_pixel_widths: Sequence[float],
    rows: Sequence[Sequence[str]],
    *,
    usable_width_cm: float = 16.5,
    is_landscape: bool = False,
) -> list[float]:
    """Scale column widths dynamically: pixel-proportional on Landscape (25.7cm),
    or standard `[1.2, 11.5, 3.0]` cm for Portrait 3-col STT/price tables.
    """
    num_cols = len(col_pixel_widths)
    if num_cols == 0:
        return []

    total_px = max(1.0, float(sum(col_pixel_widths)))
    if is_landscape:
        widths = [
            max(0.8, round(usable_width_cm * (float(w) / total_px), 2))
            for w in col_pixel_widths
        ]
        # Reconcile any rounding delta onto the widest column so total matches `usable_width_cm`
        diff = round(usable_width_cm - sum(widths), 2)
        if abs(diff) >= 0.01:
            widest_idx = max(range(num_cols), key=lambda idx: widths[idx])
            widths[widest_idx] = max(0.8, round(widths[widest_idx] + diff, 2))
        return widths

    first_hdr = [str(c).strip().lower() for c in rows[0]] if rows else []
    looks_like_stt_price = any(
        _STT_HEADER_RE.search(cell)
        or any(k in cell for k in ("danh mục", "mức giá", "đơn giá", "thành tiền"))
        for cell in first_hdr
    )
    if num_cols == 3 and (
        looks_like_stt_price
        or (
            col_pixel_widths[0] <= 0.22 * total_px
            and col_pixel_widths[1] >= 0.45 * total_px
        )
    ):
        return [1.2, 11.5, 3.0]

    if num_cols >= 4 and looks_like_stt_price:
        stt_w = 1.2
        rem = max(3.0, usable_width_cm - stt_w)
        other_px = max(1.0, float(sum(col_pixel_widths[1:])))
        return [stt_w] + [round(rem * (float(w) / other_px), 2) for w in col_pixel_widths[1:]]

    return [round(usable_width_cm * (float(w) / total_px), 2) for w in col_pixel_widths]


def _infer_header_row_count(rows: Sequence[Sequence[str]]) -> int:
    """Detect whether table has 1 header row (`STT | ...`) or 2 (including `1 | 2 | 3`)."""
    if not rows:
        return 0
    row0 = [str(c).strip().lower() for c in rows[0]]
    is_hdr = any(
        _STT_HEADER_RE.search(c)
        or any(k in c for k in _TABLE_COMPANION_KEYWORDS)
        for c in row0
    )
    if not is_hdr:
        return 0
    if len(rows) >= 2:
        non_empty = [str(c).strip() for c in rows[1] if str(c).strip()]
        if non_empty and all(val.isdigit() and int(val) <= len(rows[0]) + 1 for val in non_empty):
            return 2
    return 1


def _header_looks_like_price_col(hdr: str) -> bool:
    low = hdr.strip().lower()
    if not low or low.startswith("tên ") or any(
        excl in low for excl in ("tên dịch vụ", "danh mục", "ghi chú", "nội dung")
    ):
        return False
    return low in ("giá", "mức") or any(
        k in low for k in ("mức giá", "đơn giá", "thành tiền", "số tiền", "chi phí", "giá dịch vụ")
    )


def _infer_col_alignments(
    rows: Sequence[Sequence[str]],
    num_cols: int,
) -> list[Literal["left", "center", "right", "justify"]]:
    """Assign CENTER to STT col, RIGHT to Mức giá col, and LEFT to content col."""
    if num_cols <= 0:
        return []
    if not rows:
        return ["left"] * num_cols

    row0 = [str(rows[0][c]).strip().lower() if c < len(rows[0]) else "" for c in range(num_cols)]
    aligns: list[Literal["left", "center", "right", "justify"]] = []
    for col_idx in range(num_cols):
        hdr = row0[col_idx]
        if col_idx == 0 and (_STT_HEADER_RE.search(hdr) or "mã" in hdr or num_cols >= 3):
            aligns.append("center")
        elif col_idx > 0 and (
            _header_looks_like_price_col(hdr)
            or _column_looks_numeric_price(rows, col_idx)
        ):
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def _column_looks_numeric_price(rows: Sequence[Sequence[str]], col_idx: int) -> bool:
    vals = [
        str(r[col_idx]).strip()
        for r in rows[1:]
        if col_idx < len(r) and str(r[col_idx]).strip()
    ]
    if not vals:
        return False
    hits = sum(1 for v in vals if _PRICE_VALUE_RE.match(v))
    return (hits / float(len(vals))) >= 0.40 or (len(vals) <= 3 and hits >= 1)


def _is_group_heading_text(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return False
    return any(kw in low for kw in _GROUP_HEADING_KEYWORDS)


def _spans_to_divider_flags(spans: Sequence[int]) -> list[bool]:
    """Convert `[1, 2]` into `[True, False]` divider flags between columns."""
    flags: list[bool] = []
    for idx, span in enumerate(spans):
        for _ in range(max(1, span) - 1):
            flags.append(False)
        if idx < len(spans) - 1:
            flags.append(True)
    return flags


def _divider_flags_to_spans(flags: Sequence[bool], num_cols: int) -> list[int]:
    if num_cols <= 1:
        return [max(1, num_cols)]
    spans = [1]
    for present in flags[: num_cols - 1]:
        if present:
            spans.append(1)
        else:
            spans[-1] += 1
    return spans


def build_table_data_from_grid(
    grid: DetectedTableGrid,
    boxes: Sequence[TextBox],
    *,
    dpi: int = 200,
    y_tolerance_px: float = 10.0,
    is_landscape: bool = False,
    usable_width_cm: Optional[float] = None,
) -> tuple[
    list[list[str]],
    list[list[int]],
    list[float],
    list[Literal["left", "center", "right", "justify"]],
    int,
]:
    """Map `(center_x, center_y)` boxes inside an OpenCV table grid into structured rows & spans."""
    num_rows = grid.num_rows
    num_cols = grid.num_cols
    if num_rows == 0 or num_cols == 0:
        return [], [], [], [], 0

    scale = dpi / 200.0
    cross_tol = max(12.0 * scale, 8.0)

    cell_boxes: list[list[list[TextBox]]] = [
        [[] for _ in range(num_cols)] for _ in range(num_rows)
    ]
    row_boxes_all: list[list[TextBox]] = [[] for _ in range(num_rows)]

    for box in boxes:
        r = _locate_axis_index(box.center_y, grid.row_ys)
        c = _locate_axis_index(box.center_x, grid.col_xs)
        row_boxes_all[r].append(box)
        cell_boxes[r][c].append(box)

    divider_flags_per_row: list[list[bool]] = []
    for r in range(num_rows):
        base_spans = (
            grid.row_colspans[r]
            if r < len(grid.row_colspans) and sum(grid.row_colspans[r]) == num_cols
            else [1] * num_cols
        )
        flags = _spans_to_divider_flags(base_spans)
        # On narrow 2-3 col tables with weak vertical borders, widen merge if a single box spans across
        if num_cols <= 3:
            for box in row_boxes_all[r]:
                for div_idx in range(num_cols - 1):
                    div_x = grid.col_xs[div_idx + 1]
                    if box.x_min < (div_x - cross_tol) and box.x_max > (div_x + cross_tol):
                        flags[div_idx] = False
        divider_flags_per_row.append(flags)

    # Initial cell text assembly honoring merged horizontal spans
    rows: list[list[str]] = []
    row_colspans: list[list[int]] = []

    for r in range(num_rows):
        spans = _divider_flags_to_spans(divider_flags_per_row[r], num_cols)
        row_cells = [""] * num_cols
        col_cursor = 0
        for span_len in spans:
            end_col = min(num_cols - 1, col_cursor + span_len - 1)
            combined_boxes: list[TextBox] = []
            for c in range(col_cursor, end_col + 1):
                combined_boxes.extend(cell_boxes[r][c])
            row_cells[col_cursor] = _join_cell_boxes(
                combined_boxes, y_tolerance_px=y_tolerance_px
            )
            col_cursor = end_col + 1
        rows.append(row_cells)
        row_colspans.append(spans)

    header_rows = _infer_header_row_count(rows)

    # Mark single-cell or group-heading rows (no price in last col) as merged horizontal rows
    for r in range(header_rows, num_rows):
        spans = row_colspans[r]
        if len(spans) == 1:
            continue
        non_empty_cols = [c for c, val in enumerate(rows[r]) if val.strip()]
        if not non_empty_cols:
            continue
        last_empty = not rows[r][num_cols - 1].strip()
        tail_all_empty = all(not rows[r][c].strip() for c in range(2, num_cols))
        if (
            num_cols >= 3
            and len(non_empty_cols) == 1
            and non_empty_cols[0] == 1
            and tail_all_empty
            and (num_cols == 3 or _is_group_heading_text(rows[r][1]))
        ):
            # Pure group heading text inside col 1 with empty STT and empty remaining cols -> full colspan
            if _is_group_heading_text(rows[r][1]) or len(rows[r][1].split()) >= 4:
                rows[r][0] = rows[r][1]
                rows[r][1] = ""
                row_colspans[r] = [num_cols]
        elif (
            num_cols == 3
            and last_empty
            and len(non_empty_cols) == 2
            and non_empty_cols == [0, 1]
            and _is_group_heading_text(rows[r][1])
        ):
            # Merge Content + Price columns for 3-col group headings when last col is empty
            if sum(row_colspans[r]) == num_cols and row_colspans[r] == [1] * num_cols:
                row_colspans[r] = [1, num_cols - 1]

    target_usable_cm = (
        float(usable_width_cm)
        if usable_width_cm is not None
        else (25.7 if is_landscape else 16.5)
    )
    col_pixel_widths = [grid.col_xs[c + 1] - grid.col_xs[c] for c in range(num_cols)]
    col_widths_cm = _compute_standard_col_widths_from_geometry(
        col_pixel_widths,
        rows,
        usable_width_cm=target_usable_cm,
        is_landscape=is_landscape,
    )
    col_alignments = _infer_col_alignments(rows, num_cols)

    return rows, row_colspans, col_widths_cm, col_alignments, header_rows


# ---------------------------------------------------------------------------
# Fallback: Column Alignment Projection when OpenCV borders are faded/broken
# ---------------------------------------------------------------------------


@dataclass
class ProjectionTableBlock:
    """Structured 3-column table reconstructed via X-coordinate Column Projection fallback."""

    y_min: float
    y_max: float
    rows: list[list[str]]
    row_colspans: list[list[int]]
    col_widths_cm: list[float]
    col_alignments: list[Literal["left", "center", "right", "justify"]]
    header_rows: int


def _line_matches_table_header(text_left: str) -> bool:
    low = text_left.strip().lower()
    if not low or not _STT_HEADER_RE.search(low):
        return False
    return any(kw in low for kw in _TABLE_COMPANION_KEYWORDS)


def _find_projection_table_header(
    lines: Sequence[DetectedLine],
    idx: int,
    *,
    median_h: float,
) -> Optional[tuple[int, int, list[TextBox]]]:
    """Check whether line `idx` (or staggered `(idx, idx+1)`) forms a table header row."""
    cur = lines[idx]
    cur_text = " ".join(b.text for b in cur.boxes if b.text).strip()
    if _line_matches_table_header(cur_text):
        return idx, idx, list(cur.boxes)

    # Check staggered "STT" alone above "Danh mục ... | Mức giá"
    if (
        idx + 1 < len(lines)
        and _STT_HEADER_RE.fullmatch(cur_text.strip())
        and (lines[idx + 1].y_min - cur.y_max) <= max(35.0, 2.0 * median_h)
    ):
        nxt_text = " ".join(b.text for b in lines[idx + 1].boxes if b.text).strip()
        if any(kw in nxt_text.lower() for kw in _TABLE_COMPANION_KEYWORDS):
            combined = list(cur.boxes) + list(lines[idx + 1].boxes)
            return idx, idx + 1, combined

    return None


def _should_stop_projection_table(line: DetectedLine) -> bool:
    txt = " ".join(b.text for b in line.boxes if b.text).strip()
    low = txt.lower()
    if not low:
        return False
    if any(low.startswith(pref) for pref in _FOOTER_STOP_PREFIXES):
        return True
    if low.startswith("phụ lục") or low == "danh mục":
        return True
    if line.line_type == "two_column_row":
        r_low = line.right_text.strip().lower()
        if line.left_text.strip().lower().startswith("nơi nhận") or any(
            k in r_low for k in ("giám đốc", "chủ tịch", "kt.", "tm.", "tl.")
        ):
            return True
    return False


def _is_continuation_fragment(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s[0].islower() or s[0] in ("(", ",", ")", ";", "."):
        return True
    return False


def extract_tables_via_column_projection(
    lines: Sequence[DetectedLine],
    *,
    page_width: int,
    dpi: int = 200,
    y_tolerance_px: float = 10.0,
    is_landscape: bool = False,
    usable_width_cm: Optional[float] = None,
) -> tuple[list[DetectedLine], list[ProjectionTableBlock]]:
    """Fallback table extractor for pages whose grid borders are broken/missing.

    Triggers when an `"STT"` header accompanied by `"Danh mục"` / `"Đơn giá"` / `"Mức giá"`
    / `"Thành tiền"` is found outside OpenCV-detected grid regions. Projects boxes below the
    header into Col 0 (STT), Col 1 (Nội dung), and Col 2 (Giá tiền) without concatenating
    the 3 columns together.
    """
    if not lines:
        return [], []

    heights = [ln.height for ln in lines if ln.height > 0]
    median_h = float(statistics.median(heights)) if heights else 24.0
    scale = dpi / 200.0
    w = float(page_width)

    remaining_lines: list[DetectedLine] = []
    projected_tables: list[ProjectionTableBlock] = []

    i = 0
    n = len(lines)
    while i < n:
        hdr_match = _find_projection_table_header(lines, i, median_h=median_h)
        if hdr_match is None:
            remaining_lines.append(lines[i])
            i += 1
            continue

        hdr_start, hdr_end, hdr_boxes = hdr_match
        j = hdr_end + 1
        while j < n and not _should_stop_projection_table(lines[j]):
            # Stop if another STT table header starts below
            if _find_projection_table_header(lines, j, median_h=median_h) is not None:
                break
            j += 1

        table_lines = list(lines[hdr_start:j])
        if len(table_lines) < 2:
            remaining_lines.append(lines[i])
            i += 1
            continue

        # Determine 3 column boundaries from STT box, right price column boxes, and page width
        stt_box = min(
            (b for b in hdr_boxes if _STT_HEADER_RE.search(b.text.strip())),
            key=lambda b: b.x_min,
            default=hdr_boxes[0],
        )
        stt_width_1_5cm = (1.5 / 2.54) * float(dpi)

        right_hdr_candidates = [
            b for b in hdr_boxes if b.center_x >= 0.55 * w and not _STT_HEADER_RE.search(b.text.strip())
        ]
        price_x_mins: list[float] = [b.x_min for b in right_hdr_candidates]
        mid_x_mins: list[float] = []

        for ln in table_lines[hdr_end - hdr_start + 1 :]:
            for b in ln.boxes:
                txt = b.text.strip()
                if b.center_x >= 0.60 * w and (_PRICE_VALUE_RE.match(txt) or b.x_min >= 0.70 * w):
                    price_x_mins.append(b.x_min)
                elif b.x_min > stt_box.x_max + 8.0 * scale and b.center_x < 0.62 * w:
                    mid_x_mins.append(b.x_min)

        if mid_x_mins:
            mid_anchor = min(mid_x_mins)
            col0_max_x = min(stt_box.x_min + stt_width_1_5cm, (stt_box.x_max + mid_anchor) / 2.0)
            col0_max_x = max(col0_max_x, stt_box.x_max + 6.0 * scale)
        else:
            col0_max_x = stt_box.x_min + stt_width_1_5cm

        if price_x_mins:
            col2_min_x = max(0.55 * w, min(price_x_mins) - 14.0 * scale)
        else:
            col2_min_x = 0.72 * w

        # Ensure sane separation
        if col2_min_x <= col0_max_x + 40.0 * scale:
            col2_min_x = max(col0_max_x + 80.0 * scale, 0.72 * w)

        def _split_line_3cols(boxes_in_ln: Sequence[TextBox]) -> tuple[str, str, str, bool]:
            c0_b: list[TextBox] = []
            c1_b: list[TextBox] = []
            c2_b: list[TextBox] = []
            crosses_to_c2 = False
            for b in sorted(boxes_in_ln, key=lambda bx: bx.x_min):
                if b.center_x <= col0_max_x:
                    c0_b.append(b)
                elif b.center_x >= col2_min_x and b.x_min >= (col2_min_x - 30.0 * scale):
                    c2_b.append(b)
                else:
                    c1_b.append(b)
                    if b.x_max > (col2_min_x + 25.0 * scale) and b.x_min < (col2_min_x - 40.0 * scale):
                        crosses_to_c2 = True
            return (
                _join_cell_boxes(c0_b, y_tolerance_px=y_tolerance_px),
                _join_cell_boxes(c1_b, y_tolerance_px=y_tolerance_px),
                _join_cell_boxes(c2_b, y_tolerance_px=y_tolerance_px),
                crosses_to_c2,
            )

        rows: list[list[str]] = []
        row_colspans: list[list[int]] = []

        # Row 0: Header row from hdr_boxes
        h0, h1, h2, _ = _split_line_3cols(hdr_boxes)
        rows.append([h0, h1, h2])
        row_colspans.append([1, 1, 1])

        pending_prefix_col1: list[str] = []
        body_start_idx = (hdr_end - hdr_start) + 1
        prev_y_max = max(b.y_max for b in hdr_boxes)

        for k in range(body_start_idx, len(table_lines)):
            ln = table_lines[k]
            c0, c1, c2, crosses_c2 = _split_line_3cols(ln.boxes)
            v_gap = ln.y_min - prev_y_max
            prev_y_max = ln.y_max

            if not (c0 or c1 or c2):
                continue

            # Column numbering subheader ("1 | 2 | 3") directly under header
            if (
                len(rows) == 1
                and c0 == "1"
                and c1 == "2"
                and c2 == "3"
            ):
                rows.append([c0, c1, c2])
                row_colspans.append([1, 1, 1])
                continue

            # Check wrapped continuation or pending multi-line prefix in Col 1
            if not c0 and not c2 and c1:
                if crosses_c2 or _is_group_heading_text(c1):
                    if pending_prefix_col1 and rows and len(rows) > 1 and not rows[-1][2]:
                        # Flush stray prefix
                        rows.append(["", " ".join(pending_prefix_col1), ""])
                        row_colspans.append([3])
                        pending_prefix_col1.clear()
                    # Wide spanning group heading line -> attach or start full-colspan group row
                    if (
                        rows
                        and len(rows) > 1
                        and not rows[-1][2]
                        and len(row_colspans[-1]) < 3
                        and v_gap <= max(28.0 * scale, 1.3 * median_h)
                    ):
                        target_col = 0 if len(row_colspans[-1]) == 1 else 1
                        rows[-1][target_col] = f"{rows[-1][target_col]} {c1}".strip()
                    else:
                        rows.append([c1, "", ""])
                        row_colspans.append([3])
                    continue

                # Determine whether this Col-1-only line belongs to the previous row (if short gap
                # & fragment) or prefixes an upcoming row whose STT/Price sit on its line 2
                prev_is_data_row = len(rows) >= 2 and (bool(rows[-1][0]) or bool(rows[-1][2]))
                next_has_item = (
                    k + 1 < len(table_lines)
                    and (
                        bool(_split_line_3cols(table_lines[k + 1].boxes)[0])
                        or bool(_split_line_3cols(table_lines[k + 1].boxes)[2])
                    )
                )
                if next_has_item and not _is_continuation_fragment(c1) and (table_lines[k + 1].y_min - ln.y_max) <= max(28.0 * scale, 1.3 * median_h):
                    pending_prefix_col1.append(c1)
                elif prev_is_data_row and (v_gap <= max(28.0 * scale, 1.3 * median_h) or _is_continuation_fragment(c1)):
                    rows[-1][1] = f"{rows[-1][1]} {c1}".strip()
                else:
                    pending_prefix_col1.append(c1)
                continue

            if pending_prefix_col1:
                c1 = " ".join([*pending_prefix_col1, c1]).strip() if c1 else " ".join(pending_prefix_col1).strip()
                pending_prefix_col1.clear()

            # Check if this row is a merged group heading ("1 | Danh mục..." or "II | ...")
            if not c2 and (crosses_c2 or _is_group_heading_text(c1)):
                if not c0:
                    rows.append([c1, "", ""])
                    row_colspans.append([3])
                else:
                    rows.append([c0, c1, ""])
                    row_colspans.append([1, 2])
                continue

            rows.append([c0, c1, c2])
            row_colspans.append([1, 1, 1])

        if pending_prefix_col1:
            if len(rows) >= 2:
                rows[-1][1] = f"{rows[-1][1]} {' '.join(pending_prefix_col1)}".strip()
            else:
                rows.append(["", " ".join(pending_prefix_col1), ""])
                row_colspans.append([3])

        header_rows = _infer_header_row_count(rows)
        proj_usable = (
            float(usable_width_cm)
            if usable_width_cm is not None
            else (25.7 if is_landscape else 16.5)
        )
        proj_widths = (
            _compute_standard_col_widths_from_geometry(
                [col0_max_x - stt_box.x_min, col2_min_x - col0_max_x, w - col2_min_x],
                rows,
                usable_width_cm=proj_usable,
                is_landscape=True,
            )
            if is_landscape
            else [1.2, 11.5, 3.0]
        )
        projected_tables.append(
            ProjectionTableBlock(
                y_min=float(table_lines[0].y_min),
                y_max=float(table_lines[-1].y_max),
                rows=rows,
                row_colspans=row_colspans,
                col_widths_cm=proj_widths,
                col_alignments=_infer_col_alignments(rows, 3),
                header_rows=header_rows,
            )
        )
        i = j

    return remaining_lines, projected_tables

