"""Shared Pydantic data models for offline OCR, X/Y layout analysis, and Table Detection."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OcrLayoutConfig(BaseModel):
    """Configuration for offline RapidOCR detection, X/Y layout rules, and VietOCR."""

    dpi: int = Field(default=200, ge=72, le=1200)
    line_y_tolerance_px: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        description="Max vertical Y-center difference (px) to group boxes into the same line.",
    )
    two_col_split_ratio: float = Field(
        default=0.45,
        gt=0.2,
        lt=0.8,
        description="Horizontal split threshold (~45% page width) separating left and right columns.",
    )
    center_tolerance_ratio: float = Field(
        default=0.08,
        gt=0.0,
        lt=0.3,
        description="Allowed ±8% page-width tolerance around horizontal center for centered lines.",
    )
    indent_min_px: float = Field(
        default=30.0,
        ge=5.0,
        le=200.0,
        description="Minimum left-margin offset (px at 200 DPI) treated as first_line_indent=True.",
    )
    indent_max_px: float = Field(
        default=110.0,
        ge=30.0,
        le=300.0,
        description="Maximum left-margin offset (px at 200 DPI) treated as first_line_indent=True (covers 30-80px and 1.27cm).",
    )
    cluster_gap_px: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional override for horizontal gap separating clusters on the same line.",
    )
    crop_padding_px: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Padding (px) added around detected text boxes before VietOCR recognition.",
    )
    vietocr_config_name: str = Field(default="vgg_transformer", min_length=1)
    vietocr_device: str = Field(default="auto", min_length=1)
    vietocr_batch_size: int = Field(
        default=64,
        ge=1,
        le=256,
        description="Maximum number of cropped text boxes per VietOCR batch.",
    )
    vietocr_use_fp16: bool = Field(
        default=False,
        description="Optional FP16 autocast on CUDA GPU (disabled by default to preserve exact FP32 accuracy).",
    )
    vietocr_weights: Optional[str] = Field(default=None)
    auto_rotate_landscape: bool = Field(
        default=True,
        description="Tự động phát hiện và xoay thẳng các trang scan bị xoay 90/270 độ.",
    )
    vertical_box_aspect_threshold: float = Field(
        default=2.0,
        gt=1.0,
        le=10.0,
        description="Ngưỡng box_height > box_width * threshold để xác định chữ dựng đứng.",
    )
    vertical_box_majority_ratio: float = Field(
        default=0.60,
        gt=0.1,
        le=1.0,
        description="Tỷ lệ box dựng đứng tối thiểu (> 60%) để kích hoạt tự xoay 90°.",
    )


class TextBox(BaseModel):
    """Axis-aligned text bounding box `[x_min, y_min, x_max, y_max]` with optional text."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    text: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0


class DetectedLine(BaseModel):
    """Single horizontal line of grouped boxes with X-based layout classification."""

    line_index: int = 0
    boxes: list[TextBox] = Field(default_factory=list)
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    line_type: Literal["single", "two_column_row"] = "single"
    alignment: Literal["left", "center", "right", "justify"] = "left"
    first_line_indent: bool = False
    is_tabular_row: bool = False
    left_boxes: list[TextBox] = Field(default_factory=list)
    right_boxes: list[TextBox] = Field(default_factory=list)
    left_text: str = ""
    right_text: str = ""
    text: str = ""

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)


class LayoutSegment(BaseModel):
    """Logical page layout block (paragraph, 2-column borderless table, or bordered grid table)."""

    kind: Literal["paragraph", "two_column_table", "table"]
    page_number: int = Field(default=1, ge=1)
    y_min: float = 0.0
    y_max: float = 0.0
    alignment: Literal["left", "center", "right", "justify"] = "left"
    first_line_indent: bool = False
    bold: bool = False
    italic: bool = False
    text: str = ""
    left_lines: list[str] = Field(default_factory=list)
    right_lines: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)
    cell_colspans: Optional[list[list[int]]] = None
    col_widths_cm: Optional[list[float]] = None
    col_alignments: Optional[list[Literal["left", "center", "right", "justify"]]] = None
    header_rows: int = 0


class PageLayoutResult(BaseModel):
    """Offline OCR & layout analysis output for a single page."""

    page_number: int = Field(default=1, ge=1)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    is_landscape: bool = False
    rotation_angle: Literal[0, 90, 270] = 0
    body_left_margin: float
    body_right_margin: float
    lines: list[DetectedLine] = Field(default_factory=list)
    segments: list[LayoutSegment] = Field(default_factory=list)

