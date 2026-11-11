"""Pydantic v2 structured schemas for Vietnamese administrative documents."""

from __future__ import annotations

import unicodedata
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


def normalize_vietnamese_text(value: Optional[str]) -> Optional[str]:
    """Normalize Vietnamese text to Unicode NFC and strip surrounding whitespace."""
    if value is None:
        return None
    normalized = unicodedata.normalize("NFC", value).strip()
    return normalized


def _normalize_cell_or_text(value: Any) -> str:
    """Coerce an arbitrary scalar/text value to NFC-normalized text, mapping None to ''."""
    if value is None:
        return ""
    return normalize_vietnamese_text(str(value)) or ""


class HeaderTable(BaseModel):
    """Two-column administrative document header (Nghị định 30/2020/NĐ-CP)."""

    co_quan_chu_quan: Optional[str] = Field(
        default=None,
        description="Tên cơ quan, tổ chức chủ quản trực tiếp cấp trên ở cột trái (nếu có, nếu không để null).",
    )
    co_quan_ban_hanh: str = Field(
        default="",
        description="Tên cơ quan, tổ chức ban hành văn bản ở cột trái (in hoa, đậm).",
    )
    so_ky_hieu: str = Field(
        default="",
        description="Số và ký hiệu văn bản ở cột trái (VD: 'Số: 123/QĐ-UBND').",
    )
    quoc_hieu: str = Field(
        default="CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        description="Quốc hiệu ở cột phải.",
    )
    tieu_ngu: str = Field(
        default="Độc lập - Tự do - Hạnh phúc",
        description="Tiêu ngữ ở cột phải.",
    )
    dia_danh_ngay_thang: str = Field(
        default="",
        description="Địa danh và ngày tháng năm ban hành ở cột phải (VD: 'Hà Nội, ngày 10 tháng 11 năm 2026').",
    )

    @field_validator("co_quan_chu_quan", mode="before")
    @classmethod
    def _normalize_optional_field(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return normalize_vietnamese_text(str(value))

    @field_validator(
        "co_quan_ban_hanh",
        "so_ky_hieu",
        "quoc_hieu",
        "tieu_ngu",
        "dia_danh_ngay_thang",
        mode="before",
    )
    @classmethod
    def _normalize_required_str_fields(cls, value: Optional[str]) -> str:
        if value is None:
            return ""
        return normalize_vietnamese_text(str(value)) or ""

    def is_empty(self) -> bool:
        """Return True if all header fields are empty or None."""
        return not any(
            [
                self.co_quan_chu_quan,
                self.co_quan_ban_hanh,
                self.so_ky_hieu,
                self.quoc_hieu,
                self.tieu_ngu,
                self.dia_danh_ngay_thang,
            ]
        )


class TitleBlock(BaseModel):
    """Document type name and abstract (Tên loại văn bản và trích yếu nội dung)."""

    ten_loai_van_ban: Optional[str] = Field(
        default=None,
        description="Tên loại văn bản hành chính (VD: 'QUYẾT ĐỊNH', 'THÔNG BÁO', 'TỜ TRÌNH'; để null nếu là Công văn).",
    )
    trich_yeu: str = Field(
        default="",
        description="Trích yếu nội dung văn bản (VD: 'Về việc ban hành...').",
    )

    @field_validator("ten_loai_van_ban", mode="before")
    @classmethod
    def _normalize_optional_field(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return normalize_vietnamese_text(str(value))

    @field_validator("trich_yeu", mode="before")
    @classmethod
    def _normalize_required_str_field(cls, value: Optional[str]) -> str:
        if value is None:
            return ""
        return normalize_vietnamese_text(str(value)) or ""

    def is_empty(self) -> bool:
        """Return True if neither document type nor abstract contains text."""
        return not (self.ten_loai_van_ban or self.trich_yeu)


class BodyBlock(BaseModel):
    """Canonical ordered body block (paragraph, bordered table, or 2-column borderless table)."""

    type: Literal["paragraph", "table", "two_column_table"] = Field(
        description="Loại khối nội dung: 'paragraph', 'table', hoặc 'two_column_table'."
    )
    alignment: Literal["left", "center", "right", "justify"] = Field(
        default="justify",
        description="Căn lề của khối: 'left', 'center', 'right', hoặc 'justify'.",
    )
    first_line_indent: bool = Field(
        default=False,
        description="Có thụt đầu dòng dòng đầu tiên 1.27cm hay không.",
    )
    bold: bool = Field(default=False, description="Chữ in đậm.")
    italic: bool = Field(default=False, description="Chữ in nghiêng.")
    content: Union[str, list[list[str]]] = Field(
        description="Chuỗi văn bản nếu type='paragraph', hoặc mảng 2 chiều list[list[str]] nếu type='table' / 'two_column_table'."
    )
    cell_colspans: Optional[list[list[int]]] = Field(
        default=None,
        description="Tùy chọn: số cột gộp (colspan) theo từng hàng của bảng (VD: [[1,1,1], [3]]).",
    )
    col_widths_cm: Optional[list[float]] = Field(
        default=None,
        description="Tùy chọn: độ rộng chuẩn từng cột tính bằng cm (VD: [1.2, 11.5, 3.0]).",
    )
    col_alignments: Optional[list[Literal["left", "center", "right", "justify"]]] = Field(
        default=None,
        description="Tùy chọn: căn lề dữ liệu theo từng cột (VD: ['center', 'left', 'right']).",
    )
    header_rows: int = Field(
        default=0,
        ge=0,
        description="Số hàng tiêu đề ở đầu bảng được căn giữa và in đậm.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_null_content_and_cells(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "content" not in data:
            return data
        raw_content = data["content"]
        if raw_content is None and data.get("type") == "paragraph":
            return {**data, "content": ""}
        if isinstance(raw_content, list):
            coerced_rows = [
                ["" if cell is None else cell for cell in row]
                if isinstance(row, list)
                else row
                for row in raw_content
            ]
            return {**data, "content": coerced_rows}
        return data

    @model_validator(mode="after")
    def _validate_and_normalize_content(self) -> "BodyBlock":
        if self.type == "paragraph":
            if isinstance(self.content, list):
                lines: list[str] = []
                for row in self.content:
                    if isinstance(row, list):
                        joined = " ".join(
                            _normalize_cell_or_text(cell) for cell in row
                        ).strip()
                        if joined:
                            lines.append(joined)
                    else:
                        item = _normalize_cell_or_text(row)
                        if item:
                            lines.append(item)
                self.content = "\n".join(lines).strip()
            else:
                self.content = _normalize_cell_or_text(self.content)
        elif self.type in ("table", "two_column_table"):
            if not isinstance(self.content, list):
                raise ValueError(
                    f"BodyBlock with type={self.type!r} requires a 2D list of strings for content."
                )
            normalized_rows: list[list[str]] = []
            for row in self.content:
                if not isinstance(row, list):
                    raise ValueError(
                        "Each row in table content must be a list of cell strings."
                    )
                norm_row = [_normalize_cell_or_text(cell) for cell in row]
                if self.type == "two_column_table":
                    if len(norm_row) == 0:
                        norm_row = ["", ""]
                    elif len(norm_row) == 1:
                        norm_row = [norm_row[0], ""]
                    elif len(norm_row) > 2:
                        norm_row = [
                            norm_row[0],
                            " ".join(cell for cell in norm_row[1:] if cell).strip(),
                        ]
                normalized_rows.append(norm_row)
            self.content = normalized_rows
        return self


class FooterTable(BaseModel):
    """Two-column footer block (Nơi nhận ở cột trái, Chức vụ & Họ tên người ký ở cột phải)."""

    noi_nhan: str = Field(
        default="",
        description="Nội dung cột trái Nơi nhận (các dòng phân tách bởi '\\n').",
    )
    chuc_vu_nguoi_ky: str = Field(
        default="",
        description="Quyền hạn và chức vụ người ký ở cột phải (VD: 'KT. CHỦ TỊCH\\nPHÓ CHỦ TỊCH').",
    )
    ho_ten_nguoi_ky: str = Field(
        default="",
        description="Họ và tên người ký ở cột phải (chữ in, bỏ qua chữ ký tay viết tay).",
    )

    @field_validator("noi_nhan", mode="before")
    @classmethod
    def _normalize_noi_nhan(cls, value: Union[str, list[str], None]) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            lines = [_normalize_cell_or_text(item) for item in value]
            return "\n".join(line for line in lines if line)
        return normalize_vietnamese_text(str(value)) or ""

    @field_validator("chuc_vu_nguoi_ky", "ho_ten_nguoi_ky", mode="before")
    @classmethod
    def _normalize_signer(cls, value: Optional[str]) -> str:
        if value is None:
            return ""
        return normalize_vietnamese_text(str(value)) or ""

    def is_empty(self) -> bool:
        """Return True if all footer fields are empty."""
        return not any(
            [self.noi_nhan, self.chuc_vu_nguoi_ky, self.ho_ten_nguoi_ky]
        )


class PageStructure(BaseModel):
    """Canonical structured extraction for a single PDF page."""

    page_number: int = Field(default=1, ge=1)
    is_landscape: bool = False
    header_table: Optional[HeaderTable] = None
    title_block: Optional[TitleBlock] = None
    body_blocks: list[BodyBlock] = Field(default_factory=list)
    footer_table: Optional[FooterTable] = None


class AdministrativeDocument(BaseModel):
    """Complete structured representation of a single- or multi-page Vietnamese document."""

    source_pdf: Optional[str] = None
    total_pages: int = Field(default=1, ge=1)
    header_table: Optional[HeaderTable] = None
    title_block: Optional[TitleBlock] = None
    body_blocks: list[BodyBlock] = Field(default_factory=list)
    footer_table: Optional[FooterTable] = None
    pages: list[PageStructure] = Field(default_factory=list)


def merge_page_structures(
    pages: list[PageStructure],
    source_pdf: Optional[str] = None,
) -> AdministrativeDocument:
    """Aggregate per-page structures into a single coherent AdministrativeDocument."""
    if not pages:
        raise ValueError("Cannot merge an empty list of PageStructure objects.")

    ordered_pages = sorted(pages, key=lambda p: p.page_number)

    header_table: Optional[HeaderTable] = None
    title_block: Optional[TitleBlock] = None
    footer_table: Optional[FooterTable] = None
    body_blocks: list[BodyBlock] = []

    for page in ordered_pages:
        if header_table is None and page.header_table and not page.header_table.is_empty():
            header_table = page.header_table
        if page.title_block and not page.title_block.is_empty():
            if title_block is None:
                title_block = page.title_block
            else:
                # Preserve secondary section/annex titles (e.g., Phụ lục on later pages)
                for title_part in (page.title_block.ten_loai_van_ban, page.title_block.trich_yeu):
                    if title_part:
                        body_blocks.append(
                            BodyBlock(
                                type="paragraph",
                                alignment="center",
                                first_line_indent=False,
                                bold=True,
                                italic=False,
                                content=title_part,
                            )
                        )
        body_blocks.extend(page.body_blocks)
        if page.footer_table and not page.footer_table.is_empty():
            footer_table = page.footer_table

    return AdministrativeDocument(
        source_pdf=source_pdf,
        total_pages=len(ordered_pages),
        header_table=header_table,
        title_block=title_block,
        body_blocks=body_blocks,
        footer_table=footer_table,
        pages=ordered_pages,
    )
