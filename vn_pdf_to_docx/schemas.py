"""Pydantic v2 structured schemas for Vietnamese administrative documents."""

from __future__ import annotations

import unicodedata
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


def _strip_schema_defaults(schema: dict[str, Any]) -> None:
    """Recursively strip 'default' keys from JSON Schema for Gemini OpenAPI compatibility."""

    def _walk(node: Any, is_schema_obj: bool = True) -> None:
        if isinstance(node, dict):
            if is_schema_obj:
                node.pop("default", None)
            for key, val in node.items():
                if key in ("properties", "$defs", "definitions") and isinstance(val, dict):
                    for sub_schema in val.values():
                        _walk(sub_schema, True)
                else:
                    _walk(val, isinstance(val, dict))
        elif isinstance(node, list):
            for item in node:
                _walk(item, True)

    _walk(schema, True)


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


# ---------------------------------------------------------------------------
# Gemini Structured-Output Transport Schema
# ---------------------------------------------------------------------------
# Note:
# 1. Separating `text_content` and `table_rows` avoids non-null polymorphic
#    `Union[str, list[list[str]]]` (`anyOf`) inside `google-genai` `response_schema`.
# 2. Using dedicated Gemini sub-models without non-None string `default` values
#    keeps the JSON Schema free of `"default": "..."` keys (which Gemini schema
#    validators can reject) and prevents empty header objects from inventing
#    default Quốc hiệu / Tiêu ngữ on continuation pages.


class GeminiHeaderTable(BaseModel):
    """Gemini response_schema representation of HeaderTable without string defaults."""

    model_config = ConfigDict(json_schema_extra=_strip_schema_defaults)

    co_quan_chu_quan: Optional[str] = Field(
        description="Tên cơ quan, tổ chức chủ quản trực tiếp cấp trên ở cột trái (để null nếu không có)."
    )
    co_quan_ban_hanh: Optional[str] = Field(
        description="Tên cơ quan, tổ chức ban hành văn bản ở cột trái (để chuỗi rỗng '' hoặc null nếu không có)."
    )
    so_ky_hieu: Optional[str] = Field(
        description="Số và ký hiệu văn bản ở cột trái (VD: 'Số: 123/QĐ-UBND'; để '' hoặc null nếu không có)."
    )
    quoc_hieu: Optional[str] = Field(
        description="Quốc hiệu ở cột phải (VD: 'CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM'; để '' hoặc null nếu không có)."
    )
    tieu_ngu: Optional[str] = Field(
        description="Tiêu ngữ ở cột phải (VD: 'Độc lập - Tự do - Hạnh phúc'; để '' hoặc null nếu không có)."
    )
    dia_danh_ngay_thang: Optional[str] = Field(
        description="Địa danh và ngày tháng năm ban hành ở cột phải (để '' hoặc null nếu không có)."
    )


class GeminiTitleBlock(BaseModel):
    """Gemini response_schema representation of TitleBlock without string defaults."""

    model_config = ConfigDict(json_schema_extra=_strip_schema_defaults)

    ten_loai_van_ban: Optional[str] = Field(
        description="Tên loại văn bản hành chính (VD: 'QUYẾT ĐỊNH', 'THÔNG BÁO'; để null nếu là Công văn)."
    )
    trich_yeu: Optional[str] = Field(
        description="Trích yếu nội dung văn bản (VD: 'Về việc ban hành...'; để '' hoặc null nếu không có)."
    )


class GeminiFooterTable(BaseModel):
    """Gemini response_schema representation of FooterTable without string defaults."""

    model_config = ConfigDict(json_schema_extra=_strip_schema_defaults)

    noi_nhan: Optional[str] = Field(
        description="Nội dung cột trái Nơi nhận (các dòng phân tách bởi '\\n'; để '' hoặc null nếu không có)."
    )
    chuc_vu_nguoi_ky: Optional[str] = Field(
        description="Quyền hạn và chức vụ người ký ở cột phải (VD: 'KT. CHỦ TỊCH\\nPHÓ CHỦ TỊCH'; để '' hoặc null nếu không có)."
    )
    ho_ten_nguoi_ky: Optional[str] = Field(
        description="Họ và tên người ký ở cột phải (chữ in, bỏ qua chữ ký tay và chữ trong con dấu đỏ)."
    )


class GeminiBodyBlock(BaseModel):
    """Gemini response_schema item for a body block without polymorphic Union."""

    model_config = ConfigDict(json_schema_extra=_strip_schema_defaults)

    type: Literal["paragraph", "table"] = Field(
        description="Loại khối: 'paragraph' cho đoạn văn/tiêu đề, 'table' cho bảng biểu."
    )
    alignment: Literal["left", "center", "right", "justify"] = Field(
        description="Căn lề của khối: 'left', 'center', 'right', hoặc 'justify'."
    )
    first_line_indent: bool = Field(
        description="True nếu dòng đầu của đoạn văn có thụt lề đầu dòng (1.27cm), ngược lại False."
    )
    bold: bool = Field(description="True nếu toàn khối được in đậm.")
    italic: bool = Field(description="True nếu toàn khối được in nghiêng.")
    text_content: Optional[str] = Field(
        default=None,
        description="Nội dung văn bản đã sửa 100% lỗi chính tả nếu type='paragraph'. Để null nếu type='table'.",
    )
    table_rows: Optional[list[list[str]]] = Field(
        default=None,
        description="Mảng 2 chiều các hàng và ô (đã sửa lỗi chính tả) nếu type='table'. Để null nếu type='paragraph'.",
    )

    @field_validator("table_rows", mode="before")
    @classmethod
    def _coerce_null_table_cells(cls, value: object) -> object:
        if isinstance(value, list):
            return [
                ["" if cell is None else cell for cell in row]
                if isinstance(row, list)
                else row
                for row in value
            ]
        return value


class GeminiPageExtraction(BaseModel):
    """Gemini response_schema root model for one scanned page."""

    model_config = ConfigDict(json_schema_extra=_strip_schema_defaults)

    header_table: Optional[GeminiHeaderTable] = Field(
        default=None,
        description="Thông tin quốc hiệu, tiêu ngữ, cơ quan ban hành, số ký hiệu nếu xuất hiện ở đầu trang này; nếu không có để null.",
    )
    title_block: Optional[GeminiTitleBlock] = Field(
        default=None,
        description="Tên loại văn bản và trích yếu nếu xuất hiện trên trang này; nếu không có để null.",
    )
    body_blocks: list[GeminiBodyBlock] = Field(
        description="Danh sách các khối nội dung (đoạn văn và bảng) theo đúng thứ tự từ trên xuống dưới.",
    )
    footer_table: Optional[GeminiFooterTable] = Field(
        default=None,
        description="Khối Nơi nhận và Chức vụ/Họ tên người ký nếu xuất hiện ở cuối trang này; nếu không có để null.",
    )

    @field_validator("header_table", "title_block", "footer_table", mode="before")
    @classmethod
    def _coerce_canonical_submodels(cls, value: object) -> object:
        if isinstance(value, (HeaderTable, TitleBlock, FooterTable)):
            return value.model_dump()
        return value


def gemini_extraction_to_page_structure(
    raw: GeminiPageExtraction,
    page_number: int = 1,
) -> PageStructure:
    """Convert a Gemini transport payload into a canonical PageStructure."""
    header: Optional[HeaderTable] = None
    if raw.header_table is not None:
        candidate_header = HeaderTable(
            co_quan_chu_quan=raw.header_table.co_quan_chu_quan,
            co_quan_ban_hanh=raw.header_table.co_quan_ban_hanh or "",
            so_ky_hieu=raw.header_table.so_ky_hieu or "",
            quoc_hieu=raw.header_table.quoc_hieu or "",
            tieu_ngu=raw.header_table.tieu_ngu or "",
            dia_danh_ngay_thang=raw.header_table.dia_danh_ngay_thang or "",
        )
        if not candidate_header.is_empty():
            header = candidate_header

    title: Optional[TitleBlock] = None
    if raw.title_block is not None:
        candidate_title = TitleBlock(
            ten_loai_van_ban=raw.title_block.ten_loai_van_ban,
            trich_yeu=raw.title_block.trich_yeu or "",
        )
        if not candidate_title.is_empty():
            title = candidate_title

    footer: Optional[FooterTable] = None
    if raw.footer_table is not None:
        candidate_footer = FooterTable(
            noi_nhan=raw.footer_table.noi_nhan or "",
            chuc_vu_nguoi_ky=raw.footer_table.chuc_vu_nguoi_ky or "",
            ho_ten_nguoi_ky=raw.footer_table.ho_ten_nguoi_ky or "",
        )
        if not candidate_footer.is_empty():
            footer = candidate_footer

    canonical_blocks: list[BodyBlock] = []
    for block in raw.body_blocks:
        if block.type == "paragraph":
            text = _normalize_cell_or_text(block.text_content)
            if not text and block.table_rows:
                text = "\n".join(
                    " ".join(
                        _normalize_cell_or_text(cell) for cell in row
                    ).strip()
                    for row in block.table_rows
                ).strip()
            if not text:
                continue
            canonical_blocks.append(
                BodyBlock(
                    type="paragraph",
                    alignment=block.alignment,
                    first_line_indent=block.first_line_indent,
                    bold=block.bold,
                    italic=block.italic,
                    content=text,
                )
            )
        else:
            rows = block.table_rows or []
            cleaned_rows = [
                [_normalize_cell_or_text(cell) for cell in row]
                for row in rows
                if any(_normalize_cell_or_text(c) for c in row) or len(row) > 0
            ]
            if not cleaned_rows:
                fallback_text = _normalize_cell_or_text(block.text_content)
                if fallback_text:
                    canonical_blocks.append(
                        BodyBlock(
                            type="paragraph",
                            alignment=block.alignment,
                            first_line_indent=block.first_line_indent,
                            bold=block.bold,
                            italic=block.italic,
                            content=fallback_text,
                        )
                    )
                continue
            canonical_blocks.append(
                BodyBlock(
                    type="table",
                    alignment=block.alignment,
                    first_line_indent=False,
                    bold=block.bold,
                    italic=block.italic,
                    content=cleaned_rows,
                )
            )

    return PageStructure(
        page_number=page_number,
        header_table=header,
        title_block=title,
        body_blocks=canonical_blocks,
        footer_table=footer,
    )


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
