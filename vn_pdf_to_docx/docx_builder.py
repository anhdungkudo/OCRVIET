"""Step 3: Microsoft Word (.docx) reconstruction following Nghị định 30/2020/NĐ-CP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

try:
    from docx import Document
    from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    _DOCX_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - surfaced clearly at runtime when missing
    Document = None  # type: ignore[assignment]
    WD_ALIGN_VERTICAL = None  # type: ignore[assignment]
    WD_TABLE_ALIGNMENT = None  # type: ignore[assignment]
    WD_ALIGN_PARAGRAPH = None  # type: ignore[assignment]
    OxmlElement = None  # type: ignore[assignment]
    qn = None  # type: ignore[assignment]
    Cm = None  # type: ignore[assignment]
    Pt = None  # type: ignore[assignment]
    _DOCX_IMPORT_ERROR = exc

from .exceptions import DocxBuildError
from .schemas import (
    AdministrativeDocument,
    BodyBlock,
    FooterTable,
    HeaderTable,
    TitleBlock,
)


class DocxRenderConfig(BaseModel):
    """Layout and typography configuration (defaults match Nghị định 30/2020/NĐ-CP)."""

    page_width_cm: float = Field(default=21.0, gt=5.0)
    page_height_cm: float = Field(default=29.7, gt=5.0)
    margin_left_cm: float = Field(default=3.0, ge=0.5)
    margin_right_cm: float = Field(default=1.5, ge=0.5)
    margin_top_cm: float = Field(default=2.0, ge=0.5)
    margin_bottom_cm: float = Field(default=2.0, ge=0.5)
    font_name: str = Field(default="Times New Roman")
    font_size_pt: float = Field(default=13.0, gt=6.0, le=72.0)
    first_line_indent_cm: float = Field(default=1.27, ge=0.0)
    line_spacing: float = Field(default=1.15, ge=1.0, le=3.0)
    paragraph_space_after_pt: float = Field(default=6.0, ge=0.0)
    header_left_width_cm: float = Field(default=7.2, gt=1.0)
    footer_left_width_cm: float = Field(default=8.5, gt=1.0)
    footer_left_alignment: Literal["center", "left"] = Field(
        default="center",
        description="Căn lề cột trái 'Nơi nhận' trong footer_table: mặc định 'center' theo yêu cầu căn giữa từng cột chuẩn xác, hoặc 'left' nếu muốn căn trái theo Nghị định 30/2020/NĐ-CP.",
    )
    signature_space_pt: float = Field(default=48.0, ge=0.0)

    @property
    def usable_width_cm(self) -> float:
        """Return printable text width inside left/right margins."""
        width = self.page_width_cm - self.margin_left_cm - self.margin_right_cm
        if width <= 1.0:
            raise DocxBuildError(
                f"Usable page width ({width:.2f}cm) is too narrow for margins."
            )
        return width


def _require_docx() -> None:
    """Ensure python-docx is installed before rendering Word documents."""
    if _DOCX_IMPORT_ERROR is not None or Document is None:
        raise DocxBuildError(
            "Missing required package 'python-docx'. Install it with: pip install python-docx"
        ) from _DOCX_IMPORT_ERROR


# ECMA-376 WordprocessingML element sequence orders for table and cell properties.
_TBL_PR_ORDER: tuple[str, ...] = (
    "w:tblStyle",
    "w:tblpPr",
    "w:tblOverlap",
    "w:bidiVisual",
    "w:tblStyleRowBandSize",
    "w:tblStyleColBandSize",
    "w:tblW",
    "w:jc",
    "w:tblCellSpacing",
    "w:tblInd",
    "w:tblBorders",
    "w:shd",
    "w:tblLayout",
    "w:tblCellMar",
    "w:tblLook",
)

_TC_PR_ORDER: tuple[str, ...] = (
    "w:cnfStyle",
    "w:tcW",
    "w:gridSpan",
    "w:hMerge",
    "w:vMerge",
    "w:tcBorders",
    "w:shd",
    "w:noWrap",
    "w:tcMar",
    "w:textDirection",
    "w:tcFitText",
    "w:vAlign",
    "w:hideMark",
)


def _insert_in_schema_order(
    parent: Any,
    element: Any,
    schema_order: tuple[str, ...],
) -> None:
    """Insert an OxmlElement child into parent respecting strict OOXML sequence order."""
    order_map = {qn(tag): idx for idx, tag in enumerate(schema_order)}
    target_rank = order_map.get(element.tag, len(schema_order))
    for idx, existing in enumerate(list(parent)):
        existing_rank = order_map.get(existing.tag, len(schema_order))
        if existing_rank > target_rank:
            parent.insert(idx, element)
            return
    parent.append(element)


def _compute_two_column_widths(usable_cm: float, preferred_left_cm: float) -> tuple[float, float]:
    """Split usable page width into two strictly positive column widths."""
    min_right = min(2.0, usable_cm * 0.4)
    left_w = max(0.5, min(preferred_left_cm, usable_cm - min_right))
    right_w = max(0.5, usable_cm - left_w)
    return left_w, right_w


def _set_element_rfonts(element: Any, font_name: str) -> None:
    """Set ASCII, HAnsi, EastAsia, and ComplexScript font attributes without theme overrides."""
    r_pr = element.get_or_add_rPr()
    r_fonts = r_pr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        r_fonts.set(qn(attr), font_name)
    for theme_attr in (
        "w:asciiTheme",
        "w:hAnsiTheme",
        "w:eastAsiaTheme",
        "w:cstheme",
        "w:csTheme",
    ):
        attr_qn = qn(theme_attr)
        if attr_qn in r_fonts.attrib:
            del r_fonts.attrib[attr_qn]


def _apply_run_formatting(
    run: Any,
    *,
    font_name: str,
    size_pt: float,
    bold: bool = False,
    italic: bool = False,
) -> None:
    """Apply consistent Times New Roman font styling to a docx Run."""
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bool(bold)
    run.italic = bool(italic)
    _set_element_rfonts(run._element, font_name)


def configure_page_and_styles(doc: Any, config: DocxRenderConfig) -> None:
    """Configure A4 paper size, Nghị định 30/2020/NĐ-CP margins, and Normal style."""
    for section in doc.sections:
        section.page_width = Cm(config.page_width_cm)
        section.page_height = Cm(config.page_height_cm)
        section.left_margin = Cm(config.margin_left_cm)
        section.right_margin = Cm(config.margin_right_cm)
        section.top_margin = Cm(config.margin_top_cm)
        section.bottom_margin = Cm(config.margin_bottom_cm)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = config.font_name
    normal_style.font.size = Pt(config.font_size_pt)
    _set_element_rfonts(normal_style.element, config.font_name)


def _set_invisible_borders_xml(
    border_parent: Any,
    container_tag: str,
    edges: tuple[str, ...],
    schema_order: tuple[str, ...],
) -> None:
    """Write `<w:tblBorders>` or `<w:tcBorders>` with all specified sides set to 'none'."""
    borders_el = border_parent.find(qn(container_tag))
    if borders_el is None:
        borders_el = OxmlElement(container_tag)
        _insert_in_schema_order(border_parent, borders_el, schema_order)
    else:
        borders_el.clear()

    for edge in edges:
        edge_el = OxmlElement(f"w:{edge}")
        edge_el.set(qn("w:val"), "none")
        edge_el.set(qn("w:sz"), "0")
        edge_el.set(qn("w:space"), "0")
        edge_el.set(qn("w:color"), "auto")
        borders_el.append(edge_el)


def set_table_invisible_borders(table: Any) -> None:
    """Ensure a python-docx Table has completely invisible borders in Word & LibreOffice."""
    tbl_pr = table._tbl.tblPr
    _set_invisible_borders_xml(
        tbl_pr,
        "w:tblBorders",
        ("top", "left", "bottom", "right", "insideH", "insideV"),
        _TBL_PR_ORDER,
    )
    for row in table.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            _set_invisible_borders_xml(
                tc_pr,
                "w:tcBorders",
                ("top", "left", "bottom", "right"),
                _TC_PR_ORDER,
            )


def _set_table_grid_borders(table: Any) -> None:
    """Apply 'Table Grid' style with explicit single black border fallback if unavailable."""
    try:
        table.style = "Table Grid"
        return
    except KeyError:
        pass

    tbl_pr = table._tbl.tblPr
    borders_el = tbl_pr.find(qn("w:tblBorders"))
    if borders_el is None:
        borders_el = OxmlElement("w:tblBorders")
        _insert_in_schema_order(tbl_pr, borders_el, _TBL_PR_ORDER)
    else:
        borders_el.clear()

    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge_el = OxmlElement(f"w:{edge}")
        edge_el.set(qn("w:val"), "single")
        edge_el.set(qn("w:sz"), "4")
        edge_el.set(qn("w:space"), "0")
        edge_el.set(qn("w:color"), "000000")
        borders_el.append(edge_el)


def _set_table_column_widths(table: Any, col_widths_cm: list[float]) -> None:
    """Lock fixed total table width and fixed column/cell widths."""
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    total_twips = sum(Cm(width_cm).twips for width_cm in col_widths_cm)
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        _insert_in_schema_order(tbl_pr, tbl_w, _TBL_PR_ORDER)
    tbl_w.set(qn("w:w"), str(total_twips))
    tbl_w.set(qn("w:type"), "dxa")

    for idx, width_cm in enumerate(col_widths_cm):
        width = Cm(width_cm)
        table.columns[idx].width = width
    for row in table.rows:
        for idx, width_cm in enumerate(col_widths_cm):
            row.cells[idx].width = Cm(width_cm)


def _set_cell_margins(
    cell: Any,
    *,
    top_dxa: int = 60,
    bottom_dxa: int = 60,
    left_dxa: int = 100,
    right_dxa: int = 100,
) -> None:
    """Set internal cell padding in twips (dxa) via <w:tcMar> in schema order."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        _insert_in_schema_order(tc_pr, tc_mar, _TC_PR_ORDER)
    else:
        tc_mar.clear()

    for side, val in (
        ("top", top_dxa),
        ("bottom", bottom_dxa),
        ("left", left_dxa),
        ("right", right_dxa),
    ):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)


def _write_cell_paragraphs(
    cell: Any,
    entries: list[dict[str, Any]],
    *,
    default_alignment: Any,
    config: DocxRenderConfig,
) -> None:
    """Write formatted lines into a table cell reusing its initial empty paragraph."""
    if not entries:
        p = cell.paragraphs[0]
        p.alignment = default_alignment
        p.paragraph_format.first_line_indent = Cm(0)
        return

    for idx, entry in enumerate(entries):
        p = cell.paragraphs[0] if idx == 0 else cell.add_paragraph()
        p.alignment = entry.get("alignment", default_alignment)
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(entry.get("space_before_pt", 0.0))
        p.paragraph_format.space_after = Pt(entry.get("space_after_pt", 2.0))
        p.paragraph_format.line_spacing = config.line_spacing

        text = entry.get("text", "")
        run = p.add_run(text)
        _apply_run_formatting(
            run,
            font_name=config.font_name,
            size_pt=entry.get("size_pt", config.font_size_pt),
            bold=entry.get("bold", False),
            italic=entry.get("italic", False),
        )


def _resolve_alignment(alignment_name: str) -> Any:
    """Map schema alignment string to python-docx WD_ALIGN_PARAGRAPH enum."""
    mapping = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }
    if alignment_name not in mapping:
        raise DocxBuildError(f"Unsupported paragraph alignment: {alignment_name!r}")
    return mapping[alignment_name]


def _ensure_table_separation(
    doc: Any,
    *,
    space_before_pt: float = 0.0,
    space_after_pt: float = 0.0,
    insert_if_empty: bool = False,
) -> None:
    """Insert a compact separating paragraph whenever the preceding body element is a table."""
    sect_pr_tag = qn("w:sectPr") if qn is not None else None
    tbl_tag = qn("w:tbl") if qn is not None else None
    body_children = [
        el
        for el in doc._body._body
        if (el.tag != sect_pr_tag if sect_pr_tag is not None else not el.tag.endswith("sectPr"))
    ]
    if not body_children:
        if insert_if_empty:
            _add_compact_spacer(doc, space_before_pt=space_before_pt, space_after_pt=space_after_pt)
        return
    last_tag = body_children[-1].tag
    is_tbl = last_tag == tbl_tag if tbl_tag is not None else last_tag.endswith("tbl")
    if is_tbl:
        _add_compact_spacer(doc, space_before_pt=space_before_pt, space_after_pt=space_after_pt)


def add_header_table(doc: Any, header: Optional[HeaderTable], config: DocxRenderConfig) -> None:
    """Render two-column borderless administrative header table with centered columns."""
    if header is None or header.is_empty():
        return

    _ensure_table_separation(doc, space_after_pt=config.paragraph_space_after_pt)

    usable_cm = config.usable_width_cm
    left_w, right_w = _compute_two_column_widths(usable_cm, config.header_left_width_cm)

    table = doc.add_table(rows=1, cols=2)
    set_table_invisible_borders(table)
    _set_table_column_widths(table, [left_w, right_w])

    left_cell = table.cell(0, 0)
    right_cell = table.cell(0, 1)
    left_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    right_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    _set_cell_margins(left_cell, top_dxa=0, bottom_dxa=40, left_dxa=40, right_dxa=60)
    _set_cell_margins(right_cell, top_dxa=0, bottom_dxa=40, left_dxa=60, right_dxa=40)

    left_entries: list[dict[str, Any]] = []
    if header.co_quan_chu_quan:
        for line in header.co_quan_chu_quan.splitlines():
            if line.strip():
                left_entries.append({"text": line.strip(), "bold": False, "italic": False})
    if header.co_quan_ban_hanh:
        for line in header.co_quan_ban_hanh.splitlines():
            if line.strip():
                left_entries.append({"text": line.strip(), "bold": True, "italic": False})
    if header.so_ky_hieu:
        so_lines = [ln.strip() for ln in header.so_ky_hieu.splitlines() if ln.strip()]
        for idx, line in enumerate(so_lines):
            left_entries.append(
                {
                    "text": line,
                    "bold": False,
                    "italic": False,
                    "space_before_pt": 4.0 if idx == 0 else 0.0,
                }
            )

    right_entries: list[dict[str, Any]] = []
    if header.quoc_hieu:
        for line in header.quoc_hieu.splitlines():
            if line.strip():
                right_entries.append({"text": line.strip(), "bold": True, "italic": False})
    if header.tieu_ngu:
        for line in header.tieu_ngu.splitlines():
            if line.strip():
                right_entries.append({"text": line.strip(), "bold": True, "italic": False})
    if header.dia_danh_ngay_thang:
        date_lines = [ln.strip() for ln in header.dia_danh_ngay_thang.splitlines() if ln.strip()]
        for idx, line in enumerate(date_lines):
            right_entries.append(
                {
                    "text": line,
                    "bold": False,
                    "italic": True,
                    "space_before_pt": 4.0 if idx == 0 else 0.0,
                }
            )

    _write_cell_paragraphs(
        left_cell,
        left_entries,
        default_alignment=WD_ALIGN_PARAGRAPH.CENTER,
        config=config,
    )
    _write_cell_paragraphs(
        right_cell,
        right_entries,
        default_alignment=WD_ALIGN_PARAGRAPH.CENTER,
        config=config,
    )


def add_title_block(doc: Any, title: Optional[TitleBlock], config: DocxRenderConfig) -> None:
    """Render centered, bold document type name and abstract."""
    if title is None or title.is_empty():
        return

    if title.ten_loai_van_ban:
        lines = [ln.strip() for ln in title.ten_loai_van_ban.splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(12.0 if idx == 0 else 0.0)
            p.paragraph_format.space_after = Pt(2.0)
            p.paragraph_format.line_spacing = config.line_spacing
            run = p.add_run(line)
            _apply_run_formatting(
                run,
                font_name=config.font_name,
                size_pt=config.font_size_pt,
                bold=True,
                italic=False,
            )

    if title.trich_yeu:
        lines = [ln.strip() for ln in title.trich_yeu.splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(
                10.0 if (not title.ten_loai_van_ban and idx == 0) else 0.0
            )
            p.paragraph_format.space_after = Pt(
                10.0 if idx == len(lines) - 1 else 2.0
            )
            p.paragraph_format.line_spacing = config.line_spacing
            run = p.add_run(line)
            _apply_run_formatting(
                run,
                font_name=config.font_name,
                size_pt=config.font_size_pt,
                bold=True,
                italic=False,
            )


def _add_paragraph_block(doc: Any, block: BodyBlock, config: DocxRenderConfig) -> None:
    """Render a paragraph BodyBlock honoring alignment, indent, bold, and italic."""
    text = "" if block.content is None else str(block.content)
    if not text.strip():
        return
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return

    alignment = _resolve_alignment(block.alignment)
    indent = Cm(config.first_line_indent_cm) if block.first_line_indent else Cm(0)

    for line in lines:
        p = doc.add_paragraph()
        p.alignment = alignment
        p.paragraph_format.first_line_indent = indent
        p.paragraph_format.space_before = Pt(0.0)
        p.paragraph_format.space_after = Pt(config.paragraph_space_after_pt)
        p.paragraph_format.line_spacing = config.line_spacing

        run = p.add_run(line)
        _apply_run_formatting(
            run,
            font_name=config.font_name,
            size_pt=config.font_size_pt,
            bold=block.bold,
            italic=block.italic,
        )


def _add_compact_spacer(doc: Any, *, space_before_pt: float = 0.0, space_after_pt: float = 0.0) -> None:
    """Insert a compact non-intrusive separating paragraph (e.g. between adjacent tables)."""
    spacer = doc.add_paragraph()
    spacer.paragraph_format.first_line_indent = Cm(0)
    spacer.paragraph_format.space_before = Pt(space_before_pt)
    spacer.paragraph_format.space_after = Pt(space_after_pt)
    spacer.paragraph_format.line_spacing = Pt(4.0)
    spacer_run = spacer.add_run("")
    spacer_run.font.size = Pt(4.0)


def _add_table_block(
    doc: Any,
    block: BodyBlock,
    config: DocxRenderConfig,
    *,
    add_trailing_spacer: bool = True,
) -> None:
    """Render a 2D table BodyBlock with 'Table Grid' borders and safe ragged-row padding."""
    rows = block.content
    if not isinstance(rows, list) or not rows:
        return

    num_cols = max((len(r) for r in rows), default=0)
    if num_cols == 0:
        return

    _ensure_table_separation(doc, space_after_pt=config.paragraph_space_after_pt)

    num_rows = len(rows)
    table = doc.add_table(rows=num_rows, cols=num_cols)
    _set_table_grid_borders(table)

    col_width = config.usable_width_cm / float(num_cols)
    _set_table_column_widths(table, [col_width] * num_cols)

    alignment = _resolve_alignment(block.alignment)
    for row_idx, row_data in enumerate(rows):
        for col_idx in range(num_cols):
            raw_cell = row_data[col_idx] if col_idx < len(row_data) else ""
            cell_text = "" if raw_cell is None else str(raw_cell)
            cell = table.cell(row_idx, col_idx)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_margins(cell, top_dxa=60, bottom_dxa=60, left_dxa=90, right_dxa=90)
            cell_lines = cell_text.splitlines() if cell_text else [""]
            entries = [
                {
                    "text": ln,
                    "bold": block.bold,
                    "italic": block.italic,
                    "space_after_pt": 1.0,
                }
                for ln in cell_lines
            ]
            _write_cell_paragraphs(
                cell,
                entries,
                default_alignment=alignment,
                config=config,
            )

    # Prevent Word from merging adjacent tables when consecutive table blocks appear.
    if add_trailing_spacer:
        _add_compact_spacer(doc, space_after_pt=config.paragraph_space_after_pt)


def _build_two_col_cell_entries(
    cell_text: str,
    *,
    bold: bool,
    italic: bool,
    signature_space_pt: float,
) -> list[dict[str, Any]]:
    """Parse cell lines for a 2-column borderless table, converting blank lines to signature space."""
    if not cell_text or not cell_text.strip():
        return []
    raw_lines = cell_text.splitlines()
    entries: list[dict[str, Any]] = []
    pending_space_before = 0.0
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            if entries:
                pending_space_before = signature_space_pt
            continue
        is_noinhan = len(entries) == 0 and stripped.lower().startswith("nơi nhận")
        upper_line = stripped.upper()
        is_quochhieu = upper_line.startswith(
            ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM")
        )
        normalized_dash = stripped.lower().replace("–", "-").replace("—", "-")
        is_tieungu = normalized_dash.startswith("độc lập - tự do - hạnh phúc")
        entries.append(
            {
                "text": stripped,
                "bold": bool(bold or is_noinhan or is_quochhieu or is_tieungu),
                "italic": bool(italic or is_noinhan),
                "space_before_pt": pending_space_before,
                "space_after_pt": 2.0,
            }
        )
        pending_space_before = 0.0
    return entries


def _add_two_column_table_block(
    doc: Any,
    block: BodyBlock,
    config: DocxRenderConfig,
    *,
    add_trailing_spacer: bool = True,
) -> None:
    """Render grouped `two_column_row` lines as a 2-column invisible table with centered columns."""
    rows = block.content
    if not isinstance(rows, list) or not rows:
        return

    _ensure_table_separation(doc, space_after_pt=config.paragraph_space_after_pt)

    usable_cm = config.usable_width_cm
    all_left = "\n".join(str(r[0]) if len(r) > 0 and r[0] is not None else "" for r in rows)
    all_right = "\n".join(str(r[1]) if len(r) > 1 and r[1] is not None else "" for r in rows)

    max_right_len = max((len(ln.strip()) for ln in all_right.splitlines()), default=0)
    starts_noinhan = all_left.lstrip().lower().startswith("nơi nhận")

    if max_right_len >= 24:
        left_w, right_w = _compute_two_column_widths(usable_cm, config.header_left_width_cm)
    elif starts_noinhan:
        left_w, right_w = _compute_two_column_widths(usable_cm, config.footer_left_width_cm)
    else:
        left_w = usable_cm / 2.0
        right_w = usable_cm / 2.0

    table = doc.add_table(rows=len(rows), cols=2)
    set_table_invisible_borders(table)
    _set_table_column_widths(table, [left_w, right_w])

    left_align = (
        WD_ALIGN_PARAGRAPH.LEFT
        if (starts_noinhan and config.footer_left_alignment == "left")
        else WD_ALIGN_PARAGRAPH.CENTER
    )

    for row_idx, row_data in enumerate(rows):
        left_text = "" if len(row_data) < 1 or row_data[0] is None else str(row_data[0])
        right_text = "" if len(row_data) < 2 or row_data[1] is None else str(row_data[1])

        left_cell = table.cell(row_idx, 0)
        right_cell = table.cell(row_idx, 1)
        left_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        right_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        _set_cell_margins(left_cell, top_dxa=20, bottom_dxa=40, left_dxa=40, right_dxa=60)
        _set_cell_margins(right_cell, top_dxa=20, bottom_dxa=40, left_dxa=60, right_dxa=40)

        left_entries = _build_two_col_cell_entries(
            left_text,
            bold=block.bold,
            italic=block.italic,
            signature_space_pt=config.signature_space_pt,
        )
        right_entries = _build_two_col_cell_entries(
            right_text,
            bold=block.bold,
            italic=block.italic,
            signature_space_pt=config.signature_space_pt,
        )
        _write_cell_paragraphs(
            left_cell,
            left_entries,
            default_alignment=left_align,
            config=config,
        )
        _write_cell_paragraphs(
            right_cell,
            right_entries,
            default_alignment=WD_ALIGN_PARAGRAPH.CENTER,
            config=config,
        )

    if add_trailing_spacer:
        _add_compact_spacer(doc, space_after_pt=config.paragraph_space_after_pt)


def add_body_blocks(doc: Any, blocks: list[BodyBlock], config: DocxRenderConfig) -> None:
    """Render ordered body paragraphs, bordered tables, and 2-column invisible tables."""
    for idx, block in enumerate(blocks):
        if block.type == "paragraph":
            _add_paragraph_block(doc, block, config)
        elif block.type == "table":
            # Add trailing spacer if followed by another table or at the end (prevents
            # merging adjacent tables; footer_table checks/avoids duplicating a spacer).
            next_is_table = (
                idx + 1 < len(blocks)
                and blocks[idx + 1].type in ("table", "two_column_table")
            )
            _add_table_block(doc, block, config, add_trailing_spacer=next_is_table or idx == len(blocks) - 1)
        elif block.type == "two_column_table":
            next_is_table = (
                idx + 1 < len(blocks)
                and blocks[idx + 1].type in ("table", "two_column_table")
            )
            _add_two_column_table_block(
                doc,
                block,
                config,
                add_trailing_spacer=next_is_table or idx == len(blocks) - 1,
            )
        else:
            raise DocxBuildError(f"Unsupported BodyBlock type: {block.type!r}")


def add_footer_table(doc: Any, footer: Optional[FooterTable], config: DocxRenderConfig) -> None:
    """Render two-column borderless footer table (Nơi nhận | Chức vụ & Họ tên người ký)."""
    if footer is None or footer.is_empty():
        return

    # Add a compact separating paragraph only if not already preceded by a trailing paragraph
    # (prevents Word from merging a preceding body table into footer_table while avoiding double spacers).
    _ensure_table_separation(
        doc,
        space_before_pt=4.0,
        space_after_pt=0.0,
        insert_if_empty=True,
    )

    usable_cm = config.usable_width_cm
    left_w, right_w = _compute_two_column_widths(usable_cm, config.footer_left_width_cm)

    table = doc.add_table(rows=1, cols=2)
    set_table_invisible_borders(table)
    _set_table_column_widths(table, [left_w, right_w])

    left_cell = table.cell(0, 0)
    right_cell = table.cell(0, 1)
    left_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    right_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    _set_cell_margins(left_cell, top_dxa=40, bottom_dxa=40, left_dxa=40, right_dxa=80)
    _set_cell_margins(right_cell, top_dxa=40, bottom_dxa=40, left_dxa=80, right_dxa=40)

    left_entries: list[dict[str, Any]] = []
    if footer.noi_nhan:
        lines = [ln.strip() for ln in footer.noi_nhan.splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            is_heading = idx == 0 and line.lower().startswith("nơi nhận")
            left_entries.append(
                {
                    "text": line,
                    "bold": is_heading,
                    "italic": is_heading,
                    "space_after_pt": 1.5,
                }
            )

    right_entries: list[dict[str, Any]] = []
    if footer.chuc_vu_nguoi_ky:
        lines = [ln.strip() for ln in footer.chuc_vu_nguoi_ky.splitlines() if ln.strip()]
        for line in lines:
            right_entries.append(
                {
                    "text": line,
                    "bold": True,
                    "italic": False,
                    "space_after_pt": 2.0,
                }
            )

    if footer.ho_ten_nguoi_ky:
        name_lines = [ln.strip() for ln in footer.ho_ten_nguoi_ky.splitlines() if ln.strip()]
        has_title = bool(right_entries)
        for idx, line in enumerate(name_lines):
            right_entries.append(
                {
                    "text": line,
                    "bold": True,
                    "italic": False,
                    "space_before_pt": config.signature_space_pt if (has_title and idx == 0) else 0.0,
                    "space_after_pt": 2.0,
                }
            )

    left_align = (
        WD_ALIGN_PARAGRAPH.CENTER
        if config.footer_left_alignment == "center"
        else WD_ALIGN_PARAGRAPH.LEFT
    )
    _write_cell_paragraphs(
        left_cell,
        left_entries,
        default_alignment=left_align,
        config=config,
    )
    _write_cell_paragraphs(
        right_cell,
        right_entries,
        default_alignment=WD_ALIGN_PARAGRAPH.CENTER,
        config=config,
    )


def _has_mid_document_footer(document_data: AdministrativeDocument) -> bool:
    """Return True if an earlier page already ended with a footer before subsequent pages."""
    if len(document_data.pages) <= 1:
        return False
    ordered_pages = sorted(document_data.pages, key=lambda p: p.page_number)
    return any(
        page.footer_table is not None and not page.footer_table.is_empty()
        for page in ordered_pages[:-1]
    )


def build_docx(
    document_data: AdministrativeDocument,
    output_path: Union[str, Path],
    config: Optional[DocxRenderConfig] = None,
) -> Path:
    """Build and save a formatted Microsoft Word (.docx) from an AdministrativeDocument.

    Args:
        document_data: Structured Vietnamese administrative document extracted by Gemini.
        output_path: Destination `.docx` file path.
        config: Optional layout/typography overrides.

    Returns:
        Resolved `Path` to the written `.docx` file.

    Raises:
        DocxBuildError: If `python-docx` is unavailable, data is invalid, or saving fails.
    """
    _require_docx()
    if not isinstance(document_data, AdministrativeDocument):
        raise DocxBuildError(
            f"Expected AdministrativeDocument instance, got {type(document_data).__name__}."
        )

    cfg = config or DocxRenderConfig()
    out_path = Path(output_path)

    try:
        doc = Document()
        configure_page_and_styles(doc, cfg)
        if _has_mid_document_footer(document_data):
            ordered_pages = sorted(document_data.pages, key=lambda p: p.page_number)
            for idx, page in enumerate(ordered_pages):
                if idx > 0 and (
                    (ordered_pages[idx - 1].footer_table is not None and not ordered_pages[idx - 1].footer_table.is_empty())
                    or (page.header_table is not None and not page.header_table.is_empty())
                ):
                    doc.add_page_break()
                add_header_table(doc, page.header_table, cfg)
                add_title_block(doc, page.title_block, cfg)
                add_body_blocks(doc, page.body_blocks, cfg)
                add_footer_table(doc, page.footer_table, cfg)
        else:
            add_header_table(doc, document_data.header_table, cfg)
            add_title_block(doc, document_data.title_block, cfg)
            add_body_blocks(doc, document_data.body_blocks, cfg)
            add_footer_table(doc, document_data.footer_table, cfg)

        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
        return out_path.resolve()
    except DocxBuildError:
        raise
    except Exception as exc:
        raise DocxBuildError(f"Failed to build or save DOCX '{out_path}': {exc}") from exc
