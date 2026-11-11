"""Step 1: PDF/A rendering (PyMuPDF) and red-seal removal + enhancement (OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24 avoids fitz deprecation import warning
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

from .exceptions import ImagePreprocessError, PdfRenderError


class PreprocessConfig(BaseModel):
    """Configuration parameters for PDF rendering and OpenCV preprocessing."""

    dpi: int = Field(default=200, ge=72, le=1200)
    red_suppression_strength: float = Field(default=1.3, ge=0.0, le=3.0)
    nlm_h: float = Field(default=10.0, ge=0.0, le=50.0)
    nlm_template_window_size: int = Field(default=7, ge=3, le=21)
    nlm_search_window_size: int = Field(default=21, ge=5, le=51)
    clahe_clip_limit: float = Field(default=2.0, gt=0.0, le=20.0)
    clahe_tile_grid_size: tuple[int, int] = Field(default=(8, 8))
    image_format: Literal["png", "jpeg"] = Field(default="png")
    jpeg_quality: int = Field(default=95, ge=50, le=100)

    @field_validator("nlm_template_window_size", "nlm_search_window_size")
    @classmethod
    def _validate_odd_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError(
                f"fastNlMeansDenoising window size must be odd, got {value}."
            )
        return value

    @field_validator("clahe_tile_grid_size")
    @classmethod
    def _validate_tile_grid(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or value[0] < 1 or value[1] < 1:
            raise ValueError(
                f"clahe_tile_grid_size must be two positive integers, got {value}."
            )
        return value


@dataclass(frozen=True)
class PreprocessedPage:
    """Container holding a preprocessed page image and its encoded payload."""

    page_number: int
    image: np.ndarray
    encoded_bytes: bytes
    mime_type: str
    is_landscape: bool = False


def render_pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 200,
    *,
    page_numbers: Optional[Iterable[int]] = None,
) -> list[np.ndarray]:
    """Render pages of a PDF/A file into 200 DPI RGB numpy arrays via PyMuPDF (`fitz`).

    Args:
        pdf_path: Path to input PDF file.
        dpi: Target rasterization DPI (default 200).
        page_numbers: Optional 1-based page numbers to render. Renders all pages if None.

    Returns:
        List of RGB `np.ndarray` images with shape `(H, W, 3)` and dtype `np.uint8`.

    Raises:
        PdfRenderError: If the PDF is missing, unreadable, password-protected, empty,
            or any requested page index is invalid.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72 or dpi > 1200:
        raise PdfRenderError(f"DPI must be an integer between 72 and 1200, got {dpi!r}.")

    path = Path(pdf_path)
    if not path.exists():
        raise PdfRenderError(f"PDF file not found: {path}")
    if not path.is_file():
        raise PdfRenderError(f"Expected a PDF file path, got non-file: {path}")
    if path.stat().st_size == 0:
        raise PdfRenderError(f"PDF file is empty (0 bytes): {path}")

    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(str(path))
        if doc.needs_pass:
            raise PdfRenderError(f"PDF file is encrypted/password-protected: {path}")

        total_pages = len(doc)
        if total_pages == 0:
            raise PdfRenderError(f"PDF file contains 0 pages: {path}")

        if page_numbers is None:
            target_indices = list(range(total_pages))
        else:
            requested_pages = list(page_numbers)
            if not requested_pages:
                raise PdfRenderError("page_numbers must not be empty.")
            target_indices = []
            for pno in requested_pages:
                if isinstance(pno, bool) or not isinstance(pno, int) or pno < 1 or pno > total_pages:
                    raise PdfRenderError(
                        f"Requested page number {pno} is out of range [1, {total_pages}]."
                    )
                target_indices.append(pno - 1)

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        images: list[np.ndarray] = []

        for idx in target_indices:
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            if pix.width <= 0 or pix.height <= 0 or pix.n != 3:
                raise PdfRenderError(
                    f"Unexpected pixmap geometry on page {idx + 1}: "
                    f"({pix.width}, {pix.height}, n={pix.n})."
                )
            arr = (
                np.frombuffer(pix.samples, dtype=np.uint8)
                .reshape(pix.height, pix.width, pix.n)
                .copy()
            )
            images.append(arr)

        return images
    except PdfRenderError:
        raise
    except Exception as exc:
        raise PdfRenderError(f"Failed to render PDF '{path}' at {dpi} DPI: {exc}") from exc
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _coerce_uint8_image(image: np.ndarray) -> np.ndarray:
    """Validate an image numpy array and coerce numeric dtypes safely to uint8."""
    if not isinstance(image, np.ndarray):
        raise ImagePreprocessError(
            f"Expected numpy.ndarray image, got {type(image).__name__}."
        )
    if image.size == 0 or image.ndim not in (2, 3):
        raise ImagePreprocessError(
            f"Invalid image shape {image.shape}; expected non-empty 2D or 3D array."
        )
    if not np.issubdtype(image.dtype, np.number):
        raise ImagePreprocessError(f"Image dtype must be numeric, got {image.dtype}.")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        finite = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        max_val = float(np.max(finite))
        min_val = float(np.min(finite))
        if min_val >= 0.0 and max_val <= 1.0:
            return np.clip(np.rint(finite * 255.0), 0, 255).astype(np.uint8)
        return np.clip(finite, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def extract_red_channel_and_suppress_seal(
    image: np.ndarray,
    *,
    color_order: Literal["RGB", "BGR"] = "RGB",
    red_suppression_strength: float = 1.3,
) -> np.ndarray:
    """Isolate the Red channel and softly neutralize red-seal ink without breaking black text.

    In the Red channel, red stamp ink already has high reflectance while black ink
    (even when overlapped by a red seal) remains dark across all channels. Adding a
    brightness-gated red-excess lift matched to the page's background luminance cleanly
    eliminates darker crimson seal remnants while keeping overlapped black strokes solid.

    Args:
        image: Input image array `(H, W)`, `(H, W, 1)`, `(H, W, 3)`, or `(H, W, 4)`.
        color_order: Channel order of 3/4-channel input ('RGB' from PyMuPDF or 'BGR' from OpenCV).
        red_suppression_strength: Chrominance suppression multiplier (0.0 returns raw Red channel).

    Returns:
        Single-channel `(H, W)` `uint8` grayscale image.
    """
    if red_suppression_strength < 0.0:
        raise ImagePreprocessError(
            f"red_suppression_strength must be >= 0, got {red_suppression_strength}."
        )
    if color_order not in ("RGB", "BGR"):
        raise ImagePreprocessError(
            f"Unsupported color_order {color_order!r}; expected 'RGB' or 'BGR'."
        )

    try:
        arr = _coerce_uint8_image(image)
        if arr.ndim == 2:
            return arr.copy()

        channels = arr.shape[2]
        if channels == 1:
            return arr[:, :, 0].copy()
        if channels not in (3, 4):
            raise ImagePreprocessError(
                f"Unsupported channel count {channels}; expected 1, 3, or 4."
            )

        split_channels = cv2.split(arr)
        if color_order == "RGB":
            r, g, b = split_channels[0], split_channels[1], split_channels[2]
        else:
            b, g, r = split_channels[0], split_channels[1], split_channels[2]

        if red_suppression_strength == 0.0:
            return r.copy()

        r_f = r.astype(np.float32)
        g_f = g.astype(np.float32)
        b_f = b.astype(np.float32)

        # Estimate white paper luminance so darker scans do not get bright white halos.
        bg_level = float(np.percentile(r_f, 95))
        red_excess = np.maximum(0.0, r_f - np.maximum(g_f, b_f))
        # Background-relative gate protects both crisp black and faded text covered by
        # translucent or wet red stamp ink while cleanly fading pure stamp ink on paper.
        gate_low = max(85.0, 0.62 * bg_level)
        gate_high = max(gate_low + 20.0, 0.78 * bg_level)
        bright_gate = np.clip((r_f - gate_low) / (gate_high - gate_low), 0.0, 1.0)
        suppressed = np.minimum(
            bg_level, r_f + float(red_suppression_strength) * bright_gate * red_excess
        )
        cleaned = np.clip(np.maximum(suppressed, r_f), 0.0, 255.0).astype(np.uint8)
        return cleaned
    except ImagePreprocessError:
        raise
    except Exception as exc:
        raise ImagePreprocessError(
            f"Failed to extract Red channel and suppress red seal: {exc}"
        ) from exc


def denoise_and_enhance_contrast(
    gray_image: np.ndarray,
    *,
    nlm_h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply `cv2.fastNlMeansDenoising` and `CLAHE` to sharpen faded scanned text."""
    try:
        arr = _coerce_uint8_image(gray_image)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim != 2:
            raise ImagePreprocessError(
                f"Expected 2D grayscale image for denoising/CLAHE, got shape {arr.shape}."
            )

        if nlm_h < 0.0:
            raise ImagePreprocessError(f"nlm_h must be >= 0, got {nlm_h}.")
        if template_window_size < 3 or template_window_size % 2 == 0:
            raise ImagePreprocessError(
                f"template_window_size must be an odd integer >= 3, got {template_window_size}."
            )
        if search_window_size < 3 or search_window_size % 2 == 0:
            raise ImagePreprocessError(
                f"search_window_size must be an odd integer >= 3, got {search_window_size}."
            )
        if clahe_clip_limit <= 0.0:
            raise ImagePreprocessError(
                f"clahe_clip_limit must be > 0, got {clahe_clip_limit}."
            )
        if (
            len(clahe_tile_grid_size) != 2
            or clahe_tile_grid_size[0] < 1
            or clahe_tile_grid_size[1] < 1
        ):
            raise ImagePreprocessError(
                f"Invalid clahe_tile_grid_size: {clahe_tile_grid_size}."
            )

        if nlm_h > 0.0:
            denoised = cv2.fastNlMeansDenoising(
                arr,
                None,
                h=float(nlm_h),
                templateWindowSize=int(template_window_size),
                searchWindowSize=int(search_window_size),
            )
        else:
            denoised = arr.copy()

        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_grid_size[0]), int(clahe_tile_grid_size[1])),
        )
        enhanced = clahe.apply(denoised)
        return enhanced
    except ImagePreprocessError:
        raise
    except Exception as exc:
        raise ImagePreprocessError(
            f"Failed during fastNlMeansDenoising / CLAHE enhancement: {exc}"
        ) from exc


def preprocess_page_image(
    image: np.ndarray,
    config: Optional[PreprocessConfig] = None,
    *,
    color_order: Literal["RGB", "BGR"] = "RGB",
) -> np.ndarray:
    """Run CV preprocessing on one page image (Red channel -> NLM denoise -> CLAHE)."""
    cfg = config or PreprocessConfig()

    # Nếu người dùng tắt khử dấu đỏ và khử nhiễu NLM, giữ nguyên ảnh gốc giúp xử lý siêu tốc
    if cfg.red_suppression_strength == 0.0 and cfg.nlm_h == 0.0:
        return _coerce_uint8_image(image)

    red_cleaned = extract_red_channel_and_suppress_seal(
        image,
        color_order=color_order,
        red_suppression_strength=cfg.red_suppression_strength,
    )
    enhanced = denoise_and_enhance_contrast(
        red_cleaned,
        nlm_h=cfg.nlm_h,
        template_window_size=cfg.nlm_template_window_size,
        search_window_size=cfg.nlm_search_window_size,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_grid_size=cfg.clahe_tile_grid_size,
    )
    return enhanced


def encode_image_bytes(
    image: np.ndarray,
    image_format: Literal["png", "jpeg"] = "png",
    jpeg_quality: int = 95,
) -> tuple[bytes, str]:
    """Encode an OpenCV image array into lossless PNG or high-quality JPEG bytes."""
    arr = _coerce_uint8_image(image)
    try:
        if image_format == "png":
            ok, buf = cv2.imencode(".png", arr)
            mime_type = "image/png"
        elif image_format == "jpeg":
            ok, buf = cv2.imencode(
                ".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )
            mime_type = "image/jpeg"
        else:
            raise ImagePreprocessError(
                f"Unsupported image_format {image_format!r}; expected 'png' or 'jpeg'."
            )
        if not ok or buf is None or buf.size == 0:
            raise ImagePreprocessError(
                f"cv2.imencode returned failure for format {image_format!r}."
            )
        return buf.tobytes(), mime_type
    except ImagePreprocessError:
        raise
    except Exception as exc:
        raise ImagePreprocessError(f"Failed to encode preprocessed image: {exc}") from exc


def preprocess_pdf(
    pdf_path: Union[str, Path],
    config: Optional[PreprocessConfig] = None,
    *,
    page_numbers: Optional[Iterable[int]] = None,
    page_callback: Optional[Callable[[int, int], None]] = None,
) -> list[PreprocessedPage]:
    """Render all selected PDF pages at 200 DPI (default) and preprocess each for local OCR."""
    cfg = config or PreprocessConfig()
    selected_pages = list(page_numbers) if page_numbers is not None else None
    raw_pages = render_pdf_to_images(pdf_path, dpi=cfg.dpi, page_numbers=selected_pages)
    if selected_pages is None:
        resolved_page_numbers = list(range(1, len(raw_pages) + 1))
    else:
        resolved_page_numbers = selected_pages

    results: list[PreprocessedPage] = []
    total_pages = len(raw_pages)
    for idx, (pno, raw_img) in enumerate(zip(resolved_page_numbers, raw_pages), 1):
        if page_callback is not None:
            page_callback(idx, total_pages)
        processed = preprocess_page_image(raw_img, cfg, color_order="RGB")
        payload, mime_type = encode_image_bytes(
            processed,
            image_format=cfg.image_format,
            jpeg_quality=cfg.jpeg_quality,
        )
        results.append(
            PreprocessedPage(
                page_number=pno,
                image=processed,
                encoded_bytes=payload,
                mime_type=mime_type,
                is_landscape=bool(processed.shape[1] > processed.shape[0]),
            )
        )
    return results
