"""Step 2: Offline layout coordinate analysis (RapidOCR) and Vietnamese OCR (VietOCR)."""

from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.metadata
import inspect
import logging
import numbers
import re
import statistics
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .exceptions import LocalOcrError
from .ocr_models import (
    DetectedLine,
    LayoutSegment,
    OcrLayoutConfig,
    PageLayoutResult,
    TextBox,
)
from .preprocessor import PreprocessedPage
from .schemas import (
    AdministrativeDocument,
    BodyBlock,
    PageStructure,
    merge_page_structures,
    normalize_vietnamese_text,
)
from .table_detector import (
    DetectedTableGrid,
    build_table_data_from_grid,
    detect_table_grids,
    extract_tables_via_column_projection,
    filter_clipped_edge_boxes,
    partition_boxes_by_tables,
    split_boxes_by_grid_dividers,
)

logger = logging.getLogger(__name__)


def _ensure_pillow_antialias() -> None:
    """Restore PIL.Image.ANTIALIAS on Pillow >= 10 for VietOCR compatibility."""
    if not hasattr(Image, "ANTIALIAS"):
        resampling = getattr(Image, "Resampling", None)
        if resampling is not None and hasattr(resampling, "LANCZOS"):
            Image.ANTIALIAS = resampling.LANCZOS  # type: ignore[attr-defined]
        elif hasattr(Image, "LANCZOS"):
            Image.ANTIALIAS = Image.LANCZOS  # type: ignore[attr-defined]


def _ensure_pkg_resources_shim() -> None:
    """Provide a minimal `pkg_resources` shim when `setuptools>=82` omits it.

    `vietocr.tool.utils` imports `gdown`, whose `__init__.py` reads
    `pkg_resources.get_distribution("gdown").version`.
    """
    if sys.modules.get("pkg_resources") is not None:
        return
    try:
        importlib.import_module("pkg_resources")
        return
    except ImportError:
        pass

    shim = ModuleType("pkg_resources")
    shim.DistributionNotFound = importlib.metadata.PackageNotFoundError  # type: ignore[attr-defined]

    class _VersionConflict(Exception):
        pass

    shim.VersionConflict = _VersionConflict  # type: ignore[attr-defined]

    def _get_distribution(dist_name: str) -> SimpleNamespace:
        name_str = str(dist_name)
        try:
            ver = importlib.metadata.version(name_str)
        except importlib.metadata.PackageNotFoundError:
            ver = "0.0.0"
        return SimpleNamespace(version=ver, project_name=name_str, key=name_str.lower())

    def _require(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def _resource_filename(package_or_requirement: Any, resource_name: str) -> str:
        pkg_name = getattr(package_or_requirement, "project_name", str(package_or_requirement))
        mod = importlib.import_module(pkg_name)
        base = Path(getattr(mod, "__file__", "")).resolve().parent
        return str(base / resource_name)

    shim.get_distribution = _get_distribution  # type: ignore[attr-defined]
    shim.require = _require  # type: ignore[attr-defined]
    shim.resource_filename = _resource_filename  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim


_VIETOCR_BASE_CONFIG: dict[str, Any] = {
    "vocab": (
        "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬ"
        "bBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnN"
        "oOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStT"
        "uUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
        "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
    ),
    "device": "cpu",
    "seq_modeling": "transformer",
    "transformer": {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "max_seq_length": 1024,
        "pos_dropout": 0.1,
        "trans_dropout": 0.1,
    },
    "dataset": {
        "name": "data",
        "data_root": "./data/",
        "train_annotation": "train_annotation.txt",
        "valid_annotation": "valid_annotation.txt",
        "image_height": 32,
        "image_min_width": 32,
        "image_max_width": 512,
    },
    "predictor": {
        "beamsearch": False,
    },
    "quiet": False,
}

_VIETOCR_BUILTIN_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "vgg_transformer": {
        "weights": "https://vocr.vn/data/vietocr/vgg_transformer.pth",
        "backbone": "vgg19_bn",
        "cnn": {
            "pretrained": False,
            "ss": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "ks": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
            "hidden": 256,
        },
        "seq_modeling": "transformer",
    },
}


def _get_vietocr_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "vn_pdf_to_docx_vietocr_cache"


def _persist_yaml_cache(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_vietocr_config(
    config_name: str,
    cfg_cls: Any,
    *,
    cache_dir: Optional[Path] = None,
) -> Any:
    """Load VietOCR config offline via local file/cache/built-in defaults before network."""
    # 1. Explicit YAML file path
    local_file = Path(config_name)
    if local_file.is_file():
        return cfg_cls.load_config_from_file(str(local_file))

    resolved_cache = cache_dir or _get_vietocr_cache_dir()
    base_cache_path = resolved_cache / "base.yml"
    model_cache_path = resolved_cache / f"{config_name}.yml"

    # 2. Local cached YAML pair (zero network)
    if base_cache_path.is_file() and model_cache_path.is_file():
        try:
            import yaml

            base_dict = yaml.safe_load(base_cache_path.read_text(encoding="utf-8"))
            model_dict = yaml.safe_load(model_cache_path.read_text(encoding="utf-8"))
            if isinstance(base_dict, dict) and isinstance(model_dict, dict):
                merged = copy.deepcopy(base_dict)
                merged.update(model_dict)
                return cfg_cls(merged)
        except Exception:
            pass

    # 3. Embedded offline config for known architectures (e.g. vgg_transformer)
    if config_name in _VIETOCR_BUILTIN_MODEL_CONFIGS:
        base_dict = copy.deepcopy(_VIETOCR_BASE_CONFIG)
        model_dict = copy.deepcopy(_VIETOCR_BUILTIN_MODEL_CONFIGS[config_name])
        _persist_yaml_cache(base_cache_path, base_dict)
        _persist_yaml_cache(model_cache_path, model_dict)
        merged = copy.deepcopy(base_dict)
        merged.update(model_dict)
        return cfg_cls(merged)

    # 4. Fallback to upstream VietOCR online fetch and cache locally if successful
    cfg = cfg_cls.load_config_from_name(config_name)
    _persist_yaml_cache(base_cache_path, copy.deepcopy(_VIETOCR_BASE_CONFIG))
    _persist_yaml_cache(model_cache_path, dict(cfg))
    return cfg


def _resolve_local_vietocr_weights(weights_value: str) -> str:
    """Return local cached `.pth` path if available before letting VietOCR download."""
    if not weights_value.startswith("http"):
        return weights_value
    filename = weights_value.rsplit("/", 1)[-1].strip()
    if not filename:
        return weights_value

    candidates: list[Path] = [Path(tempfile.gettempdir()) / filename]
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / filename)
    candidates.append(Path(__file__).resolve().parent.parent / filename)

    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate)
        except OSError:
            continue
    return weights_value


def _is_real_number(value: Any) -> bool:
    """Return True for Python/NumPy real numeric scalars while excluding bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _poly_or_bbox_to_rect(raw_box: Any) -> Optional[tuple[float, float, float, float]]:
    """Convert a 4-point polygon or 4-element bbox into `(x_min, y_min, x_max, y_max)`."""
    if raw_box is None:
        return None
    if isinstance(raw_box, np.ndarray):
        raw_box = raw_box.tolist()
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) == 0:
        return None

    # Flat 4-number bbox [x_min, y_min, x_max, y_max]
    if len(raw_box) == 4 and all(_is_real_number(v) for v in raw_box):
        x1, y1, x2, y2 = (float(v) for v in raw_box)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    # Polygon of (x, y) points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    xs: list[float] = []
    ys: list[float] = []
    for pt in raw_box:
        if isinstance(pt, np.ndarray):
            pt = pt.tolist()
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
            except (TypeError, ValueError):
                return None
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def parse_rapidocr_boxes(
    raw_output: Any,
    *,
    img_width: int,
    img_height: int,
) -> list[TextBox]:
    """Normalize outputs from `rapidocr_onnxruntime` into clamped `TextBox` objects."""
    if raw_output is None:
        return []

    payload = raw_output
    if isinstance(payload, tuple) and len(payload) >= 1:
        payload = payload[0]

    if payload is None:
        return []

    raw_boxes: Any = None
    if hasattr(payload, "boxes"):
        raw_boxes = getattr(payload, "boxes")
    else:
        raw_boxes = payload

    if raw_boxes is None:
        return []

    if isinstance(raw_boxes, np.ndarray):
        items = raw_boxes.tolist()
    elif isinstance(raw_boxes, (list, tuple)):
        items = list(raw_boxes)
    else:
        return []

    boxes: list[TextBox] = []
    for item in items:
        candidate = item
        preset_text = ""
        # Handle `[poly, text, score]` if full OCR result was returned
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[0], (list, tuple, np.ndarray))
        ):
            first_el = item[0].tolist() if isinstance(item[0], np.ndarray) else item[0]
            if isinstance(first_el, (list, tuple)) and not all(
                _is_real_number(v) for v in first_el
            ):
                candidate = item[0]
                if isinstance(item[1], str):
                    preset_text = item[1]

        rect = _poly_or_bbox_to_rect(candidate)
        if rect is None:
            continue

        x_min, y_min, x_max, y_max = rect
        x_min = max(0.0, min(float(img_width), x_min))
        x_max = max(0.0, min(float(img_width), x_max))
        y_min = max(0.0, min(float(img_height), y_min))
        y_max = max(0.0, min(float(img_height), y_max))

        if (x_max - x_min) < 2.0 or (y_max - y_min) < 2.0:
            continue

        boxes.append(
            TextBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                text=preset_text,
            )
        )
    return boxes


class RapidOcrDetector:
    """Wrapper around `rapidocr_onnxruntime.RapidOCR` for text box detection."""

    def __init__(self, *, engine: Optional[Any] = None) -> None:
        if engine is not None:
            self._engine = engine
            return
        if sys.platform == "win32":
            from . import _preload_win_msvc_runtime

            _preload_win_msvc_runtime()
            try:
                import ctypes

                k32 = ctypes.WinDLL("kernel32")
                k32.GetModuleHandleW.restype = ctypes.c_void_p
                k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
                handle = k32.GetModuleHandleW("msvcp140.dll")
                if handle:
                    buf = ctypes.create_unicode_buffer(512)
                    k32.GetModuleFileNameW(handle, buf, 512)
                    loaded_path = buf.value
                    if "pyqt5" in loaded_path.lower() and "qt5\\bin" in loaded_path.lower():
                        logger.warning(
                            "Detected '%s' loaded before vn_pdf_to_docx; import vn_pdf_to_docx before PyQt5 to avoid onnxruntime ABI crashes.",
                            loaded_path,
                        )
            except Exception:
                pass
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - surfaced clearly when missing
            raise LocalOcrError(
                "Missing required package 'rapidocr-onnxruntime'. "
                "Install it with: pip install rapidocr-onnxruntime"
            ) from exc

        try:
            self._engine = RapidOCR()
        except Exception as exc:
            raise LocalOcrError(f"Failed to initialize RapidOCR engine: {exc}") from exc

    def detect_boxes(
        self,
        image: np.ndarray,
        *,
        color_order: Literal["RGB", "BGR"] = "RGB",
    ) -> list[TextBox]:
        """Detect `[x_min, y_min, x_max, y_max]` text boxes on one page image."""
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            raise LocalOcrError("Invalid image passed to RapidOCR detector.")

        h, w = image.shape[:2]
        if image.ndim == 2:
            bgr_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            bgr_img = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 3:
            bgr_img = (
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if color_order == "RGB"
                else image
            )
        elif image.shape[2] == 4:
            bgr_img = (
                cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                if color_order == "RGB"
                else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            )
        else:
            raise LocalOcrError(f"Unsupported image channel count: {image.shape[2]}.")

        try:
            raw_output = self._engine(bgr_img, use_det=True, use_cls=False, use_rec=False)
        except TypeError:
            # Compatibility fallback if a mock or older wrapper takes positional arg only
            raw_output = self._engine(bgr_img)
        except Exception as exc:
            raise LocalOcrError(f"RapidOCR box detection failed: {exc}") from exc

        return parse_rapidocr_boxes(raw_output, img_width=w, img_height=h)


def _crop_to_pil_rgb(
    image: np.ndarray,
    box: TextBox,
    *,
    padding_px: int = 3,
    color_order: Literal["RGB", "BGR"] = "RGB",
) -> Image.Image:
    """Crop a `TextBox` from `image` with padding and return an RGB PIL Image."""
    h, w = image.shape[:2]
    x1 = max(0, int(np.floor(box.x_min)) - padding_px)
    y1 = max(0, int(np.floor(box.y_min)) - padding_px)
    x2 = min(w, int(np.ceil(box.x_max)) + padding_px)
    y2 = min(h, int(np.ceil(box.y_max)) + padding_px)

    if x2 <= x1 or y2 <= y1:
        raise LocalOcrError(f"Invalid empty crop region for box {box}.")

    crop = image[y1:y2, x1:x2]
    if crop.ndim == 2:
        return Image.fromarray(crop).convert("RGB")
    if crop.shape[2] == 1:
        return Image.fromarray(crop[:, :, 0]).convert("RGB")
    if crop.shape[2] == 3:
        rgb_crop = (
            crop
            if color_order == "RGB"
            else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        )
        return Image.fromarray(rgb_crop)
    if crop.shape[2] == 4:
        rgb_crop = (
            cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
            if color_order == "RGB"
            else cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        )
        return Image.fromarray(rgb_crop)
    raise LocalOcrError(f"Unsupported crop channel count: {crop.shape[2]}.")


@contextlib.contextmanager
def _compat_torch_load_for_vietocr() -> Iterator[None]:
    """Temporarily default `torch.load(..., weights_only=False)` on PyTorch >= 2.6 for VietOCR."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        yield
        return

    orig_load = getattr(torch, "load", None)
    if orig_load is None:
        yield
        return

    try:
        has_weights_only = "weights_only" in inspect.signature(orig_load).parameters
    except (TypeError, ValueError):
        has_weights_only = False

    if not has_weights_only:
        yield
        return

    def _wrapped_load(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    torch.load = _wrapped_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig_load  # type: ignore[assignment]


class VietOcrRecognizer:
    """Sequential CPU wrapper around `vietocr` (`vgg_transformer`)."""

    def __init__(
        self,
        *,
        config_name: str = "vgg_transformer",
        device: str = "cpu",
        weights: Optional[str] = None,
        predictor: Optional[Any] = None,
    ) -> None:
        self.config_name = config_name
        self.device = device
        self.weights = weights

        if predictor is not None:
            self._predictor = predictor
            return

        _ensure_pillow_antialias()
        _ensure_pkg_resources_shim()
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:  # pragma: no cover - surfaced clearly when missing
            missing_name = getattr(exc, "name", None)
            if missing_name == "vietocr":
                raise LocalOcrError(
                    "Missing required package 'vietocr' (and PyTorch). "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc
            raise LocalOcrError(
                f"Failed to import 'vietocr' dependencies ({exc}). "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        try:
            cfg = _load_vietocr_config(config_name, Cfg)
            cfg["device"] = device
            if "cnn" in cfg and isinstance(cfg["cnn"], dict):
                cfg["cnn"]["pretrained"] = False
            if weights:
                cfg["weights"] = weights
            elif isinstance(cfg.get("weights"), str):
                cfg["weights"] = _resolve_local_vietocr_weights(cfg["weights"])
            with _compat_torch_load_for_vietocr():
                self._predictor = Predictor(cfg)
        except LocalOcrError:
            raise
        except Exception as exc:
            raise LocalOcrError(
                f"Failed to initialize VietOCR predictor ({config_name}, {device}): {exc}"
            ) from exc

    def recognize_box(
        self,
        image: np.ndarray,
        box: TextBox,
        *,
        padding_px: int = 3,
        color_order: Literal["RGB", "BGR"] = "RGB",
    ) -> str:
        """Crop `box` from `image` and predict Vietnamese text sequentially."""
        pil_img = _crop_to_pil_rgb(
            image,
            box,
            padding_px=padding_px,
            color_order=color_order,
        )
        try:
            raw_text = self._predictor.predict(pil_img)
        except Exception as exc:
            raise LocalOcrError(f"VietOCR recognition failed on box {box}: {exc}") from exc
        return normalize_vietnamese_text(str(raw_text)) if raw_text is not None else ""


def group_boxes_into_lines(
    boxes: Sequence[TextBox],
    *,
    y_tolerance_px: float = 10.0,
) -> list[list[TextBox]]:
    """Group bounding boxes whose Y-coordinates differ by <= `y_tolerance_px` into lines."""
    if not boxes:
        return []
    if y_tolerance_px <= 0:
        raise ValueError(f"y_tolerance_px must be > 0, got {y_tolerance_px}.")

    sorted_boxes = sorted(boxes, key=lambda b: (b.center_y, b.x_min))
    lines: list[list[TextBox]] = []
    current_line: list[TextBox] = [sorted_boxes[0]]
    anchor_y = sorted_boxes[0].center_y
    mean_y = anchor_y

    for box in sorted_boxes[1:]:
        if (
            abs(box.center_y - mean_y) <= y_tolerance_px
            and abs(box.center_y - anchor_y) <= y_tolerance_px * 1.5
        ):
            current_line.append(box)
            mean_y = sum(b.center_y for b in current_line) / len(current_line)
        else:
            lines.append(sorted(current_line, key=lambda b: (b.x_min, b.y_min)))
            current_line = [box]
            anchor_y = box.center_y
            mean_y = anchor_y

    lines.append(sorted(current_line, key=lambda b: (b.x_min, b.y_min)))
    lines.sort(key=lambda ln: sum(b.center_y for b in ln) / len(ln))
    return lines


def cluster_line_boxes(
    line_boxes: Sequence[TextBox],
    page_width: int,
    *,
    cluster_gap_px: Optional[float] = None,
) -> list[list[TextBox]]:
    """Merge horizontally close boxes on the same line into X-separated clusters."""
    if not line_boxes:
        return []
    ordered = sorted(line_boxes, key=lambda b: (b.x_min, b.y_min))
    gap_threshold = (
        float(cluster_gap_px)
        if cluster_gap_px is not None
        else max(12.0, 0.016 * float(page_width))
    )

    clusters: list[list[TextBox]] = [[ordered[0]]]
    cur_x_max = ordered[0].x_max

    for box in ordered[1:]:
        if (box.x_min - cur_x_max) <= gap_threshold:
            clusters[-1].append(box)
            cur_x_max = max(cur_x_max, box.x_max)
        else:
            clusters.append([box])
            cur_x_max = box.x_max

    return clusters


def _is_two_column_split(
    clusters: list[list[TextBox]],
    page_width: int,
    config: OcrLayoutConfig,
) -> bool:
    """Check whether a line has 2 clusters separated across the ~45% page split."""
    if len(clusters) != 2:
        return False

    w = float(page_width)
    left_boxes, right_boxes = clusters[0], clusters[1]
    lx1 = min(b.x_min for b in left_boxes)
    lx2 = max(b.x_max for b in left_boxes)
    rx1 = min(b.x_min for b in right_boxes)
    rx2 = max(b.x_max for b in right_boxes)

    l_cx = (lx1 + lx2) / 2.0
    r_cx = (rx1 + rx2) / 2.0
    split_x = config.two_col_split_ratio * w
    middle_gap = rx1 - lx2

    min_gap = max(40.0, 0.04 * w)
    return (
        l_cx < split_x
        and r_cx > split_x
        and lx2 <= (config.two_col_split_ratio + 0.08) * w
        and rx1 >= (config.two_col_split_ratio - 0.05) * w
        and middle_gap >= min_gap
    )


def _raw_is_left_col(x_min: float, x_max: float, page_width: int, split_ratio: float) -> bool:
    return x_max <= (split_ratio + 0.07) * float(page_width)


def _raw_is_right_col(x_min: float, x_max: float, page_width: int, split_ratio: float) -> bool:
    cx = (x_min + x_max) / 2.0
    return (
        x_min >= (split_ratio - 0.02) * float(page_width)
        and cx >= (split_ratio + 0.08) * float(page_width)
    )


def _find_header_footer_column_indices(
    clustered: Sequence[tuple[list[TextBox], list[list[TextBox]], bool]],
    page_width: int,
    config: OcrLayoutConfig,
    *,
    excluded_indices: Optional[set[int]] = None,
) -> set[int]:
    """Identify indices of lines belonging to 2-column header/footer blocks."""
    n = len(clustered)
    if n == 0:
        return set()

    excluded = excluded_indices or set()
    bounds: list[tuple[float, float, float, float]] = []
    heights: list[float] = []
    max_page_y = 0.0
    for boxes, _, _ in clustered:
        x1 = min(b.x_min for b in boxes)
        y1 = min(b.y_min for b in boxes)
        x2 = max(b.x_max for b in boxes)
        y2 = max(b.y_max for b in boxes)
        bounds.append((x1, y1, x2, y2))
        if y2 > y1:
            heights.append(y2 - y1)
        if y2 > max_page_y:
            max_page_y = y2

    median_h = float(statistics.median(heights)) if heights else 24.0
    scale = config.dpi / 200.0
    est_page_h = max(max_page_y, float(page_width) * 1.41)
    split_ratio = config.two_col_split_ratio
    tight_gap = max(45.0 * scale, 2.5 * median_h)
    sig_gap = max(420.0 * scale, 14.0 * median_h)
    stagger_tol = max(50.0 * scale, 2.5 * median_h)

    hf_indices: set[int] = set()

    for idx, (_, clusters, is_two_col) in enumerate(clustered):
        if not is_two_col or idx in excluded:
            continue
        hf_indices.add(idx)
        _, y1, _, y2 = bounds[idx]
        in_lower_half = y1 >= 0.45 * est_page_h

        # Walk backward up to 3 lines for staggered header/footer column lines
        anchor_y1 = y1
        for k in range(idx - 1, max(-1, idx - 4), -1):
            if k in excluded:
                break
            _, cl_k, two_k = clustered[k]
            if two_k or len(cl_k) >= 3:
                break
            kx1, ky1, kx2, ky2 = bounds[k]
            if (anchor_y1 - ky2) > tight_gap:
                break
            if _raw_is_left_col(kx1, kx2, page_width, split_ratio) or _raw_is_right_col(
                kx1, kx2, page_width, split_ratio
            ):
                hf_indices.add(k)
                anchor_y1 = ky1
            else:
                break

        # Walk forward: absorb sandwiched lines (between two_column_rows) and lower-half footer lines
        last_y2 = y2
        for k in range(idx + 1, n):
            if k in excluded:
                break
            _, cl_k, two_k = clustered[k]
            if two_k:
                break
            if len(cl_k) >= 3:
                break
            kx1, ky1, kx2, ky2 = bounds[k]
            is_left_k = _raw_is_left_col(kx1, kx2, page_width, split_ratio)
            is_right_k = _raw_is_right_col(kx1, kx2, page_width, split_ratio)
            if not (is_left_k or is_right_k):
                break

            sandwiched = any(
                m not in excluded
                and clustered[m][2]
                and (bounds[m][1] - ky2) <= max(100.0 * scale, 4.5 * median_h)
                for m in range(k + 1, min(n, k + 4))
            )
            if sandwiched:
                hf_indices.add(k)
                last_y2 = ky2
                continue
            if in_lower_half and (
                (is_left_k and (ky1 - last_y2) <= tight_gap)
                or (is_right_k and (ky1 - y2) <= sig_gap)
            ):
                hf_indices.add(k)
                if is_left_k:
                    last_y2 = ky2
                continue
            break

    # Also detect staggered 2-column blocks with zero exact two_column_row lines
    i = 0
    while i < n:
        if i in hf_indices or i in excluded:
            i += 1
            continue
        _, cl_i, two_i = clustered[i]
        if two_i or len(cl_i) >= 3:
            i += 1
            continue
        ix1, iy1, ix2, iy2 = bounds[i]
        in_band = iy1 < 0.30 * est_page_h or iy1 >= 0.45 * est_page_h
        is_left_i = _raw_is_left_col(ix1, ix2, page_width, split_ratio)
        is_right_i = _raw_is_right_col(ix1, ix2, page_width, split_ratio)
        if not (in_band and (is_left_i or is_right_i)):
            i += 1
            continue

        j = i + 1
        has_left = is_left_i
        has_right = is_right_i
        first_opp_cy: Optional[float] = None
        i_cy = (iy1 + iy2) / 2.0
        prev_y2 = iy2

        while j < n:
            if j in excluded:
                break
            _, cl_j, two_j = clustered[j]
            if two_j or len(cl_j) >= 3:
                break
            jx1, jy1, jx2, jy2 = bounds[j]
            is_left_j = _raw_is_left_col(jx1, jx2, page_width, split_ratio)
            is_right_j = _raw_is_right_col(jx1, jx2, page_width, split_ratio)
            if not (is_left_j or is_right_j):
                break
            gap_lim = sig_gap if (iy1 >= 0.45 * est_page_h and is_right_j) else tight_gap
            if (jy1 - prev_y2) > gap_lim:
                break
            if is_left_j:
                has_left = True
                if not is_left_i and first_opp_cy is None:
                    first_opp_cy = (jy1 + jy2) / 2.0
            if is_right_j:
                has_right = True
                if not is_right_i and first_opp_cy is None:
                    first_opp_cy = (jy1 + jy2) / 2.0
            prev_y2 = jy2
            j += 1

        if has_left and has_right and first_opp_cy is not None and abs(first_opp_cy - i_cy) <= stagger_tol:
            for k in range(i, j):
                hf_indices.add(k)
            i = j
        else:
            i += 1

    return hf_indices


def estimate_text_margins(
    clustered_lines: Sequence[tuple[list[TextBox], list[list[TextBox]], bool]],
    page_width: int,
    config: OcrLayoutConfig,
    *,
    fallback_margins: Optional[tuple[float, float]] = None,
    excluded_indices: Optional[set[int]] = None,
) -> tuple[float, float]:
    """Estimate standard body `(left_margin, right_margin)` in pixels."""
    w = float(page_width)
    scale = config.dpi / 200.0
    indent_min = config.indent_min_px * scale
    indent_max = config.indent_max_px * scale
    excluded = excluded_indices or set()

    left_candidates: list[tuple[float, float]] = []  # (x_min, line_width)
    right_candidates: list[float] = []
    two_col_left_xs: list[float] = []

    for idx, (boxes, clusters, is_two_col) in enumerate(clustered_lines):
        x_min = min(b.x_min for b in boxes)
        x_max = max(b.x_max for b in boxes)
        if is_two_col:
            two_col_left_xs.append(min(b.x_min for b in clusters[0]))
            continue
        if idx in excluded:
            continue
        if len(clusters) >= 3:
            continue

        cx = (x_min + x_max) / 2.0
        width = x_max - x_min
        tentative_center = (
            abs(cx - 0.5 * w) <= config.center_tolerance_ratio * w
            and x_min > 0.16 * w
            and x_max < 0.84 * w
            and width < 0.70 * w
        )
        if tentative_center:
            continue

        if x_min < 0.35 * w:
            left_candidates.append((x_min, width))
        if x_max > 0.65 * w:
            right_candidates.append(x_max)

    if right_candidates:
        max_r = max(right_candidates)
        # Guard against a single spurious far-right edge artifact when >= 2 tighter body lines exist
        near_max = [x for x in right_candidates if x >= max_r - 70.0 * scale]
        if len(near_max) == 1 and max_r > 0.96 * w and len(right_candidates) >= 3:
            others = sorted([x for x in right_candidates if x < max_r - 70.0 * scale], reverse=True)
            second_max = others[0]
            second_cluster = [x for x in others if x >= second_max - 45.0 * scale]
            if len(second_cluster) >= 2:
                near_max = second_cluster
        body_right = float(statistics.median(near_max))
    elif fallback_margins is not None:
        body_right = float(fallback_margins[1])
    else:
        body_right = 0.92 * w

    if left_candidates:
        wide_left = [x for x, line_w in left_candidates if line_w >= 0.40 * w]
        filtered_left = [x for x, _ in left_candidates]
        if wide_left:
            min_wide = min(wide_left)
            kept = [x for x in filtered_left if x >= min_wide - indent_max]
            if kept:
                filtered_left = kept

        min_l = min(filtered_left)
        near_min = [x for x in filtered_left if x <= min_l + 15.0 * scale]
        body_left = float(statistics.median(near_min))

        # If all body lines share a single left coordinate (e.g., all 1-line paragraphs are indented),
        # check whether a fallback margin or 2-column left edge sits 30..110px to the left.
        if max(filtered_left) - min_l <= 20.0 * scale:
            ref_candidates: list[float] = []
            if fallback_margins is not None:
                ref_candidates.append(float(fallback_margins[0]))
            if two_col_left_xs:
                ref_candidates.append(float(statistics.median(two_col_left_xs)))
            for ref in ref_candidates:
                diff = body_left - ref
                if indent_min <= diff <= indent_max:
                    body_left = ref
                    break
    elif fallback_margins is not None:
        body_left = float(fallback_margins[0])
    elif two_col_left_xs:
        body_left = float(min(two_col_left_xs))
    else:
        body_left = 0.14 * w

    return body_left, body_right


_TABULAR_HEADER_RE = re.compile(r"^(?:stt|số\s*tt|tt)\b", re.IGNORECASE)
_TABULAR_RIGHT_HEADER_RE = re.compile(
    r"^(?:mức\s*giá|đơn\s*giá|thành\s*tiền|số\s*tiền|ghi\s*chú|đơn\s*vị(?:\s*tính)?|tỷ\s*lệ|số\s*lượng|tiến\s*độ)\b",
    re.IGNORECASE,
)
_TABULAR_NUMERIC_VALUE_RE = re.compile(
    r"^\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?(?:\s*(?:%|đ|đồng|vnđ))?$",
    re.IGNORECASE,
)
_FOOTER_SIGNER_PREFIXES: tuple[str, ...] = (
    "KT.",
    "TM.",
    "Q.",
    "TL.",
    "TUQ.",
    "CHỦ TỊCH",
    "PHÓ CHỦ TỊCH",
    "GIÁM ĐỐC",
    "PHÓ GIÁM ĐỐC",
    "BỘ TRƯỞNG",
    "THỨ TRƯỞNG",
    "CỤC TRƯỞNG",
    "CHI CỤC TRƯỞNG",
    "TRƯỞNG PHÒNG",
    "HIỆU TRƯỞNG",
    "CHÁNH VĂN PHÒNG",
)


def _is_footer_or_header_marker(
    left_text: str,
    right_text: str,
    *,
    is_two_col: bool = False,
    is_right_single: bool = False,
) -> bool:
    """Return True if a line has unmistakable document header/footer cues."""
    l_low = left_text.lower().strip()
    r_low = right_text.lower().strip()
    l_up = left_text.upper().strip()
    r_up = right_text.upper().strip()
    if l_low.startswith(("nơi nhận", "ủy ban", "cộng hòa", "cộng hoà", "phụ lục")):
        return True
    if is_two_col and (l_low == "danh mục" or r_low.startswith("phụ lục")):
        return True
    if r_low.startswith(("cộng hòa", "cộng hoà", "độc lập", "phụ lục")):
        return True
    if is_two_col and any(r_up.startswith(prefix) for prefix in _FOOTER_SIGNER_PREFIXES):
        return True
    if is_right_single and any(l_up.startswith(prefix) for prefix in _FOOTER_SIGNER_PREFIXES):
        return True
    return False


def _detect_multicolumn_table_rows(
    clustered: Sequence[tuple[list[TextBox], list[list[TextBox]], bool]],
    page_width: int,
    config: OcrLayoutConfig,
) -> tuple[set[int], set[int]]:
    """Detect rows belonging to multi-column tables so 2-cluster rows aren't split into 2-col tables.

    Returns:
        `(demoted_two_col_rows, tabular_single_rows)`:
        - `demoted_two_col_rows`: indices where `is_two_col` was True and must be demoted to
          `single` (`is_tabular_row=True`).
        - `tabular_single_rows`: indices of 1-, 2-, or >=3-cluster rows inside the same table
          region that should also be marked `is_tabular_row=True` (no false indent / header-footer wrap).
    """
    n = len(clustered)
    if n == 0:
        return set(), set()

    bounds: list[tuple[float, float, float, float]] = []
    cluster_spans: list[list[tuple[float, float]]] = []
    heights: list[float] = []
    max_page_y = 0.0

    for boxes, clusters, _ in clustered:
        x1 = min(b.x_min for b in boxes)
        y1 = min(b.y_min for b in boxes)
        x2 = max(b.x_max for b in boxes)
        y2 = max(b.y_max for b in boxes)
        bounds.append((x1, y1, x2, y2))
        cluster_spans.append(
            [(min(b.x_min for b in cl), max(b.x_max for b in cl)) for cl in clusters]
        )
        if y2 > y1:
            heights.append(y2 - y1)
        if y2 > max_page_y:
            max_page_y = y2

    median_h = float(statistics.median(heights)) if heights else 24.0
    w = float(page_width)
    scale = config.dpi / 200.0
    est_page_h = max(max_page_y, w * 1.41)
    col_tol = max(60.0 * scale, 0.05 * w)
    min_table_gutter = max(32.0 * scale, 0.02 * w)
    upward_gap = max(45.0 * scale, 2.2 * median_h)
    max_step_gap = max(90.0 * scale, 3.8 * median_h)
    max_tabular_gap = max(160.0 * scale, 6.5 * median_h)
    max_seed_jump = max(320.0 * scale, 14.0 * median_h)

    def _has_real_table_gutter(spans: Sequence[tuple[float, float]]) -> bool:
        if len(spans) < 3:
            return False
        return max(spans[c + 1][0] - spans[c][1] for c in range(len(spans) - 1)) >= min_table_gutter

    seed_indices: list[int] = []
    staggered_stt_anchors: dict[int, int] = {}

    for idx, (_, clusters, is_two_col) in enumerate(clustered):
        if len(clusters) >= 3 and _has_real_table_gutter(cluster_spans[idx]):
            seed_indices.append(idx)
        elif is_two_col:
            l_txt = " ".join(b.text for b in clusters[0] if b.text).strip()
            r_txt = " ".join(b.text for b in clusters[1] if b.text).strip()
            if not _is_footer_or_header_marker(l_txt, r_txt, is_two_col=True):
                is_stt_hdr = _TABULAR_HEADER_RE.match(l_txt) and len(r_txt) <= 40
                is_right_hdr = bool(_TABULAR_RIGHT_HEADER_RE.match(r_txt))
                is_right_num = bool(
                    _TABULAR_NUMERIC_VALUE_RE.match(r_txt)
                    and cluster_spans[idx][1][0] >= 0.62 * w
                )
                if is_stt_hdr or is_right_hdr or is_right_num:
                    seed_indices.append(idx)
        elif len(clusters) == 1 and cluster_spans[idx][0][1] < 0.35 * w:
            l_txt = " ".join(b.text for b in clusters[0] if b.text).strip()
            if _TABULAR_HEADER_RE.match(l_txt):
                for m in range(idx + 1, min(n, idx + 3)):
                    if (bounds[m][1] - bounds[idx][3]) > max_step_gap:
                        break
                    m_clusters = clustered[m][1]
                    if len(m_clusters) >= 2:
                        m_left = " ".join(b.text for b in m_clusters[0] if b.text).strip()
                        m_right = " ".join(b.text for b in m_clusters[-1] if b.text).strip()
                        if not _is_footer_or_header_marker(
                            m_left, m_right, is_two_col=clustered[m][2]
                        ):
                            seed_indices.append(m)
                            staggered_stt_anchors[m] = idx
                        break

    if not seed_indices:
        return set(), set()

    # Deduplicate seed_indices while preserving ascending order
    seed_indices = sorted(set(seed_indices))
    seed_set = set(seed_indices)
    demoted_two_col: set[int] = set()
    tabular_single: set[int] = set()
    visited: set[int] = set()

    def _has_multi_seed_below(pos: int) -> bool:
        for s_idx in seed_indices:
            if pos < s_idx <= pos + 6 and (bounds[s_idx][1] - bounds[pos][3]) <= max_seed_jump:
                return True
        return False

    for seed_idx in seed_indices:
        if seed_idx in visited:
            continue
        visited.add(seed_idx)
        _, seed_clusters, seed_two_col = clustered[seed_idx]
        if seed_two_col:
            demoted_two_col.add(seed_idx)
        else:
            tabular_single.add(seed_idx)

        spans = cluster_spans[seed_idx]
        has_middle_col = len(spans) >= 3
        table_left = spans[0][0]
        col0_right = spans[0][1] if has_middle_col else min(spans[0][1], table_left + 120.0 * scale)
        col1_left = spans[1][0] if has_middle_col else table_left
        last_col_left = spans[-1][0]
        last_col_right = spans[-1][1]
        last_tabular_y2 = bounds[seed_idx][3]

        stt_anchor_idx = staggered_stt_anchors.get(seed_idx)
        if stt_anchor_idx is not None and stt_anchor_idx not in visited:
            stt_span = cluster_spans[stt_anchor_idx][0]
            tabular_single.add(stt_anchor_idx)
            visited.add(stt_anchor_idx)
            table_left = min(table_left, stt_span[0])
            col0_right = stt_span[1]
            if not has_middle_col and spans[0][0] > stt_span[1] + 10.0:
                has_middle_col = True
                col1_left = spans[0][0]

        # Walk upward up to 3 lines for staggered table column headers (e.g. "STT | Mức giá" or "STT")
        anchor_y1 = bounds[seed_idx][1]
        for k in range(seed_idx - 1, max(-1, seed_idx - 4), -1):
            if k in visited or k in seed_set:
                break
            _, k_clusters, k_two_col = clustered[k]
            kx1, ky1, kx2, ky2 = bounds[k]
            if (anchor_y1 - ky2) > upward_gap:
                break
            k_spans = cluster_spans[k]
            k_left_txt = " ".join(b.text for b in k_clusters[0] if b.text).strip()
            k_right_txt = (
                " ".join(b.text for b in k_clusters[-1] if b.text).strip()
                if len(k_clusters) >= 2
                else ""
            )
            if _is_footer_or_header_marker(k_left_txt, k_right_txt, is_two_col=k_two_col):
                break
            if k_two_col:
                if (
                    abs(k_spans[0][0] - table_left) <= col_tol
                    and k_spans[0][1] < last_col_left - 20.0
                    and k_spans[1][0] >= last_col_left - col_tol
                    and k_spans[1][1] <= max(last_col_right + col_tol, 0.94 * w)
                ):
                    demoted_two_col.add(k)
                    visited.add(k)
                    anchor_y1 = ky1
                    continue
                break
            if (
                len(k_clusters) == 1
                and _TABULAR_HEADER_RE.match(k_left_txt)
                and k_spans[0][0] >= table_left - col_tol
                and k_spans[0][1] <= col0_right + col_tol
            ):
                tabular_single.add(k)
                visited.add(k)
                anchor_y1 = ky1
                continue
            break

        # Walk forward across table continuation rows until the table ends
        k = seed_idx + 1
        while k < n:
            _, k_clusters, k_two_col = clustered[k]
            kx1, ky1, kx2, ky2 = bounds[k]
            k_spans = cluster_spans[k]

            if len(k_clusters) >= 3 and _has_real_table_gutter(k_spans):
                if (ky1 - last_tabular_y2) > max_seed_jump:
                    break
                has_middle_col = True
                table_left = min(table_left, k_spans[0][0])
                col0_right = max(col0_right, k_spans[0][1])
                col1_left = (
                    k_spans[1][0]
                    if col1_left == table_left
                    else min(col1_left, k_spans[1][0])
                )
                last_col_left = min(last_col_left, k_spans[-1][0])
                last_col_right = max(last_col_right, k_spans[-1][1])
                last_tabular_y2 = ky2
                tabular_single.add(k)
                visited.add(k)
                k += 1
                continue

            gap_from_tabular = ky1 - last_tabular_y2
            gap_from_prev = ky1 - bounds[k - 1][3]
            if gap_from_tabular > max_tabular_gap and gap_from_prev > max_step_gap:
                break

            k_left_txt = " ".join(b.text for b in k_clusters[0] if b.text).strip()
            k_right_txt = (
                " ".join(b.text for b in k_clusters[-1] if b.text).strip()
                if len(k_clusters) >= 2
                else ""
            )
            is_right_single = (
                len(k_clusters) == 1
                and kx1 >= (config.two_col_split_ratio - 0.02) * w
            )

            if _is_footer_or_header_marker(
                k_left_txt,
                k_right_txt,
                is_two_col=k_two_col,
                is_right_single=is_right_single,
            ):
                break

            if k_left_txt.lower().strip() == "danh mục" and k + 1 < n:
                nxt_txt = " ".join(b.text for b in clustered[k + 1][0] if b.text).lower()
                if "phụ lục" in nxt_txt:
                    break

            if (
                k_two_col
                and ky1 >= 0.45 * est_page_h
                and not _has_multi_seed_below(k)
                and k_spans[1][0] < last_col_left - max(150.0 * scale, 0.10 * w)
            ):
                break

            if len(k_clusters) == 2:
                left_ok = k_spans[0][0] >= table_left - col_tol and k_spans[0][1] < last_col_left
                min_right_x = (col1_left - col_tol) if has_middle_col else (table_left - col_tol)
                right_ok = (
                    k_spans[1][0] >= min_right_x
                    and k_spans[1][1] <= max(last_col_right + col_tol, 0.94 * w)
                )
                if left_ok and right_ok:
                    if k_two_col:
                        demoted_two_col.add(k)
                    else:
                        tabular_single.add(k)
                    visited.add(k)
                    last_tabular_y2 = ky2
                    k += 1
                    continue
                break

            # Single-cluster line: stop if full-width prose bridges into the rightmost column
            last_col_w = max(40.0, last_col_right - last_col_left)
            mid_left_ref = col1_left if has_middle_col else table_left
            bridges_to_right_col = (
                kx1 <= mid_left_ref + 35.0 * scale
                and kx2 >= last_col_left + 0.55 * last_col_w
            )
            if bridges_to_right_col:
                break

            # Stop on annex headings ("Phụ lục ...") or centered section titles after right-aligned marker
            if k_left_txt.lower().startswith("phụ lục"):
                break
            if kx1 >= last_col_left - col_tol and k + 1 < n:
                nx1, _, nx2, _ = bounds[k + 1]
                ncx = (nx1 + nx2) / 2.0
                title_left_floor = (col1_left + 60.0 * scale) if has_middle_col else (col0_right + 120.0 * scale)
                if (
                    len(clustered[k + 1][1]) == 1
                    and abs(ncx - 0.5 * w) <= config.center_tolerance_ratio * w
                    and nx1 > title_left_floor
                ):
                    break

            k_cx = (kx1 + kx2) / 2.0
            center_left_floor = (
                col1_left + 40.0 * scale if has_middle_col else table_left + 160.0 * scale
            )
            is_centered_heading = (
                abs(k_cx - 0.5 * w) <= config.center_tolerance_ratio * w
                and kx1 > center_left_floor
                and kx2 < last_col_left - 20.0 * scale
                and not _TABULAR_NUMERIC_VALUE_RE.match(k_left_txt)
            )
            if is_centered_heading:
                break

            if kx1 >= table_left - 30.0 * scale and kx2 <= max(last_col_right + 35.0 * scale, 0.94 * w):
                tabular_single.add(k)
                visited.add(k)
                last_tabular_y2 = ky2
                k += 1
                continue

            break

    return demoted_two_col, tabular_single


def classify_page_lines(
    lines_of_boxes: Sequence[Sequence[TextBox]],
    page_width: int,
    config: Optional[OcrLayoutConfig] = None,
    *,
    fallback_margins: Optional[tuple[float, float]] = None,
) -> tuple[list[DetectedLine], float, float]:
    """Classify each line as `two_column_row` or `single` with alignment and indent."""
    cfg = config or OcrLayoutConfig()
    if page_width <= 0:
        raise LocalOcrError(f"page_width must be positive, got {page_width}.")

    clustered: list[tuple[list[TextBox], list[list[TextBox]], bool]] = []
    for raw_line in lines_of_boxes:
        boxes = sorted(raw_line, key=lambda b: (b.x_min, b.y_min))
        if not boxes:
            continue
        clusters = cluster_line_boxes(boxes, page_width, cluster_gap_px=cfg.cluster_gap_px)
        is_two_col = _is_two_column_split(clusters, page_width, cfg)
        clustered.append((boxes, clusters, is_two_col))

    demoted_table_rows, tabular_single_rows = _detect_multicolumn_table_rows(
        clustered, page_width, cfg
    )
    for idx in demoted_table_rows:
        boxes, clusters, _ = clustered[idx]
        clustered[idx] = (boxes, clusters, False)

    tabular_indices = demoted_table_rows | tabular_single_rows
    hf_indices = _find_header_footer_column_indices(
        clustered, page_width, cfg, excluded_indices=tabular_indices
    )
    body_left, body_right = estimate_text_margins(
        clustered,
        page_width,
        cfg,
        fallback_margins=fallback_margins,
        excluded_indices=hf_indices | tabular_indices,
    )

    w = float(page_width)
    scale = cfg.dpi / 200.0
    left_touch_tol = max(18.0, 0.012 * w)
    right_touch_tol = max(55.0, 0.055 * w)
    indent_min = cfg.indent_min_px * scale
    indent_max = cfg.indent_max_px * scale

    detected_lines: list[DetectedLine] = []
    for idx, (boxes, clusters, is_two_col) in enumerate(clustered):
        x_min = min(b.x_min for b in boxes)
        y_min = min(b.y_min for b in boxes)
        x_max = max(b.x_max for b in boxes)
        y_max = max(b.y_max for b in boxes)

        if is_two_col:
            left_boxes = clusters[0]
            right_boxes = clusters[1]
            left_text = " ".join(b.text for b in left_boxes if b.text).strip()
            right_text = " ".join(b.text for b in right_boxes if b.text).strip()
            detected_lines.append(
                DetectedLine(
                    line_index=idx,
                    boxes=boxes,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    line_type="two_column_row",
                    alignment="center",
                    first_line_indent=False,
                    left_boxes=left_boxes,
                    right_boxes=right_boxes,
                    left_text=left_text,
                    right_text=right_text,
                    text=" | ".join(part for part in (left_text, right_text) if part),
                )
            )
            continue

        line_text = " ".join(b.text for b in boxes if b.text).strip()
        if idx in tabular_indices:
            detected_lines.append(
                DetectedLine(
                    line_index=idx,
                    boxes=boxes,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    line_type="single",
                    alignment="left",
                    first_line_indent=False,
                    is_tabular_row=True,
                    text=line_text,
                )
            )
            continue

        cx = (x_min + x_max) / 2.0
        offset_left = x_min - body_left
        right_gap = body_right - x_max
        touches_left_exact = x_min <= (body_left + left_touch_tol)
        is_indented = indent_min <= offset_left <= indent_max
        touches_left_zone = touches_left_exact or is_indented
        touches_right = x_max >= (body_right - right_touch_tol)
        near_center = abs(cx - 0.5 * w) <= cfg.center_tolerance_ratio * w
        asymmetric_indented_left = (
            is_indented
            and not touches_right
            and right_gap > (offset_left + max(60.0 * scale, 0.04 * w))
            and x_min <= 0.25 * w
        )

        if (
            near_center
            and not touches_left_zone
            and not touches_right
        ):
            alignment: Literal["left", "center", "right", "justify"] = "center"
            first_line_indent = False
        elif (
            near_center
            and not touches_left_exact
            and not touches_right
            and not asymmetric_indented_left
            and abs(offset_left - right_gap) <= max(45.0 * scale, 0.03 * w)
        ):
            alignment = "center"
            first_line_indent = False
        elif touches_left_zone and touches_right:
            alignment = "justify"
            first_line_indent = is_indented
        elif touches_right and not touches_left_zone:
            if x_min >= max(body_left + indent_max + 20.0 * scale, 0.35 * w):
                alignment = "right"
                first_line_indent = False
            else:
                alignment = "justify"
                first_line_indent = indent_min <= offset_left <= indent_max * 1.6
        else:
            alignment = "left"
            first_line_indent = is_indented

        detected_lines.append(
            DetectedLine(
                line_index=idx,
                boxes=boxes,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                line_type="single",
                alignment=alignment,
                first_line_indent=first_line_indent,
                text=line_text,
            )
        )

    return detected_lines, body_left, body_right


def _is_left_column_only(line: DetectedLine, page_width: int, split_ratio: float) -> bool:
    return (
        line.line_type == "single"
        and not line.is_tabular_row
        and line.x_max <= (split_ratio + 0.07) * float(page_width)
    )


def _is_right_column_only(line: DetectedLine, page_width: int, split_ratio: float) -> bool:
    return (
        line.line_type == "single"
        and not line.is_tabular_row
        and line.x_min >= (split_ratio - 0.02) * float(page_width)
        and line.center_x >= (split_ratio + 0.08) * float(page_width)
    )


def _interval_blocked(
    y_start: float,
    y_end: float,
    blocked_y_ranges: Sequence[tuple[float, float]],
) -> bool:
    """Return True if a detected table block sits vertically between two non-grid lines."""
    for b_y1, b_y2 in blocked_y_ranges:
        mid_y = (b_y1 + b_y2) / 2.0
        if y_start <= mid_y <= y_end:
            return True
    return False


def group_lines_into_segments(
    lines: Sequence[DetectedLine],
    *,
    page_number: int = 1,
    page_width: int,
    page_height: int,
    body_right_margin: float,
    config: Optional[OcrLayoutConfig] = None,
    blocked_y_ranges: Optional[Sequence[tuple[float, float]]] = None,
) -> list[LayoutSegment]:
    """Group consecutive `two_column_row` lines into 2-column tables and wrap prose lines."""
    cfg = config or OcrLayoutConfig()
    if not lines:
        return []

    blocked = list(blocked_y_ranges or [])
    heights = [ln.height for ln in lines if ln.height > 0]
    median_h = float(statistics.median(heights)) if heights else 24.0
    scale = cfg.dpi / 200.0
    right_wrap_tol = max(55.0, 0.055 * float(page_width))
    para_gap_limit = max(14.0, 1.35 * median_h)
    tight_col_gap = max(45.0 * scale, 2.5 * median_h)
    sig_gap_limit = max(420.0 * scale, 14.0 * median_h)
    sig_break_gap = max(40.0 * scale, 1.8 * median_h)
    stagger_tol = max(50.0 * scale, 2.5 * median_h)

    segments: list[LayoutSegment] = []
    n = len(lines)
    i = 0

    while i < n:
        cur = lines[i]
        pre_left: list[str] = []
        pre_right: list[str] = []
        in_col_zone = (
            (cur.y_min < 0.30 * float(page_height) and not segments)
            or cur.y_min >= 0.45 * float(page_height)
        )

        # Handle 1-2 narrow header/footer column lines staggered just above a two_column_row
        if (
            cur.line_type == "single"
            and not cur.is_tabular_row
            and in_col_zone
            and (
                _is_left_column_only(cur, page_width, cfg.two_col_split_ratio)
                or _is_right_column_only(cur, page_width, cfg.two_col_split_ratio)
            )
        ):
            look_ahead = i + 1
            while (
                look_ahead < min(n, i + 3)
                and lines[look_ahead].line_type == "single"
                and not lines[look_ahead].is_tabular_row
                and not _interval_blocked(lines[look_ahead - 1].y_max, lines[look_ahead].y_min, blocked)
                and (lines[look_ahead].y_min - lines[look_ahead - 1].y_max) <= max(35.0 * scale, 2.2 * median_h)
                and (
                    _is_left_column_only(lines[look_ahead], page_width, cfg.two_col_split_ratio)
                    or _is_right_column_only(lines[look_ahead], page_width, cfg.two_col_split_ratio)
                )
            ):
                look_ahead += 1
            if (
                look_ahead < n
                and lines[look_ahead].line_type == "two_column_row"
                and not _interval_blocked(lines[look_ahead - 1].y_max, lines[look_ahead].y_min, blocked)
                and (lines[look_ahead].y_min - lines[look_ahead - 1].y_max) <= max(35.0 * scale, 2.2 * median_h)
            ):
                for k in range(i, look_ahead):
                    if _is_left_column_only(lines[k], page_width, cfg.two_col_split_ratio) and lines[k].text:
                        pre_left.append(lines[k].text)
                    elif _is_right_column_only(lines[k], page_width, cfg.two_col_split_ratio) and lines[k].text:
                        pre_right.append(lines[k].text)
                i = look_ahead
                cur = lines[i]

        # Handle fully staggered 2-column header/footer block (when Y diff > 10px gives zero two_column_row lines)
        if (
            cur.line_type == "single"
            and not cur.is_tabular_row
            and in_col_zone
            and (
                _is_left_column_only(cur, page_width, cfg.two_col_split_ratio)
                or _is_right_column_only(cur, page_width, cfg.two_col_split_ratio)
            )
        ):
            in_lower = cur.y_min >= 0.45 * float(page_height)
            cur_is_left = _is_left_column_only(cur, page_width, cfg.two_col_split_ratio)
            last_l_y2: Optional[float] = cur.y_max if cur_is_left else None
            last_r_y2: Optional[float] = cur.y_max if not cur_is_left else None
            first_opp_cy: Optional[float] = None
            k = i + 1
            while k < n:
                nxt = lines[k]
                if nxt.line_type != "single" or nxt.is_tabular_row:
                    break
                if _interval_blocked(lines[k - 1].y_max, nxt.y_min, blocked):
                    break
                nxt_is_left = _is_left_column_only(nxt, page_width, cfg.two_col_split_ratio)
                nxt_is_right = _is_right_column_only(nxt, page_width, cfg.two_col_split_ratio)
                if not (nxt_is_left or nxt_is_right):
                    break
                if nxt_is_left:
                    ref_y2 = last_l_y2 if last_l_y2 is not None else lines[k - 1].y_max
                    if (nxt.y_min - ref_y2) > tight_col_gap:
                        break
                    last_l_y2 = nxt.y_max
                    if not cur_is_left and first_opp_cy is None:
                        first_opp_cy = nxt.center_y
                else:
                    ref_y2 = last_r_y2 if last_r_y2 is not None else lines[k - 1].y_max
                    gap_lim = sig_gap_limit if (in_lower and last_r_y2 is not None) else tight_col_gap
                    if (nxt.y_min - ref_y2) > gap_lim:
                        break
                    last_r_y2 = nxt.y_max
                    if cur_is_left and first_opp_cy is None:
                        first_opp_cy = nxt.center_y
                k += 1

            if first_opp_cy is not None and abs(first_opp_cy - cur.center_y) <= stagger_tol:
                stagger_left: list[str] = []
                stagger_right: list[str] = []
                prev_r_y2: Optional[float] = None
                for m in range(i, k):
                    ln_m = lines[m]
                    if _is_left_column_only(ln_m, page_width, cfg.two_col_split_ratio):
                        if ln_m.text:
                            stagger_left.append(ln_m.text)
                    elif _is_right_column_only(ln_m, page_width, cfg.two_col_split_ratio):
                        if ln_m.text:
                            if prev_r_y2 is not None and (ln_m.y_min - prev_r_y2) >= sig_break_gap:
                                stagger_right.append("")
                            stagger_right.append(ln_m.text)
                            prev_r_y2 = ln_m.y_max
                if stagger_left and stagger_right:
                    segments.append(
                        LayoutSegment(
                            kind="two_column_table",
                            page_number=page_number,
                            y_min=float(lines[i].y_min),
                            y_max=float(lines[k - 1].y_max),
                            alignment="center",
                            first_line_indent=False,
                            left_lines=stagger_left,
                            right_lines=stagger_right,
                        )
                    )
                    i = k
                    continue

        if cur.line_type == "two_column_row":
            seg_y_min = float(pre_left and lines[max(0, i - len(pre_left))].y_min or cur.y_min)
            seg_y_max = float(cur.y_max)
            left_lines: list[str] = pre_left + ([cur.left_text] if cur.left_text else [])
            right_lines: list[str] = pre_right + ([cur.right_text] if cur.right_text else [])
            last_left_y_max = cur.y_max if left_lines else None
            last_right_y_max = cur.y_max if right_lines else None
            j = i + 1

            while j < n:
                nxt = lines[j]
                prev_anchor_y = lines[j - 1].y_max
                if _interval_blocked(prev_anchor_y, nxt.y_min, blocked):
                    break

                if nxt.line_type == "two_column_row":
                    if (nxt.y_min - prev_anchor_y) > max(120.0 * scale, 5.5 * median_h):
                        break
                    if nxt.left_text:
                        left_lines.append(nxt.left_text)
                        last_left_y_max = nxt.y_max
                    if nxt.right_text:
                        if (
                            last_right_y_max is not None
                            and (nxt.y_min - last_right_y_max) >= sig_break_gap
                        ):
                            right_lines.append("")
                        right_lines.append(nxt.right_text)
                        last_right_y_max = nxt.y_max
                    seg_y_max = max(seg_y_max, float(nxt.y_max))
                    j += 1
                    continue

                # Absorb single-column lines sandwiched before another two_column_row
                sandwiched_two_col = any(
                    lines[k].line_type == "two_column_row"
                    and not _interval_blocked(nxt.y_max, lines[k].y_min, blocked)
                    and (lines[k].y_min - nxt.y_max) <= max(100.0 * scale, 4.5 * median_h)
                    for k in range(j + 1, min(n, j + 4))
                )
                is_left_only = _is_left_column_only(nxt, page_width, cfg.two_col_split_ratio)
                is_right_only = _is_right_column_only(nxt, page_width, cfg.two_col_split_ratio)

                if sandwiched_two_col and (is_left_only or is_right_only):
                    if is_left_only and nxt.text:
                        left_lines.append(nxt.text)
                        last_left_y_max = nxt.y_max
                    elif is_right_only and nxt.text:
                        if (
                            last_right_y_max is not None
                            and (nxt.y_min - last_right_y_max) >= sig_break_gap
                        ):
                            right_lines.append("")
                        right_lines.append(nxt.text)
                        last_right_y_max = nxt.y_max
                    seg_y_max = max(seg_y_max, float(nxt.y_max))
                    j += 1
                    continue

                # Absorb footer left continuation ("Nơi nhận:", "- ...") & right signer name
                left_started_recipient = any(
                    ln.lower().startswith(("nơi nhận", "-", "+")) for ln in left_lines
                )
                has_upcoming_right_signer = any(
                    _is_right_column_only(lines[k], page_width, cfg.two_col_split_ratio)
                    and not _interval_blocked(cur.y_max, lines[k].y_min, blocked)
                    and (lines[k].y_min - cur.y_max) <= sig_gap_limit
                    for k in range(j + 1, min(n, j + 7))
                )
                in_lower_half = cur.y_min >= 0.45 * float(page_height)

                if (
                    is_left_only
                    and (nxt.y_min - (last_left_y_max or prev_anchor_y)) <= tight_col_gap
                    and (left_started_recipient or has_upcoming_right_signer or (in_lower_half and nxt.text.startswith("-")))
                ):
                    if nxt.text:
                        left_lines.append(nxt.text)
                        last_left_y_max = nxt.y_max
                    seg_y_max = max(seg_y_max, float(nxt.y_max))
                    j += 1
                    continue

                if (
                    is_right_only
                    and last_right_y_max is not None
                    and (nxt.y_min - last_right_y_max) <= sig_gap_limit
                    and (in_lower_half or left_started_recipient or (nxt.y_min - last_right_y_max) <= max(45.0 * scale, 2.2 * median_h))
                ):
                    if nxt.text:
                        if (nxt.y_min - last_right_y_max) >= sig_break_gap:
                            right_lines.append("")
                        right_lines.append(nxt.text)
                        last_right_y_max = nxt.y_max
                    seg_y_max = max(seg_y_max, float(nxt.y_max))
                    j += 1
                    continue

                break

            segments.append(
                LayoutSegment(
                    kind="two_column_table",
                    page_number=page_number,
                    y_min=min(seg_y_min, float(cur.y_min)),
                    y_max=seg_y_max,
                    alignment="center",
                    first_line_indent=False,
                    left_lines=left_lines,
                    right_lines=right_lines,
                )
            )
            i = j
            continue

        # Normal single-column line -> merge wrapped paragraph lines when appropriate
        para_lines: list[str] = [cur.text] if cur.text else []
        para_alignment = cur.alignment
        para_indent = cur.first_line_indent
        prev_line = cur
        seg_y_min = float(cur.y_min)
        seg_y_max = float(cur.y_max)
        j = i + 1

        if not cur.is_tabular_row and cur.alignment in ("justify", "left"):
            while j < n:
                nxt = lines[j]
                if (
                    nxt.line_type != "single"
                    or nxt.is_tabular_row
                    or nxt.alignment not in ("justify", "left")
                    or nxt.first_line_indent
                    or _interval_blocked(prev_line.y_max, nxt.y_min, blocked)
                ):
                    break

                prev_wrapped = prev_line.x_max >= (body_right_margin - right_wrap_tol)
                v_gap = nxt.y_min - prev_line.y_max
                if not prev_wrapped or v_gap > para_gap_limit:
                    break

                if nxt.text:
                    para_lines.append(nxt.text)
                prev_line = nxt
                seg_y_max = max(seg_y_max, float(nxt.y_max))
                j += 1

        joined_text = " ".join(part for part in para_lines if part).strip()
        if joined_text:
            segments.append(
                LayoutSegment(
                    kind="paragraph",
                    page_number=page_number,
                    y_min=seg_y_min,
                    y_max=seg_y_max,
                    alignment=para_alignment,
                    first_line_indent=para_indent,
                    text=joined_text,
                )
            )
        i = j

    return segments


_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

_LANDSCAPE_HEADER_STRONG_KWS = (
    "mã tương đương",
    "mã tương",
    "tên dịch vụ",
    "phê duyệt giá",
    "mức giá",
    "đơn giá",
    "thành tiền",
    "ghi chú",
    "danh mục",
    "nội dung",
    "stt",
    "cộng hòa",
    "cộng hoà",
    "quyết định",
    "thông tư",
    "phụ lục",
)
_VN_DIACRITIC_CHARS = frozenset(
    "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
)
_WEIRD_UPSIDEDOWN_CHAR_RE = re.compile(r"[\[\]'?;\\]")
_MIXED_LETTER_DIGIT_RE = re.compile(r"[a-zà-ỹ][0-9]|[0-9][a-zà-ỹ]{2,}")


def detect_vertical_text_ratio(
    boxes: Sequence[TextBox],
    *,
    min_aspect_ratio: float = 2.0,
) -> tuple[float, int]:
    """Return `(vertical_ratio, valid_box_count)` where vertical boxes satisfy `h > w * ratio`."""
    valid = [b for b in boxes if b.width >= 4.0 and b.height >= 4.0]
    if not valid:
        valid = [b for b in boxes if b.width > 0.0 and b.height > 0.0]
    if not valid:
        return 0.0, 0
    vertical_count = sum(1 for b in valid if b.height > (b.width * min_aspect_ratio))
    return float(vertical_count) / float(len(valid)), len(valid)


def rotate_boxes_90(
    boxes: Sequence[TextBox],
    img_height: int,
    img_width: int,
    rotate_code: int,
) -> list[TextBox]:
    """Project `(x_min, y_min, x_max, y_max)` coordinates across OpenCV 90°/270° rotation."""
    h = float(img_height)
    w = float(img_width)
    rotated: list[TextBox] = []
    if rotate_code == cv2.ROTATE_90_CLOCKWISE:
        for b in boxes:
            rotated.append(
                b.model_copy(
                    update={
                        "x_min": h - b.y_max,
                        "y_min": b.x_min,
                        "x_max": h - b.y_min,
                        "y_max": b.x_max,
                    }
                )
            )
    elif rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        for b in boxes:
            rotated.append(
                b.model_copy(
                    update={
                        "x_min": b.y_min,
                        "y_min": w - b.x_max,
                        "x_max": b.y_max,
                        "y_max": w - b.x_min,
                    }
                )
            )
    else:
        raise ValueError(f"Unsupported rotate_code: {rotate_code}")
    return rotated


def score_landscape_orientation_texts(
    top_texts: Sequence[str],
    *,
    bottom_texts: Sequence[str] = (),
) -> float:
    """Score upright Vietnamese readability & header keyword presence for 90° vs 270°."""
    score = 0.0
    joined_top = " | ".join(t.strip().lower() for t in top_texts if t.strip())
    for kw in _LANDSCAPE_HEADER_STRONG_KWS:
        if kw in joined_top:
            score += 100.0

    for txt in top_texts:
        stripped = txt.strip()
        if not stripped:
            continue
        if stripped.isdigit() and len(stripped) <= 3:
            score -= 15.0
            continue
        for tok in stripped.split():
            low = tok.lower()
            has_vn = any(c in _VN_DIACRITIC_CHARS for c in low)
            has_weird = bool(
                _WEIRD_UPSIDEDOWN_CHAR_RE.search(tok)
                or _MIXED_LETTER_DIGIT_RE.search(tok)
            )
            cleaned_alpha = tok.replace("-", "").replace(",", "").replace(".", "")
            if has_weird:
                score -= 8.0
            elif has_vn and cleaned_alpha.isalpha():
                score += 6.0
            elif tok.isalpha() and len(tok) >= 2:
                score += 2.0

    for btxt in bottom_texts:
        s = btxt.strip()
        if s.isdigit() and 1 <= len(s) <= 3:
            score += 20.0
    return score


def _select_orientation_probe_boxes(
    boxes: Sequence[TextBox],
    *,
    img_width: int,
    img_height: int,
) -> tuple[list[TextBox], list[TextBox]]:
    """Select up to 5 top-region boxes and 1 bottom-center page-number candidate."""
    top_cands = sorted(
        [b for b in boxes if b.center_y < (0.22 * img_height) and b.width >= 16.0],
        key=lambda b: (b.y_min, b.x_min),
    )[:5]
    if len(top_cands) < 3:
        top_cands = sorted(boxes, key=lambda b: (b.y_min, b.x_min))[:5]

    bottom_cands = sorted(
        [
            b
            for b in boxes
            if b.center_y > (0.88 * img_height)
            and (0.25 * img_width) <= b.center_x <= (0.75 * img_width)
            and b.width <= 120.0
        ],
        key=lambda b: -b.center_y,
    )[:1]
    return top_cands, bottom_cands


def detect_and_fix_orientation(
    image: np.ndarray,
    *,
    detector: RapidOcrDetector,
    recognizer: VietOcrRecognizer,
    config: OcrLayoutConfig,
    color_order: Literal["RGB", "BGR"] = "RGB",
    pdf_is_landscape: bool = False,
) -> tuple[np.ndarray, bool, Literal[0, 90, 270], list[TextBox]]:
    """Detect Landscape PDFs or rotated-90° vertical-text scans and return `(upright_img, is_landscape, angle, boxes)`."""
    h, w = image.shape[:2]
    raw_boxes = detector.detect_boxes(image, color_order=color_order)

    if w > h or pdf_is_landscape:
        return image, True, 0, raw_boxes

    vertical_ratio, valid_count = detect_vertical_text_ratio(
        raw_boxes,
        min_aspect_ratio=config.vertical_box_aspect_threshold,
    )
    should_rotate = config.auto_rotate_landscape and (
        (valid_count >= 3 and vertical_ratio > config.vertical_box_majority_ratio)
        or (1 <= valid_count <= 2 and vertical_ratio >= 0.99)
    )
    if not should_rotate:
        return image, False, 0, raw_boxes

    # Evaluate both 90° CCW (270) and 90° CW (90) candidates using top-band VietOCR readability
    img_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ccw_h, ccw_w = img_ccw.shape[:2]
    ccw_boxes = rotate_boxes_90(raw_boxes, h, w, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ccw_top_boxes, ccw_bot_boxes = _select_orientation_probe_boxes(
        ccw_boxes, img_width=ccw_w, img_height=ccw_h
    )
    ccw_top_texts = [
        recognizer.recognize_box(
            img_ccw, b, padding_px=config.crop_padding_px, color_order=color_order
        )
        for b in ccw_top_boxes
    ]
    ccw_bot_texts = [
        recognizer.recognize_box(
            img_ccw, b, padding_px=config.crop_padding_px, color_order=color_order
        )
        for b in ccw_bot_boxes
    ]
    score_ccw = score_landscape_orientation_texts(
        ccw_top_texts, bottom_texts=ccw_bot_texts
    )

    img_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cw_h, cw_w = img_cw.shape[:2]
    cw_boxes = rotate_boxes_90(raw_boxes, h, w, cv2.ROTATE_90_CLOCKWISE)
    cw_top_boxes, cw_bot_boxes = _select_orientation_probe_boxes(
        cw_boxes, img_width=cw_w, img_height=cw_h
    )
    cw_top_texts = [
        recognizer.recognize_box(
            img_cw, b, padding_px=config.crop_padding_px, color_order=color_order
        )
        for b in cw_top_boxes
    ]
    cw_bot_texts = [
        recognizer.recognize_box(
            img_cw, b, padding_px=config.crop_padding_px, color_order=color_order
        )
        for b in cw_bot_boxes
    ]
    score_cw = score_landscape_orientation_texts(
        cw_top_texts, bottom_texts=cw_bot_texts
    )

    if score_cw > score_ccw:
        winning_img = img_cw
        winning_angle: Literal[0, 90, 270] = 90
        winning_boxes = cw_boxes
    else:
        winning_img = img_ccw
        winning_angle = 270
        winning_boxes = ccw_boxes

    # Re-run RapidOCR once on the upright image for tighter horizontal text boxes,
    # keeping projected boxes if detector output looks unchanged (e.g., mocked detector).
    redetected = detector.detect_boxes(winning_img, color_order=color_order)
    if redetected:
        redet_ratio, redet_count = detect_vertical_text_ratio(
            redetected,
            min_aspect_ratio=config.vertical_box_aspect_threshold,
        )
        if redet_count > 0 and redet_ratio <= 0.50:
            winning_boxes = redetected

    logger.info(
        "Auto-rotated page (%dx%d -> %dx%d, %.0f%% vertical boxes) by %d deg into Landscape (CCW=%.1f vs CW=%.1f).",
        w,
        h,
        winning_img.shape[1],
        winning_img.shape[0],
        vertical_ratio * 100.0,
        winning_angle,
        score_ccw,
        score_cw,
    )
    return winning_img, True, winning_angle, winning_boxes


def _recognize_lines_sequentially(
    image: np.ndarray,
    lines_of_boxes: Sequence[Sequence[TextBox]],
    recognizer: VietOcrRecognizer,
    config: OcrLayoutConfig,
    *,
    color_order: Literal["RGB", "BGR"] = "RGB",
) -> list[list[TextBox]]:
    """Run VietOCR strictly sequentially on each box in every detected line."""
    recognized_lines: list[list[TextBox]] = []
    for line in lines_of_boxes:
        out_line: list[TextBox] = []
        for box in line:
            text = recognizer.recognize_box(
                image,
                box,
                padding_px=config.crop_padding_px,
                color_order=color_order,
            )
            out_line.append(box.model_copy(update={"text": text}))
        recognized_lines.append(out_line)
    return recognized_lines


class LocalOcrLayoutAnalyzer:
    """Sequential Step 2 analyzer combining OpenCV grid tables, RapidOCR, X/Y layout, and VietOCR."""

    def __init__(
        self,
        config: Optional[OcrLayoutConfig] = None,
        *,
        detector: Optional[RapidOcrDetector] = None,
        recognizer: Optional[VietOcrRecognizer] = None,
    ) -> None:
        self.config = config or OcrLayoutConfig()
        self._detector = detector
        self._recognizer = recognizer

    def _get_detector(self) -> RapidOcrDetector:
        if self._detector is None:
            self._detector = RapidOcrDetector()
        return self._detector

    def _get_recognizer(self) -> VietOcrRecognizer:
        if self._recognizer is None:
            self._recognizer = VietOcrRecognizer(
                config_name=self.config.vietocr_config_name,
                device=self.config.vietocr_device,
                weights=self.config.vietocr_weights,
            )
        return self._recognizer

    def process_page(
        self,
        image: np.ndarray,
        *,
        page_number: int = 1,
        color_order: Literal["RGB", "BGR"] = "RGB",
        fallback_margins: Optional[tuple[float, float]] = None,
        pdf_is_landscape: bool = False,
    ) -> PageLayoutResult:
        """Auto-orient rotated landscape scans, detect OpenCV grids + RapidOCR boxes, run VietOCR, and build segments."""
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            raise LocalOcrError(f"Invalid image on page {page_number}.")

        detector = self._get_detector()
        recognizer = self._get_recognizer()

        # 0. Auto-detect Portrait vs Landscape (width > height or >60% vertical text boxes)
        working_image, is_landscape, rotation_angle, raw_boxes = detect_and_fix_orientation(
            image,
            detector=detector,
            recognizer=recognizer,
            config=self.config,
            color_order=color_order,
            pdf_is_landscape=pdf_is_landscape,
        )
        h, w = working_image.shape[:2]
        usable_width_cm = 25.7 if is_landscape else 16.5

        # 1. Detect bordered table grids via OpenCV morphology BEFORE grouping lines
        grids = detect_table_grids(working_image, dpi=self.config.dpi, color_order=color_order)

        # 2. Split any wide RapidOCR boxes crossing confirmed vertical grid rules, then run VietOCR
        raw_boxes = split_boxes_by_grid_dividers(raw_boxes, grids, dpi=self.config.dpi)
        ordered_lines = group_boxes_into_lines(
            raw_boxes, y_tolerance_px=self.config.line_y_tolerance_px
        )
        recognized_lines = _recognize_lines_sequentially(
            working_image,
            ordered_lines,
            recognizer,
            self.config,
            color_order=color_order,
        )
        recognized_boxes = [box for ln in recognized_lines for box in ln]

        # 3. Assign boxes whose center lies inside any OpenCV table bbox directly to (r, c) cells
        populated_grids, outside_boxes = partition_boxes_by_tables(
            recognized_boxes, grids, dpi=self.config.dpi
        )
        outside_boxes = filter_clipped_edge_boxes(
            outside_boxes, grids, page_width=w, dpi=self.config.dpi
        )

        table_segments: list[LayoutSegment] = []
        blocked_y_ranges: list[tuple[float, float]] = []

        for grid, grid_boxes in populated_grids:
            rows, colspans, col_widths, col_aligns, hdr_rows = build_table_data_from_grid(
                grid,
                grid_boxes,
                dpi=self.config.dpi,
                y_tolerance_px=self.config.line_y_tolerance_px,
                is_landscape=is_landscape,
                usable_width_cm=usable_width_cm,
            )
            if not rows:
                continue
            blocked_y_ranges.append((float(grid.y_min), float(grid.y_max)))
            table_segments.append(
                LayoutSegment(
                    kind="table",
                    page_number=page_number,
                    y_min=float(grid.y_min),
                    y_max=float(grid.y_max),
                    alignment="center",
                    first_line_indent=False,
                    table_rows=rows,
                    cell_colspans=colspans,
                    col_widths_cm=col_widths,
                    col_alignments=col_aligns,
                    header_rows=hdr_rows,
                )
            )

        # 4. Run line grouping, X-alignment rules, and Column Projection fallback on outside boxes
        if outside_boxes:
            outside_lines = group_boxes_into_lines(
                outside_boxes, y_tolerance_px=self.config.line_y_tolerance_px
            )
            classified_lines, body_left, body_right = classify_page_lines(
                outside_lines,
                page_width=w,
                config=self.config,
                fallback_margins=fallback_margins,
            )
            remaining_lines, projected_tables = extract_tables_via_column_projection(
                classified_lines,
                page_width=w,
                dpi=self.config.dpi,
                y_tolerance_px=self.config.line_y_tolerance_px,
                is_landscape=is_landscape,
                usable_width_cm=usable_width_cm,
            )
            for proj in projected_tables:
                blocked_y_ranges.append((float(proj.y_min), float(proj.y_max)))
                table_segments.append(
                    LayoutSegment(
                        kind="table",
                        page_number=page_number,
                        y_min=float(proj.y_min),
                        y_max=float(proj.y_max),
                        alignment="center",
                        first_line_indent=False,
                        table_rows=proj.rows,
                        cell_colspans=proj.row_colspans,
                        col_widths_cm=proj.col_widths_cm,
                        col_alignments=proj.col_alignments,
                        header_rows=proj.header_rows,
                    )
                )

            non_table_segments = group_lines_into_segments(
                remaining_lines,
                page_number=page_number,
                page_width=w,
                page_height=h,
                body_right_margin=body_right,
                config=self.config,
                blocked_y_ranges=blocked_y_ranges,
            )
        else:
            classified_lines = []
            if fallback_margins is not None:
                body_left, body_right = fallback_margins
            elif table_segments:
                body_left = float(min(g.x_min for g, _ in populated_grids))
                body_right = float(max(g.x_max for g, _ in populated_grids))
            else:
                body_left = 0.14 * float(w)
                body_right = 0.92 * float(w)
            non_table_segments = []

        segments = sorted(
            [*non_table_segments, *table_segments],
            key=lambda seg: (seg.y_min, 0 if seg.kind != "paragraph" else 1),
        )

        return PageLayoutResult(
            page_number=page_number,
            page_width=w,
            page_height=h,
            is_landscape=is_landscape,
            rotation_angle=rotation_angle,
            body_left_margin=body_left,
            body_right_margin=body_right,
            lines=classified_lines,
            segments=segments,
        )

    def process_document(
        self,
        pages: Iterable[PreprocessedPage],
        *,
        page_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[PageLayoutResult]:
        """Process all preprocessed pages sequentially without multithreading."""
        page_list = list(pages)
        if not page_list:
            raise LocalOcrError("No preprocessed pages supplied for local OCR.")

        results: list[PageLayoutResult] = []
        total = len(page_list)
        portrait_margins: Optional[tuple[float, float]] = None

        for idx, page in enumerate(page_list, 1):
            if page_callback is not None:
                page_callback(idx, total)
            page_is_wide = bool(
                getattr(page, "is_landscape", False)
                or (isinstance(page.image, np.ndarray) and page.image.ndim >= 2 and page.image.shape[1] > page.image.shape[0])
            )
            page_res = self.process_page(
                page.image,
                page_number=page.page_number,
                color_order="RGB",
                fallback_margins=None if page_is_wide else portrait_margins,
                pdf_is_landscape=bool(getattr(page, "is_landscape", False)),
            )
            if not page_res.is_landscape and portrait_margins is None and page_res.lines:
                portrait_margins = (page_res.body_left_margin, page_res.body_right_margin)
            results.append(page_res)

        return results


def collect_document_text_batch(
    pages: Sequence[PageLayoutResult],
) -> tuple[dict[str, str], list[tuple[str, int, int, str, int]]]:
    """Flatten all VietOCR text entries across all pages into `{"id_1": "...", ...}`.

    Returns:
        `(flat_text_dict, bindings)` where each binding tuple maps an `id_k` back to
        `(key, page_idx, segment_idx, slot_kind, line_idx)`.
    """
    flat_dict: dict[str, str] = {}
    bindings: list[tuple[str, int, int, str, int]] = []
    counter = 1

    for p_idx, page in enumerate(pages):
        for s_idx, seg in enumerate(page.segments):
            if seg.kind == "paragraph":
                if seg.text.strip():
                    key = f"id_{counter}"
                    counter += 1
                    flat_dict[key] = seg.text.strip()
                    bindings.append((key, p_idx, s_idx, "paragraph", 0))
            elif seg.kind == "two_column_table":
                for l_idx, line in enumerate(seg.left_lines):
                    if line.strip():
                        key = f"id_{counter}"
                        counter += 1
                        flat_dict[key] = line.strip()
                        bindings.append((key, p_idx, s_idx, "left_line", l_idx))
                for r_idx, line in enumerate(seg.right_lines):
                    if line.strip():
                        key = f"id_{counter}"
                        counter += 1
                        flat_dict[key] = line.strip()
                        bindings.append((key, p_idx, s_idx, "right_line", r_idx))
            elif seg.kind == "table":
                for r_idx, row in enumerate(seg.table_rows):
                    for c_idx, cell in enumerate(row):
                        cleaned_cell = cell.strip()
                        if cleaned_cell and _HAS_LETTER_RE.search(cleaned_cell):
                            key = f"id_{counter}"
                            counter += 1
                            flat_dict[key] = cleaned_cell
                            bindings.append((key, p_idx, s_idx, f"table_cell:{r_idx}", c_idx))

    return flat_dict, bindings


def build_administrative_document_from_layouts(
    pages: Sequence[PageLayoutResult],
    corrected_texts: Optional[dict[str, str]] = None,
    bindings: Optional[Sequence[tuple[str, int, int, str, int]]] = None,
    *,
    source_pdf: Optional[str] = None,
) -> AdministrativeDocument:
    """Assemble offline OCR & layout blocks into an `AdministrativeDocument`."""
    if not pages:
        raise LocalOcrError("Cannot build AdministrativeDocument from empty page layouts.")

    # Deep copy segments cleanly so original layouts are not mutated unexpectedly
    cloned_pages = [page.model_copy(deep=True) for page in pages]

    if corrected_texts and bindings:
        for key, p_idx, s_idx, slot_kind, line_idx in bindings:
            if key not in corrected_texts:
                continue
            new_text = normalize_vietnamese_text(corrected_texts[key]) or ""
            if not new_text:
                continue
            seg = cloned_pages[p_idx].segments[s_idx]
            if slot_kind == "paragraph":
                seg.text = new_text
            elif slot_kind == "left_line":
                seg.left_lines[line_idx] = new_text
            elif slot_kind == "right_line":
                seg.right_lines[line_idx] = new_text
            elif slot_kind.startswith("table_cell:"):
                r_idx = int(slot_kind.split(":", 1)[1])
                if 0 <= r_idx < len(seg.table_rows) and 0 <= line_idx < len(seg.table_rows[r_idx]):
                    seg.table_rows[r_idx][line_idx] = new_text

    page_structures: list[PageStructure] = []
    for page in cloned_pages:
        body_blocks: list[BodyBlock] = []
        for seg in page.segments:
            if seg.kind == "paragraph":
                if not seg.text.strip():
                    continue
                body_blocks.append(
                    BodyBlock(
                        type="paragraph",
                        alignment=seg.alignment,
                        first_line_indent=seg.first_line_indent,
                        bold=seg.bold,
                        italic=seg.italic,
                        content=seg.text,
                    )
                )
            elif seg.kind == "two_column_table":
                left_text = "\n".join(seg.left_lines).strip()
                right_text = "\n".join(seg.right_lines).strip()
                if not left_text and not right_text:
                    continue
                body_blocks.append(
                    BodyBlock(
                        type="two_column_table",
                        alignment="center",
                        first_line_indent=False,
                        bold=False,
                        italic=False,
                        content=[[left_text, right_text]],
                    )
                )
            elif seg.kind == "table":
                if not seg.table_rows:
                    continue
                body_blocks.append(
                    BodyBlock(
                        type="table",
                        alignment="center",
                        first_line_indent=False,
                        bold=False,
                        italic=False,
                        content=seg.table_rows,
                        cell_colspans=seg.cell_colspans,
                        col_widths_cm=seg.col_widths_cm,
                        col_alignments=seg.col_alignments,
                        header_rows=seg.header_rows,
                    )
                )
        page_structures.append(
            PageStructure(
                page_number=page.page_number,
                is_landscape=page.is_landscape,
                body_blocks=body_blocks,
            )
        )

    return merge_page_structures(page_structures, source_pdf=source_pdf)
