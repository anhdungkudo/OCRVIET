"""Step 3: 1-Shot text-only Vietnamese spelling & OCR correction via Google GenAI."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, Field, field_validator

try:
    from google import genai
    from google.genai import types
    _GENAI_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - surfaced clearly at runtime when missing
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    _GENAI_IMPORT_ERROR = exc

from .exceptions import GeminiExtractionError
from .schemas import normalize_vietnamese_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ROTATION: list[str] = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

DEFAULT_TEXT_CORRECTION_INSTRUCTION = """Bạn là chuyên gia hiệu đính văn bản hành chính Việt Nam (Nghị định 30/2020/NĐ-CP) và sửa lỗi nhận dạng OCR (VietOCR).

NHIỆM VỤ BẮT BUỘC:
1. Đầu vào là một JSON Dictionary phẳng dạng {"id_1": "văn bản thô 1", "id_2": "văn bản thô 2", ...} được trích xuất theo đúng thứ tự đọc từ tài liệu scan.
2. SỬA SẠCH 100% LỖI CHÍNH TẢ TIẾNG VIỆT dựa theo ngữ cảnh hành chính Việt Nam:
   - Phục hồi chính xác dấu thanh (sắc, huyền, hỏi, ngã, nặng) và dấu mũ/móc (ă, â, ê, ô, ơ, ư, đ).
   - Sửa triệt để lỗi nhận dạng ký tự OCR kinh điển: nhầm i/l/1 (VD: 'Điểu 1' -> 'Điều 1', '1àm việc' -> 'làm việc', 'Công lỷ' -> 'Công lý'), nhầm o/O/0 (VD: '3O/2O2O' -> '30/2020', 'tháng O5' -> 'tháng 05'), nhầm rn/m, cl/d, d/đ, mất dấu hoặc sai dấu do chữ mờ/dấu mộc đỏ.
   - Bảo toàn nguyên văn tên riêng, chức danh, số hiệu văn bản, ngày tháng, số tiền, thuật ngữ pháp lý; tuyệt đối không tự ý tóm tắt, không thêm bớt ý, không gộp hay tách key.
3. ĐẦU RA BẮT BUỘC:
   - Trả về ĐÚNG 1 JSON Object hợp lệ, giữ nguyên TẤT CẢ các key `id_...` như đầu vào, giá trị là chuỗi văn bản đã được sửa sạch chính tả tiếng Việt."""


class GeminiExtractorConfig(BaseModel):
    """Configuration for 1-Shot text-only Gemini spelling correction."""

    model: str = Field(default="gemini-3.8-flash", min_length=1)
    fallback_models: list[str] = Field(default_factory=list)
    max_words_per_batch: int = Field(default=5000, ge=50, le=50_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(default=None, ge=256, le=65536)
    max_retries: int = Field(default=1, ge=0, le=20)
    retry_base_delay: float = Field(default=1.5, gt=0.0, le=60.0)
    timeout_ms: Optional[int] = Field(
        default=35_000,
        ge=1_000,
        description="HTTP request timeout in milliseconds passed to google.genai.types.HttpOptions.",
    )

    @field_validator("model")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Gemini model name must not be empty.")
        return stripped


def build_model_rotation(
    primary_model: str,
    fallback_models: Sequence[str] = (),
) -> list[str]:
    """Build ordered unique model list starting with `primary_model` (excluding gemini-3.6-flash)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in [primary_model, *fallback_models]:
        name = raw.strip()
        if name and name != "gemini-3.6-flash" and name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered or ["gemini-3.8-flash"]


def split_text_dict_into_batches(
    text_map: dict[str, str],
    max_words: int = 5000,
) -> list[dict[str, str]]:
    """Split a flat `{id: text}` dict into sequential batches of at most `max_words` words."""
    if max_words < 1:
        raise ValueError(f"max_words must be >= 1, got {max_words}.")
    if not text_map:
        return []

    batches: list[dict[str, str]] = []
    current_batch: dict[str, str] = {}
    current_words = 0

    for key, val in text_map.items():
        text = "" if val is None else str(val).strip()
        if not text:
            continue
        word_count = max(1, len(text.split()))
        if current_batch and (current_words + word_count) > max_words:
            batches.append(current_batch)
            current_batch = {key: text}
            current_words = word_count
        else:
            current_batch[key] = text
            current_words += word_count

    if current_batch:
        batches.append(current_batch)

    return batches


def build_batch_correction_prompt(text_batch: dict[str, str]) -> str:
    """Build text-only user prompt containing the flat JSON dictionary to spell-correct."""
    payload_json = json.dumps(text_batch, ensure_ascii=False, indent=2)
    return (
        "Hãy sửa sạch lỗi chính tả tiếng Việt, lỗi mất dấu thanh/dấu mũ và lỗi nhầm ký tự OCR "
        "(i/l/1, o/O/0, d/đ, rn/m, cl/d...) theo ngữ cảnh văn bản hành chính Việt Nam cho toàn bộ "
        "các đoạn văn bản trong JSON dưới đây.\n"
        "Yêu cầu BẮT BUỘC: Chỉ trả về đúng 1 JSON Object giữ nguyên tất cả các key `id` ban đầu.\n\n"
        f"{payload_json}"
    )


_PERMANENT_STATUS_CODE_RE = re.compile(
    r"(?<![\d.-])\b(?:400|401|403|404)\b(?![\d.-])"
    r"(?!\s*(?:ms|s|sec|seconds|min|minutes|bytes|kb|mb|gb|tokens|retries|attempts|pages|items|rows|cols|px|dpi|chars)\b)"
)
_QUOTA_STATUS_CODE_RE = re.compile(
    r"(?<![\d.-])\b429\b(?![\d.-])"
    r"(?!\s*(?:ms|s|sec|seconds|min|minutes|bytes|kb|mb|gb|tokens|retries|attempts|pages|items|rows|cols|px|dpi|chars)\b)"
)
_TRANSIENT_STATUS_CODE_RE = re.compile(
    r"(?<![\d.-])\b(?:408|500|502|503|504)\b(?![\d.-])"
    r"(?!\s*(?:ms|s|sec|seconds|min|minutes|bytes|kb|mb|gb|tokens|retries|attempts|pages|items|rows|cols|px|dpi|chars)\b)"
)
_NOT_FOUND_CODE_RE = re.compile(
    r"(?<![\d.-])\b404\b(?![\d.-])"
    r"(?!\s*(?:ms|s|sec|seconds|min|minutes|bytes|kb|mb|gb|tokens|retries|attempts|pages|items|rows|cols|px|dpi|chars)\b)"
)
_TRACE_ID_KV_RE = re.compile(
    r"""\b(?:[\w.-]*?(?:request|req|trace|span|correlation|session|job|operation|op|debug|error|client)[_-]?id|trace(?:[_-]?id)?|id)["']?\s*[=:]\s*["']?[\w.-]+["']?"""
)


def _extract_status_code(exc: Exception) -> Any:
    status_code = getattr(exc, "code", None)
    if status_code is None:
        status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "status", None)
    return status_code


def _is_model_not_found_error(exc: Exception) -> bool:
    """Return True if an exception indicates the requested model name returned 404 / NOT_FOUND."""
    status_code = _extract_status_code(exc)
    if not isinstance(status_code, bool) and isinstance(status_code, int):
        return status_code == 404
    if isinstance(status_code, str) and status_code.strip().upper() in ("404", "NOT_FOUND"):
        return True
    msg = str(exc).lower()
    msg_for_codes = _TRACE_ID_KV_RE.sub(" ", msg)
    return (
        "not_found" in msg
        or "not found" in msg
        or "no longer available" in msg
        or bool(_NOT_FOUND_CODE_RE.search(msg_for_codes))
    )


def _is_quota_exhausted_error(exc: Exception) -> bool:
    """Return True for 429 / RESOURCE_EXHAUSTED so we abort retries and fall back to VietOCR."""
    status_code = _extract_status_code(exc)
    if not isinstance(status_code, bool) and isinstance(status_code, int):
        if status_code == 429:
            return True
    elif isinstance(status_code, str):
        if status_code.strip().upper() in ("429", "RESOURCE_EXHAUSTED"):
            return True

    msg = str(exc).lower()
    msg_for_codes = _TRACE_ID_KV_RE.sub(" ", msg)
    quota_markers = (
        "resource_exhausted",
        "quota exceeded",
        "generate_content_free_tier_requests",
        "retrydelay",
    )
    return any(marker in msg for marker in quota_markers) or bool(
        _QUOTA_STATUS_CODE_RE.search(msg_for_codes)
    )


def _is_transient_api_error(exc: Exception) -> bool:
    """Determine whether an exception from google-genai is transient and safe to retry."""
    if _is_quota_exhausted_error(exc):
        return False

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    status_code = _extract_status_code(exc)

    if not isinstance(status_code, bool) and isinstance(status_code, int):
        if status_code in (408, 500, 502, 503, 504):
            return True
        if 400 <= status_code < 500:
            return False
    elif isinstance(status_code, str):
        code_upper = status_code.strip().upper()
        if code_upper in (
            "408",
            "500",
            "502",
            "503",
            "504",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
        ):
            return True
        if code_upper in (
            "400",
            "401",
            "403",
            "404",
            "429",
            "RESOURCE_EXHAUSTED",
            "INVALID_ARGUMENT",
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
            "NOT_FOUND",
        ):
            return False

    exc_type_name = type(exc).__name__.lower()
    if any(tok in exc_type_name for tok in ("timeout", "connect", "protocol")):
        return True

    msg = str(exc).lower()
    msg_for_codes = _TRACE_ID_KV_RE.sub(" ", msg)

    permanent_markers = (
        "invalid_argument",
        "unauthenticated",
        "permission_denied",
        "api key not valid",
        "api_key_invalid",
        "not_found",
        "not found",
        "no longer available",
        "resource_exhausted",
    )
    if any(marker in msg for marker in permanent_markers) or _PERMANENT_STATUS_CODE_RE.search(
        msg_for_codes
    ):
        return False

    transient_markers = (
        "unavailable",
        "deadline_exceeded",
        "timeout",
        "timed out",
        "temporarily",
        "connection",
        "reset by peer",
    )
    return any(marker in msg for marker in transient_markers) or bool(
        _TRANSIENT_STATUS_CODE_RE.search(msg_for_codes)
    )


def _strip_markdown_json_fence(text: str) -> str:
    """Strip accidental outer ```json ... ``` markdown fences if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_response_diagnostics(response: Any) -> str:
    """Summarize finish_reason and prompt_feedback from a Gemini response object."""
    finish_reason = None
    candidates = getattr(response, "candidates", None)
    if candidates and isinstance(candidates, list) and len(candidates) > 0:
        finish_reason = getattr(candidates[0], "finish_reason", None)
    prompt_feedback = getattr(response, "prompt_feedback", None)
    return f"finish_reason={finish_reason!r}, prompt_feedback={prompt_feedback!r}"


def _parse_corrected_json_response(
    response: Any,
    expected_keys: Sequence[str],
) -> dict[str, str]:
    """Validate and extract `{id: corrected_text}` from a Gemini text-only JSON response."""
    raw_text: Optional[str] = None
    try:
        candidate_text = getattr(response, "text", None)
        if isinstance(candidate_text, str):
            raw_text = candidate_text
    except Exception:
        raw_text = None

    parsed_obj: Any = None
    if raw_text and raw_text.strip():
        cleaned_json = _strip_markdown_json_fence(raw_text)
        try:
            parsed_obj = json.loads(cleaned_json)
        except json.JSONDecodeError as exc:
            first_brace = cleaned_json.find("{")
            last_brace = cleaned_json.rfind("}")
            if first_brace != -1 and last_brace > first_brace and (
                first_brace > 0 or last_brace < len(cleaned_json) - 1
            ):
                try:
                    parsed_obj = json.loads(cleaned_json[first_brace : last_brace + 1])
                except json.JSONDecodeError:
                    parsed_obj = None
            if parsed_obj is None:
                diag = _extract_response_diagnostics(response)
                raise GeminiExtractionError(
                    f"Gemini returned invalid JSON during text correction ({diag}): {exc}"
                ) from exc
    else:
        try:
            parsed_obj = getattr(response, "parsed", None)
        except Exception:
            parsed_obj = None

    if parsed_obj is None:
        diag = _extract_response_diagnostics(response)
        raise GeminiExtractionError(
            f"Gemini returned empty response during text correction ({diag})."
        )

    if isinstance(parsed_obj, list) and len(parsed_obj) == 1 and isinstance(parsed_obj[0], dict):
        parsed_obj = parsed_obj[0]

    if not isinstance(parsed_obj, dict):
        raise GeminiExtractionError(
            f"Expected JSON object dict from Gemini, got {type(parsed_obj).__name__}."
        )

    expected_set = set(expected_keys)
    if expected_set and not any(k in parsed_obj for k in expected_set):
        raise GeminiExtractionError(
            "Gemini JSON response did not preserve any expected input id keys."
        )

    normalized: dict[str, str] = {}
    for key, val in parsed_obj.items():
        if val is None or isinstance(val, (dict, list)):
            continue
        cleaned_val = normalize_vietnamese_text(str(val))
        if cleaned_val is not None:
            normalized[str(key)] = cleaned_val
    return normalized


class GeminiTextCorrector:
    """Text-only 1-Shot Vietnamese spelling corrector wrapping `google-genai`."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[GeminiExtractorConfig] = None,
        *,
        client: Optional[Any] = None,
    ) -> None:
        if _GENAI_IMPORT_ERROR is not None or genai is None or types is None:
            raise GeminiExtractionError(
                "Missing required package 'google-genai'. Install it with: pip install google-genai"
            ) from _GENAI_IMPORT_ERROR

        self.config = config or GeminiExtractorConfig()
        self.model_rotation = build_model_rotation(
            self.config.model, self.config.fallback_models
        )

        if client is not None:
            self._client = client
            return

        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not resolved_key or not resolved_key.strip():
            raise GeminiExtractionError(
                "Gemini API key is required. Pass `api_key` or set GEMINI_API_KEY / GOOGLE_API_KEY."
            )

        clean_key = resolved_key.strip()
        try:
            if self.config.timeout_ms is not None:
                http_opts = types.HttpOptions(timeout=self.config.timeout_ms)
                self._client = genai.Client(api_key=clean_key, http_options=http_opts)
            else:
                self._client = genai.Client(api_key=clean_key)
        except Exception as exc:
            raise GeminiExtractionError(
                f"Failed to initialize google.genai.Client: {exc}"
            ) from exc

    def correct_batch(
        self,
        text_batch: dict[str, str],
        *,
        batch_index: int = 1,
        total_batches: int = 1,
    ) -> dict[str, str]:
        """Send one text-only JSON dictionary batch to Gemini with model rotation."""
        if not text_batch:
            return {}

        user_prompt = build_batch_correction_prompt(text_batch)
        config_kwargs: dict[str, Any] = {
            "system_instruction": DEFAULT_TEXT_CORRECTION_INSTRUCTION,
            "temperature": self.config.temperature,
            "response_mime_type": "application/json",
        }
        if self.config.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.config.max_output_tokens
        generation_config = types.GenerateContentConfig(**config_kwargs)

        last_exc: Optional[Exception] = None
        attempts = self.config.max_retries + 1
        model_idx = 0
        tried_models: set[str] = set()

        for attempt in range(1, attempts + 1):
            current_model = self.model_rotation[model_idx % len(self.model_rotation)]
            tried_models.add(current_model)
            try:
                response = self._client.models.generate_content(
                    model=current_model,
                    contents=user_prompt,
                    config=generation_config,
                )
                parsed = _parse_corrected_json_response(
                    response, expected_keys=list(text_batch.keys())
                )

                result: dict[str, str] = {}
                for key, raw_val in text_batch.items():
                    corrected = parsed.get(key, "")
                    if corrected:
                        result[key] = corrected
                    else:
                        logger.warning(
                            "Key '%s' thiếu trong JSON trả về từ model '%s'; giữ nguyên text VietOCR.",
                            key,
                            current_model,
                        )
                        result[key] = raw_val
                return result
            except GeminiExtractionError as exc:
                last_exc = exc
                has_untried_model = len(tried_models) < len(self.model_rotation)
                if has_untried_model and attempt < attempts:
                    next_idx = (model_idx + 1) % len(self.model_rotation)
                    next_model = self.model_rotation[next_idx]
                    logger.warning(
                        "Batch %d/%d: Model '%s' trả về JSON không hợp lệ (%s). Xoay vòng sang '%s'...",
                        batch_index,
                        total_batches,
                        current_model,
                        exc,
                        next_model,
                    )
                    model_idx = next_idx
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if _is_quota_exhausted_error(exc):
                    logger.warning(
                        "Batch %d/%d: Model '%s' gặp lỗi 429 RESOURCE_EXHAUSTED — dừng retry để chuyển ngay sang văn bản VietOCR.",
                        batch_index,
                        total_batches,
                        current_model,
                    )
                    break

                is_404 = _is_model_not_found_error(exc)
                has_untried_model = len(tried_models) < len(self.model_rotation)

                if is_404 and has_untried_model and attempt < attempts:
                    next_idx = (model_idx + 1) % len(self.model_rotation)
                    next_model = self.model_rotation[next_idx]
                    logger.warning(
                        "Batch %d/%d: Model '%s' không khả dụng (404). Xoay vòng sang '%s'...",
                        batch_index,
                        total_batches,
                        current_model,
                        next_model,
                    )
                    model_idx = next_idx
                    continue

                if attempt >= attempts or not _is_transient_api_error(exc):
                    break

                # Rotate immediately to the next model in the rotation list
                if len(self.model_rotation) > 1:
                    next_idx = (model_idx + 1) % len(self.model_rotation)
                    next_model = self.model_rotation[next_idx]
                    logger.warning(
                        "Batch %d/%d: Model '%s' gặp lỗi tạm thời (%s). Xoay vòng sang '%s'...",
                        batch_index,
                        total_batches,
                        current_model,
                        exc,
                        next_model,
                    )
                    model_idx = next_idx

                cycle = max(0, (attempt - 1) // len(self.model_rotation))
                delay = min(self.config.retry_base_delay * (1.7 ** cycle), 30.0)
                logger.warning(
                    "Batch %d/%d thử lại (lần %d/%d) sau %.1fs...",
                    batch_index,
                    total_batches,
                    attempt,
                    attempts,
                    delay,
                )
                time.sleep(delay)

        actual_attempts = attempt if "attempt" in locals() else 1
        raise GeminiExtractionError(
            f"Gemini text correction failed on batch {batch_index}/{total_batches} "
            f"after {actual_attempts} attempt(s): {last_exc}"
        ) from last_exc

    def correct_document_texts(
        self,
        text_map: dict[str, str],
        *,
        batch_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, str]:
        """Correct all document text entries sequentially (1 request for <= 5000 words)."""
        batches = split_text_dict_into_batches(
            text_map, max_words=self.config.max_words_per_batch
        )
        if not batches:
            return {}

        merged: dict[str, str] = {}
        total_batches = len(batches)
        for idx, batch in enumerate(batches, 1):
            if batch_callback is not None:
                batch_callback(idx, total_batches)
            corrected_batch = self.correct_batch(
                batch, batch_index=idx, total_batches=total_batches
            )
            merged.update(corrected_batch)

        return merged
