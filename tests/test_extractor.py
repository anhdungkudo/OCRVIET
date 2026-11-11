"""Tests for Step 3: 1-Shot text-only Gemini batch correction & model rotation."""

from __future__ import annotations

import importlib.util
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import vn_pdf_to_docx.extractor as extractor_mod
from vn_pdf_to_docx.exceptions import GeminiExtractionError
from vn_pdf_to_docx.extractor import (
    DEFAULT_MODEL_ROTATION,
    GeminiExtractorConfig,
    GeminiTextCorrector,
    _is_model_not_found_error,
    _is_transient_api_error,
    _parse_corrected_json_response,
    build_batch_correction_prompt,
    build_model_rotation,
    split_text_dict_into_batches,
)

HAS_GENAI = importlib.util.find_spec("google.genai") is not None


class TestExtractorLogic(unittest.TestCase):
    """Test text-only batch splitting, model rotation, error handling, and JSON parsing."""

    def test_batch_splitting_respects_5000_word_limit(self) -> None:
        self.assertEqual(split_text_dict_into_batches({}), [])

        small_doc = {"id_1": "Căn cứ Nghị định", "id_2": "Điều 1. Nội dung"}
        batches_small = split_text_dict_into_batches(small_doc, max_words=5000)
        self.assertEqual(len(batches_small), 1)
        self.assertEqual(batches_small[0], small_doc)

        # Document exceeding max_words splits sequentially
        large_doc = {
            "id_1": " ".join(["từ"] * 60),
            "id_2": " ".join(["chữ"] * 60),
            "id_3": " ".join(["câu"] * 30),
        }
        batches_large = split_text_dict_into_batches(large_doc, max_words=100)
        self.assertEqual(len(batches_large), 2)
        self.assertEqual(list(batches_large[0].keys()), ["id_1"])
        self.assertEqual(list(batches_large[1].keys()), ["id_2", "id_3"])

    def test_build_batch_prompt_and_model_rotation(self) -> None:
        self.assertEqual(
            DEFAULT_MODEL_ROTATION,
            ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"],
        )
        self.assertNotIn("gemini-3.6-flash", DEFAULT_MODEL_ROTATION)
        rotation = build_model_rotation("gemini-3.8-flash", ["gemini-3.6-flash", "gemini-3.5-flash"])
        self.assertEqual(
            rotation,
            ["gemini-3.8-flash", "gemini-3.5-flash"],
        )

        prompt = build_batch_correction_prompt({"id_1": "Điểu 1. Văn ban"})
        self.assertIn("id_1", prompt)
        self.assertIn("i/l/1", prompt)

    def test_transient_vs_permanent_and_404_error_classification(self) -> None:
        # 429 / RESOURCE_EXHAUSTED must NEVER be retried (immediate fallback to VietOCR)
        self.assertFalse(_is_transient_api_error(RuntimeError("429 ResourceExhausted")))
        self.assertFalse(_is_transient_api_error(RuntimeError("429 RESOURCE_EXHAUSTED trace_id=94041")))
        self.assertTrue(_is_transient_api_error(RuntimeError("503 Service Unavailable")))
        self.assertTrue(_is_transient_api_error(RuntimeError("504 Deadline Exceeded")))
        self.assertTrue(_is_transient_api_error(TimeoutError("Request timed out")))
        self.assertTrue(_is_transient_api_error(ConnectionError("")))

        self.assertTrue(_is_model_not_found_error(RuntimeError("404 NOT_FOUND: model not found")))
        self.assertFalse(_is_model_not_found_error(RuntimeError("429 RESOURCE_EXHAUSTED request_id=404")))

        class StringCodeError(Exception):
            def __init__(self, code: str) -> None:
                self.code = code

        self.assertFalse(_is_transient_api_error(StringCodeError("RESOURCE_EXHAUSTED")))
        self.assertTrue(_is_transient_api_error(StringCodeError("UNAVAILABLE")))
        self.assertFalse(_is_transient_api_error(StringCodeError("INVALID_ARGUMENT")))
        self.assertFalse(_is_transient_api_error(RuntimeError("400 INVALID_ARGUMENT")))
        self.assertFalse(_is_transient_api_error(RuntimeError("403 PERMISSION_DENIED")))

        # Substring false-positive guards
        self.assertTrue(_is_transient_api_error(RuntimeError("503 Service Unavailable request_id=124009")))
        self.assertFalse(_is_transient_api_error(RuntimeError("400 INVALID_ARGUMENT request-id=503")))

        class CodeZeroWithStatusError(Exception):
            def __init__(self, code: object, status_code: object, msg: str = "", status: object = None) -> None:
                super().__init__(msg)
                self.code = code
                self.status_code = status_code
                self.status = status

        self.assertFalse(_is_transient_api_error(CodeZeroWithStatusError(0, 503, "")))
        self.assertTrue(_is_transient_api_error(CodeZeroWithStatusError(0, 400, "503 Service Unavailable")))
        self.assertTrue(_is_transient_api_error(CodeZeroWithStatusError(None, 503, "")))
        self.assertFalse(_is_transient_api_error(CodeZeroWithStatusError(None, 400, "")))

    def test_parse_corrected_json_response(self) -> None:
        payload = {"id_1": "Điều 1. Quy định chung.", "id_2": "Nơi nhận:"}
        parsed = _parse_corrected_json_response(
            SimpleNamespace(text=json.dumps(payload, ensure_ascii=False)),
            expected_keys=["id_1", "id_2"],
        )
        self.assertEqual(parsed, payload)

        # Markdown-fenced JSON
        fenced = _parse_corrected_json_response(
            SimpleNamespace(text=f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"),
            expected_keys=["id_1", "id_2"],
        )
        self.assertEqual(fenced, payload)

        # Empty response raises GeminiExtractionError
        with self.assertRaises(GeminiExtractionError):
            _parse_corrected_json_response(
                SimpleNamespace(text="", parsed=None, candidates=[SimpleNamespace(finish_reason="SAFETY")]),
                expected_keys=["id_1"],
            )

        # Malformed JSON raises GeminiExtractionError
        with self.assertRaises(GeminiExtractionError) as ctx:
            _parse_corrected_json_response(
                SimpleNamespace(text="{bad", parsed=None, candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")]),
                expected_keys=["id_1"],
            )
        self.assertIn("MAX_TOKENS", str(ctx.exception))

        # Completely mismatched keys raise GeminiExtractionError
        with self.assertRaises(GeminiExtractionError):
            _parse_corrected_json_response(
                SimpleNamespace(text='{"wrong_key": "abc"}'),
                expected_keys=["id_1"],
            )

        # Nested dict/list values are skipped safely so caller falls back to raw text
        nested_resp = SimpleNamespace(text='{"id_1": "Hợp lệ", "id_2": {"bad": 1}, "id_3": ["x"]}')
        self.assertEqual(
            _parse_corrected_json_response(nested_resp, expected_keys=["id_1", "id_2", "id_3"]),
            {"id_1": "Hợp lệ"},
        )

    def test_corrector_requires_api_key_and_validates_config(self) -> None:
        cfg = GeminiExtractorConfig()
        self.assertEqual(cfg.model, "gemini-3.8-flash")
        self.assertEqual(cfg.fallback_models, [])
        self.assertEqual(cfg.max_retries, 1)
        self.assertEqual(cfg.max_words_per_batch, 5000)
        self.assertEqual(cfg.timeout_ms, 35_000)
        self.assertIsNone(cfg.max_output_tokens)

        if HAS_GENAI:
            old_gemini = os.environ.pop("GEMINI_API_KEY", None)
            old_google = os.environ.pop("GOOGLE_API_KEY", None)
            try:
                with self.assertRaises(GeminiExtractionError):
                    GeminiTextCorrector(api_key="")
            finally:
                if old_gemini is not None:
                    os.environ["GEMINI_API_KEY"] = old_gemini
                if old_google is not None:
                    os.environ["GOOGLE_API_KEY"] = old_google

    def test_1_shot_primary_model_only_and_429_fails_immediately_without_retry(self) -> None:
        fake_types = SimpleNamespace(
            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw),
            HttpOptions=lambda **kw: SimpleNamespace(**kw),
        )

        mock_client = MagicMock()
        corrected_json = json.dumps(
            {
                "id_1": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
                "id_2": "Điều 1. Quy định chung ngày 05/03/2020.",
            },
            ensure_ascii=False,
        )
        # Transient 503 retries once on the SAME primary model and succeeds
        mock_client.models.generate_content.side_effect = [
            RuntimeError("503 UNAVAILABLE"),
            SimpleNamespace(text=corrected_json),
        ]

        raw_input = {
            "id_1": "CONG HOA XA HOI CHU NGHIA VIET NAM",
            "id_2": "Điểu 1. Quy định chung ngày O5/O3/2O2O.",
        }

        with (
            patch.object(extractor_mod, "_GENAI_IMPORT_ERROR", None),
            patch.object(extractor_mod, "genai", SimpleNamespace(Client=MagicMock())),
            patch.object(extractor_mod, "types", fake_types),
            patch.object(extractor_mod.time, "sleep") as sleep_mock,
        ):
            corrector = GeminiTextCorrector(
                config=GeminiExtractorConfig(retry_base_delay=0.5),
                client=mock_client,
            )
            self.assertEqual(corrector.model_rotation, ["gemini-3.8-flash"])
            result = corrector.correct_document_texts(raw_input)

            self.assertEqual(result["id_1"], "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
            self.assertEqual(result["id_2"], "Điều 1. Quy định chung ngày 05/03/2020.")

            called_models = [c.kwargs["model"] for c in mock_client.models.generate_content.call_args_list]
            self.assertEqual(called_models, ["gemini-3.8-flash", "gemini-3.8-flash"])
            for call in mock_client.models.generate_content.call_args_list:
                self.assertIsInstance(call.kwargs["contents"], str)
                self.assertEqual(call.kwargs["config"].response_mime_type, "application/json")
            self.assertEqual(sleep_mock.call_count, 1)

            # 429 RESOURCE_EXHAUSTED must fail IMMEDIATELY on attempt 1 (NO 7 retries, NO sleep)
            mock_client.models.generate_content.reset_mock()
            mock_client.models.generate_content.side_effect = RuntimeError(
                "429 RESOURCE_EXHAUSTED: retryDelay: 62693s"
            )
            sleep_mock.reset_mock()
            with self.assertRaises(GeminiExtractionError) as ctx429:
                corrector_high_retries = GeminiTextCorrector(
                    config=GeminiExtractorConfig(max_retries=6, retry_base_delay=1.5),
                    client=mock_client,
                )
                corrector_high_retries.correct_batch({"id_1": "test"})
            self.assertEqual(mock_client.models.generate_content.call_count, 1)
            sleep_mock.assert_not_called()
            self.assertIn("after 1 attempt(s)", str(ctx429.exception))

            # Permanent 400 error fails immediately without sleep
            mock_client.models.generate_content.reset_mock()
            mock_client.models.generate_content.side_effect = RuntimeError("400 INVALID_ARGUMENT")
            sleep_mock.reset_mock()
            with self.assertRaises(GeminiExtractionError):
                corrector.correct_batch({"id_1": "test"})
            self.assertEqual(mock_client.models.generate_content.call_count, 1)
            sleep_mock.assert_not_called()

    def test_multi_batch_sequential_execution_when_exceeding_max_words(self) -> None:
        fake_types = SimpleNamespace(
            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw),
            HttpOptions=lambda **kw: SimpleNamespace(**kw),
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            SimpleNamespace(text=json.dumps({"id_1": "Đoạn 1 đã sửa."}, ensure_ascii=False)),
            SimpleNamespace(text=json.dumps({"id_2": "Đoạn 2 đã sửa."}, ensure_ascii=False)),
        ]
        long_map = {
            "id_1": " ".join(["chữ"] * 40),
            "id_2": " ".join(["văn"] * 40),
        }
        batch_events: list[tuple[int, int]] = []
        with (
            patch.object(extractor_mod, "_GENAI_IMPORT_ERROR", None),
            patch.object(extractor_mod, "genai", SimpleNamespace(Client=MagicMock())),
            patch.object(extractor_mod, "types", fake_types),
        ):
            corrector = GeminiTextCorrector(
                config=GeminiExtractorConfig(max_words_per_batch=50),
                client=mock_client,
            )
            res = corrector.correct_document_texts(
                long_map, batch_callback=lambda c, t: batch_events.append((c, t))
            )
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        self.assertEqual(batch_events, [(1, 2), (2, 2)])
        self.assertEqual(res, {"id_1": "Đoạn 1 đã sửa.", "id_2": "Đoạn 2 đã sửa."})

    def test_malformed_json_with_explicit_fallback_excludes_36_flash_and_parses_prose_json(self) -> None:
        # 1. Brief prose wrapping a valid JSON object should parse cleanly
        prose_resp = SimpleNamespace(text='Kết quả hiệu đính:\n{"id_1": "Điều 1. Nội dung."}\nHoàn tất.')
        self.assertEqual(
            _parse_corrected_json_response(prose_resp, expected_keys=["id_1"]),
            {"id_1": "Điều 1. Nội dung."},
        )

        # 2. Explicit fallback_models filters out gemini-3.6-flash and falls back to gemini-3.5-flash
        fake_types = SimpleNamespace(
            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw),
            HttpOptions=lambda **kw: SimpleNamespace(**kw),
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            SimpleNamespace(text="{truncated bad json"),
            SimpleNamespace(text=json.dumps({"id_1": "Điều 1. Đã sửa."}, ensure_ascii=False)),
        ]
        with (
            patch.object(extractor_mod, "_GENAI_IMPORT_ERROR", None),
            patch.object(extractor_mod, "genai", SimpleNamespace(Client=MagicMock())),
            patch.object(extractor_mod, "types", fake_types),
        ):
            corrector = GeminiTextCorrector(
                config=GeminiExtractorConfig(fallback_models=["gemini-3.6-flash", "gemini-3.5-flash"]),
                client=mock_client,
            )
            res = corrector.correct_batch({"id_1": "Điểu 1. Da sua."})
        self.assertEqual(res, {"id_1": "Điều 1. Đã sửa."})
        called_models = [c.kwargs["model"] for c in mock_client.models.generate_content.call_args_list]
        self.assertEqual(called_models, ["gemini-3.8-flash", "gemini-3.5-flash"])


if __name__ == "__main__":
    unittest.main()
