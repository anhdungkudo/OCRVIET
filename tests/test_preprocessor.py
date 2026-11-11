"""Real PyMuPDF + OpenCV tests for Step 1 preprocessing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

from vn_pdf_to_docx.exceptions import ImagePreprocessError, PdfRenderError
from vn_pdf_to_docx.preprocessor import (
    PreprocessConfig,
    denoise_and_enhance_contrast,
    encode_image_bytes,
    extract_red_channel_and_suppress_seal,
    preprocess_page_image,
    preprocess_pdf,
    render_pdf_to_images,
)


class TestPreprocessor(unittest.TestCase):
    """Test PDF rendering at 300 DPI and OpenCV red-seal removal + denoising + CLAHE."""

    def _create_sample_pdf(self, path: Path, *, num_pages: int = 1, encrypt: bool = False) -> None:
        doc = fitz.open()
        for page_idx in range(num_pages):
            page = doc.new_page(width=595, height=842)  # A4 in PDF points (72 DPI)
            page.insert_text(
                (72, 120),
                f"TRANG {page_idx + 1} - VAN BAN HANH CHINH",
                fontsize=18,
                color=(0, 0, 0),
            )
            # Red circular official seal overlapping part of the text
            page.draw_circle((200, 115), 45, color=(0.85, 0.08, 0.08), width=4)
            page.insert_text((170, 122), "CONG DAU DO", fontsize=11, color=(0.85, 0.08, 0.08))
        if encrypt:
            doc.save(
                str(path),
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="owner123",
                user_pw="user123",
            )
        else:
            doc.save(str(path))
        doc.close()

    def test_render_pdf_at_300_dpi_and_unicode_path(self) -> None:
        self.assertEqual(PreprocessConfig().dpi, 200)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "văn_bản_mẫu_300dpi.pdf"
            self._create_sample_pdf(pdf_path, num_pages=2)

            images = render_pdf_to_images(pdf_path, dpi=300)
            self.assertEqual(len(images), 2)
            for img in images:
                self.assertEqual(img.dtype, np.uint8)
                self.assertEqual(img.ndim, 3)
                self.assertEqual(img.shape[2], 3)
                # 595 pt * 300 / 72 = 2479..2480 px; 842 pt * 300 / 72 = 3508..3509 px
                self.assertAlmostEqual(img.shape[1], 2480, delta=2)
                self.assertAlmostEqual(img.shape[0], 3508, delta=2)

            # Page subset selection
            page_two_only = render_pdf_to_images(pdf_path, dpi=150, page_numbers=[2])
            self.assertEqual(len(page_two_only), 1)
            self.assertAlmostEqual(page_two_only[0].shape[1], 1240, delta=2)

    def test_red_seal_removal_preserves_black_text_and_blue_signature(self) -> None:
        # Synthetic RGB scanned page with white-ish background
        img = np.full((300, 500, 3), 240, dtype=np.uint8)

        # Black administrative text
        cv2.putText(
            img, "QUYET DINH", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 4
        )
        # Blue pen signature
        cv2.line(img, (320, 220), (450, 200), (20, 50, 195), 3)

        # Overlay red stamp multiplicatively across text and background
        stamp_mask = np.zeros((300, 500), dtype=np.uint8)
        cv2.circle(stamp_mask, (160, 140), 70, 255, 6)
        cv2.putText(
            stamp_mask, "DAU DO", (105, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 3
        )
        stamped = img.astype(np.float32)
        stamped[stamp_mask > 0] *= np.array([195 / 240.0, 25 / 240.0, 30 / 240.0])
        stamped_u8 = np.clip(stamped, 0, 255).astype(np.uint8)

        # Masks for pixel regions
        pure_stamp = (stamp_mask > 0) & (img[:, :, 0] > 200)
        black_under_stamp = (stamp_mask > 0) & (img[:, :, 0] < 40)
        blue_sig = (img[:, :, 2] > 180) & (img[:, :, 0] < 40)

        cleaned = extract_red_channel_and_suppress_seal(
            stamped_u8, color_order="RGB", red_suppression_strength=1.3
        )
        self.assertEqual(cleaned.shape, (300, 500))
        self.assertEqual(cleaned.dtype, np.uint8)

        # Red stamp on paper should fade to >= 230 (near paper white 240)
        self.assertGreater(float(cleaned[pure_stamp].mean()), 230.0)
        # Black text underneath the red stamp must stay dark (< 50) and unbroken
        self.assertLess(float(cleaned[black_under_stamp].mean()), 50.0)
        # Blue signature must remain dark (< 50)
        self.assertLess(float(cleaned[blue_sig].mean()), 50.0)

        # BGR input ordering must behave identically when color_order="BGR"
        bgr_u8 = cv2.cvtColor(stamped_u8, cv2.COLOR_RGB2BGR)
        cleaned_bgr = extract_red_channel_and_suppress_seal(
            bgr_u8, color_order="BGR", red_suppression_strength=1.3
        )
        np.testing.assert_array_equal(cleaned, cleaned_bgr)

    def test_denoising_and_clahe_reduce_noise_and_enhance_text(self) -> None:
        rng = np.random.default_rng(123)
        base = np.full((240, 320), 210, dtype=np.uint8)
        cv2.putText(base, "CONG HOA", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 80, 2)
        noisy = np.clip(base.astype(np.float32) + rng.normal(0, 12, base.shape), 0, 255).astype(
            np.uint8
        )

        enhanced = denoise_and_enhance_contrast(noisy, nlm_h=10.0, clahe_clip_limit=2.0)
        self.assertEqual(enhanced.shape, noisy.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Background std should decrease significantly after NLM denoising
        bg_region = (slice(10, 60), slice(10, 100))
        self.assertLess(float(enhanced[bg_region].std()), float(noisy[bg_region].std()))

    def test_preprocess_pdf_and_encoding_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "scan.pdf"
            self._create_sample_pdf(pdf_path, num_pages=1)

            pages = preprocess_pdf(pdf_path, PreprocessConfig(dpi=150, image_format="png"))
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0].page_number, 1)
            self.assertEqual(pages[0].mime_type, "image/png")
            self.assertTrue(pages[0].encoded_bytes.startswith(b"\x89PNG"))

            decoded = cv2.imdecode(
                np.frombuffer(pages[0].encoded_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, pages[0].image.shape)

            jpg_bytes, jpg_mime = encode_image_bytes(pages[0].image, image_format="jpeg")
            self.assertEqual(jpg_mime, "image/jpeg")
            self.assertTrue(jpg_bytes.startswith(b"\xff\xd8\xff"))

    def test_preprocess_pdf_with_generator_page_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "multipage.pdf"
            self._create_sample_pdf(pdf_path, num_pages=3)

            gen = (p for p in (2, 3))
            pages = preprocess_pdf(pdf_path, PreprocessConfig(dpi=150), page_numbers=gen)
            self.assertEqual([p.page_number for p in pages], [2, 3])
            self.assertEqual(len(pages), 2)

    def test_red_seal_removal_preserves_faded_and_wet_overlapped_text(self) -> None:
        base = np.full((320, 480, 3), 245, dtype=np.uint8)
        # Faded gray text (intensity ~125) and crisp black text (intensity ~25)
        cv2.putText(base, "FADED", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (125, 125, 125), 3)
        cv2.putText(base, "BLACK", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (25, 25, 25), 3)

        seal = np.zeros((320, 480), dtype=np.uint8)
        cv2.circle(seal, (110, 110), 65, 255, 8)   # overlaps faded text
        cv2.circle(seal, (110, 230), 65, 255, 8)   # overlaps black text
        cv2.circle(seal, (360, 180), 65, 255, 8)   # blank paper region

        # Translucent multiplicative red ink
        mult = base.astype(np.float32)
        mult[seal > 0] *= np.array([0.88, 0.18, 0.18])
        mult_u8 = np.clip(mult, 0, 255).astype(np.uint8)

        # Semi-opaque wet red ink over text
        wet = base.astype(np.float32)
        wet[seal > 0] = 0.65 * (wet[seal > 0] * np.array([0.88, 0.18, 0.18])) + 0.35 * np.array([210.0, 20.0, 20.0])
        wet_u8 = np.clip(wet, 0, 255).astype(np.uint8)

        blank_seal_m = (seal > 0) & (base[:, :, 0] > 230)
        faded_under_m = (seal > 0) & (base[:, :, 0] > 100) & (base[:, :, 0] < 140)
        black_under_m = (seal > 0) & (base[:, :, 0] < 40)

        cleaned_mult = extract_red_channel_and_suppress_seal(mult_u8, color_order="RGB")
        cleaned_wet = extract_red_channel_and_suppress_seal(wet_u8, color_order="RGB")

        # Pure red seal on paper must fade cleanly to paper white
        self.assertGreater(float(cleaned_mult[blank_seal_m].mean()), 235.0)
        self.assertGreater(float(cleaned_wet[blank_seal_m].mean()), 235.0)
        # Faded and black text under red seal must stay dark and unbroken
        self.assertLess(float(cleaned_mult[faded_under_m].mean()), 130.0)
        self.assertLess(float(cleaned_wet[faded_under_m].mean()), 165.0)
        self.assertLess(float(cleaned_wet[black_under_m].mean()), 95.0)

        # After CLAHE, blank tile containing red seal must stay bright white (no black smudge)
        enhanced = preprocess_page_image(mult_u8, PreprocessConfig(nlm_h=0.0))
        self.assertGreater(float(enhanced[blank_seal_m].mean()), 235.0)

    def test_pdf_with_transparent_red_seal_over_black_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "transparent_seal.pdf"
            doc = fitz.open()
            page = doc.new_page(width=400, height=250)
            page.insert_text((40, 130), "DIEU 1. VAN BAN", fontsize=22, color=(0, 0, 0))
            shape = page.new_shape()
            shape.draw_circle((140, 125), 55)
            shape.finish(color=(0.88, 0.08, 0.08), fill=(0.9, 0.1, 0.1), stroke_opacity=0.65, fill_opacity=0.25, width=5)
            shape.commit()
            doc.save(str(pdf_path))
            doc.close()

            pages = preprocess_pdf(pdf_path, PreprocessConfig(dpi=150, nlm_h=6.0))
            self.assertEqual(len(pages), 1)
            processed = pages[0].image
            # Pure paper region outside text & seal stays white, while black text stays dark
            self.assertGreater(float(processed[10:40, 10:40].mean()), 230.0)
            self.assertLess(int(processed.min()), 40)

    def test_error_and_edge_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            # Missing PDF
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(tmp_dir / "missing.pdf")

            # Empty 0-byte PDF
            empty_pdf = tmp_dir / "empty.pdf"
            empty_pdf.write_bytes(b"")
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(empty_pdf)

            # Corrupt PDF
            bad_pdf = tmp_dir / "corrupt.pdf"
            bad_pdf.write_bytes(b"not a valid pdf")
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(bad_pdf)

            # Encrypted PDF
            enc_pdf = tmp_dir / "encrypted.pdf"
            self._create_sample_pdf(enc_pdf, encrypt=True)
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(enc_pdf)

            # Invalid page numbers, bools, empty list & invalid DPI
            valid_pdf = tmp_dir / "valid.pdf"
            self._create_sample_pdf(valid_pdf, num_pages=1)
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(valid_pdf, page_numbers=[2])
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(valid_pdf, page_numbers=[])
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(valid_pdf, page_numbers=[True])  # type: ignore[list-item]
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(valid_pdf, dpi=30)
            with self.assertRaises(PdfRenderError):
                render_pdf_to_images(valid_pdf, dpi=True)  # type: ignore[arg-type]

        # Invalid numpy arrays for preprocessing
        with self.assertRaises(ImagePreprocessError):
            preprocess_page_image(np.array([], dtype=np.uint8))
        with self.assertRaises(ImagePreprocessError):
            extract_red_channel_and_suppress_seal(np.zeros((10, 10, 5), dtype=np.uint8))
        with self.assertRaises(ImagePreprocessError):
            denoise_and_enhance_contrast(np.zeros((10, 10), dtype=np.uint8), template_window_size=4)

        # Already grayscale 2D image should pass cleanly
        gray = np.full((64, 64), 200, dtype=np.uint8)
        out_gray = preprocess_page_image(gray)
        self.assertEqual(out_gray.shape, (64, 64))

        # 4-channel RGBA + float [0, 1] input + raw red channel (strength=0.0)
        rgba_float = np.ones((40, 40, 4), dtype=np.float32) * 0.9
        rgba_float[10:20, 10:20, :3] = [0.85, 0.1, 0.1]  # pure red patch
        raw_red = extract_red_channel_and_suppress_seal(
            rgba_float, color_order="RGB", red_suppression_strength=0.0
        )
        self.assertEqual(raw_red.shape, (40, 40))
        self.assertEqual(raw_red.dtype, np.uint8)
        self.assertGreater(int(raw_red[15, 15]), 210)

        # Float array with NaN should coerce cleanly without crashing
        nan_img = np.full((24, 24, 3), 0.8, dtype=np.float32)
        nan_img[0, 0, 0] = np.nan
        cleaned_nan = extract_red_channel_and_suppress_seal(nan_img, color_order="RGB")
        self.assertEqual(cleaned_nan.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
