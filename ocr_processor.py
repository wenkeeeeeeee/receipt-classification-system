"""
ocr_processor.py
-----------------
PaddleOCR 3.x wrapper used by both training (OCR cache) and the demo.

The OCR text fed to DistilBERT is the recognised lines joined with a
single SPACE — that is exactly the format stored in ocr_text_cache.json
at training time, so it must not change. The LLM, however, benefits from
seeing the real line structure (header vs. items vs. totals), so the
lines are also exposed separately via extract_lines_from_image().
"""

import os

# --- Global Engine Flags (Required for 2026 Stability) ---
# These MUST be set before paddleocr is imported, otherwise Paddle has
# already read them and runs its slow model-source connectivity check.
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import logging
import tempfile
import numpy as np
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)


class OCRProcessor:
    def __init__(self, device="cpu"):
        """Initializes PaddleOCR 3.x with the Modern PIR Pipeline."""
        print(f"[OCR] Initializing Paddle 3.x on {device.upper()}...")
        # Note: use_textline_orientation replaces use_angle_cls in 2026 API
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            device=device,
            det_limit_side_len=2560,  # Forces PaddleOCR to allow high-res images
            det_limit_type='max'
        )
        print("[OCR] Engine Ready.")

    def _recursive_find_text(self, obj):
        """Internal helper to find 'rec_texts' in the complex Paddle 3.x result object."""
        if isinstance(obj, dict):
            if 'rec_texts' in obj: return obj['rec_texts']
            for v in obj.values():
                res = self._recursive_find_text(v)
                if res: return res
        elif hasattr(obj, '__dict__'):
            return self._recursive_find_text(vars(obj))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                res = self._recursive_find_text(item)
                if res: return res
        return None

    def extract_lines_from_image(self, image_path):
        """Returns the recognised text lines in reading order (list of str)."""
        try:
            if not os.path.exists(image_path):
                return []

            # Safe byte-loader for Windows paths (handles WhatsApp spaces/dots)
            img_bytes = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            if img is None:
                return []

            # In Paddle 3.x, predict() returns a generator
            results = list(self.ocr.predict(img))

            # Use recursive search to find the 'rec_texts' list
            text_list = self._recursive_find_text(results)
            return [t for t in (text_list or []) if t and t.strip()]

        except Exception as e:
            print(f"[OCR Image Error] {e}")
            return []

    def extract_text_from_image(self, image_path):
        """Space-joined OCR text — the same format DistilBERT was trained on."""
        return " ".join(self.extract_lines_from_image(image_path))

    def extract_text_from_pdf(self, pdf_path):
        """Converts PDF pages to images and runs PaddleOCR on each."""
        try:
            from pdf2image import convert_from_path

            # 300 DPI is the standard for receipt clarity
            pages = convert_from_path(pdf_path, dpi=300)
            results = []

            for i, page in enumerate(pages):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                page.save(tmp_path, "PNG")
                text = self.extract_text_from_image(tmp_path)
                os.remove(tmp_path)
                results.append({"page": i + 1, "text": text})
            return results
        except Exception as e:
            print(f"[OCR PDF Error] {e}")
            return []
