"""
preprocessing.py
-------------------
Image preprocessing for receipts before OCR, adapted from the
adaptive scan/photo logic in ground.py — but with the LLM-based
"is this a scan or photo" routing step REMOVED for the demo, since
that step required an API call and added latency/dependency that
isn't needed for a fast, offline-capable live demo.

Instead, we use a lightweight heuristic (image sharpness / Laplacian
variance) to decide between "scan" and "photo" processing strategies,
which is instant and runs entirely locally.

The preprocessed image is used for BOTH OCR and the CNN: the training
set in Datasets/split was built from these preprocessed grayscale
images (Structured_Dataset_V5), not from the raw colour photos.
"""

import cv2
import numpy as np
from PIL import Image, ImageOps

SCAN_SHARPNESS_THRESHOLD = 800


def _laplacian_variance(gray_img) -> float:
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def _looks_like_clean_scan(gray_img) -> bool:
    """
    Heuristic replacement for the LLM routing call in ground.py.
    A clean digital scan tends to have high edge sharpness and low
    noise variance compared to a noisy mobile photo. This is a
    coarse but fast, fully local substitute for the original
    LLM-based 'scan vs photo' classification.
    """
    # Empirically, clean scans tend to have higher Laplacian variance
    # (sharper edges) than blurry/noisy phone photos of thermal receipts.
    return _laplacian_variance(gray_img) > SCAN_SHARPNESS_THRESHOLD


def _load_gray(image_path: str) -> np.ndarray:
    """
    Loads an image as grayscale with EXIF orientation applied.
    PIL is used instead of cv2.imread because cv2.imread cannot open
    non-ASCII Windows paths, and phone photos often rely on the EXIF
    orientation tag to appear upright.
    """
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            return np.array(im.convert("L"))
    except Exception as e:
        raise ValueError(f"Could not read image: {image_path} ({e})")


def preprocess_receipt_with_info(image_path: str):
    """
    Applies the same adaptive preprocessing pipeline used when building
    the training dataset (ground.py), minus the LLM routing call.
    Returns (processed grayscale image, info dict).
    """
    gray = _load_gray(image_path)

    # Resize to a consistent width (1800px), preserving aspect ratio —
    # identical to the training-time preprocessing in ground.py.
    target_width = 1800
    scale = target_width / gray.shape[1]
    resized = cv2.resize(
        gray,
        (target_width, int(gray.shape[0] * scale)),
        interpolation=cv2.INTER_LANCZOS4  # Upgraded algorithm for sharper text edges
    )

    # Increase contrast to make tiny text pop against the white paper
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    resized = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)

    sharpness = _laplacian_variance(resized)

    if sharpness > SCAN_SHARPNESS_THRESHOLD:
        strategy = "scan"
        # "Scan" strategy: Light sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        final_img = cv2.filter2D(resized, -1, kernel)
    else:
        strategy = "photo"
        # "Photo" strategy: Denoise + Adaptive Threshold + Slimming
        denoised = cv2.bilateralFilter(resized, 9, 75, 75)

        # 1. THE THRESHOLD: Bump C slightly up to 6.
        # (4 was too thick/bonded, 8 was too thin/disconnected. 6 is the middle ground).
        binarized = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 71, 6
        )

        # 2. THE SLIMMING FILTER: Expand the white background to shave the black text.
        # A 2x2 kernel is very gentle. It will separate bonded letters without breaking loops.
        kernel = np.ones((2, 2), np.uint8)
        final_img = cv2.dilate(binarized, kernel, iterations=1)

    info = {
        "strategy": strategy,
        "sharpness": round(sharpness, 1),
        "original_size": [int(gray.shape[1]), int(gray.shape[0])],
        "processed_size": [int(final_img.shape[1]), int(final_img.shape[0])],
    }
    return final_img, info


def preprocess_receipt(image_path: str) -> np.ndarray:
    """Returns the processed grayscale image as a numpy array, ready for OCR."""
    return preprocess_receipt_with_info(image_path)[0]


def save_image(img: np.ndarray, output_path: str) -> str:
    """cv2.imwrite equivalent that also works with non-ASCII Windows paths."""
    ext = "." + str(output_path).rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}")
    buf.tofile(str(output_path))
    return str(output_path)


def preprocess_and_save(image_path: str, output_path: str) -> str:
    """Convenience wrapper that preprocesses and saves to disk, returning the path."""
    return save_image(preprocess_receipt(image_path), output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py <input_image> <output_image>")
        sys.exit(1)
    out = preprocess_and_save(sys.argv[1], sys.argv[2])
    print(f"Saved preprocessed image to: {out}")
