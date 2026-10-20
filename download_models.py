"""
download_models.py
-------------------
Downloads the trained EfficientNet-B0 V1 and DistilBERT V1 weights from the
GitHub release and unpacks them into ./models, where inference.py finds them.

    python download_models.py
"""

import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

URL = "https://github.com/wenkeeeeeeee/receipt-classification-system/releases/download/v1.0/receipt-models-v1.zip"
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ZIP_PATH = BASE_DIR / "receipt-models-v1.zip"

REQUIRED = [
    "CNN_aug/best_efficientnet.pth",
    "CNN_aug/class_names.json",
    "BERT_aug/best_distilbert/model.safetensors",
    "BERT_aug/best_distilbert/config.json",
    "BERT_aug/label_mapping.json",
]


def _progress(block, block_size, total):
    if total > 0:
        done = min(block * block_size, total)
        sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
        sys.stdout.flush()


def main():
    if all((MODELS_DIR / f).exists() for f in REQUIRED):
        print(f"Models already present in {MODELS_DIR}")
        return

    print(f"Downloading models from\n  {URL}")
    urllib.request.urlretrieve(URL, ZIP_PATH, _progress)
    print("\nExtracting...")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.namelist():
            target = (MODELS_DIR / member).resolve()
            if MODELS_DIR.resolve() not in target.parents and target != MODELS_DIR.resolve():
                raise RuntimeError(f"Unsafe path in archive: {member}")
        zf.extractall(MODELS_DIR)
    ZIP_PATH.unlink()

    missing = [f for f in REQUIRED if not (MODELS_DIR / f).exists()]
    if missing:
        shutil.rmtree(MODELS_DIR, ignore_errors=True)
        raise SystemExit(f"Archive is missing files: {missing}")
    print(f"Done. Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
