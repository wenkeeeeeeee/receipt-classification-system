"""
inference.py
-------------
Loads the trained EfficientNet-B0 V1 and DistilBERT V1 models (both
trained on the augmented dataset) and runs inference on a single
receipt. The architecture, preprocessing, and label mapping here MUST
exactly match what was used during training, or predictions will be
silently wrong (the classic "runs fine but garbage output" bug).

Things this file keeps in sync with training:
  1. Rebuilding the EXACT classifier head (Dropout + Linear) before
     load_state_dict — loading into a fresh efficientnet_b0() with the
     default 1000-class head will silently mismatch.
  2. Using the SAME normalization mean/std as training
     ([0.485,0.456,0.406] / [0.229,0.224,0.225]).
  3. Using the SAME image size (224x224) as training.
  4. Using the SAME class_to_idx mapping saved during training rather
     than an alphabetical guess — always load the saved JSON.
  5. Using the SAME MAX_LEN for the BERT tokenizer as training
     (256 for DistilBERT V1).
  6. Feeding the CNN the PREPROCESSED grayscale image — the training
     images in Datasets/split are the preprocessed V5 images.
  7. Calling model.eval() before inference.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from labels import normalize_label

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Models are looked up in ./models (filled by download_models.py), then in
# the original FYP layout <project root>/Models. Override with MODELS_DIR.
BASE_DIR   = Path(__file__).resolve().parent
_DEFAULT_MODELS = BASE_DIR / "models" if (BASE_DIR / "models").exists() else BASE_DIR.parent / "Models"
MODELS_DIR = Path(os.getenv("MODELS_DIR", _DEFAULT_MODELS))

# EfficientNet-B0 V1 (augmented) — the CNN used in the late-fusion experiment
CNN_MODEL_PATH   = MODELS_DIR / "CNN_aug" / "best_efficientnet.pth"
CNN_CLASSES_PATH = MODELS_DIR / "CNN_aug" / "class_names.json"

# DistilBERT V1 (augmented) — the primary classifier
BERT_MODEL_DIR   = MODELS_DIR / "BERT_aug" / "best_distilbert"
BERT_LABELS_PATH = MODELS_DIR / "BERT_aug" / "label_mapping.json"

# Must match training exactly
IMG_SIZE   = 224
MAX_LEN    = 256          # DistilBERT V1 was trained with 256 (V2 used 512)
DROPOUT    = 0.3          # EfficientNet-B0 V1 used 0.3 dropout in the classifier head
DEVICE     = "cpu"        # Demo runs on CPU laptop

# Late fusion weights from Experiment 5 (BERT-favoured, best result)
FUSION_W_CNN  = 0.3
FUSION_W_BERT = 0.7


# ─── IMAGE TRANSFORM (must match val_tf from training, NOT train_tf) ──────────
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _sorted_scores(scores: dict) -> dict:
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def _top(scores: dict) -> dict:
    label = next(iter(scores))
    return {"prediction": label, "confidence": scores[label], "scores": scores}


class ReceiptClassifier:
    """
    Loads both trained models once at startup and exposes a single
    predict() method that returns CNN, BERT, and fused predictions.
    All returned labels are display names (see labels.py).
    """

    def __init__(self):
        print("[Inference] Loading models, this may take a moment...")
        self._load_cnn()
        self._load_bert()
        print("[Inference] All models loaded and ready.")

    # ── CNN LOADING ─────────────────────────────────────────────────────────
    def _load_cnn(self):
        with open(CNN_CLASSES_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # idx_to_class keys are strings in JSON ("0", "1", ...) — convert to int
        self.cnn_idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
        self.num_classes = len(self.cnn_idx_to_class)

        # Rebuild EXACTLY the same architecture as training:
        # efficientnet_b0 backbone + Dropout(DROPOUT) + Linear(in_features, num_classes)
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )

        state_dict = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()   # CRITICAL: disables dropout/batchnorm training behaviour

        self.cnn_model = model
        print(f"[Inference] CNN loaded ({CNN_MODEL_PATH.name}). Classes: {list(self.cnn_idx_to_class.values())}")

    # ── BERT LOADING ────────────────────────────────────────────────────────
    def _load_bert(self):
        with open(BERT_LABELS_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.bert_id_to_label = {int(k): v for k, v in mapping["id2label"].items()}

        # Loading via from_pretrained on the SAVED MODEL DIRECTORY restores the
        # fine-tuned classifier head and tokenizer together.
        self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained(str(BERT_MODEL_DIR))
        self.bert_model = DistilBertForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        self.bert_model.to(DEVICE)
        self.bert_model.eval()
        print(f"[Inference] BERT loaded ({BERT_MODEL_DIR.name}). Classes: {list(self.bert_id_to_label.values())}")

    # ── CNN PREDICTION ──────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_cnn(self, image_path):
        # Grayscale preprocessed image -> 3 channels, exactly like ImageFolder's loader
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
        logits = self.cnn_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        return _sorted_scores({
            self.cnn_idx_to_class[i]: float(probs[i])
            for i in range(self.num_classes)
        })

    # ── BERT PREDICTION ─────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_bert(self, ocr_text):
        if not ocr_text or not ocr_text.strip():
            # Empty OCR text — return uniform confidence rather than feeding an
            # empty string through the tokenizer, which can produce misleadingly
            # confident garbage predictions.
            n = len(self.bert_id_to_label)
            return {label: 1.0 / n for label in self.bert_id_to_label.values()}

        encoding = self.bert_tokenizer(
            ocr_text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        logits = self.bert_model(**encoding).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

        return _sorted_scores({
            self.bert_id_to_label[i]: float(probs[i])
            for i in range(len(self.bert_id_to_label))
        })

    # ── LATE FUSION ──────────────────────────────────────────────────────────
    def fuse(self, cnn_scores, bert_scores):
        """
        Weighted average from Experiment 5 (BERT-favoured: 0.3 CNN / 0.7 BERT).
        Fuses by class NAME, not index, and refuses to fuse mismatched label sets.
        """
        cnn_keys = set(cnn_scores.keys())
        bert_keys = set(bert_scores.keys())

        if cnn_keys != bert_keys:
            raise ValueError(
                f"Label mismatch between CNN and BERT class sets! "
                f"In CNN but not BERT: {cnn_keys - bert_keys}. "
                f"In BERT but not CNN: {bert_keys - cnn_keys}."
            )

        return _sorted_scores({
            label: FUSION_W_CNN * cnn_scores[label] + FUSION_W_BERT * bert_scores[label]
            for label in cnn_keys
        })

    # ── FULL PIPELINE ───────────────────────────────────────────────────────
    def predict(self, image_path, ocr_text):
        cnn_scores = self.predict_cnn(image_path)
        bert_scores = self.predict_bert(ocr_text)
        fused_scores = self.fuse(cnn_scores, bert_scores)

        def display(scores):
            return {normalize_label(k): v for k, v in scores.items()}

        return {
            "cnn": _top(display(cnn_scores)),
            "bert": _top(display(bert_scores)),
            "fused": _top(display(fused_scores)),
        }


# ─── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from ocr_processor import OCRProcessor
    from preprocessing import preprocess_and_save

    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_receipt_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    processed_path = str(Path(image_path).with_name("processed_" + Path(image_path).name))
    preprocess_and_save(image_path, processed_path)

    ocr = OCRProcessor(device="cpu")
    ocr_text = ocr.extract_text_from_image(processed_path)
    print(f"\nOCR extracted text ({len(ocr_text)} chars):")
    print(ocr_text[:300])

    classifier = ReceiptClassifier()
    result = classifier.predict(processed_path, ocr_text)

    for name in ("cnn", "bert", "fused"):
        print(f"\n--- {name.upper()} ---")
        print(f"{result[name]['prediction']} ({result[name]['confidence']:.4f})")
