"""
app.py
--------
Flask backend for the receipt classification demo.

Pipeline per upload:
  1. Save uploaded image
  2. Preprocess (preprocessing.py — adaptive scan/photo logic from ground.py)
  3. OCR (ocr_processor.py — PaddleOCR 3.x)
  4. Field extraction (field_extraction.py — regex first, LLM validator)
  5. CNN + BERT + Fusion classification (inference.py)
  6. LLM safety net (safety_net.py)

Uploads are processed by ONE background worker thread, one job at a time.
PaddleOCR and the torch models are not thread-safe, so when several people
use the public tunnel at once their scans queue up instead of crashing the
server. The browser polls /api/jobs/<id> to show live per-stage progress.

Models are loaded ONCE at startup (not per-request).

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import time
import uuid
import queue
import threading
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

from ocr_processor import OCRProcessor
from preprocessing import preprocess_receipt_with_info, save_image
from field_extraction import run_field_extraction, GROQ_API_KEY, GROQ_MODEL
from inference import ReceiptClassifier, FUSION_W_CNN, FUSION_W_BERT, MAX_LEN
from safety_net import decide, CONFIDENCE_THRESHOLD
from labels import CATEGORIES, BLANK_LABEL

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLES_DIR = BASE_DIR / "static" / "samples"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_FILE_SIZE_MB = 15
MAX_STORED_UPLOADS = 60        # oldest files beyond this are deleted (raw + processed)
MAX_QUEUE_LENGTH = 8           # reject new scans when this many are already waiting
JOB_TTL_SECONDS = 15 * 60      # finished jobs are forgotten after this

STAGES = ["preprocessing", "ocr", "field_extraction", "classification", "safety_net"]

# Validation-set results (440 images) from the training experiments
MODEL_CARD = {
    "dataset": {"images": 2197, "categories": 7, "split": "Stratified 80/20", "val_images": 440},
    "models": [
        {"key": "bert", "name": "DistilBERT V1", "input": "OCR text", "role": "Primary classifier",
         "accuracy": 0.9045, "macro_f1": 0.8653, "weighted_f1": 0.9042},
        {"key": "fused", "name": f"Late Fusion ({FUSION_W_CNN} / {FUSION_W_BERT})", "input": "Image + text",
         "role": "Comparison", "accuracy": 0.9114, "macro_f1": 0.8812, "weighted_f1": 0.9109},
        {"key": "cnn", "name": "EfficientNet-B0 V1", "input": "Preprocessed image", "role": "Comparison",
         "accuracy": 0.8614, "macro_f1": 0.8028, "weighted_f1": 0.8629},
    ],
    "bert_per_class_f1": {
        "Food & Beverage": 0.9830, "Transportation": 1.0000, "Electronics & Electrical": 0.9333,
        "Others": 0.8481, "Recreation & Lifestyle": 0.8182, "Household & Apparel": 0.7451,
        "Home & Hardware": 0.7297,
    },
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

# ─── LOAD MODELS ONCE AT STARTUP ───────────────────────────────────────────────
print("=" * 60)
print("  Starting Receipt Classification Demo")
print("=" * 60)

print("\n[Startup] Loading OCR engine...")
ocr_engine = OCRProcessor(device="cpu")

print("\n[Startup] Loading CNN + BERT models...")
classifier = ReceiptClassifier()

if not GROQ_API_KEY:
    print("\n[Startup] WARNING: GROQ_API_KEY not set — store name and the LLM safety net are disabled.")

print("\n[Startup] All systems ready. Starting Flask server...\n")


# ─── JOB QUEUE ─────────────────────────────────────────────────────────────────
jobs = {}                    # job_id -> job dict
jobs_lock = threading.Lock()
job_queue = queue.Queue()


def _new_job(raw_name: str) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "status": "queued",          # queued | running | done | error
        "stage": None,
        "timings": {},
        "result": None,
        "error": None,
        "raw_name": raw_name,
        "created": time.time(),
        "finished": None,
    }
    with jobs_lock:
        jobs[job["id"]] = job
    return job


def _queue_position(job_id: str) -> int:
    with jobs_lock:
        waiting = sorted((j for j in jobs.values() if j["status"] == "queued"), key=lambda j: j["created"])
    for i, j in enumerate(waiting):
        if j["id"] == job_id:
            return i + 1
    return 0


def _prune_jobs():
    now = time.time()
    with jobs_lock:
        for jid in [jid for jid, j in jobs.items() if j["finished"] and now - j["finished"] > JOB_TTL_SECONDS]:
            del jobs[jid]


def _cleanup_uploads():
    """Keeps only the newest MAX_STORED_UPLOADS files so the folder can't grow forever."""
    with jobs_lock:
        in_use = set()
        for j in jobs.values():
            if j["status"] in ("queued", "running"):
                in_use |= {j["raw_name"], f"processed_{Path(j['raw_name']).stem}.jpg"}
    files = sorted((p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.name not in in_use),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[MAX_STORED_UPLOADS:]:
        try:
            p.unlink()
        except OSError:
            pass


def run_pipeline(job: dict) -> dict:
    timings = job["timings"]
    t_start = time.time()
    safe_name = job["raw_name"]
    raw_path = UPLOAD_DIR / safe_name

    def stage(name):
        job["stage"] = name
        return time.time()

    # ── Step 1: Preprocess ──────────────────────────────────────────────────
    t0 = stage("preprocessing")
    # Saved as JPEG like the training images in Structured_Dataset_V5
    processed_filename = f"processed_{Path(safe_name).stem}.jpg"
    processed_path = UPLOAD_DIR / processed_filename
    processed_img, prep_info = preprocess_receipt_with_info(str(raw_path))
    save_image(processed_img, str(processed_path))
    timings["preprocessing"] = round(time.time() - t0, 2)

    # ── Step 2: OCR ─────────────────────────────────────────────────────────
    t0 = stage("ocr")
    ocr_lines = ocr_engine.extract_lines_from_image(str(processed_path))
    ocr_text = " ".join(ocr_lines)   # same space-joined format DistilBERT was trained on
    timings["ocr"] = round(time.time() - t0, 2)

    # ── Step 3: Field extraction (regex + LLM validator) ────────────────────
    t0 = stage("field_extraction")
    extraction = run_field_extraction(ocr_lines)
    timings["field_extraction"] = round(time.time() - t0, 2)

    # ── Step 4: Classification (CNN + BERT + Fusion) ────────────────────────
    # The CNN gets the PREPROCESSED image: the training set (Datasets/split)
    # consists of the preprocessed grayscale V5 images, not raw colour photos.
    t0 = stage("classification")
    result = classifier.predict(str(processed_path), ocr_text)
    timings["classification"] = round(time.time() - t0, 2)

    # ── Step 5: The Safety Net ──────────────────────────────────────────────
    t0 = stage("safety_net")
    final = decide(result, extraction["llm"], ocr_text)
    timings["safety_net"] = round(time.time() - t0, 3)

    timings["total"] = round(time.time() - t_start, 2)

    preds = {k: result[k]["prediction"] for k in ("bert", "fused", "cnn")}
    return {
        "image_url": f"/uploads/{safe_name}",
        "processed_image_url": f"/uploads/{processed_filename}",
        "preprocess": prep_info,
        "ocr_text": ocr_text,
        "ocr_lines": ocr_lines,
        "fields": extraction["fields"],
        "field_sources": extraction["sources"],
        "llm": extraction["llm"],
        "cnn": result["cnn"],
        "bert": result["bert"],
        "fused": result["fused"],
        "final": final,
        "agreement": {
            "predictions": preds,
            "all_agree": len(set(preds.values())) == 1,
        },
        "timings": timings,
    }


def worker():
    while True:
        job = job_queue.get()
        job["status"] = "running"
        try:
            job["result"] = run_pipeline(job)
            job["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            job["error"] = str(e) or type(e).__name__
            job["status"] = "error"
        finally:
            job["finished"] = time.time()
            job_queue.task_done()
            _prune_jobs()
            _cleanup_uploads()


threading.Thread(target=worker, daemon=True, name="pipeline-worker").start()


# ─── ROUTES ────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def info():
    samples = sorted(p.name for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS) \
        if SAMPLES_DIR.exists() else []
    return jsonify({
        "categories": CATEGORIES,
        "blank_label": BLANK_LABEL,
        "stages": STAGES,
        "llm": {"model": GROQ_MODEL, "enabled": bool(GROQ_API_KEY)},
        "threshold": CONFIDENCE_THRESHOLD,
        "fusion": {"cnn": FUSION_W_CNN, "bert": FUSION_W_BERT},
        "bert_max_len": MAX_LEN,
        "max_file_mb": MAX_FILE_SIZE_MB,
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "model_card": MODEL_CARD,
        "samples": [{"name": n, "url": f"/static/samples/{n}"} for n in samples],
    })


@app.route("/api/scan", methods=["POST"])
def scan():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False,
                        "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    if job_queue.qsize() >= MAX_QUEUE_LENGTH:
        return jsonify({"success": False, "error": "The server is busy right now — please try again in a minute."}), 503

    # Timestamped ASCII-safe name: avoids collisions when the same file is
    # uploaded repeatedly, and avoids Windows path issues with emoji/unicode.
    suffix = Path(file.filename).suffix.lower()
    safe_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}{suffix}"
    raw_path = UPLOAD_DIR / safe_name
    file.save(raw_path)

    # Reject files that only *look* like images
    try:
        with Image.open(raw_path) as im:
            im.verify()
    except Exception:
        raw_path.unlink(missing_ok=True)
        return jsonify({"success": False, "error": "That file could not be read as an image."}), 400

    job = _new_job(safe_name)
    job_queue.put(job)
    return jsonify({"success": True, "job_id": job["id"], "position": _queue_position(job["id"])}), 202


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found or expired"}), 404

    payload = {
        "success": job["status"] != "error",
        "status": job["status"],
        "stage": job["stage"],
        "timings": job["timings"],
        "position": _queue_position(job_id) if job["status"] == "queued" else 0,
    }
    if job["status"] == "done":
        payload["result"] = job["result"]
    if job["status"] == "error":
        payload["error"] = job["error"]
    return jsonify(payload)


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "queued": job_queue.qsize()})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"success": False, "error": f"File is too large (max {MAX_FILE_SIZE_MB} MB)."}), 413


if __name__ == "__main__":
    import socket

    _cleanup_uploads()

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("=" * 60)
    print(f"  Demo running locally at:    http://127.0.0.1:5000")
    print(f"  Others on your WiFi/LAN at: http://{local_ip}:5000")
    print("  For a public URL, run start_tunnel.bat in another terminal")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
