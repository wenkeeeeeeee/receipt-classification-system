"""
field_extraction.py (demo version)
-------------------------------------
Extracts key receipt fields: store name, date, time, total amount, tax.

Strategy (Verification Pipeline):
  1. Regex does a fast "first pass" to catch structured fields (Date, Time, totals).
  2. The LLM acts as an autocorrecting validator. It isolates the header to find
     the store name, ignores address lines, and fixes OCR typos caused by thresholding.
     It also returns an independent category used by the safety net (app.py).

The LLM outcome is reported explicitly (status "ok" / "error" / "disabled") so
that an API failure is never confused with the LLM saying "this is not a receipt".
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

from labels import normalize_label

# Prefer a .env next to the demo, then fall back to the project root .env
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")
load_dotenv(_BASE_DIR.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "30"))
LLM_MAX_TOKENS = 4096   # reasoning + answer budget (see llm_validator)
LLM_ATTEMPTS = 2

FIELD_KEYS = ["store_name", "transaction_date", "transaction_time",
              "tax_amount", "total_amount", "category"]

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT, max_retries=1)
    return _client


# ─── REGEX-BASED EXTRACTION (First Pass) ────────────────────────────────────────

# Amounts like 12.50, 12,50 or 1,234.56 (the last one used to be cut to "234.56")
_AMOUNT = r'(\d{1,3}(?:,\d{3})+\.\d{2}|\d+[.,]\d{2})'


def regex_extract(ocr_text: str) -> dict:
    result = {k: "" for k in FIELD_KEYS}  # store_name always empty -> LLM handles it

    if not ocr_text or not ocr_text.strip():
        return result

    # ── Date: supports DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD
    date_patterns = [
        r'\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})\b',
        r'\b(\d{4}-\d{1,2}-\d{1,2})\b',
    ]
    for pat in date_patterns:
        m = re.search(pat, ocr_text)
        if m:
            result["transaction_date"] = m.group(1)
            break

    # ── Time: HH:MM with optional AM/PM
    time_match = re.search(r'\b(\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AaPp][Mm])?)\b', ocr_text)
    if time_match:
        result["transaction_time"] = time_match.group(1).strip()

    # ── Total amount: Multiline support
    total_patterns = [
        r'(?:Grand\s*Total|Total\s*Amount|Amount\s*Due)[\s\S]{0,30}?' + _AMOUNT,
        r'(?<!Sub)(?<!Sub\s)\bTotal\b[\s\S]{0,30}?' + _AMOUNT,
    ]
    for pat in total_patterns:
        m = re.search(pat, ocr_text, re.IGNORECASE)
        if m:
            result["total_amount"] = m.group(1)
            break

    # ── Tax amount: Multiline support.
    # The negative lookahead skips headers such as "Tax Invoice No." and
    # "GST ID : 0011..." which previously grabbed an unrelated number.
    tax_patterns = [
        r'(?:GST|SST|VAT|Service\s*Charge|Tax)'
        r'(?!\s*(?:Invoice|ID\b|Reg|No\b|Number|Summary))'
        r'\s*(?:@?\s*\d{1,2}\s*%)?[\s\S]{0,30}?' + _AMOUNT,
    ]
    for pat in tax_patterns:
        m = re.search(pat, ocr_text, re.IGNORECASE)
        if m:
            result["tax_amount"] = m.group(1)
            break

    return result


# ─── LLM-BASED EXTRACTION (Autocorrector & Validator) ───────────────────────────

def _build_prompt(ocr_lines: list, initial_regex_data: dict) -> str:
    # Grab only the first 12 lines to prevent the LLM from getting confused by footers
    header_text = "\n".join(ocr_lines[:12])
    full_text = "\n".join(ocr_lines)

    return f"""You are an expert OCR autocorrection and data extraction assistant.
The provided OCR text comes from a receipt, but the text contains heavy typos, merged letters, and corrupted characters due to image thresholding.

Your task:
1. Extract the 'store_name'.
   - RULE: Look in the first 1-8 lines of the Header Text.
   - RULE: If you find BOTH a highly recognizable consumer brand (e.g., "MarryBrown") AND a formal legal corporate entity (e.g., ending in Sdn Bhd, LLC, Ltd), combine them using a slash. Format: "Brand Name / Legal Name" (e.g., "MarryBrown / Ali Restaurants Sdn Bhd").
   - RULE: If only one exists, just output that single name. Do not assume the legal entity is the brand name. If you cannot find a recognizable brand, output an empty string.
   - RULE: AGGRESSIVE AUTOCORRECT: You MUST fix OCR typos using your knowledge of real-world brands. The OCR often swaps letters (e.g., 'hadonalds' -> 'McDonalds', 'SODATS' -> 'SPORTS DIRECT'). Do not output the corrupted spelling.
   - RULE: DO NOT extract address lines (ignore keywords like Jalan, Persiaran, Bandar, KL, Level, Lvl).
   - RULE: DO NOT extract queue or order numbers.
2. Verify the tax_amount and total_amount. If the regex missed them due to line breaks or typos, extract them.
3. Keep the date and time if they are correct.
4. Classify the receipt into exactly ONE of these predefined categories based on the store and line items (if a receipt has mixed items, prioritize the category with the highest quantity of items):
   - "Food & Beverage" (Restaurants, cafes, fast food, coffee shops, bakeries)
   - "Transportation" (Petrol/gas stations, parking tickets, tolls, automotive services)
   - "Household & Apparel" (Supermarkets, groceries, convenience stores, clothing, department stores)
   - "Home & Hardware" (Hardware stores, furniture, home improvement, renovation)
   - "Electronics & Electrical" (Gadgets, appliances, computer stores, phone shops, lighting equipments)
   - "Recreation & Lifestyle" (Gyms, entertainment, hobbies, sporting goods)
   - "Others" (Stationery, medical/clinics, banking, unknown, or miscellaneous receipts)

   *CRITICAL RULE: Stationery, medical, and banking receipts MUST strictly be categorized as "Others". If the receipt details do not clearly fit into any of the first six categories, you MUST classify it as "Others".*
   *CRITICAL RULE 2: If the OCR text is completely blank, or clearly DOES NOT belong to a receipt (e.g., a random document, or pure garbage text), you MUST return an empty string "" for the category.*

5. Return ONLY a valid JSON object with exactly these 6 keys: store_name, transaction_date, transaction_time, tax_amount, total_amount, category. Do not include markdown code block formatting.

Header Text (For Context):
{header_text}

Full OCR Text (one recognised line per row):
{full_text}

Initial Regex Extraction:
{json.dumps(initial_regex_data, indent=2)}

Corrected JSON:"""


def _parse_llm_json(text: str) -> dict:
    text = (text or "").strip()
    # Clean markdown formatting if the LLM included it
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a JSON object")
    # Keep only the expected keys, coerce everything to clean strings
    return {k: ("" if data.get(k) is None else str(data.get(k)).strip()) for k in FIELD_KEYS}


def llm_validator(ocr_lines: list, initial_regex_data: dict) -> dict:
    """
    Returns {"status": "ok"|"error"|"disabled", "data": {...}, "message": str}.
    "data" is only meaningful when status == "ok".
    """
    if not GROQ_API_KEY:
        print("[Field Extraction] Warning: GROQ_API_KEY is missing.")
        return {"status": "disabled", "data": {}, "message": "GROQ_API_KEY is not set"}

    prompt = _build_prompt(ocr_lines, initial_regex_data)
    last_error = None

    # gpt-oss is a reasoning model: max_tokens covers the hidden reasoning AND
    # the JSON answer. With the old budget of 800, long receipts used it all up
    # on reasoning, returned nothing, and Groq raised json_validate_failed.
    for attempt in range(LLM_ATTEMPTS):
        try:
            response = _get_client().chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                temperature=0.1,
                max_tokens=LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
                reasoning_format="hidden"
            )
            data = _parse_llm_json(response.choices[0].message.content)
            return {"status": "ok", "data": data, "message": ""}

        except Exception as e:
            last_error = e
            print(f"[Field Extraction] Groq LLM Validator failed (attempt {attempt + 1}/{LLM_ATTEMPTS}): {e}")

    return {"status": "error", "data": {}, "message": f"{type(last_error).__name__}: {last_error}"[:300]}


# ─── HYBRID ENTRY POINT ─────────────────────────────────────────────────────────

def run_field_extraction(ocr_lines: list) -> dict:
    """
    Full extraction with provenance.
    Returns {
        "fields":  {store_name, transaction_date, transaction_time, tax_amount, total_amount},
        "sources": {field: "regex" | "llm" | ""},
        "llm":     {"status", "model", "message", "category" (display name or "")},
    }
    """
    ocr_text = " ".join(ocr_lines)
    regex_result = regex_extract(ocr_text)
    fields = {k: regex_result[k] for k in FIELD_KEYS if k != "category"}
    sources = {k: ("regex" if v else "") for k, v in fields.items()}

    llm = {"status": "disabled", "model": GROQ_MODEL, "message": "", "category": ""}

    if ocr_text.strip():
        verdict = llm_validator(ocr_lines, regex_result)
        llm.update(status=verdict["status"], message=verdict["message"])

        if verdict["status"] == "ok":
            data = verdict["data"]
            llm["category"] = normalize_label(data.get("category", ""))
            for k in fields:
                v = data.get(k, "")
                if v:
                    if v != fields[k]:
                        sources[k] = "llm"
                    fields[k] = v
    else:
        llm["message"] = "No OCR text — LLM not called"

    _sanity_check(fields, sources)
    return {"fields": fields, "sources": sources, "llm": llm}


def _to_amount(value: str):
    m = re.search(_AMOUNT, value or "")
    if not m:
        return None
    s = m.group(1)
    s = s.replace(",", "") if re.search(r",\d{3}", s) else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _sanity_check(fields: dict, sources: dict):
    """A tax amount can never be >= the total — that means a pattern grabbed the total twice."""
    tax, total = _to_amount(fields["tax_amount"]), _to_amount(fields["total_amount"])
    if tax is not None and total is not None and total > 0 and tax >= total:
        fields["tax_amount"] = ""
        sources["tax_amount"] = ""


def extract_fields(ocr_text: str) -> dict:
    """Backwards-compatible helper: flat dict of fields + LLM category."""
    out = run_field_extraction(ocr_text.splitlines() if "\n" in ocr_text else [ocr_text])
    return {**out["fields"], "category": out["llm"]["category"]}


if __name__ == "__main__":
    sample = [
        "STARBUCKS COFFEE",
        "123 Jalan Bukit Bintang",
        "Date: 15/03/2026  Time: 14:32",
        "Caffe Latte        RM 12.50",
        "Croissant          RM 8.90",
        "Subtotal           RM 21.40",
        "SST 6%             RM 1.28",
        "Total              RM 22.68",
    ]
    print(json.dumps(run_field_extraction(sample), indent=2))
