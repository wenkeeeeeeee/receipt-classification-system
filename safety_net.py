"""
safety_net.py
--------------
The LLM "safety net" that sits on top of DistilBERT (the primary model).

Two jobs:
  1. Low-confidence override — when DistilBERT's confidence is below
     CONFIDENCE_THRESHOLD, or the LLM flags an out-of-distribution receipt
     ("Others"), the LLM's independent category replaces DistilBERT's.
  2. Input validation — when the LLM says the text is not a receipt (empty
     category), or OCR found no text at all, the upload is rejected instead
     of being given a meaningless category.

Model scores are never rewritten. The decision is returned separately as
"final", with a human-readable trail explaining how it was reached.
If the LLM is unavailable, the system falls back to DistilBERT alone.
"""

from labels import BLANK_LABEL, VALID_CATEGORIES

CONFIDENCE_THRESHOLD = 0.95
OOD_CATEGORIES = {"Others", BLANK_LABEL}


def decide(result: dict, llm: dict, ocr_text: str) -> dict:
    """
    result : output of ReceiptClassifier.predict() (display labels)
    llm    : the "llm" block from field_extraction.run_field_extraction()
    Returns the "final" verdict dict.
    """
    bert_pred = result["bert"]["prediction"]
    bert_conf = result["bert"]["confidence"]
    llm_status = llm.get("status")
    llm_cat = llm.get("category", "")
    model_name = (llm.get("model") or "LLM").split("/")[-1]

    trail = [f"DistilBERT predicted {bert_pred} ({bert_conf:.1%})"]

    def verdict(prediction, source, confidence, rejected=False):
        return {
            "prediction": prediction,
            "confidence": confidence,
            "source": source,            # distilbert | llm_override | llm_confirmed | rejected | fallback
            "rejected": rejected,
            "llm_category": llm_cat,
            "llm_status": llm_status,
            "threshold": CONFIDENCE_THRESHOLD,
            "trail": trail,
        }

    # --- BLANK INPUT: nothing to classify, no need for the LLM to tell us ---
    if not ocr_text.strip():
        trail[0] = "OCR found no readable text"
        trail.append("Rejected: image does not look like a receipt")
        print("[Safety Net] REJECT: OCR returned no text.")
        return verdict(BLANK_LABEL, "rejected", None, rejected=True)

    # --- LLM UNAVAILABLE: fall back to the local model instead of rejecting ---
    if llm_status != "ok":
        trail.append(f"{model_name} unavailable ({llm.get('message') or llm_status}) — using DistilBERT only")
        print(f"[Safety Net] LLM unavailable ({llm_status}). Falling back to DistilBERT.")
        return verdict(bert_pred, "fallback", bert_conf)

    print(f"\n[Safety Net] DistilBERT predicted: '{bert_pred}' (Confidence: {bert_conf:.3f})")
    print(f"[Safety Net] {model_name} predicted: '{llm_cat or 'BLANK / NO RECEIPT DETECTED'}'")

    # --- BLANK RECEIPT HANDLER: the LLM says this is not a receipt ---
    if not llm_cat or llm_cat == BLANK_LABEL:
        trail.append(f"{model_name} judged the text is not a receipt")
        trail.append("Rejected: input validation failed")
        print("[Safety Net] OVERRIDE EXECUTED: Forcing category to 'Blank / Not a Receipt'")
        return verdict(BLANK_LABEL, "rejected", None, rejected=True)

    if llm_cat not in VALID_CATEGORIES:
        trail.append(f"{model_name} returned an unknown category '{llm_cat}' — ignored")
        return verdict(bert_pred, "distilbert", bert_conf)

    # Trigger 1: DistilBERT is unsure
    bert_unsure = bert_conf < CONFIDENCE_THRESHOLD
    # Trigger 2: The LLM caught an Out-Of-Distribution (OOD) receipt
    is_ood_catch = bert_pred != llm_cat and llm_cat in OOD_CATEGORIES

    if bert_unsure:
        trail.append(f"Below the {CONFIDENCE_THRESHOLD:.0%} threshold → asked {model_name}")
    else:
        trail.append(f"Above the {CONFIDENCE_THRESHOLD:.0%} threshold")

    # We ONLY override if DistilBERT is unsure OR if the LLM caught an OOD receipt.
    # If DistilBERT is highly confident about a standard category, we trust DistilBERT.
    if bert_unsure or is_ood_catch:
        if llm_cat == bert_pred:
            trail.append(f"{model_name} agrees: {llm_cat}")
            print("[Safety Net] PASS: Models agree. Local model trusted.")
            return verdict(llm_cat, "llm_confirmed", bert_conf)

        if is_ood_catch:
            trail.append(f"{model_name} flagged an out-of-distribution receipt → {llm_cat}")
            print(f"[Safety Net] OVERRIDE EXECUTED: OOD receipt caught. Forcing category to '{llm_cat}'")
        else:
            trail.append(f"{model_name} disagrees → overridden to {llm_cat}")
            print(f"[Safety Net] OVERRIDE EXECUTED: Local model was unsure. Using '{llm_cat}'")
        return verdict(llm_cat, "llm_override", None)

    if llm_cat != bert_pred:
        trail.append(f"{model_name} suggested {llm_cat}, but DistilBERT is confident — kept {bert_pred}")
        print("[Safety Net] PASS: Models disagreed, but DistilBERT is highly confident. Trusting LOCAL model.")
    else:
        trail.append(f"{model_name} agrees: {llm_cat}")
        print("[Safety Net] PASS: Models agree. Local model trusted.")
    return verdict(bert_pred, "distilbert", bert_conf)
