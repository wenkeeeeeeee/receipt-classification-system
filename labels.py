"""
labels.py
----------
Single source of truth for category names.

The trained models use the dataset folder names ("Food and beverages",
"Other", ...) while the LLM prompt uses display names ("Food & Beverage",
"Others", ...). Everything shown to the user or compared in the safety
net goes through normalize_label() so the two vocabularies never drift.
"""

BLANK_LABEL = "Blank / Not a Receipt"

# Display names, in the order the UI lists them.
CATEGORIES = [
    "Food & Beverage",
    "Transportation",
    "Household & Apparel",
    "Home & Hardware",
    "Electronics & Electrical",
    "Recreation & Lifestyle",
    "Others",
]

VALID_CATEGORIES = set(CATEGORIES) | {BLANK_LABEL}

# --- TRANSLATION DICTIONARY ---
# Catching common dataset naming variations (plurals, 'and' vs '&', casing).
# Keys are compared case-insensitively after collapsing '_' to spaces.
_LABEL_MAP = {
    "food and beverages": "Food & Beverage",
    "food and beverage": "Food & Beverage",
    "food & beverages": "Food & Beverage",
    "food & beverage": "Food & Beverage",
    "household and apparel": "Household & Apparel",
    "household & apparel": "Household & Apparel",
    "home and hardware": "Home & Hardware",
    "home & hardware": "Home & Hardware",
    "electronics and electrical": "Electronics & Electrical",
    "electronics & electrical": "Electronics & Electrical",
    "recreation and lifestyle": "Recreation & Lifestyle",
    "recreation & lifestyle": "Recreation & Lifestyle",
    "transportation": "Transportation",
    "transport": "Transportation",
    "other": "Others",
    "others": "Others",
    "blank / not a receipt": BLANK_LABEL,
    "blank": BLANK_LABEL,
}


def normalize_label(label: str) -> str:
    """Maps any known spelling of a category to its display name."""
    if not label:
        return ""
    key = " ".join(str(label).replace("_", " ").split()).lower()
    return _LABEL_MAP.get(key, str(label).strip())
