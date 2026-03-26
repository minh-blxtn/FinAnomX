# src/utils/text_utils.py
from __future__ import annotations
import re

def normalize_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q

def expand_abbreviations(q: str) -> str:
    # placeholder: implement later for chatbot (e.g., "tsla" -> "TSLA", "ig" -> "integrated gradients")
    return q
