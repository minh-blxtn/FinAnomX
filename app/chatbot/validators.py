# =========================================================
# app/chatbot/validators.py
# Output validation and user-facing cleanup for FinAnomX chatbot
# =========================================================

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional
import re

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chatbot.rag_engine import RetrievalResult  # noqa: E402


def _extract_counterfactual_strength(result: RetrievalResult) -> Optional[float]:
    ctx = result.context or {}
    cf_rows = ctx.get("counterfactual", [])
    if isinstance(cf_rows, list) and cf_rows:
        row = cf_rows[0]
    elif isinstance(cf_rows, dict):
        row = cf_rows
    else:
        row = {}
    for c in ["cf_strength", "cf_drop_proxy", "cf_top1_contribution_proxy", "top1_contribution_drop_proxy"]:
        if c in row and row[c] is not None:
            try:
                return float(row[c])
            except Exception:
                pass
    return None


def _extract_top_news_weight(result: RetrievalResult) -> Optional[float]:
    ctx = result.context or {}
    news_rows = ctx.get("top_news", [])
    if isinstance(news_rows, list) and news_rows:
        row = news_rows[0]
        for c in ["contribution", "weight", "confidence_score"]:
            if row.get(c) is not None:
                try:
                    return float(row.get(c))
                except Exception:
                    pass
    event_evidence = ctx.get("event_evidence", {})
    if isinstance(event_evidence, dict):
        for c in ["top_event_confidence_score", "top_event_retrieval_score"]:
            if event_evidence.get(c) is not None:
                try:
                    return float(event_evidence.get(c))
                except Exception:
                    pass
    return None


def _has_evidence_for_strong_cause_claim(result: RetrievalResult) -> bool:
    news_weight = _extract_top_news_weight(result)
    cf_drop = _extract_counterfactual_strength(result)
    if isinstance(result.context, dict):
        cf_rows = result.context.get("counterfactual", [])
        if isinstance(cf_rows, list) and cf_rows:
            support_level = str(cf_rows[0].get("cf_support_level", "")).lower()
            if support_level == "strong":
                return True
        if isinstance(cf_rows, dict):
            support_level = str(cf_rows.get("cf_support_level", "")).lower()
            news_conf = str((result.context.get("event_evidence", {}) or {}).get("news_confidence_level", "")).lower()
            if support_level == "strong" and news_conf in {"high", "medium"}:
                return True
    if news_weight is None or cf_drop is None:
        return False
    return (news_weight >= 0.30) and (cf_drop >= 0.10)


def _downgrade_overstrong_phrasing(text: str, allow_strong: bool) -> str:
    if allow_strong:
        return text
    replacements = {
        r"\bconfirmed cause\b": "plausible explanation",
        r"\broot cause\b": "strong candidate explanation",
        r"\bdefinitive cause\b": "plausible explanation",
        r"\bproves that\b": "suggests that",
        r"\bproves\b": "suggests",
        r"\bthis confirms\b": "this supports",
        r"\bwas caused by\b": "was plausibly associated with",
        r"\bis caused by\b": "is plausibly associated with",
        r"\bthe cause is\b": "the strongest candidate explanation is",
    }
    out = text
    for pat, repl in replacements.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def _strip_internal_sections(text: str) -> str:
    out = text or ""
    patterns = [
        r"(?is)\n*how this query was interpreted.*$",
        r"(?is)\n*sources used.*$",
        r"(?is)\n*nguồn:.*$",
        r"(?is)\n*sources:.*$",
    ]
    for pat in patterns:
        out = re.sub(pat, "", out)
    out = re.sub(r"(?im)^\s*[-*]?\s*data/results/.*$", "", out)
    out = re.sub(r"(?im)^\s*[-*]?\s*data\\results\\.*$", "", out)
    out = re.sub(r"(?im)^\s*[-*]?\s*ticker=.*$", "", out)
    return out


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()




def _extract_event_title_and_confidence(result: RetrievalResult) -> tuple[str, str]:
    ctx = result.context or {}
    event_evidence = ctx.get("event_evidence", {}) if isinstance(ctx, dict) else {}
    if isinstance(event_evidence, dict):
        title = str(event_evidence.get("top_event_title") or "").strip()
        conf = str(event_evidence.get("news_confidence_level") or "").strip().lower()
        return title, conf
    return "", ""


def _looks_like_event_denial(text: str) -> bool:
    t = (text or "").lower()
    phrases = [
        "no specific external event",
        "cannot identify a reliable specific related event",
        "do not have enough grounded evidence",
        "not enough evidence to identify",
        "no reliable specific news trigger",
        "no event was confirmed",
    ]
    return any(p in t for p in phrases)

def build_unanswerable_response(result: RetrievalResult) -> str:
    msg = result.context.get("message") if isinstance(result.context, dict) else None
    msg = msg or "The current artifacts do not contain enough evidence to answer this question reliably."
    return _normalize_whitespace(
        f"I do not have enough grounded evidence in the current artifacts to answer that reliably.\n\n{msg}"
    )


def build_minimal_grounded_response(result: RetrievalResult) -> str:
    return "I could not generate a stable grounded answer from the local model, so I am falling back to a safe artifact-based response."


def validate_and_finalize_answer(result: RetrievalResult, model_text: Optional[str]) -> str:
    if not result.answerable:
        return build_unanswerable_response(result)
    text = (model_text or "").strip()
    if not text:
        return build_minimal_grounded_response(result)

    title, conf = _extract_event_title_and_confidence(result)
    if title and conf in {"high", "medium"} and _looks_like_event_denial(text):
        # fall back to the grounded deterministic answer when the model contradicts a supported event in artifacts
        from app.chatbot.responders import build_grounded_response  # local import to avoid cycles at import time
        text = build_grounded_response(result)

    allow_strong = _has_evidence_for_strong_cause_claim(result)
    text = _downgrade_overstrong_phrasing(text, allow_strong=allow_strong)
    text = _strip_internal_sections(text)
    text = _normalize_whitespace(text)
    return text


