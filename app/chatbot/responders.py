# =========================================================
# app/chatbot/responders.py
# Deterministic grounded responders for FinAnomX chatbot
# =========================================================

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional
import math

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chatbot.rag_engine import RetrievalResult  # noqa: E402


def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return x is None


def _fmt_num(x: Any, digits: int = 2) -> str:
    try:
        if _is_nan(x):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _metric_line(label: str, value: Any, suffix: str = "") -> Optional[str]:
    if value is None or _is_nan(value):
        return None
    return f"- {label}: {_fmt_num(value)}{suffix}"


def _extract_top_features(model_evidence: Dict[str, Any], top_k: int = 4) -> List[str]:
    rows = model_evidence.get("top_features", []) if isinstance(model_evidence, dict) else []
    out: List[str] = []
    for row in rows[:top_k]:
        feat = row.get("feature_pretty") or row.get("feature") or "N/A"
        group = row.get("feature_group") or row.get("group")
        bucket = row.get("z_bucket")
        bits = [feat]
        if group:
            bits.append(str(group))
        if bucket:
            bits.append(str(bucket))
        out.append(" | ".join(bits))
    return out


def _extract_supporting_articles(event_evidence: Dict[str, Any], top_k: int = 2) -> List[str]:
    rows = (event_evidence.get("supporting_articles") or []) if isinstance(event_evidence, dict) else []
    out: List[str] = []
    for row in rows[:top_k]:
        title = str(row.get("title") or "").strip()
        domain = str(row.get("domain") or "").strip()
        if title:
            out.append(f"{title} ({domain})" if domain else title)
    return out


def _cf_row(ctx: Dict[str, Any]) -> Dict[str, Any]:
    cf = ctx.get("counterfactual", {})
    if isinstance(cf, list):
        return cf[0] if cf else {}
    return cf if isinstance(cf, dict) else {}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text




def _query_mentions_why_event(query: str) -> bool:
    q = (query or '').lower()
    patterns = [
        'why does that event', 'why does this event', 'why does that explain',
        'why does this explain', 'plausibly explain', 'why is that event',
        'why is this event', 'how does that event', 'how does this event',
    ]
    return any(p in q for p in patterns)


def _event_confidence_phrase(conf: str) -> str:
    c = (conf or '').strip().lower()
    if c == 'high':
        return 'The current artifact set treats this as a high-confidence related event.'
    if c == 'medium':
        return 'The current artifact set treats this as a medium-confidence related event.'
    if c == 'low':
        return 'The current artifact set only treats this as a low-confidence related event.'
    return ''


def _coalesce_event_summary(event_evidence: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    candidates = [
        event_evidence.get('top_event_user_summary_long'),
        ctx.get('top_event_user_summary_long'),
        event_evidence.get('top_event_user_summary_short'),
        ctx.get('top_event_user_summary_short'),
        ctx.get('chat_ready_event_summary'),
        event_evidence.get('news_reasoning_summary'),
    ]
    for val in candidates:
        txt = _clean_text(val)
        if txt:
            return txt
    return ''


def _summarize_article_topic(event_evidence: Dict[str, Any]) -> str:
    raw = _clean_text(event_evidence.get('top_event_article_summary') or event_evidence.get('top_event_evidence_excerpt'))
    if not raw:
        return ''
    raw = re.sub(r"\s+", ' ', raw).strip()
    raw = re.sub(r"^We recently compiled a list of .*?\.\s*", '', raw, flags=re.IGNORECASE)
    raw = re.sub(r"^In this article,.*?\.\s*", '', raw, flags=re.IGNORECASE)
    raw = re.sub(r"^This article .*?\.\s*", '', raw, flags=re.IGNORECASE)
    # cut to first sentence-like chunk without breaking decimals too much
    m = re.search(r'(.{40,260}?[.!?])(?:\s|$)', raw)
    txt = (m.group(1) if m else raw[:220]).strip()
    txt = re.sub(r'\s+',' ', txt)
    return txt

def _respond_out_of_scope(result: RetrievalResult) -> str:
    msg = result.context.get(
        "message",
        "The requested information is not available in the current FinAnomX artifacts.",
    )
    return (
        "I do not have enough grounded evidence in the current FinAnomX artifacts "
        f"to answer that reliably.\n\n{msg}"
    ).strip()


def _respond_project_overview(result: RetrievalResult) -> str:
    ctx = result.context
    tickers = ", ".join(ctx.get("tickers", [])) or "N/A"
    return (
        "FinAnomX is an explainable financial anomaly detection system that combines "
        "anomaly scoring, numeric attribution, event grounding, and dashboard artifacts. "
        f"It currently tracks {ctx.get('ticker_count', 'N/A')} tickers: {tickers}. "
        f"The analysis window runs from {ctx.get('analysis_window', {}).get('start', 'N/A')} "
        f"to {ctx.get('analysis_window', {}).get('end', 'N/A')}."
    )


def _respond_data_query(result: RetrievalResult) -> str:
    ctx = result.context
    return (
        f"The current system is built from market data ({ctx.get('market_data_source', 'N/A')}), "
        f"macro data ({ctx.get('macro_data_source', 'N/A')}), and news/event data ({ctx.get('news_source', 'N/A')}). "
        f"It tracks {ctx.get('ticker_count', 'N/A')} tickers and uses macro variables such as "
        f"{', '.join(ctx.get('macro_variables', [])) or 'N/A'}."
    )


def _respond_anomaly_lookup(result: RetrievalResult) -> str:
    ctx = result.context
    abnormality = ctx.get("abnormality", {})
    model_evidence = ctx.get("model_evidence", {})
    event_evidence = ctx.get("event_evidence", {})
    summary = _clean_text(ctx.get("chat_ready_full_summary") or ctx.get("chat_ready_numeric_summary") or ctx.get("summary"))

    intro = f"Yes — the system marks this date as unusual for {ctx.get('ticker', 'the selected ticker')} on {ctx.get('actual_date', 'the selected date')}."
    short = _clean_text(abnormality.get("what_was_abnormal_short"))
    if short:
        intro += f" {short}"

    lines: List[str] = []
    for item in [
        _metric_line("1-day return", abnormality.get("ret_1d_pct"), "%"),
        _metric_line("5-day return", abnormality.get("ret_5d_pct"), "%"),
        _metric_line("20-day drawdown", abnormality.get("drawdown_20_pct"), "%"),
        _metric_line("Intraday range", abnormality.get("range_pct"), "%"),
        _metric_line("Gap", abnormality.get("gap_pct"), "%"),
        _metric_line("Volume z-score", abnormality.get("volume_z_20")),
    ]:
        if item:
            lines.append(item)

    feats = _extract_top_features(model_evidence, top_k=3)
    if feats:
        lines.append("- Key model signals: " + "; ".join(feats))

    title = _clean_text(event_evidence.get("top_event_title"))
    confidence = _clean_text(event_evidence.get("news_confidence_level"))
    short_event = _clean_text(event_evidence.get("top_event_user_summary_short") or ctx.get("top_event_user_summary_short"))
    if title:
        extra = f" (confidence: {confidence})" if confidence else ""
        lines.append(f"- Leading related event: {title}{extra}")
        if short_event:
            lines.append(f"- Why it may matter: {short_event}")

    body = summary or _clean_text(model_evidence.get("xai_summary_human")) or (
        "The anomaly reflects unusual market behaviour supported by the strongest "
        "model signals in the current artifact set."
    )
    return "\n\n".join([intro, "\n".join(lines), body]).strip()


def _respond_anomaly_explain_numeric(result: RetrievalResult) -> str:
    ctx = result.context
    abnormality = ctx.get("abnormality", {})
    model_evidence = ctx.get("model_evidence", {})
    summary = _clean_text(ctx.get("chat_ready_numeric_summary") or ctx.get("summary") or ctx.get("chat_ready_full_summary"))

    header = _clean_text(abnormality.get("what_was_abnormal_short")) or (
        "This anomaly is mainly defined by unusual price, range, and volume behaviour."
    )
    lines: List[str] = []
    for item in [
        _metric_line("1-day return", abnormality.get("ret_1d_pct"), "%"),
        _metric_line("5-day return", abnormality.get("ret_5d_pct"), "%"),
        _metric_line("20-day drawdown", abnormality.get("drawdown_20_pct"), "%"),
        _metric_line("20-day drawup", abnormality.get("drawup_20_pct"), "%"),
        _metric_line("Gap", abnormality.get("gap_pct"), "%"),
        _metric_line("Intraday range", abnormality.get("range_pct"), "%"),
        _metric_line("Volume ratio vs 20-day baseline", abnormality.get("volume_ratio_20_pct"), "%"),
        _metric_line("Volume z-score", abnormality.get("volume_z_20")),
    ]:
        if item:
            lines.append(item)
    feats = _extract_top_features(model_evidence, top_k=4)
    if feats:
        lines.append("- Key model signals: " + "; ".join(feats))
    body = summary or _clean_text(model_evidence.get("xai_summary_human")) or (
        "The strongest model evidence comes from the highest-attribution numeric signals "
        "rather than one isolated feature."
    )
    return "\n\n".join([header, "\n".join(lines), body]).strip()


def _respond_anomaly_explain_news(result: RetrievalResult) -> str:
    ctx = result.context
    event_evidence = ctx.get("event_evidence", {})
    title = _clean_text(event_evidence.get("top_event_title"))
    if not title:
        return (
            "The current artifact set does not provide a strong grounded article for this anomaly, "
            "so I cannot identify a reliable specific news trigger."
        )

    domain = _clean_text(event_evidence.get("top_event_domain"))
    etype = _clean_text(event_evidence.get("top_event_content_source_type"))
    confidence = _clean_text(event_evidence.get("news_confidence_level"))
    reasoning = _clean_text(event_evidence.get("top_event_user_summary_long") or ctx.get("top_event_user_summary_long") or ctx.get("chat_ready_event_summary") or event_evidence.get("news_reasoning_summary") or ctx.get("chat_ready_full_summary"))
    limits = _clean_text(event_evidence.get("news_evidence_limitations"))
    article_summary = _clean_text(event_evidence.get("top_event_article_summary") or event_evidence.get("top_event_evidence_excerpt"))

    meta: List[str] = []
    if domain:
        meta.append(domain)
    if etype:
        meta.append(f"evidence type: {etype}")
    if confidence:
        meta.append(f"confidence: {confidence}")

    supporters = _extract_supporting_articles(event_evidence, top_k=2)
    lines = [f"Top candidate event: {title}"]
    if meta:
        lines.append("- " + " | ".join(meta))
    if supporters:
        lines.append("- Supporting coverage: " + "; ".join(supporters))

    body = reasoning or (
        "This is the strongest available event explanation in the current artifact set, "
        "but it should still be read as evidence-backed plausibility rather than definitive proof."
    )
    if "why" in str(result.raw_query).lower() and reasoning:
        body = reasoning
    if article_summary:
        body = f"The article is mainly about: {article_summary}\n\n{body}"
    if limits:
        body += f" {limits}"
    return "\n\n".join([lines[0], "\n".join(lines[1:]) if len(lines) > 1 else "", body]).strip()


def _respond_counterfactual_validation(result: RetrievalResult) -> str:
    ctx = result.context
    cf = _cf_row(ctx)
    if not cf:
        return "Counterfactual validation is not available for this anomaly in the current artifact set."

    lines: List[str] = []
    for label, key in [
        ("Support level", "cf_support_level"),
        ("Strength", "cf_strength"),
        ("Stability", "cf_stability"),
        ("Specificity", "cf_specificity"),
        ("Ambiguity risk", "cf_ambiguity_risk"),
    ]:
        val = cf.get(key)
        if val is None or _is_nan(val):
            continue
        if isinstance(val, (int, float)):
            lines.append(f"- {label}: {_fmt_num(val, 3)}")
        else:
            lines.append(f"- {label}: {val}")

    summary = _clean_text(cf.get("counterfactual_summary") or ctx.get("chat_ready_event_summary") or ctx.get("summary"))
    intro = f"Counterfactual support for this explanation is currently {_clean_text(cf.get('cf_support_level')).lower() or 'available but limited'}."
    return "\n\n".join([intro, "\n".join(lines), summary or "This check measures evidence sensitivity and robustness, not definitive causality."]).strip()


def _respond_model_compare(result: RetrievalResult) -> str:
    ctx = result.context
    static_records = ctx.get("static", [])
    adapt_records = ctx.get("adapt", [])
    bits: List[str] = []
    if static_records:
        row = static_records[0]
        bits.append(f"Static: F1 {_fmt_num(row.get('f1'))}, AUROC {_fmt_num(row.get('auroc'))}, AUPRC {_fmt_num(row.get('auprc'))}")
    if adapt_records:
        row = adapt_records[0]
        bits.append(f"Adaptive: F1 {_fmt_num(row.get('f1'))}, AUROC {_fmt_num(row.get('auroc'))}, AUPRC {_fmt_num(row.get('auprc'))}")
    return "The current evaluation compares static and adaptive variants across the main detection metrics.\n\n" + ("\n".join(f"- {b}" for b in bits) if bits else "No comparison rows are available in the current artifacts.")


def _respond_metrics_query(result: RetrievalResult) -> str:
    ctx = result.context
    metric_name = ctx.get("requested_metric", result.parsed.get("metric_name"))
    model_name = ctx.get("requested_model", result.parsed.get("model_name"))
    detection_rows = ctx.get("detection_rows", [])
    lines: List[str] = []
    for block in detection_rows:
        variant = block.get("variant")
        rows = block.get("rows", [])
        if not rows:
            continue
        row = rows[0]
        if metric_name and metric_name in row:
            lines.append(f"- {variant}: {metric_name} = {_fmt_num(row.get(metric_name))}")
        else:
            vals = []
            for c in ["auroc", "auprc", "f1", "precision", "recall"]:
                if c in row:
                    vals.append(f"{c.upper()} {_fmt_num(row.get(c))}")
            if vals:
                lines.append(f"- {variant}: " + ", ".join(vals))
    header = f"Here are the available evaluation metrics for {model_name}." if model_name else "Here are the available evaluation metrics."
    return "\n\n".join([header, "\n".join(lines) if lines else "No matching metrics were found in the current evaluation artifacts."]).strip()


def _respond_event_window_query(result: RetrievalResult) -> str:
    ctx = result.context
    rows = ctx.get("event_windows", [])[:5]
    count = ctx.get("count", len(rows))
    lines = [f"- Event windows found: {count}"]
    for row in rows:
        lines.append(f"- {row.get('ticker', 'N/A')} | {row.get('event_type', row.get('type', 'N/A'))} | {row.get('event_date', row.get('date', 'N/A'))}")
    return "The dashboard can query event windows from the current artifact set.\n\n" + "\n".join(lines)


def _respond_dashboard_help(result: RetrievalResult) -> str:
    return "Use Overview to scan episode peaks, Detail to inspect one anomaly in plain language, and Chatbot to ask for explanations, candidate events, confidence, or evaluation metrics."


def _respond_topk(result: RetrievalResult) -> str:
    ctx = result.context
    rows = ctx.get("results", [])
    ticker = ctx.get("ticker", "the selected ticker")
    top_k = ctx.get("top_k", len(rows))
    if not rows:
        return f"No anomalies were found for {ticker} under the current filter."
    lines: List[str] = []
    for i, row in enumerate(rows[:top_k], start=1):
        lines.append(f"- {i}) {row.get('date')} | score {_fmt_num(row.get('score'))} | subtype {row.get('subtype', 'N/A')} | top event {row.get('top_event_title', 'N/A')}")
    return f"Here are the most prominent anomalies for {ticker} in the current filter.\n\n" + "\n".join(lines)


def build_grounded_response(result: RetrievalResult) -> str:
    if not result.answerable:
        return _respond_out_of_scope(result)
    intent = result.effective_intent
    if intent == "PROJECT_OVERVIEW":
        return _respond_project_overview(result)
    if intent == "DATA_QUERY":
        return _respond_data_query(result)
    if intent == "ANOMALY_LOOKUP":
        return _respond_anomaly_lookup(result)
    if intent == "ANOMALY_EXPLAIN_NUMERIC":
        return _respond_anomaly_explain_numeric(result)
    if intent == "ANOMALY_EXPLAIN_NEWS":
        return _respond_anomaly_explain_news(result)
    if intent == "COUNTERFACTUAL_VALIDATION":
        return _respond_counterfactual_validation(result)
    if intent == "MODEL_COMPARE":
        return _respond_model_compare(result)
    if intent == "METRICS_QUERY":
        return _respond_metrics_query(result)
    if intent == "EVENT_WINDOW_QUERY":
        return _respond_event_window_query(result)
    if intent == "DASHBOARD_HELP":
        return _respond_dashboard_help(result)
    if intent == "ANOMALY_TOPK":
        return _respond_topk(result)
    return _respond_out_of_scope(result)




