# =========================================================
# app/chatbot/rag_engine.py
# Grounded retrieval layer for FinAnomX chatbot
# =========================================================

from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_store import load_artifacts
from app.chatbot.intent import (
    parse_user_query,
    IntentResult,
    INTENT_PROJECT_OVERVIEW,
    INTENT_DATA_QUERY,
    INTENT_ANOMALY_LOOKUP,
    INTENT_ANOMALY_TOPK,
    INTENT_ANOMALY_EXPLAIN_NUMERIC,
    INTENT_ANOMALY_EXPLAIN_NEWS,
    INTENT_COUNTERFACTUAL_VALIDATION,
    INTENT_MODEL_COMPARE,
    INTENT_METRICS_QUERY,
    INTENT_EVENT_WINDOW_QUERY,
    INTENT_DASHBOARD_HELP,
    INTENT_OUT_OF_SCOPE,
)


# =========================================================
# 1. Dataclass
# =========================================================
@dataclass
class RetrievalResult:
    raw_query: str
    parsed: Dict[str, Any]
    effective_intent: str
    answerable: bool
    context: Dict[str, Any]
    sources: List[Dict[str, Any]]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# 2. Helpers
# =========================================================
def _safe_ts(x: Any) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x).normalize()
    except Exception:
        return None


def _top_feature_from_aux(store, ticker: str, date: pd.Timestamp) -> Optional[str]:
    xai_df = store.get_xai_features(ticker, date, top_k=1)
    if xai_df.empty:
        return None
    if "feature" in xai_df.columns and pd.notna(xai_df.iloc[0]["feature"]):
        return str(xai_df.iloc[0]["feature"]).strip()
    return None


def _top_event_from_aux(store, ticker: str, date: pd.Timestamp) -> Optional[str]:
    news_df = store.get_top_news(ticker, date, k=1)
    if news_df.empty:
        return None
    if "title" in news_df.columns and pd.notna(news_df.iloc[0]["title"]):
        return str(news_df.iloc[0]["title"]).strip()
    return None


def _load_event_windows() -> pd.DataFrame:
    store = load_artifacts()
    path = store.root / "data" / "results" / "event_windows.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for c in ["event_date", "start_date", "end_date", "date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
    return df


def _top_feature_1_from_row(row: pd.Series) -> Optional[str]:
    for c in ["xai_top_features", "top_feature_1"]:
        if c in row.index and pd.notna(row[c]):
            text = str(row[c]).strip()
            if not text:
                continue
            for sep in ["|", ",", ";"]:
                if sep in text:
                    return text.split(sep)[0].strip()
            return text
    return None


def _top_event_title_from_row(row: pd.Series) -> Optional[str]:
    for c in ["news1_title", "top_event_title", "title"]:
        if c in row.index and pd.notna(row[c]):
            text = str(row[c]).strip()
            if text:
                return text
    return None


def _as_source(file_name: str, key: str, url: Optional[str] = None) -> Dict[str, Any]:
    src = {"file": file_name, "key": key}
    if url:
        src["url"] = url
    return src


FOLLOWUP_EVENT_TERMS = [
    "that event", "this event", "the event", "that article", "this article", "that explanation",
    "why does that event", "why does this event", "how does that event",
    "why does that explain", "why does this explain", "why would that event",
    "why is that event", "why is this event", "plausibly explain",
]
FOLLOWUP_COUNTERFACTUAL_TERMS = [
    "how confident", "are you sure", "how sure", "how reliable", "counterfactual",
    "what is the evidence", "how strong is the evidence", "how strong is that evidence",
    "does that really explain", "does this really explain", "how convincing",
]
FOLLOWUP_ANOMALY_TERMS = [
    "that anomaly", "this anomaly", "it", "that move", "this move",
    "what was unusual", "what happened that day", "tell me more",
]

def _contains_any_phrase(query: str, phrases: List[str]) -> bool:
    query = (query or "").lower()
    return any(p in query for p in phrases)


def _apply_conversation_memory(parsed: IntentResult, conversation_context: Optional[Dict[str, Any]]) -> IntentResult:
    if not conversation_context:
        return parsed

    q = (parsed.normalized_query or "").lower()
    has_explicit_target = bool(parsed.ticker and parsed.date)

    followup_event = _contains_any_phrase(q, FOLLOWUP_EVENT_TERMS)
    followup_cf = _contains_any_phrase(q, FOLLOWUP_COUNTERFACTUAL_TERMS)
    followup_anomaly = _contains_any_phrase(q, FOLLOWUP_ANOMALY_TERMS)

    if has_explicit_target or not (followup_event or followup_cf or followup_anomaly):
        return parsed

    ticker = conversation_context.get("ticker")
    date = conversation_context.get("date")
    model_name = conversation_context.get("model_name")

    if ticker and not parsed.ticker:
        parsed.ticker = str(ticker)
    if date and not parsed.date:
        parsed.date = str(date)
    if model_name and not parsed.model_name:
        parsed.model_name = str(model_name)

    if followup_cf:
        parsed.intent = INTENT_COUNTERFACTUAL_VALIDATION
    elif followup_event:
        parsed.intent = INTENT_ANOMALY_EXPLAIN_NEWS
    elif followup_anomaly:
        parsed.intent = INTENT_ANOMALY_LOOKUP

    parsed.notes.append("Follow-up query was resolved using the most recent anomaly context from the chat history.")
    parsed.entities["resolved_from_history"] = True
    parsed.entities["history_ticker"] = parsed.ticker
    parsed.entities["history_date"] = parsed.date
    return parsed


def _pick_effective_intent(parsed: IntentResult) -> str:
    q = (parsed.normalized_query or "").lower()

    if parsed.metric_name == "event_hit_rate":
        return INTENT_EVENT_WINDOW_QUERY

    if parsed.count_only and parsed.ticker and (parsed.start_date or parsed.end_date or "anomaly" in q):
        return INTENT_ANOMALY_TOPK

    if parsed.count_only and not parsed.ticker and ("ma" in q or "ticker" in q or "dang phan tich" in q):
        return INTENT_PROJECT_OVERVIEW

    if not parsed.ticker and ("du lieu" in q or "lay tu dau" in q or "nguon" in q):
        return INTENT_DATA_QUERY

    if not parsed.ticker and (
        "pipeline" in q or "project" in q or "he thong" in q or "dang phan tich" in q or "co may ma" in q or "bao nhieu ma" in q
    ):
        return INTENT_PROJECT_OVERVIEW

    if parsed.ticker and parsed.date and (
        "counterfactual" in q or "co chac khong" in q or "phản biện" in q or "phan bien" in q
    ):
        return INTENT_COUNTERFACTUAL_VALIDATION

    if parsed.ticker and parsed.date and (
        "news" in q or "cause" in q or "nguyen nhan" in q or "tin nao" in q or
        " event" in q or q.startswith("event") or "that event" in q or "this event" in q or
        "explain the anomaly" in q or "plausibly explain" in q
    ):
        return INTENT_ANOMALY_EXPLAIN_NEWS

    if parsed.ticker and parsed.date and (
        "feature" in q or "xai" in q or "importance" in q or "integrated gradients" in q
    ):
        return INTENT_ANOMALY_EXPLAIN_NUMERIC

    if parsed.ticker and (parsed.start_date or parsed.end_date) and ("anomaly" in q or parsed.count_only):
        return INTENT_ANOMALY_TOPK

    if parsed.intent:
        return parsed.intent

    return INTENT_OUT_OF_SCOPE


def _default_top_anomaly_for_range(
    ticker: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Optional[pd.Series]:
    store = load_artifacts()
    if ticker is None:
        return None

    df = store.filter_anomalies(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        model_name="main",
        flagged_only=True,
    )
    if df.empty:
        return None

    return df.sort_values("ui_score", ascending=False).iloc[0]


def _default_top_anomaly_any(ticker: Optional[str] = None) -> Optional[pd.Series]:
    store = load_artifacts()
    df = store.filter_anomalies(
        ticker=ticker,
        model_name="main",
        flagged_only=True,
    )
    if df.empty:
        return None
    return df.sort_values("ui_score", ascending=False).iloc[0]


def _resolve_target_anomaly_for_explain(parsed: IntentResult) -> Optional[IntentResult]:
    if parsed.ticker and parsed.date:
        return parsed

    default_row = _default_top_anomaly_for_range(
        ticker=parsed.ticker,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
    )
    if default_row is None and parsed.ticker:
        default_row = _default_top_anomaly_any(parsed.ticker)

    if default_row is None:
        return None

    parsed.ticker = str(default_row["ticker"])
    parsed.date = pd.to_datetime(default_row["date"]).strftime("%Y-%m-%d")
    parsed.notes.append("Không có ngày anomaly cụ thể; hệ thống dùng anomaly có score cao nhất trong phạm vi hiện tại.")
    return parsed


# =========================================================
# 3. Retrieval blocks
# =========================================================
def retrieve_project_overview(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()

    df_main = store.filter_anomalies(
        start_date="2023-01-01",
        end_date="2024-12-31",
        model_name="main",
        flagged_only=True,
    )

    counts_by_ticker = []
    if not df_main.empty and "ticker" in df_main.columns:
        grp = df_main.groupby("ticker").size().sort_values(ascending=False)
        counts_by_ticker = [
            {"ticker": str(ticker), "count": int(count)}
            for ticker, count in grp.items()
        ]

    context = {
        "project_name": "FinAnomX",
        "objective": (
            "Explainable financial anomaly detection with numeric attribution, "
            "event grounding, and artifact-first deployment."
        ),
        "tickers": store.available_tickers(),
        "ticker_count": int(len(store.available_tickers())),
        "analysis_window": {
            "start": "2023-01-01",
            "end": "2024-12-31",
        },
        "analysis_years": [2023, 2024],
        "pipeline_steps": [
            "00_preflight",
            "01_ingest_raw",
            "02_process_data",
            "03_build_event_windows",
            "04_baselines",
            "05_train_models",
            "06_infer_anomalies",
            "07_xai_numeric",
            "08_news_on_demand",
            "09_event_grounding",
            "10_evaluate",
            "11_pack_for_web",
        ],
        "core_artifacts": [
            "anomalies.csv",
            "xai_temporal.json",
            "event_attribution.csv",
            "event_counterfactual.csv",
            "eval_detection_static.csv",
            "eval_detection_adapt.csv",
            "eval_explain.csv",
            "model_comparison_static.csv",
            "model_comparison_adapt.csv",
        ],
        "main_flagged_count": int(len(df_main)),
        "main_flagged_counts_by_ticker": counts_by_ticker,
    }

    sources = [
        _as_source("data/results/anomalies.csv", "master artifact"),
        _as_source("data/results/eval_explain.csv", "explainability summary"),
    ]

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_PROJECT_OVERVIEW,
        answerable=True,
        context=context,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_data_query(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()
    event_windows = _load_event_windows()

    context = {
        "market_data_source": "Yahoo Finance (yfinance)",
        "macro_data_source": "FRED",
        "news_source": "GDELT DOC API",
        "event_sources": [
            "BLS/FRED CPI release proxy",
            "Federal Reserve FOMC calendar",
            "SEC 8-K Item 2.02 earnings proxy",
        ],
        "tickers": store.available_tickers(),
        "ticker_count": int(len(store.available_tickers())),
        "raw_market_shape": list(store.raw_market.shape),
        "anomalies_shape": list(store.anomalies.shape),
        "anomalies_base_shape": list(store.anomalies_base.shape),
        "xai_feature_shape": list(store.xai_feature.shape),
        "event_attribution_shape": list(store.event_attribution.shape),
        "event_counterfactual_shape": list(store.event_counterfactual.shape),
        "event_windows_shape": list(event_windows.shape) if not event_windows.empty else None,
        "analysis_window": {
            "start": "2023-01-01",
            "end": "2024-12-31",
        },
        "macro_variables": ["vix", "dgs10", "fedfunds", "cpi"],
    }

    sources = [
        _as_source("data/raw/raw_merged.parquet", "market + macro merged"),
        _as_source("data/results/anomalies.csv", "master anomaly table"),
    ]
    if not event_windows.empty:
        sources.append(_as_source("data/results/event_windows.csv", "event window artifact"))

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_DATA_QUERY,
        answerable=True,
        context=context,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_anomaly_lookup(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()

    if parsed.ticker is None:
        return RetrievalResult(
            raw_query=parsed.raw_query,
            parsed=parsed.to_dict(),
            effective_intent=INTENT_ANOMALY_LOOKUP,
            answerable=False,
            context={"message": "Cần có ticker để tra cứu anomaly cụ thể."},
            sources=[],
            notes=parsed.notes,
        )

    if parsed.date is None:
        default_row = _default_top_anomaly_for_range(parsed.ticker, parsed.start_date, parsed.end_date)
        if default_row is None:
            default_row = _default_top_anomaly_any(parsed.ticker)
        if default_row is None:
            return RetrievalResult(
                raw_query=parsed.raw_query,
                parsed=parsed.to_dict(),
                effective_intent=INTENT_ANOMALY_LOOKUP,
                answerable=False,
                context={"message": "Không tìm thấy anomaly phù hợp cho ticker được hỏi trong phạm vi hiện tại."},
                sources=[],
                notes=parsed.notes,
            )
        parsed.date = pd.to_datetime(default_row["date"]).strftime("%Y-%m-%d")
        parsed.notes.append("Không có ngày cụ thể; hệ thống dùng anomaly có điểm cao nhất trong phạm vi hiện tại.")

    context = store.build_chat_context(
        ticker=parsed.ticker,
        date=parsed.date,
        model_name=parsed.model_name or "main",
        news_k=3,
        nearest_previous=True,
    )

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_ANOMALY_LOOKUP,
        answerable=bool(context.get("found", False)),
        context=context,
        sources=context.get("sources", []),
        notes=parsed.notes,
    )


def retrieve_anomaly_topk(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()

    if parsed.ticker is None:
        return RetrievalResult(
            raw_query=parsed.raw_query,
            parsed=parsed.to_dict(),
            effective_intent=INTENT_ANOMALY_TOPK,
            answerable=False,
            context={"message": "Cần có ticker để tra cứu anomaly theo khoảng thời gian."},
            sources=[],
            notes=parsed.notes,
        )

    requested_top_k = parsed.top_k or 5
    count_only = bool(getattr(parsed, "count_only", False))

    df = store.filter_anomalies(
        ticker=parsed.ticker,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        model_name=parsed.model_name or "main",
        flagged_only=True,
    )

    if df.empty:
        return RetrievalResult(
            raw_query=parsed.raw_query,
            parsed=parsed.to_dict(),
            effective_intent=INTENT_ANOMALY_TOPK,
            answerable=False,
            context={"message": "Không tìm thấy anomaly nào dưới bộ lọc hiện tại."},
            sources=[],
            notes=parsed.notes,
        )

    df_sorted = df.sort_values("ui_score", ascending=False)
    preview_n = min(3, len(df_sorted)) if count_only else min(requested_top_k, len(df_sorted))

    out = []
    for _, row in df_sorted.head(preview_n).iterrows():
        row_date = pd.to_datetime(row["date"]).normalize()
        top_feature = _top_feature_from_aux(store, parsed.ticker, row_date) or _top_feature_1_from_row(row)
        top_event = _top_event_from_aux(store, parsed.ticker, row_date) or _top_event_title_from_row(row)

        out.append({
            "ticker": row["ticker"],
            "date": row_date.strftime("%Y-%m-%d"),
            "score": float(row["ui_score"]) if pd.notna(row["ui_score"]) else None,
            "subtype": row.get("subtype", None),
            "top_feature_1": top_feature,
            "top_event_title": top_event,
        })

    context = {
        "ticker": parsed.ticker,
        "start_date": parsed.start_date,
        "end_date": parsed.end_date,
        "top_k": requested_top_k,
        "count": int(len(df_sorted)),
        "count_only": count_only,
        "results": out,
    }

    sources = [
        _as_source(
            "data/results/anomalies.csv",
            f"ticker={parsed.ticker}, range={parsed.start_date or 'ALL'} to {parsed.end_date or 'ALL'}",
        ),
        _as_source(
            "data/results/xai_feature.csv",
            f"ticker={parsed.ticker}, range={parsed.start_date or 'ALL'} to {parsed.end_date or 'ALL'}",
        ),
        _as_source(
            "data/results/event_attribution.csv",
            f"ticker={parsed.ticker}, range={parsed.start_date or 'ALL'} to {parsed.end_date or 'ALL'}",
        ),
    ]

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_ANOMALY_TOPK,
        answerable=True,
        context=context,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_anomaly_explain_numeric(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()

    parsed = _resolve_target_anomaly_for_explain(parsed)
    if parsed is None or parsed.ticker is None or parsed.date is None:
        return RetrievalResult(
            raw_query=parsed.raw_query if parsed else "",
            parsed=parsed.to_dict() if parsed else {},
            effective_intent=INTENT_ANOMALY_EXPLAIN_NUMERIC,
            answerable=False,
            context={"message": "Không resolve được anomaly hợp lệ để giải thích numeric."},
            sources=[],
            notes=parsed.notes if parsed else [],
        )

    context = store.build_chat_context(
        ticker=parsed.ticker,
        date=parsed.date,
        model_name=parsed.model_name or "main",
        news_k=5,
        nearest_previous=True,
    )

    if not context.get("found", False):
        return RetrievalResult(
            raw_query=parsed.raw_query,
            parsed=parsed.to_dict(),
            effective_intent=INTENT_ANOMALY_EXPLAIN_NUMERIC,
            answerable=False,
            context={"message": "Không tìm thấy anomaly context phù hợp trong hệ thống hiện tại."},
            sources=[],
            notes=parsed.notes,
        )

    numeric_context = {
        "ticker": context["ticker"],
        "requested_date": context["requested_date"],
        "actual_date": context["actual_date"],
        "abnormality": context.get("abnormality", {}),
        "model_evidence": context.get("model_evidence", {}),
        "summary": context.get("chat_ready_numeric_summary") or context.get("chat_ready_full_summary"),
        "full_summary": context.get("chat_ready_full_summary"),
    }

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_ANOMALY_EXPLAIN_NUMERIC,
        answerable=True,
        context=numeric_context,
        sources=context.get("sources", []),
        notes=parsed.notes,
    )


def retrieve_anomaly_explain_news(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()

    parsed = _resolve_target_anomaly_for_explain(parsed)
    if parsed is None or parsed.ticker is None or parsed.date is None:
        return RetrievalResult(
            raw_query=parsed.raw_query if parsed else "",
            parsed=parsed.to_dict() if parsed else {},
            effective_intent=INTENT_ANOMALY_EXPLAIN_NEWS,
            answerable=False,
            context={"message": "Không resolve được anomaly hợp lệ để giải thích bằng news."},
            sources=[],
            notes=parsed.notes if parsed else [],
        )

    context = store.build_anomaly_context(
        ticker=parsed.ticker,
        date=parsed.date,
        model_name=parsed.model_name or "main",
        news_k=5,
        nearest_previous=True,
    )

    if not context.get("found", False):
        return RetrievalResult(
            raw_query=parsed.raw_query,
            parsed=parsed.to_dict(),
            effective_intent=INTENT_ANOMALY_EXPLAIN_NEWS,
            answerable=False,
            context={"message": "Không tìm thấy anomaly context phù hợp trong hệ thống hiện tại."},
            sources=[],
            notes=parsed.notes,
        )

    event_evidence = context.get("event_evidence", {})
    supporting_articles = event_evidence.get("supporting_articles", [])
    counterfactual = context.get("counterfactual", {})

    out = {
        "ticker": context["ticker"],
        "requested_date": context["requested_date"],
        "actual_date": context["actual_date"],
        "abnormality": context.get("abnormality", {}),
        "event_evidence": event_evidence,
        "counterfactual": counterfactual,
        "summary": context.get("chat_ready_event_summary") or context.get("chat_ready_full_summary"),
        "full_summary": context.get("chat_ready_full_summary"),
    }

    sources = list(context.get("sources", []))
    for row in supporting_articles[:3]:
        if row.get("url"):
            sources.append(
                _as_source(
                    "data/results/event_attribution.csv",
                    f"ticker={parsed.ticker}, peak_date={context['actual_date']}",
                    url=row.get("url"),
                )
            )

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_ANOMALY_EXPLAIN_NEWS,
        answerable=True,
        context=out,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_counterfactual_validation(parsed: IntentResult) -> RetrievalResult:
    parsed = _resolve_target_anomaly_for_explain(parsed)
    if parsed is None or parsed.ticker is None or parsed.date is None:
        return RetrievalResult(
            raw_query=parsed.raw_query if parsed else "",
            parsed=parsed.to_dict() if parsed else {},
            effective_intent=INTENT_COUNTERFACTUAL_VALIDATION,
            answerable=False,
            context={"message": "Không resolve được anomaly hợp lệ cho counterfactual validation."},
            sources=[],
            notes=parsed.notes if parsed else [],
        )

    store = load_artifacts()
    context = store.build_chat_context(
        ticker=parsed.ticker,
        date=parsed.date,
        model_name=parsed.model_name or "main",
        news_k=3,
        nearest_previous=True,
    )

    out = {
        "ticker": context.get("ticker"),
        "actual_date": context.get("actual_date"),
        "top_news": context.get("top_news", []),
        "counterfactual": context.get("counterfactual", []),
    }

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_COUNTERFACTUAL_VALIDATION,
        answerable=bool(context.get("found", False)),
        context=out,
        sources=context.get("sources", []),
        notes=parsed.notes,
    )


def retrieve_model_compare(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()

    mc_static = store.get_model_comparison("static")
    mc_adapt = store.get_model_comparison("adapt")

    context = {
        "static": mc_static.to_dict(orient="records"),
        "adapt": mc_adapt.to_dict(orient="records"),
        "summary_note": (
            "Use static/adapt model comparison artifacts for trade-off analysis. "
            "The main detector should not be claimed as the best-performing model by default."
        ),
    }

    sources = [
        _as_source("data/results/model_comparison_static.csv", "full table"),
        _as_source("data/results/model_comparison_adapt.csv", "full table"),
    ]

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_MODEL_COMPARE,
        answerable=True,
        context=context,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_metrics_query(parsed: IntentResult) -> RetrievalResult:
    store = load_artifacts()

    metric_name = parsed.metric_name
    model_name = parsed.model_name or "main"

    static_df = store.get_detection_metrics("static")
    adapt_df = store.get_detection_metrics("adapt")
    explain_df = store.get_explain_metrics()

    detection_rows = []
    for variant_name, df in [("static", static_df), ("adapt", adapt_df)]:
        if "model" not in df.columns:
            continue

        cand = df[df["model"].astype(str).str.lower() == str(model_name).lower()].copy()
        if cand.empty and model_name == "main":
            cand = df[df["model"].astype(str).str.contains("patchtst_tranad", case=False, na=False)].copy()

        if not cand.empty:
            detection_rows.append({
                "variant": variant_name,
                "rows": cand.to_dict(orient="records"),
            })

    context = {
        "requested_metric": metric_name,
        "requested_model": model_name,
        "detection_rows": detection_rows,
        "explain_metrics": explain_df.to_dict(orient="records"),
    }

    sources = [
        _as_source("data/results/eval_detection_static.csv", "detection static metrics"),
        _as_source("data/results/eval_detection_adapt.csv", "detection adaptive metrics"),
        _as_source("data/results/eval_explain.csv", "explainability metrics"),
    ]

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_METRICS_QUERY,
        answerable=True,
        context=context,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_event_window_query(parsed: IntentResult) -> RetrievalResult:
    event_windows = _load_event_windows()

    if event_windows.empty:
        return RetrievalResult(
            raw_query=parsed.raw_query,
            parsed=parsed.to_dict(),
            effective_intent=INTENT_EVENT_WINDOW_QUERY,
            answerable=False,
            context={"message": "event_windows.csv hiện không có trong hệ thống."},
            sources=[],
            notes=parsed.notes,
        )

    df = event_windows.copy()
    date_col = "event_date" if "event_date" in df.columns else ("date" if "date" in df.columns else None)

    if parsed.ticker and "ticker" in df.columns:
        df = df[(df["ticker"] == parsed.ticker) | (df["ticker"] == "ALL")]

    if date_col and parsed.start_date:
        df = df[df[date_col] >= pd.to_datetime(parsed.start_date)]
    if date_col and parsed.end_date:
        df = df[df[date_col] <= pd.to_datetime(parsed.end_date)]

    context = {
        "ticker": parsed.ticker,
        "start_date": parsed.start_date,
        "end_date": parsed.end_date,
        "event_windows": df.head(50).to_dict(orient="records"),
        "count": int(len(df)),
    }

    sources = [
        _as_source(
            "data/results/event_windows.csv",
            f"ticker={parsed.ticker or 'ALL'}, range={parsed.start_date or 'ALL'} to {parsed.end_date or 'ALL'}",
        )
    ]

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_EVENT_WINDOW_QUERY,
        answerable=True,
        context=context,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_dashboard_help(parsed: IntentResult) -> RetrievalResult:
    context = {
        "views": [
            {
                "name": "Overview",
                "purpose": "Filter ticker/date/subtype and inspect anomaly markers on the price chart.",
            },
            {
                "name": "Detail",
                "purpose": "Inspect one anomaly in depth: 60-day context, top features, temporal segment, news, counterfactual.",
            },
            {
                "name": "Chatbot",
                "purpose": "Ask grounded questions about anomalies, metrics, event windows, and model comparisons.",
            },
        ],
        "usage_tips": [
            "Choose ticker and date range in the sidebar.",
            "Click an anomaly marker in Overview to jump to Detail.",
            "Use Chatbot for anomaly lookup, top-k anomalies, and metrics queries.",
        ],
    }

    sources = [
        _as_source("data/results/anomalies.csv", "used by Overview/Detail"),
    ]

    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_DASHBOARD_HELP,
        answerable=True,
        context=context,
        sources=sources,
        notes=parsed.notes,
    )


def retrieve_out_of_scope(parsed: IntentResult) -> RetrievalResult:
    return RetrievalResult(
        raw_query=parsed.raw_query,
        parsed=parsed.to_dict(),
        effective_intent=INTENT_OUT_OF_SCOPE,
        answerable=False,
        context={
            "message": "Hệ thống FinAnomX hiện chưa có đủ artifact phù hợp để trả lời câu hỏi này."
        },
        sources=[],
        notes=parsed.notes,
    )


# =========================================================
# 4. Main entry
# =========================================================
def retrieve_for_query(raw_query: str, conversation_context: Optional[Dict[str, Any]] = None) -> RetrievalResult:
    parsed = parse_user_query(raw_query)
    parsed = _apply_conversation_memory(parsed, conversation_context)
    effective_intent = _pick_effective_intent(parsed)

    if effective_intent == INTENT_PROJECT_OVERVIEW:
        return retrieve_project_overview(parsed)
    elif effective_intent == INTENT_DATA_QUERY:
        return retrieve_data_query(parsed)
    elif effective_intent == INTENT_ANOMALY_LOOKUP:
        return retrieve_anomaly_lookup(parsed)
    elif effective_intent == INTENT_ANOMALY_TOPK:
        return retrieve_anomaly_topk(parsed)
    elif effective_intent == INTENT_ANOMALY_EXPLAIN_NUMERIC:
        return retrieve_anomaly_explain_numeric(parsed)
    elif effective_intent == INTENT_ANOMALY_EXPLAIN_NEWS:
        return retrieve_anomaly_explain_news(parsed)
    elif effective_intent == INTENT_COUNTERFACTUAL_VALIDATION:
        return retrieve_counterfactual_validation(parsed)
    elif effective_intent == INTENT_MODEL_COMPARE:
        return retrieve_model_compare(parsed)
    elif effective_intent == INTENT_METRICS_QUERY:
        return retrieve_metrics_query(parsed)
    elif effective_intent == INTENT_EVENT_WINDOW_QUERY:
        return retrieve_event_window_query(parsed)
    elif effective_intent == INTENT_DASHBOARD_HELP:
        return retrieve_dashboard_help(parsed)
    else:
        return retrieve_out_of_scope(parsed)


# =========================================================
# 5. Quick sanity tests
# =========================================================
if __name__ == "__main__":
    examples = [
        "NVDA ngày 2024-09-17 có gì bất thường không",
        "top 5 anomaly của TSLA năm 2024",
        "tin nào gây ra NVDA 2024-09-17",
        "có chắc không, phản biện thử cho NVDA 2024-09-17",
        "model nào tốt hơn giữa main và baseline",
        "F1 của iforest là bao nhiêu",
        "dữ liệu lấy từ đâu",
        "web này xài sao",
        "event hit rate là gì",
        "MSFT tháng 5/2024 có gì nổi bật",
        "NVDA q1 2024",
        "trong năm 2023 thì amazone có mấy điểm bất thường",
        "có mấy mã đang phân tích",
    ]

    for q in examples:
        print("=" * 120)
        print("RAW:", q)
        result = retrieve_for_query(q)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))


