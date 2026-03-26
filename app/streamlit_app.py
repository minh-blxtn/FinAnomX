# =========================================================
# app/streamlit_app.py
# FinAnomX Streamlit app (Level 1 deterministic chatbot)
# =========================================================

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_store import load_artifacts
from app.components import (
    render_kpi_cards,
    build_overview_chart,
    build_detail_window_chart,
    build_feature_bar_chart,
    build_temporal_chart,
    render_executive_summary,
    render_abnormality_panel,
    render_model_signal_panel,
    render_event_snapshot,
    render_news_table,
    render_counterfactual_panel,
    render_verdict_box,
    build_anomaly_selector_df,
)
from app.chatbot.rag_engine import retrieve_for_query
from app.chatbot.responders import build_grounded_response
from app.chatbot.prompts import build_prompt_package
from app.chatbot.ollama_client import OllamaClient
from app.chatbot.validators import validate_and_finalize_answer
import config


# =========================================================
# 1. Page config
# =========================================================
st.set_page_config(
    page_title="FinAnomX",
    page_icon="📈",
    layout="wide",
)

st.title("FinAnomX")
st.caption("Explainable Financial Anomaly Detection Dashboard")

st.markdown(
    """
<style>
div[data-baseweb="select"] * {
    cursor: pointer !important;
}
input, button, [role="button"] {
    cursor: pointer !important;
}
[data-testid="stDateInput"] input {
    cursor: pointer !important;
}
div[role="radiogroup"] label {
    cursor: pointer !important;
}
.block-container {
    padding-top: 3rem;
}
.finomx-typing {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 24px;
    padding: 2px 0 6px 0;
}
.finomx-typing span {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: rgba(255,255,255,0.82);
    animation: finomxPulse 1.25s infinite ease-in-out;
}
.finomx-typing span:nth-child(2) { animation-delay: 0.16s; }
.finomx-typing span:nth-child(3) { animation-delay: 0.32s; }
@keyframes finomxPulse {
    0%, 80%, 100% { opacity: 0.25; transform: translateY(0) scale(0.9); }
    40% { opacity: 1; transform: translateY(-2px) scale(1); }
}
</style>
""",
    unsafe_allow_html=True,
)


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# =========================================================
# 2. Load artifacts
# =========================================================
@st.cache_resource(show_spinner=False)
def get_store():
    return load_artifacts()


@st.cache_resource(show_spinner=False)
def get_ollama_client():
    return OllamaClient(
        base_url=config.OLLAMA_BASE_URL,
        model_name=config.OLLAMA_LLM_MODEL,
        timeout_sec=180,
    )


@st.cache_data(ttl=15, show_spinner=False)
def get_ollama_health():
    return get_ollama_client().healthcheck().to_dict()


store = get_store()

# =========================================================
# 3. Session state
# =========================================================
if "selected_anomaly" not in st.session_state:
    st.session_state["selected_anomaly"] = None

if "active_view" not in st.session_state:
    st.session_state["active_view"] = "Overview"

if "last_filter_signature" not in st.session_state:
    st.session_state["last_filter_signature"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "pending_chat_query" not in st.session_state:
    st.session_state["pending_chat_query"] = None


# =========================================================
# 4. Sidebar filters
# =========================================================
default_start = pd.Timestamp("2023-01-01").date()
default_end = pd.Timestamp("2024-12-31").date()
model_name = "main"

with st.sidebar:
    st.header("Filters")

    ticker_options = store.available_tickers()
    default_ticker_idx = ticker_options.index("NVDA") if "NVDA" in ticker_options else 0
    ticker = st.selectbox("Ticker", ticker_options, index=default_ticker_idx)

    col_start, col_end = st.columns(2)

    with col_start:
        start_date_input = st.date_input("Start date", value=default_start)

    with col_end:
        end_date_input = st.date_input("End date", value=default_end)

    start_date = pd.to_datetime(start_date_input).normalize()
    end_date = pd.to_datetime(end_date_input).normalize()

    if start_date > end_date:
        st.error("Start date must be earlier than or equal to end date.")
        st.stop()

    subtype_options = ["All"] + store.available_subtypes()
    subtype_filter = st.selectbox("Subtype", subtype_options, index=0)

    st.markdown("---")
    st.caption("Click an anomaly marker to jump straight into Detail.")


current_filter_signature = (ticker, str(start_date.date()), str(end_date.date()), subtype_filter)
if st.session_state["last_filter_signature"] != current_filter_signature:
    st.session_state["last_filter_signature"] = current_filter_signature
    if st.session_state["selected_anomaly"] is not None:
        if st.session_state["selected_anomaly"].get("ticker") != ticker:
            st.session_state["selected_anomaly"] = None


# =========================================================
# 5. Prepare filtered data
# =========================================================
filtered_all = store.filter_anomalies(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    subtype=subtype_filter,
    model_name=model_name,
    flagged_only=False,
)

peak_mask = (
    (filtered_all["ui_flag"] == 1)
    & (pd.to_numeric(filtered_all.get("is_episode_peak", 0), errors="coerce").fillna(0).astype(int) == 1)
)
filtered_flagged = filtered_all[peak_mask].copy()

price_df = store.raw_market.copy()
price_df = price_df[
    (price_df["ticker"] == ticker)
    & (price_df["date"] >= start_date)
    & (price_df["date"] <= end_date)
].copy().sort_values("date").reset_index(drop=True)

price_close_map = price_df[["date", "close"]].drop_duplicates().copy()

filtered_flagged = filtered_flagged.drop(columns=["close"], errors="ignore")
filtered_flagged = filtered_flagged.merge(
    price_close_map,
    on="date",
    how="left",
)


# =========================================================
# 6. Navigation
# =========================================================
view_options = ["Overview", "Detail", "Chatbot"]

active_view = st.radio(
    "Navigation",
    options=view_options,
    index=view_options.index(st.session_state["active_view"]),
    horizontal=True,
    label_visibility="collapsed",
)

if active_view != st.session_state["active_view"]:
    st.session_state["active_view"] = active_view


# =========================================================
# 7. Helpers
# =========================================================
def choose_detail_target() -> Optional[Tuple[str, pd.Timestamp, str]]:
    state_val = st.session_state.get("selected_anomaly")
    if state_val:
        return (
            state_val["ticker"],
            pd.to_datetime(state_val["date"]).normalize(),
            state_val.get("model_name", model_name),
        )

    if not filtered_flagged.empty:
        row = filtered_flagged.sort_values("ui_score", ascending=False).iloc[0]
        return (
            row["ticker"],
            pd.to_datetime(row["date"]).normalize(),
            model_name,
        )

    return None


def compute_time_segment_dates(ticker_value: str, date_value: pd.Timestamp):
    temporal_obj = store.get_top_time_segment(ticker_value, date_value)
    if not temporal_obj:
        return None, None, None

    peak_idx = temporal_obj["peak_idx"]
    window = temporal_obj["window"]

    price_hist = store.raw_market.copy()
    price_hist = price_hist[
        (price_hist["ticker"] == ticker_value)
        & (price_hist["date"] <= pd.to_datetime(date_value).normalize())
    ].sort_values("date")

    if len(price_hist) < window:
        return temporal_obj, None, None

    window_df = price_hist.tail(window).reset_index(drop=True)

    seg_left = max(0, peak_idx - 1)
    seg_right = min(len(window_df) - 1, peak_idx + 1)

    segment_start = window_df.loc[seg_left, "date"]
    segment_end = window_df.loc[seg_right, "date"]

    return temporal_obj, segment_start, segment_end


def _typing_indicator_html() -> str:
    return '<div class="finomx-typing"><span></span><span></span><span></span></div>'


def _latest_chat_context() -> Optional[dict]:
    history = st.session_state.get("chat_history", []) or []
    for item in reversed(history):
        if item.get("role") != "assistant":
            continue
        meta = item.get("meta") or {}
        target = meta.get("resolved_target")
        if isinstance(target, dict) and target.get("ticker") and target.get("date"):
            return target
    return None


def _extract_resolved_target(retrieval) -> dict:
    ctx = retrieval.context if isinstance(retrieval.context, dict) else {}
    parsed = retrieval.parsed if isinstance(retrieval.parsed, dict) else {}
    event_evidence = ctx.get("event_evidence", {}) if isinstance(ctx.get("event_evidence"), dict) else {}
    return {
        "ticker": ctx.get("ticker") or parsed.get("ticker"),
        "date": ctx.get("actual_date") or parsed.get("date"),
        "model_name": parsed.get("model_name") or "main",
        "effective_intent": retrieval.effective_intent,
        "top_event_title": event_evidence.get("top_event_title") or ctx.get("top_event_title"),
        "top_event_user_summary_long": event_evidence.get("top_event_user_summary_long") or ctx.get("top_event_user_summary_long"),
        "top_event_user_summary_short": event_evidence.get("top_event_user_summary_short") or ctx.get("top_event_user_summary_short"),
    }


def _submit_chat_query(query: str, conversation_context: Optional[dict] = None) -> str:
    query = (query or "").strip()
    if not query:
        return

    retrieval = retrieve_for_query(query, conversation_context=conversation_context)

    # deterministic fallback luôn có sẵn
    deterministic_answer = build_grounded_response(retrieval)
    final_answer = deterministic_answer
    response_mode = "deterministic"
    llm_error = None
    llm_meta = {}

    # chỉ thử local LLM nếu retrieval answerable
    if retrieval.answerable:
        try:
            health = get_ollama_health()
            llm_meta["health"] = health

            if health.get("ok") and health.get("model_available"):
                client = get_ollama_client()
                pkg = build_prompt_package(retrieval)

                gen = client.generate_grounded_answer(
                    system_prompt=pkg["system_prompt"],
                    user_prompt=pkg["user_prompt"],
                )

                llm_meta["generation"] = {
                    "ok": gen.ok,
                    "model": gen.model,
                    "done_reason": gen.done_reason,
                    "error": gen.error,
                    "prompt_eval_count": gen.prompt_eval_count,
                    "eval_count": gen.eval_count,
                }

                if gen.ok and (gen.response_text or "").strip():
                    final_answer = validate_and_finalize_answer(retrieval, gen.response_text)
                    response_mode = "local_llm"
                else:
                    llm_error = gen.error or "Local LLM returned no usable final answer."
            else:
                if not health.get("ok"):
                    llm_error = health.get("error") or "Ollama server is unreachable."
                elif not health.get("model_available"):
                    llm_error = f"Model '{health.get('model_requested')}' is not available in Ollama."
        except Exception as e:
            llm_error = str(e)

    st.session_state["chat_history"].append({
        "role": "user",
        "content": query,
    })

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": final_answer,
        "meta": {
            "effective_intent": retrieval.effective_intent,
            "parsed": retrieval.parsed,
            "sources": retrieval.sources,
            "response_mode": response_mode,
            "llm_error": llm_error,
            "llm_meta": llm_meta,
            "resolved_target": _extract_resolved_target(retrieval),
        },
    })

    return final_answer

# =========================================================
# 8. Overview
# =========================================================
if st.session_state["active_view"] == "Overview":
    st.subheader("Overview")

    kpis = store.get_overview_kpis(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        subtype=subtype_filter,
        model_name=model_name,
        peaks_only=True,
    )
    render_kpi_cards(kpis)

    if price_df.empty:
        st.warning("No price data is available for the selected filter.")
    else:
        fig = build_overview_chart(
            price_df=price_df,
            anomaly_df=filtered_flagged,
            ticker=ticker,
            score_col="ui_score",
        )

        selected_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=560,
            key="overview_plot",
        )

        if selected_points:
            point = selected_points[0]
            x_val = pd.to_datetime(point.get("x")).normalize() if point.get("x") else None
            y_val = point.get("y", None)

            if x_val is not None and not filtered_flagged.empty:
                cand = filtered_flagged.copy()
                cand["date"] = pd.to_datetime(cand["date"]).dt.normalize()
                cand = cand[cand["date"] == x_val]

                if y_val is not None and "close" in cand.columns and not cand.empty:
                    cand["y_gap"] = (cand["close"] - float(y_val)).abs()
                    cand = cand.sort_values("y_gap")

                if not cand.empty:
                    row = cand.iloc[0]
                    st.session_state["selected_anomaly"] = {
                        "ticker": row["ticker"],
                        "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                        "model_name": model_name,
                    }
                    st.session_state["active_view"] = "Detail"
                    safe_rerun()

    st.markdown("### Episode peaks in current filter")
    selector_df = build_anomaly_selector_df(
        filtered_flagged.sort_values("ui_score", ascending=False)
    )

    if selector_df.empty:
        st.info("No flagged anomaly is available under the current filter.")
    else:
        st.dataframe(selector_df, use_container_width=True, hide_index=True)


# =========================================================
# 9. Detail
# =========================================================
elif st.session_state["active_view"] == "Detail":
    st.subheader("Detail")

    _, top_bar_right = st.columns([0.82, 0.18])
    with top_bar_right:
        if st.button("← Back to Overview", use_container_width=True):
            st.session_state["active_view"] = "Overview"
            safe_rerun()

    target = choose_detail_target()

    if target is None:
        st.info("No anomaly is selected. Please click a marker in Overview or choose a filter with flagged anomalies.")
    else:
        target_ticker, target_date, target_model = target

        st.markdown(
            f"**Selected anomaly:** `{target_ticker}` | `{target_date.date()}`"
        )

        detail_selector_df = build_anomaly_selector_df(
            filtered_flagged.sort_values("ui_score", ascending=False)
        )

        if not detail_selector_df.empty:
            labels = detail_selector_df["label"].tolist()
            current_label = None

            for _, r in detail_selector_df.iterrows():
                if (
                    r["ticker"] == target_ticker
                    and pd.to_datetime(r["date"]).normalize() == target_date
                ):
                    current_label = r["label"]
                    break

            chosen_label = st.selectbox(
                "Or choose anomaly manually",
                options=labels,
                index=labels.index(current_label) if current_label in labels else 0,
            )

            chosen_row = detail_selector_df[detail_selector_df["label"] == chosen_label].iloc[0]
            target_ticker = chosen_row["ticker"]
            target_date = pd.to_datetime(chosen_row["date"]).normalize()

            st.session_state["selected_anomaly"] = {
                "ticker": target_ticker,
                "date": target_date.strftime("%Y-%m-%d"),
                "model_name": model_name,
            }

        context = store.build_anomaly_context(
            ticker=target_ticker,
            date=target_date,
            model_name=target_model,
            news_k=5,
            nearest_previous=True,
        )

        if not context["found"]:
            st.error(context["message"])
        else:
            actual_date = pd.to_datetime(context["actual_date"]).normalize()

            if context.get("used_nearest_previous", False):
                st.warning(
                    f"The requested date is not a trading date. "
                    f"The nearest previous trading day was used: {actual_date.date()}."
                )

            anomaly_row = context["anomaly"]
            xai_records = pd.DataFrame(context["xai_features"])
            news_records = pd.DataFrame(context["top_news"])
            cf_records = pd.DataFrame(context["counterfactual"])

            temporal_obj, segment_start, segment_end = compute_time_segment_dates(
                target_ticker, actual_date
            )

            render_executive_summary(anomaly_row)
            render_abnormality_panel(anomaly_row)
            overview_left, overview_right = st.columns([1.05, 0.95])
            with overview_left:
                render_model_signal_panel(anomaly_row, xai_records)
            with overview_right:
                render_event_snapshot(anomaly_row)

            st.markdown("### 60-Day Context Window")
            detail_price = store.get_price_window(target_ticker, actual_date, window_days=30)
            detail_fig = build_detail_window_chart(
                price_df=detail_price,
                center_date=actual_date,
                segment_start=segment_start,
                segment_end=segment_end,
            )
            st.plotly_chart(detail_fig, use_container_width=True)

# =========================================================
# 10. Chatbot
# =========================================================
else:
    st.subheader("Chatbot")

    _ = get_ollama_health()

    _, top_right = st.columns([0.82, 0.18])
    with top_right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["pending_chat_query"] = None
            safe_rerun()

    for item in st.session_state["chat_history"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    typed_query = st.chat_input("Ask about an anomaly, market behaviour, related events, confidence, or dashboard metrics...")

    pending_query = st.session_state.get("pending_chat_query")
    user_query = pending_query or typed_query

    if user_query:
        if pending_query:
            st.session_state["pending_chat_query"] = None

        conversation_context = _latest_chat_context()

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(_typing_indicator_html(), unsafe_allow_html=True)
            final_answer = _submit_chat_query(user_query, conversation_context=conversation_context)
            placeholder.empty()
            st.markdown(final_answer)

        safe_rerun()


