# =========================================================
# app/components.py
# Reusable UI components for FinAnomX Streamlit app
# =========================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =========================================================
# 1. Constants / display helpers
# =========================================================
SUBTYPE_COLOR_MAP = {
    "other": "#636EFA",
    "shift": "#EF553B",
    "spike": "#00CC96",
    "vol-burst": "#AB63FA",
}

SENTIMENT_COLOR_MAP = {
    "positive": "#00CC96",
    "neutral": "#636EFA",
    "negative": "#EF553B",
}


def _safe_float(value: Any, ndigits: int = 4) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return round(float(value), ndigits)
    except Exception:
        return None


def _safe_str(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _friendly_bucket_text(bucket: Any) -> str:
    text = _safe_str(bucket, default="")
    if not text:
        return ""
    mapping = {
        "strongly elevated": "well above its recent baseline",
        "elevated": "above its recent baseline",
        "near baseline": "close to its recent baseline",
        "depressed": "below its recent baseline",
        "strongly depressed": "well below its recent baseline",
    }
    return mapping.get(text.lower(), text)


def _first_sentences(text: str, limit: int = 2) -> str:
    text = text.strip()
    if not text:
        return ""
    parts = re.split(r"(?<!\d)\.\s+(?=[A-Z])", text)
    parts = [p.strip().rstrip(".") for p in parts if p.strip()]
    if not parts:
        return text
    clipped = ". ".join(parts[:limit]).strip()
    return clipped + ("." if clipped and not clipped.endswith(".") else "")


def _clean_event_reasoning_text(text: Any) -> str:
    s = _safe_str(text, default="")
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"Additional context:.*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"window context:\s*nan;?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"ambiguity risk\s*[0-9\.]+;?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"cluster consensus\s*[0-9\.]+;?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s{2,}", " ", s).strip(" ;")
    return _first_sentences(s, limit=2)


def _compact_theme_list(text: Any, max_items: int = 3) -> str:
    raw = _safe_str(text, default="")
    if not raw or raw == "N/A":
        return ""
    items = [i.strip() for i in str(raw).split("|") if i.strip()]
    cleaned = []
    for item in items:
        friendly = item.replace("_", " ").title()
        if friendly not in cleaned:
            cleaned.append(friendly)
    return ", ".join(cleaned[:max_items])


def _build_detail_summary(anomaly_row: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    abnormal = _safe_str(anomaly_row.get("what_was_abnormal_short"), default="")
    if abnormal and abnormal != "N/A":
        lines.append(abnormal)

    xai_summary = _safe_str(anomaly_row.get("xai_summary_human"), default="")
    if xai_summary and xai_summary != "N/A":
        lines.append(_first_sentences(xai_summary, limit=1))

    title = _safe_str(anomaly_row.get("top_event_title"), default="")
    conf = _safe_str(anomaly_row.get("news_confidence_level"), default="")
    source = _safe_str(anomaly_row.get("top_event_domain"), default="")
    if title and title != "N/A":
        short_event = _safe_str(anomaly_row.get("top_event_user_summary_short"), default="")
        if short_event:
            lines.append(short_event.rstrip(".") + ".")
        else:
            event_line = f"Most relevant related event: {title}"
            extras = []
            if source and source != "N/A":
                extras.append(source)
            if conf and conf != "N/A":
                extras.append(f"confidence: {conf.lower()}")
            if extras:
                event_line += f" ({', '.join(extras)})"
            lines.append(event_line + ".")

    return lines


def _extract_top_feature(feature_text: Any) -> str:
    """
    Extract a readable top feature from packed text fields like:
    - 'fedfunds_diff, cpi, macd'
    - 'fedfunds_diff | cpi | macd'
    - '1) fedfunds_diff; 2) cpi'
    """
    text = _safe_str(feature_text, default="")
    if not text:
        return "N/A"

    for sep in ["|", ",", ";"]:
        if sep in text:
            first = text.split(sep)[0].strip()
            return first if first else "N/A"

    return text


def _pick_news_title(row: pd.Series) -> str:
    for c in ["news1_title", "top_event_title", "title"]:
        if c in row.index and _safe_str(row[c], default="") != "":
            return _safe_str(row[c])
    return "N/A"


def _coerce_price_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "close" in out.columns:
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=[c for c in ["date", "close"] if c in out.columns])
    if "date" in out.columns:
        out = out.sort_values("date")
    return out.reset_index(drop=True)


def _coerce_anomaly_df(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "close" in out.columns:
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if score_col in out.columns:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.dropna(subset=[c for c in ["date", "close"] if c in out.columns])
    if "date" in out.columns:
        out = out.sort_values("date")
    return out.reset_index(drop=True)


# =========================================================
# 2. KPI cards
# =========================================================
def render_kpi_cards(kpis: Dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total anomalies", int(kpis.get("total_anomalies", 0)))

    with c2:
        st.metric("Top subtype", _safe_str(kpis.get("top_subtype", "N/A")))

    with c3:
        max_score = kpis.get("max_score", None)
        st.metric("Max score", f"{max_score:.4f}" if max_score is not None else "N/A")


# =========================================================
# 3. Overview chart
# =========================================================
def build_overview_chart(
    price_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    ticker: str,
    score_col: str = "ui_score",
) -> go.Figure:
    """
    Overview chart:
    - close price line
    - episode-peak anomaly markers colored by subtype
    - rich hover with score / subtype / top feature / top event
    """
    price_df = _coerce_price_df(price_df)
    anomaly_df = _coerce_anomaly_df(anomaly_df, score_col=score_col)

    fig = go.Figure()

    # Close line
    if not price_df.empty:
        fig.add_trace(
            go.Scatter(
                x=price_df["date"].tolist(),
                y=price_df["close"].astype(float).tolist(),
                mode="lines",
                name=f"{ticker} Close",
                line=dict(width=2),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close=%{y:.2f}<extra></extra>",
            )
        )

    # Anomaly markers split by subtype for color legend
    subtype_values = (
        anomaly_df["subtype"].dropna().unique().tolist()
        if ("subtype" in anomaly_df.columns and not anomaly_df.empty)
        else ["other"]
    )

    for subtype in subtype_values:
        sub_df = anomaly_df.copy()
        if "subtype" in sub_df.columns:
            sub_df = sub_df[sub_df["subtype"] == subtype]

        if sub_df.empty:
            continue

        hover_text = []
        for _, row in sub_df.iterrows():
            top_feat = (
                _extract_top_feature(row.get("xai_top_features"))
                if "xai_top_features" in row.index
                else "N/A"
            )
            top_event = _pick_news_title(row)
            score_val = row.get(score_col, None)

            hover_text.append(
                f"date: {_safe_str(row.get('date'))}"
                f"<br>score: {_safe_float(score_val, 4)}"
                f"<br>subtype: {_safe_str(row.get('subtype'))}"
                f"<br>top_feature_1: {top_feat}"
                f"<br>top_event_title: {top_event}"
            )

        fig.add_trace(
            go.Scatter(
                x=sub_df["date"].tolist(),
                y=sub_df["close"].astype(float).tolist(),
                mode="markers",
                name=f"Anomaly - {subtype}",
                marker=dict(
                    size=10,
                    color=SUBTYPE_COLOR_MAP.get(subtype, "#636EFA"),
                    line=dict(width=1, color="white"),
                ),
                customdata=sub_df[["ticker", "date"]].astype(str).values
                if {"ticker", "date"}.issubset(sub_df.columns)
                else None,
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{ticker} Close Price with Episode-Peak Markers",
        xaxis_title="Date",
        yaxis_title="Close Price",
        hovermode="closest",
        legend_title="Series",
        template="plotly_white",
        height=520,
        margin=dict(l=30, r=20, t=60, b=30),
    )
    fig.update_xaxes(type="date")
    fig.update_yaxes(type="linear")

    return fig


# =========================================================
# 4. Detail window chart
# =========================================================
def build_detail_window_chart(
    price_df: pd.DataFrame,
    center_date: pd.Timestamp,
    segment_start: Optional[pd.Timestamp] = None,
    segment_end: Optional[pd.Timestamp] = None,
) -> go.Figure:
    """
    3-panel detail chart:
    - Close
    - Log return
    - Rolling volatility / volume
    """
    df = _coerce_price_df(price_df)

    # compute extra series safely
    df["log_return"] = np.log(df["close"]).diff()
    df["vol20"] = df["log_return"].rolling(20, min_periods=5).std()
    df["volume_m"] = df["volume"] / 1_000_000.0 if "volume" in df.columns else np.nan

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Close Price", "Log Return", "20-Day Rolling Volatility / Volume"),
        row_heights=[0.45, 0.25, 0.30],
    )

    # Row 1: Close
    fig.add_trace(
        go.Scatter(
            x=df["date"].tolist(),
            y=df["close"].astype(float).tolist(),
            mode="lines",
            name="Close",
            line=dict(width=2),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Center date marker
    fig.add_vline(
        x=pd.to_datetime(center_date),
        line_width=1.5,
        line_dash="dash",
        line_color="red",
    )

    # Optional highlight segment
    if segment_start is not None and segment_end is not None:
        fig.add_vrect(
            x0=pd.to_datetime(segment_start),
            x1=pd.to_datetime(segment_end),
            fillcolor="gold",
            opacity=0.18,
            line_width=0,
            row="all",
            col=1,
        )

    # Row 2: Returns
    fig.add_trace(
        go.Scatter(
            x=df["date"].tolist(),
            y=df["log_return"].astype(float).tolist(),
            mode="lines",
            name="Log Return",
            line=dict(width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Log return=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)

    # Row 3: Volatility + Volume
    fig.add_trace(
        go.Scatter(
            x=df["date"].tolist(),
            y=df["vol20"].astype(float).tolist(),
            mode="lines",
            name="Vol20",
            line=dict(width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Vol20=%{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    if "volume_m" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["date"].tolist(),
                y=df["volume_m"].astype(float).tolist(),
                name="Volume (M)",
                opacity=0.25,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume(M)=%{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=780,
        margin=dict(l=30, r=20, t=80, b=30),
        title=f"60-Day Window around {_safe_str(center_date)}",
        hovermode="x unified",
        legend_title="Series",
    )

    fig.update_yaxes(title_text="Close", row=1, col=1, type="linear")
    fig.update_yaxes(title_text="Return", row=2, col=1, type="linear")
    fig.update_yaxes(title_text="Vol / Vol(M)", row=3, col=1, type="linear")
    fig.update_xaxes(title_text="Date", row=3, col=1, type="date")

    return fig


# =========================================================
# 5. Top features bar chart
# =========================================================
def build_feature_bar_chart(
    xai_df: pd.DataFrame,
    top_k: int = 10,
) -> go.Figure:
    df = xai_df.copy()

    value_col = None
    for candidate in ["importance_norm", "importance", "attr_value", "score"]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Top Features (No data available)",
            height=420,
        )
        return fig

    if "rank" in df.columns:
        df = df.sort_values("rank")
    else:
        df = df.sort_values(value_col, ascending=False)

    df = df.head(top_k).copy()
    df["feature"] = df["feature"].astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    fig = go.Figure(
        go.Bar(
            x=df[value_col].astype(float).tolist(),
            y=df["feature"].tolist(),
            orientation="h",
            hovertemplate="Feature=%{y}<br>Importance=%{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Top Feature Attribution",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=420,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="linear")
    return fig


# =========================================================
# 6. Temporal importance chart
# =========================================================
def build_temporal_chart(
    temporal_obj: Optional[Dict[str, Any]],
) -> go.Figure:
    if not temporal_obj or "time_importance" not in temporal_obj:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Temporal Importance (No data available)",
            height=320,
        )
        return fig

    values = pd.to_numeric(pd.Series(temporal_obj["time_importance"]), errors="coerce").fillna(0.0)
    peak_idx = temporal_obj.get("peak_idx", None)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=values.astype(float).tolist(),
            mode="lines+markers",
            name="Temporal Importance",
            hovertemplate="Index=%{x}<br>Importance=%{y:.4f}<extra></extra>",
        )
    )

    if peak_idx is not None and 0 <= int(peak_idx) < len(values):
        peak_idx = int(peak_idx)
        fig.add_vline(
            x=peak_idx,
            line_width=1.5,
            line_dash="dash",
            line_color="red",
        )

    fig.update_layout(
        template="plotly_white",
        title="Temporal Attribution Profile",
        xaxis_title="Relative Time Index in Window",
        yaxis_title="Importance",
        height=320,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    fig.update_xaxes(type="linear")
    fig.update_yaxes(type="linear")
    return fig


# =========================================================
# 7. News display helpers
# =========================================================

def render_executive_summary(anomaly_row: Dict[str, Any]) -> None:
    st.subheader("Summary")
    lines = _build_detail_summary(anomaly_row)
    if not lines:
        st.info("No summary is available for this anomaly.")
        return
    for line in lines:
        st.markdown(f"- {line}")


def render_abnormality_panel(anomaly_row: Dict[str, Any]) -> None:
    st.subheader("What Was Abnormal")
    short = _safe_str(anomaly_row.get("what_was_abnormal_short"), default="No abnormality summary available.")
    market = _safe_str(anomaly_row.get("market_context_summary"), default="")
    st.markdown(f"**{short}**")
    if market != "N/A":
        st.caption(market)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        ("1D return", anomaly_row.get("ret_1d_pct"), "%"),
        ("5D return", anomaly_row.get("ret_5d_pct"), "%"),
        ("20D drawdown", anomaly_row.get("drawdown_20_pct"), "%"),
        ("Range", anomaly_row.get("range_pct"), "%"),
        ("Gap", anomaly_row.get("gap_pct"), "%"),
        ("Volume z", anomaly_row.get("volume_z_20"), ""),
    ]
    for col, (label, val, suf) in zip([c1, c2, c3, c4, c5, c6], metrics):
        with col:
            num = _safe_float(val, 2)
            st.metric(label, f"{num}{suf}" if num is not None else "N/A")


def render_model_signal_panel(anomaly_row: Dict[str, Any], xai_df: pd.DataFrame) -> None:
    st.subheader("Key Signals")
    summary = _safe_str(anomaly_row.get("xai_summary_human"), default="No model-signal summary is available.")
    st.markdown(summary)

    if xai_df.empty:
        st.info("No feature-level attribution is available for this anomaly.")
        return

    view = xai_df.copy().head(4)
    rows = []
    for _, row in view.iterrows():
        signal = _safe_str(row.get("feature_pretty") or row.get("feature"), default="N/A")
        category = _safe_str(row.get("feature_group"), default="N/A")
        why = _friendly_bucket_text(row.get("z_bucket")) or "Highlighted by the model"
        if why.lower() == "unknown":
            why = "helped the anomaly stand out"
        rows.append({
            "Signal": signal,
            "Category": category.title() if category != "N/A" else category,
            "Why it stood out": why,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



def render_event_snapshot(anomaly_row: Dict[str, Any]) -> None:
    st.subheader("Related Event")
    title = _safe_str(anomaly_row.get("top_event_title"), default="No grounded event is available.")
    if title == "No grounded event is available.":
        st.info(title)
        return

    confidence = _safe_str(anomaly_row.get("news_confidence_level"), default="N/A")
    domain = _safe_str(anomaly_row.get("top_event_domain"), default="N/A")
    tier = _safe_str(anomaly_row.get("top_event_domain_tier"), default="N/A")
    etype = _safe_str(anomaly_row.get("top_event_content_source_type"), default="N/A")
    themes = _compact_theme_list(anomaly_row.get("top_event_article_themes"))
    reasoning = _clean_event_reasoning_text(anomaly_row.get("news_reasoning_summary"))
    user_summary = _safe_str(anomaly_row.get("top_event_user_summary_short"), default="")

    st.markdown(f"**{title}**")
    meta = [domain]
    if tier != "N/A":
        meta.append(f"tier: {tier}")
    if etype != "N/A":
        meta.append(f"evidence: {etype}")
    if confidence != "N/A":
        meta.append(f"confidence: {confidence.lower()}")
    st.caption(" | ".join([m for m in meta if m]))

    if themes:
        st.write(f"**Main topic:** {themes}")

    if user_summary:
        st.write(user_summary)
    elif reasoning:
        st.write(reasoning)


def prepare_news_display_df(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["rank", "title", "domain", "sentiment", "weight", "evidence_type", "url", "pub_date"])

    df = news_df.copy()
    out = pd.DataFrame()
    out["rank"] = df["rank_news"] if "rank_news" in df.columns else None
    out["title"] = df["title"] if "title" in df.columns else None
    out["domain"] = df["domain"] if "domain" in df.columns else None
    if "sent_label" in df.columns:
        out["sentiment"] = df["sent_label"]
    else:
        out["sentiment"] = df["sentiment"] if "sentiment" in df.columns else None
    out["weight"] = df["contribution"] if "contribution" in df.columns else df["weight"] if "weight" in df.columns else None
    out["evidence_type"] = df["content_source_type"] if "content_source_type" in df.columns else None
    out["url"] = df["url"] if "url" in df.columns else None
    if "seendate" in df.columns:
        out["pub_date"] = df["seendate"]
    else:
        out["pub_date"] = df["pub_date"] if "pub_date" in df.columns else None

    if "weight" in out.columns:
        out["weight"] = out["weight"].apply(lambda x: _safe_float(x, 4))

    return out


def render_news_table(news_df: pd.DataFrame) -> None:
    st.subheader("Top News Evidence")

    display_df = prepare_news_display_df(news_df)

    if display_df.empty:
        st.info("No grounded news evidence is available for this anomaly.")
        return

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("URL"),
            "weight": st.column_config.NumberColumn("Weight", format="%.4f"),
        },
    )


# =========================================================
# 8. Counterfactual panel
# =========================================================
def render_counterfactual_panel(cf_df: pd.DataFrame, primary_title: str = "N/A") -> None:
    st.subheader("Counterfactual Validation")

    if cf_df.empty:
        st.info("No counterfactual result is available for this anomaly.")
        return

    row = cf_df.iloc[0]

    changed_val = row["top1_changed"] if "top1_changed" in row.index else row.get("cf_top1_changed")
    drop_proxy_val = None
    for c in ["cf_strength", "cf_drop_proxy", "cf_top1_contribution_proxy", "top1_contribution_drop_proxy"]:
        if c in row.index and pd.notna(row[c]):
            drop_proxy_val = row[c]
            break

    cf_topk_val = None
    for c in ["cf_topk", "topk"]:
        if c in row.index and pd.notna(row[c]):
            cf_topk_val = row[c]
            break

    support_level = row.get("cf_support_level", "N/A") if hasattr(row, 'get') else "N/A"
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Top-1 changed", str(changed_val) if changed_val is not None else "N/A")

    with c2:
        st.metric("CF strength", f"{float(drop_proxy_val):.4f}" if drop_proxy_val is not None else "N/A")

    with c3:
        st.metric("Support", str(support_level) if support_level is not None else "N/A")

    with st.expander("Show counterfactual detail"):
        compact = pd.DataFrame(
            [{
                "primary_news_title": primary_title,
                "top1_changed": changed_val,
                "cf_strength": drop_proxy_val,
                "counterfactual_topk": cf_topk_val,
                "support_level": support_level,
                "cluster_consensus": _safe_float(row.get("cf_cluster_consensus"), 3),
                "article_penalty": _safe_float(row.get("cf_article_penalty"), 3),
            }]
        )
        st.dataframe(compact, use_container_width=True, hide_index=True)


# =========================================================
# 9. Verdict logic
# =========================================================
def compute_primary_cause_verdict(
    news_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Conservative verdict logic:
    - if no news: insufficient external evidence
    - if top news weight high + counterfactual proxy high + top1 changed: stronger candidate
    - else: plausible but weakly supported
    """
    if news_df.empty:
        return {
            "label": "Insufficient external evidence",
            "reason": "No grounded news article is available for the selected anomaly.",
            "primary_title": None,
        }

    news_sorted = news_df.copy()
    if "rank_news" in news_sorted.columns:
        news_sorted = news_sorted.sort_values("rank_news")
    elif "contribution" in news_sorted.columns:
        news_sorted = news_sorted.sort_values("contribution", ascending=False)

    top_row = news_sorted.iloc[0]
    title = _safe_str(top_row.get("title"))
    weight = _safe_float(top_row.get("contribution", None), 4)

    if cf_df.empty:
        return {
            "label": "Plausible event candidate",
            "reason": (
                f"The highest-ranked article is '{title}', but no counterfactual result is "
                "available to strengthen the evidence."
            ),
            "primary_title": title,
        }

    cf_row = cf_df.iloc[0]
    changed = str(cf_row.get("top1_changed", "")).lower() in ["true", "1", "yes"]
    drop_proxy = None
    for c in ["cf_strength", "cf_drop_proxy", "cf_top1_contribution_proxy", "top1_contribution_drop_proxy"]:
        if c in cf_row.index and pd.notna(cf_row[c]):
            drop_proxy = _safe_float(cf_row.get(c), 4)
            break

    support_level = str(cf_row.get("cf_support_level", "")).lower()
    if support_level == "strong" or ((weight is not None and weight >= 0.30) and (drop_proxy is not None and drop_proxy >= 0.10) and changed):
        return {
            "label": "Primary candidate cause",
            "reason": (
                f"The article '{title}' has the highest evidence weight ({weight:.4f}) "
                f"and is supported by a counterfactual drop proxy of {drop_proxy:.4f}."
            ),
            "primary_title": title,
        }

    return {
        "label": "Plausible but weakly supported",
        "reason": (
            f"The article '{title}' is the top-ranked event candidate"
            + (f" with weight {weight:.4f}" if weight is not None else "")
            + f", and the current counterfactual support level is {support_level or 'weak'}, so it should still be treated as a candidate cause."
        ),
        "primary_title": title,
    }


def render_verdict_box(news_df: pd.DataFrame, cf_df: pd.DataFrame) -> str:
    st.subheader("Primary Cause")

    verdict = compute_primary_cause_verdict(news_df, cf_df)
    primary_title = verdict.get("primary_title") or "N/A"

    st.markdown(
        f"""
### {verdict['label']}
**Candidate event:** {primary_title}

{verdict['reason']}
"""
    )

    return primary_title


# =========================================================
# 10. Sources box
# =========================================================
def render_sources_box(sources: List[Dict[str, Any]]) -> None:
    st.subheader("Sources Used")

    if not sources:
        st.info("No sources were recorded.")
        return

    for src in sources:
        file_name = _safe_str(src.get("file"))
        key = _safe_str(src.get("key"))
        st.markdown(f"- **{file_name}** | `{key}`")


# =========================================================
# 11. Optional helper: anomaly selector options
# =========================================================
def build_anomaly_selector_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["label", "ticker", "date", "score", "subtype"])

    out = df.copy()
    out["score"] = out["ui_score"].apply(lambda x: _safe_float(x, 4))
    out["label"] = out.apply(
        lambda r: f"{r['ticker']} | {pd.to_datetime(r['date']).date()} | score={r['score']} | subtype={_safe_str(r.get('subtype'))}",
        axis=1,
    )

    return out[["label", "ticker", "date", "score", "subtype"]].reset_index(drop=True)



