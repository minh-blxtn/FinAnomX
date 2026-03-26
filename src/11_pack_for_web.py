# src/11_pack_for_web.py
# STEP 11 — Pack final web/chat artifacts from already-computed results

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, fail, require_columns, require_unique_key  # noqa: E402
from utils.io import ensure_dir, read_csv, read_parquet, write_csv, sha256_file, write_json  # noqa: E402


FEATURE_PRETTY = {
    "log_return": "daily return",
    "volume_z_20": "abnormal volume",
    "vol_5": "5-day volatility",
    "vol_10": "10-day volatility",
    "vol_20": "20-day volatility",
    "vol_60": "60-day volatility",
    "rsi": "RSI momentum",
    "macd": "MACD momentum",
    "macd_signal": "MACD signal",
    "macd_diff": "MACD spread",
    "vix_diff": "VIX change",
    "dgs10_diff": "10Y yield change",
    "fedfunds_diff": "Fed funds change",
    "cpi_diff": "CPI change",
}


def _run_dir_latest() -> Path:
    d = config.RUNS_DIR / "latest"
    ensure_dir(d)
    return d


def _read_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _parse_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None).dt.normalize()
    return df


def _to_native(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        return None if pd.isna(x) else float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def _fmt_pct(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{float(x):+.{nd}f}%"


def _fmt_num(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.{nd}f}"


def _feature_group(name: str) -> str:
    n = str(name)
    if n in {"log_return"}:
        return "price"
    if n in {"volume_z_20"}:
        return "volume"
    if n.startswith("vol_"):
        return "volatility"
    if n in {"rsi", "macd", "macd_signal", "macd_diff"}:
        return "technical"
    if n.endswith("_diff") or n in {"vix", "dgs10", "fedfunds", "cpi"}:
        return "macro"
    return "other"


def _feature_pretty(name: str) -> str:
    return FEATURE_PRETTY.get(str(name), str(name).replace("_", " "))


def _safe_text(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip() if x is not None else ""
    return "" if s.lower() in {"", "nan", "none", "n/a"} else s


def _resolve_main_columns(base: pd.DataFrame, step06_manifest: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    if "score_main" in base.columns and "flag_main_adapt" in base.columns:
        name = None
        if step06_manifest:
            name = (step06_manifest.get("main_selection", {}) or {}).get("chosen_candidate")
        return "score_main", "flag_main_adapt", str(name or "main")
    if "score_main" in base.columns and "flag_main" in base.columns:
        name = None
        if step06_manifest:
            name = (step06_manifest.get("main_selection", {}) or {}).get("chosen_candidate")
        return "score_main", "flag_main", str(name or "main")

    cand = None
    if step06_manifest:
        cand = (step06_manifest.get("main_selection", {}) or {}).get("chosen_candidate")
    cand = str(cand or "patchtst_tranad")

    score_map = {
        "patchtst": "score_patchtst",
        "tranad": "score_tranad",
        "patchtst_tranad": "score_patchtst_tranad",
        "chronos": "score_chronos",
    }
    flag_adapt_map = {
        "patchtst": "flag_patchtst_adapt",
        "tranad": "flag_tranad_adapt",
        "patchtst_tranad": "flag_patchtst_tranad_adapt",
        "chronos": "flag_chronos_adapt",
    }
    flag_static_map = {
        "patchtst": "flag_patchtst",
        "tranad": "flag_tranad",
        "patchtst_tranad": "flag_patchtst_tranad",
        "chronos": "flag_chronos",
    }
    sc = score_map.get(cand)
    fa = flag_adapt_map.get(cand)
    fs = flag_static_map.get(cand)
    if sc in base.columns and fa in base.columns:
        return sc, fa, cand
    if sc in base.columns and fs in base.columns:
        return sc, fs, cand

    if "score_patchtst_tranad" in base.columns and "flag_patchtst_tranad_adapt" in base.columns:
        return "score_patchtst_tranad", "flag_patchtst_tranad_adapt", "patchtst_tranad"
    fail("Could not resolve main score/flag columns for Step 11.")


def _add_market_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    df["ret_1d_pct"] = np.expm1(pd.to_numeric(df["log_return"], errors="coerce")) * 100.0
    g = df.groupby("ticker", group_keys=False)
    df["ret_5d_pct"] = g["close"].pct_change(5) * 100.0
    df["ret_20d_pct"] = g["close"].pct_change(20) * 100.0
    roll_max20 = g["close"].rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
    vol_ma20 = g["volume"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    df["drawdown_20_pct"] = (df["close"] / roll_max20 - 1.0) * 100.0
    df["volume_ratio_20_pct"] = (df["volume"] / vol_ma20 - 1.0) * 100.0
    if {"open", "high", "low"}.issubset(df.columns):
        prev_close = g["close"].shift(1)
        df["gap_pct"] = ((df["open"] - prev_close) / prev_close) * 100.0
        df["range_pct"] = ((df["high"] - df["low"]) / df["close"]) * 100.0
    else:
        df["gap_pct"] = np.nan
        df["range_pct"] = np.nan
    return df


def _display_subtype(row: pd.Series) -> str:
    st = _safe_text(row.get("episode_subtype") or row.get("subtype"))
    if st:
        return st
    direction = _safe_text(row.get("episode_direction")) or "flat"
    abs_ret_z = pd.to_numeric(row.get("episode_max_abs_ret_z", np.nan), errors="coerce")
    vol_z = pd.to_numeric(row.get("episode_max_vol_z", np.nan), errors="coerce")
    ep_len = pd.to_numeric(row.get("episode_len", np.nan), errors="coerce")
    if pd.notna(abs_ret_z) and abs_ret_z >= 2.5:
        return f"elevated-price-move-{direction}" if direction in {"up", "down"} else "elevated-price-move"
    if pd.notna(vol_z) and vol_z >= 2.5:
        return "elevated-volume"
    if pd.notna(ep_len) and ep_len >= 5 and direction in {"up", "down"}:
        return f"persistent-{direction}-move"
    return "other"


def _what_was_abnormal_short(row: pd.Series) -> str:
    st = _safe_text(row.get("subtype_display") or row.get("episode_subtype") or row.get("subtype"))
    ep_len = pd.to_numeric(row.get("episode_len", np.nan), errors="coerce")
    direction = _safe_text(row.get("episode_direction")) or "flat"
    r1 = pd.to_numeric(row.get("ret_1d_pct", np.nan), errors="coerce")
    r5 = pd.to_numeric(row.get("ret_5d_pct", np.nan), errors="coerce")
    volz = pd.to_numeric(row.get("volume_z_20", np.nan), errors="coerce")
    if st == "price-spike-up":
        return f"Upward price shock: 1-day move {_fmt_pct(r1)} and 5-day move {_fmt_pct(r5)}."
    if st == "price-spike-down":
        return f"Downward price shock: 1-day move {_fmt_pct(r1)} and 5-day move {_fmt_pct(r5)}."
    if st == "volume-burst":
        return f"Volume burst: volume z-score {_fmt_num(volz)} with 1-day return {_fmt_pct(r1)}."
    if st == "regime-shift":
        return f"Regime shift: {direction} episode lasting {int(ep_len) if pd.notna(ep_len) else 'n/a'} day(s), 5-day move {_fmt_pct(r5)}."
    if st.startswith("persistent-"):
        return f"Persistent {direction} move over {int(ep_len) if pd.notna(ep_len) else 'n/a'} day(s), 5-day move {_fmt_pct(r5)}."
    return f"Unusual episode lasting {int(ep_len) if pd.notna(ep_len) else 'n/a'} day(s), 1-day return {_fmt_pct(r1)} and volume z-score {_fmt_num(volz)}."


def _market_context_summary(row: pd.Series) -> str:
    parts = [
        f"1D return {_fmt_pct(row.get('ret_1d_pct', np.nan))}",
        f"5D return {_fmt_pct(row.get('ret_5d_pct', np.nan))}",
        f"20D drawdown {_fmt_pct(row.get('drawdown_20_pct', np.nan))}",
        f"volume z {_fmt_num(row.get('volume_z_20', np.nan))}",
    ]
    if "range_pct" in row:
        parts.insert(2, f"intraday range {_fmt_pct(row.get('range_pct', np.nan))}")
    return "; ".join(parts) + "."


def _z_bucket(z: float) -> str:
    if pd.isna(z):
        return "unknown"
    z = float(z)
    if z >= 2.0:
        return "strongly elevated"
    if z >= 1.0:
        return "elevated"
    if z <= -2.0:
        return "strongly depressed"
    if z <= -1.0:
        return "depressed"
    return "near baseline"


def _summarize_xai_peak(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("rank").copy()
    if "feature_pretty" not in df.columns:
        df["feature_pretty"] = df["feature"].map(_feature_pretty)
    if "feature_group" not in df.columns:
        df["feature_group"] = df["feature"].map(_feature_group)
    if "z_bucket" not in df.columns:
        df["z_bucket"] = df["z_lookback"].map(_z_bucket) if "z_lookback" in df.columns else "unknown"

    top1 = df.iloc[0]
    grp_weights = df.groupby("feature_group")["importance_norm"].sum().sort_values(ascending=False)
    primary_group = grp_weights.index[0] if len(grp_weights) else "other"

    macro_count = int((df["feature_group"] == "macro").sum())
    tech_count = int((df["feature_group"] == "technical").sum())
    price_volume_count = int(df["feature_group"].isin(["price", "volume", "volatility"]).sum())

    human = f"Top model signal was {top1['feature_pretty']} ({top1['z_bucket']})."
    if primary_group == "macro":
        human += f" Top-5 signals were dominated by macro context; technical count={tech_count}/5."
    elif primary_group == "technical":
        human += f" Top-5 signals were dominated by technical momentum; macro count={macro_count}/5."
    else:
        human += f" Top-5 signals included {price_volume_count}/5 price, volume, or volatility features."

    return pd.Series({
        "xai_top_features": ", ".join(df["feature"].astype(str).tolist()),
        "xai_top_features_pretty": ", ".join(df["feature_pretty"].astype(str).tolist()),
        "xai_top_features_imp": ", ".join(df["importance_norm"].round(4).astype(str).tolist()),
        "xai_primary_group": primary_group,
        "xai_macro_count_top5": macro_count,
        "xai_technical_count_top5": tech_count,
        "xai_price_volume_count_top5": price_volume_count,
        "xai_top1_feature": str(top1["feature"]),
        "xai_top1_feature_pretty": str(top1["feature_pretty"]),
        "xai_top1_z_bucket": str(top1["z_bucket"]),
        "xai_summary_human": human,
    })


def _feature_human_phrase(row: pd.Series) -> str:
    pretty = str(row.get("feature_pretty", row.get("feature", "feature")))
    bucket = str(row.get("z_bucket", "near baseline"))
    contrib = pd.to_numeric(row.get("importance_norm", np.nan), errors="coerce")
    contrib_txt = "" if pd.isna(contrib) else f" (importance {_fmt_num(contrib, 3)})"
    return f"{pretty} was {bucket}{contrib_txt}."


def _make_top_feature_records(grp: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    if grp is None or len(grp) == 0:
        return []
    rows = grp.sort_values("rank").head(top_k).copy()
    if "feature_pretty" not in rows.columns:
        rows["feature_pretty"] = rows["feature"].map(_feature_pretty)
    if "feature_group" not in rows.columns:
        rows["feature_group"] = rows["feature"].map(_feature_group)
    if "z_bucket" not in rows.columns:
        rows["z_bucket"] = rows["z_lookback"].map(_z_bucket) if "z_lookback" in rows.columns else "unknown"
    cols = [c for c in ["feature", "feature_pretty", "feature_group", "importance_norm", "value_raw", "z_lookback", "z_bucket"] if c in rows.columns]
    out = []
    for _, r in rows.iterrows():
        item = {c: _to_native(r[c]) for c in cols}
        item["human_phrase"] = _feature_human_phrase(r)
        out.append(item)
    return out


def _make_supporting_article_records(grp: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
    if grp is None or len(grp) == 0:
        return []
    rows = grp.sort_values("rank_news").head(top_k).copy()
    cols = [c for c in [
        "rank_news", "title", "domain", "domain_tier", "content_source_type", "confidence_score",
        "article_themes", "sent_label", "article_summary", "evidence_excerpt", "url", "canonical_url",
        "market_consistency", "event_window_match", "xai_consistency", "event_cluster_consensus",
        "generic_wrap_flag", "holdings_flag", "direction_mismatch_penalty"
    ] if c in rows.columns]
    return [{c: _to_native(r[c]) for c in cols} for _, r in rows.iterrows()]


def _clean_article_snippet(text: Any, max_chars: int = 220) -> str:
    s = _safe_text(text)
    if not s:
        return ""
    s = " ".join(s.split())
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip() + "…"
    return s


def _theme_phrase(theme_text: Any) -> str:
    raw = _safe_text(theme_text)
    if not raw:
        return ""
    mapping = {
        "EARNINGS": "company results and earnings expectations",
        "ANALYST": "analyst sentiment and recommendation changes",
        "PRODUCT_AI": "AI and product-related demand",
        "TECHNICAL_FLOW": "technical trading flow and positioning",
        "MACRO": "broader macro conditions",
        "LEGAL": "legal or regulatory issues",
        "LAYOFFS_DELIVERIES": "operational and delivery-related news",
    }
    items = []
    for token in raw.split("|"):
        tok = token.strip()
        if tok and tok in mapping and mapping[tok] not in items:
            items.append(mapping[tok])
    if not items:
        return ""
    return ", ".join(items[:2])


def _market_move_phrase(row: pd.Series) -> str:
    subtype = _safe_text(row.get("subtype_display") or row.get("episode_subtype") or row.get("subtype")).lower()
    r1 = pd.to_numeric(row.get("ret_1d_pct", np.nan), errors="coerce")
    r5 = pd.to_numeric(row.get("ret_5d_pct", np.nan), errors="coerce")
    range_pct = pd.to_numeric(row.get("range_pct", np.nan), errors="coerce")
    if "regime-shift" in subtype:
        return "a regime-shift pattern"
    if "volume" in subtype:
        return "a volume-led anomaly"
    if "persistent" in subtype:
        return "a persistent short-term move"
    if pd.notna(r1) and abs(r1) >= 3:
        return "a sharp one-day move"
    if pd.notna(r5) and abs(r5) >= 8:
        return "a strong multi-day move"
    if pd.notna(range_pct) and range_pct >= 4:
        return "an unusually volatile trading day"
    return "an unusual short-term market move"


def _event_alignment_phrase(row: pd.Series) -> str:
    mc = pd.to_numeric(row.get("top_event_market_consistency", np.nan), errors="coerce")
    xc = pd.to_numeric(row.get("top_event_xai_consistency", np.nan), errors="coerce")
    if pd.notna(mc) and pd.notna(xc) and mc >= 0.75 and xc >= 0.75:
        return "This aligns well with both the market behaviour and the model signals."
    if pd.notna(mc) and mc >= 0.65:
        return "This broadly matches the market move observed on that date."
    if pd.notna(xc) and xc >= 0.65:
        return "This is directionally consistent with the strongest model signals."
    return "This is treated as related evidence rather than definitive proof of causality."


def _confidence_from_row(row: pd.Series) -> str:
    if pd.isna(row.get("top_event_title", np.nan)) or pd.isna(row.get("cf_strength", np.nan)):
        return "not_evaluated"
    body = str(row.get("top_event_content_source_type", "title_only"))
    cf = str(row.get("cf_support_level", "weak"))
    risk = pd.to_numeric(row.get("cf_ambiguity_risk", np.nan), errors="coerce")
    top_conf = pd.to_numeric(row.get("top_event_confidence_score", np.nan), errors="coerce")
    cluster = pd.to_numeric(row.get("cf_cluster_consensus", np.nan), errors="coerce")
    penalty = pd.to_numeric(row.get("cf_article_penalty", np.nan), errors="coerce")
    cluster = 0.0 if pd.isna(cluster) else float(cluster)
    penalty = 0.0 if pd.isna(penalty) else float(penalty)
    if cf == "strong" and body in {"full_text", "summary"} and (pd.isna(risk) or risk <= 0.42) and top_conf >= 0.62 and cluster >= 0.45 and penalty <= 0.22:
        return "high"
    if cf in {"strong", "moderate"} and body in {"full_text", "summary"} and top_conf >= 0.50 and (pd.isna(risk) or risk <= 0.62) and penalty <= 0.42:
        return "medium"
    return "low"


def _news_evidence_limitations(row: pd.Series) -> str:
    if pd.isna(row.get("top_event_title", np.nan)):
        if int(pd.to_numeric(row.get("is_episode_peak", 0), errors="coerce") or 0) == 0:
            return "Not evaluated: grounded news is only computed for analysis peaks."
        return "No grounded external article was linked to this peak."
    ctype = str(row.get("top_event_content_source_type", "title_only"))
    risk = pd.to_numeric(row.get("cf_ambiguity_risk", np.nan), errors="coerce")
    risk_txt = ""
    if pd.notna(risk):
        if risk >= 0.60:
            risk_txt = " Alternative explanations remain material."
        elif risk >= 0.40:
            risk_txt = " Some alternative explanations remain plausible."
    if ctype == "title_only":
        return "Evidence is title-only, so causal interpretation should stay conservative." + risk_txt
    if ctype == "summary":
        return "Evidence mainly comes from article summary/metadata rather than full body text." + risk_txt
    return "Evidence includes article body text, but the retrieval result is still a candidate explanation rather than proof." + risk_txt


def _counterfactual_summary(row: pd.Series) -> str:
    if pd.isna(row.get("cf_strength", np.nan)):
        return "Not evaluated for this row."
    return (
        f"Counterfactual support is {row.get('cf_support_level', 'weak')} with strength {_fmt_num(row.get('cf_strength', np.nan), 3)}; "
        f"stability {_fmt_num(row.get('cf_stability', np.nan), 3)}, specificity {_fmt_num(row.get('cf_specificity', np.nan), 3)}, "
        f"margin {_fmt_num(row.get('cf_margin_score', np.nan), 3)}, ambiguity risk {_fmt_num(row.get('cf_ambiguity_risk', np.nan), 3)}."
    )


def _news_reasoning_summary(row: pd.Series) -> str:
    if pd.isna(row.get("top_event_title", np.nan)):
        if int(pd.to_numeric(row.get("is_episode_peak", 0), errors="coerce") or 0) == 0:
            return "Not evaluated because grounded news reasoning is only computed for analysis peaks."
        return "No grounded article was retrieved for this anomaly peak."
    title = str(row.get("top_event_title"))
    context = str(row.get("market_context_summary", "")).strip()
    xai = str(row.get("xai_summary_human", "")).strip()
    support = str(row.get("cf_support_level", "weak"))
    ctype = str(row.get("top_event_content_source_type", "title_only")).replace("_", " ")
    themes = str(row.get("top_event_article_themes", "")).replace("|", ", ").strip(", ")
    extra = f" Additional context: article themes: {themes}." if themes else ""
    return (
        f"Top linked article is '{title}'. The market anomaly looked like: {context} "
        f"Model signals indicate: {xai} Evidence type is {ctype}, and counterfactual support is {support}.{extra}"
    ).strip()


def _explanation_summary_long(row: pd.Series) -> str:
    parts = [
        _safe_text(row.get("what_was_abnormal_short")),
        _safe_text(row.get("xai_summary_human")),
        _safe_text(row.get("news_reasoning_summary")),
        _safe_text(row.get("counterfactual_summary")),
        _safe_text(row.get("news_evidence_limitations")),
    ]
    return " ".join([p for p in parts if p])


def _top_event_user_summary_short(row: pd.Series) -> str:
    title = _safe_text(row.get("top_event_title"))
    if not title:
        return "No strong related event is available for this anomaly."
    snippet = _clean_article_snippet(row.get("top_event_article_summary") or row.get("top_event_evidence_excerpt"))
    move = _market_move_phrase(row)
    conf = _safe_text(row.get("news_confidence_level")).lower()
    lead = snippet or f"The linked article is '{title}'"
    if conf == "low":
        return f"{lead}. It may be related to {move.lower()}, but the evidence is limited."
    return f"{lead}. It is treated as a {conf or 'related'} event because it lines up with {move.lower()}."


def _top_event_user_summary_long(row: pd.Series) -> str:
    title = _safe_text(row.get("top_event_title"))
    if not title:
        return "The current artifact set does not provide a strong event explanation for this anomaly."
    snippet = _clean_article_snippet(row.get("top_event_article_summary") or row.get("top_event_evidence_excerpt"))
    theme = _theme_phrase(row.get("top_event_article_themes"))
    what = _safe_text(row.get("what_was_abnormal_short"))
    xai = _safe_text(row.get("xai_summary_human"))
    align = _event_alignment_phrase(row)
    conf = _confidence_from_row(row)
    conf_txt = {
        "high": "The system treats this as a high-confidence related event.",
        "medium": "The system treats this as a medium-confidence related event.",
        "low": "The system treats this as a weak candidate explanation.",
    }.get(conf, "")
    pieces = []
    if snippet:
        pieces.append(snippet + ".")
    elif theme:
        pieces.append(f"The linked article focuses on {theme}.")
    else:
        pieces.append(f"The linked article is '{title}'.")
    if what:
        pieces.append(f"That matters because the anomaly looked like {what[0].lower() + what[1:] if len(what) > 1 else what.lower()}")
        if not pieces[-1].endswith('.'):
            pieces[-1] += '.'
    if xai:
        pieces.append(f"The strongest model evidence also pointed in that direction: {xai}")
        if not pieces[-1].endswith('.'):
            pieces[-1] += '.'
    pieces.append(align)
    if conf_txt:
        pieces.append(conf_txt)
    return " ".join(pieces).strip()


def _chat_ready_numeric_summary(row: pd.Series) -> str:
    pieces = []
    for field in ["what_was_abnormal_short", "market_context_summary", "xai_summary_human"]:
        txt = _safe_text(row.get(field))
        if txt:
            pieces.append(txt if txt.endswith(".") else txt + ".")
    feats = _safe_text(row.get("xai_top_features_pretty"))
    if feats:
        pieces.append(f"Top features: {feats}.")
    return " ".join(pieces).strip()


def _chat_ready_event_summary(row: pd.Series) -> str:
    return _top_event_user_summary_long(row)


def _chat_ready_full_summary(row: pd.Series) -> str:
    parts = [
        _chat_ready_numeric_summary(row),
        _chat_ready_event_summary(row),
        _safe_text(row.get("counterfactual_summary")),
    ]
    return " ".join([p for p in parts if p]).strip()


def _build_chat_evidence_rows(
    master: pd.DataFrame,
    xai_top_map: Dict[tuple, List[Dict[str, Any]]],
    news_top_map: Dict[tuple, List[Dict[str, Any]]],
    cf_map: Dict[tuple, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in master.sort_values(["ticker", "date"]).iterrows():
        key = (str(r.get("ticker")), pd.to_datetime(r.get("date")).strftime("%Y-%m-%d"))
        rows.append({
            "ticker": str(r.get("ticker")),
            "date": pd.to_datetime(r.get("date")).strftime("%Y-%m-%d"),
            "split": _to_native(r.get("split")),
            "is_episode_peak": int(pd.to_numeric(r.get("is_episode_peak", 0), errors="coerce") or 0),
            "main_detector_name": _to_native(r.get("main_detector_name")),
            "subtype": _to_native(r.get("subtype_display") or r.get("subtype")),
            "abnormality": {
                "what_was_abnormal_short": _to_native(r.get("what_was_abnormal_short")),
                "market_context_summary": _to_native(r.get("market_context_summary")),
                "ret_1d_pct": _to_native(r.get("ret_1d_pct")),
                "ret_5d_pct": _to_native(r.get("ret_5d_pct")),
                "drawdown_20_pct": _to_native(r.get("drawdown_20_pct")),
                "volume_z_20": _to_native(r.get("volume_z_20")),
                "episode_peak_score": _to_native(r.get("episode_peak_score")),
                "main_score": _to_native(r.get("main_score")),
            },
            "model_evidence": {
                "xai_primary_group": _to_native(r.get("xai_primary_group")),
                "xai_summary_human": _to_native(r.get("xai_summary_human")),
                "xai_top_features_pretty": _to_native(r.get("xai_top_features_pretty")),
                "top_features": xai_top_map.get(key, []),
                "temporal": {
                    "peak_idx": _to_native(r.get("xai_time_peak_idx")),
                    "peak_val": _to_native(r.get("xai_time_peak_val")),
                },
            },
            "event_evidence": {
                "top_event_title": _to_native(r.get("top_event_title")),
                "top_event_url": _to_native(r.get("top_event_url")),
                "top_event_domain": _to_native(r.get("top_event_domain")),
                "top_event_content_source_type": _to_native(r.get("top_event_content_source_type")),
                "top_event_confidence_score": _to_native(r.get("top_event_confidence_score")),
                "news_confidence_level": _to_native(r.get("news_confidence_level")),
                "news_reasoning_summary": _to_native(r.get("news_reasoning_summary")),
                "supporting_articles": news_top_map.get(key, []),
            },
            "counterfactual": cf_map.get(key, {}),
            "chat_ready_numeric_summary": _to_native(r.get("chat_ready_numeric_summary")),
            "chat_ready_event_summary": _to_native(r.get("chat_ready_event_summary")),
            "chat_ready_full_summary": _to_native(r.get("chat_ready_full_summary")),
            "top_event_user_summary_short": _to_native(r.get("top_event_user_summary_short")),
            "top_event_user_summary_long": _to_native(r.get("top_event_user_summary_long")),
        })
    return rows


def main() -> None:
    logger.info("=== Step 11: Pack anomalies.csv MASTER ===")

    results_dir = config.RESULTS_DIR
    p_base = results_dir / "anomalies_base.csv"
    p_xai_feat = results_dir / "xai_feature.csv"
    p_xai_temp = results_dir / "xai_temporal.json"
    p_ea = results_dir / "event_attribution.csv"
    p_ec = results_dir / "event_counterfactual.csv"
    p_raw = config.RAW_MERGED_PATH

    for p in [p_base, p_xai_feat, p_xai_temp, p_ea, p_ec, p_raw]:
        if not p.exists():
            fail(f"Missing required file for step11: {p}")

    step06_manifest = _read_optional_json(_run_dir_latest() / "manifest_step06.json")

    base = read_csv(p_base)
    base = _parse_dates(base, "date")
    require_unique_key(base, ["ticker", "date"], where="anomalies_base")

    raw = read_parquet(p_raw)
    raw = _parse_dates(raw, "date")
    require_unique_key(raw, ["ticker", "date"], where="raw_merged")

    keep_raw = [c for c in ["ticker", "date", "open", "high", "low"] if c in raw.columns]
    base = base.merge(raw[keep_raw].copy(), on=["ticker", "date"], how="left")

    main_score_col, main_flag_col, main_detector_name = _resolve_main_columns(base, step06_manifest)
    base["main_score"] = pd.to_numeric(base[main_score_col], errors="coerce")
    base["main_flag"] = pd.to_numeric(base[main_flag_col], errors="coerce").fillna(0).astype(int)
    base["main_detector_name"] = main_detector_name
    base["main_score_col"] = main_score_col
    base["main_flag_col"] = main_flag_col

    base = _add_market_descriptors(base)
    if "subtype" in base.columns:
        base["subtype_model_raw"] = base["subtype"]
    base["subtype_display"] = base.apply(_display_subtype, axis=1)
    base["subtype"] = base["subtype_display"]
    base["what_was_abnormal_short"] = base.apply(_what_was_abnormal_short, axis=1)
    base["market_context_summary"] = base.apply(_market_context_summary, axis=1)

    xai = read_csv(p_xai_feat)
    require_columns(xai, ["ticker", "date", "rank", "feature", "importance_norm"], where="xai_feature")
    xai["date"] = pd.to_datetime(xai["date"]).dt.normalize()
    xai5 = xai[xai["rank"] <= 5].sort_values(["ticker", "date", "rank"]).copy()
    xai_rows = []
    for (ticker_key, date_key), grp in xai5.groupby(["ticker", "date"], sort=False):
        s = _summarize_xai_peak(grp.copy())
        s["ticker"] = ticker_key
        s["date"] = date_key
        xai_rows.append(s)
    xai_sum = pd.DataFrame(xai_rows)
    require_unique_key(xai_sum, ["ticker", "date"], where="xai_feature summary")

    with open(p_xai_temp, "r", encoding="utf-8") as f:
        temp = json.load(f)
    trows = []
    for _, v in temp.items():
        t = v.get("ticker")
        d = pd.to_datetime(v.get("date")).normalize()
        imp = v.get("time_importance", [])
        if isinstance(imp, list) and len(imp) > 0:
            arr = np.asarray(imp, dtype=float)
            idx = int(np.nanargmax(arr))
            val = float(arr[idx])
        else:
            idx = np.nan
            val = np.nan
        trows.append({"ticker": t, "date": d, "xai_time_peak_idx": idx, "xai_time_peak_val": val})
    xai_time = pd.DataFrame(trows)
    require_unique_key(xai_time, ["ticker", "date"], where="xai_temporal summary")

    ea = read_csv(p_ea)
    require_columns(
        ea,
        ["ticker", "peak_date", "rank_news", "retrieval_score", "contribution", "sent_label", "sent_score", "domain", "seendate", "title", "url"],
        where="event_attribution",
    )
    ea["peak_date"] = pd.to_datetime(ea["peak_date"]).dt.normalize()
    top1 = ea[ea["rank_news"] == 1].copy().rename(columns={
        "peak_date": "date",
        "title": "top_event_title",
        "url": "top_event_url",
        "final_url": "top_event_final_url",
        "canonical_url": "top_event_canonical_url",
        "domain": "top_event_domain",
        "domain_tier": "top_event_domain_tier",
        "seendate": "top_event_seendate",
        "sent_label": "top_event_sent_label",
        "sent_score": "top_event_sent_score",
        "contribution": "top_event_contribution",
        "retrieval_score": "top_event_retrieval_score",
        "article_summary": "top_event_article_summary",
        "evidence_text": "top_event_evidence_text",
        "evidence_excerpt": "top_event_evidence_excerpt",
        "content_source_type": "top_event_content_source_type",
        "body_support": "top_event_body_support",
        "domain_score": "top_event_domain_score",
        "evidence_len_norm": "top_event_evidence_len_norm",
        "market_consistency": "top_event_market_consistency",
        "event_window_match": "top_event_event_window_match",
        "xai_consistency": "top_event_xai_consistency",
        "article_themes": "top_event_article_themes",
        "peak_event_window_types": "top_event_peak_event_window_types",
        "peak_primary_group": "top_event_peak_primary_group",
        "peak_macro_share": "top_event_peak_macro_share",
        "confidence_score": "top_event_confidence_score",
    })
    keep_top1 = [c for c in [
        "ticker", "date", "top_event_title", "top_event_url", "top_event_final_url", "top_event_canonical_url",
        "top_event_domain", "top_event_domain_tier", "top_event_seendate", "top_event_sent_label", "top_event_sent_score",
        "top_event_contribution", "top_event_retrieval_score", "top_event_article_summary", "top_event_evidence_text",
        "top_event_evidence_excerpt", "top_event_content_source_type", "top_event_body_support", "top_event_domain_score",
        "top_event_evidence_len_norm", "top_event_market_consistency", "top_event_event_window_match",
        "top_event_xai_consistency", "top_event_article_themes", "top_event_peak_event_window_types",
        "top_event_peak_primary_group", "top_event_peak_macro_share", "top_event_confidence_score",
    ] if c in top1.columns]
    top1 = top1[keep_top1]
    require_unique_key(top1, ["ticker", "date"], where="event_attribution top1")

    ec = read_csv(p_ec)
    require_columns(ec, ["ticker", "peak_date", "cf_strength", "cf_support_level"], where="event_counterfactual")
    ec["peak_date"] = pd.to_datetime(ec["peak_date"]).dt.normalize()
    ec = ec.rename(columns={"peak_date": "date"})
    require_unique_key(ec, ["ticker", "date"], where="event_counterfactual")

    master = base.merge(xai_sum, on=["ticker", "date"], how="left")
    master = master.merge(xai_time, on=["ticker", "date"], how="left")
    master = master.merge(top1, on=["ticker", "date"], how="left")
    master = master.merge(ec, on=["ticker", "date"], how="left")

    master["news_confidence_level"] = master.apply(_confidence_from_row, axis=1)
    master["news_evidence_limitations"] = master.apply(_news_evidence_limitations, axis=1)
    master["counterfactual_summary"] = master.apply(_counterfactual_summary, axis=1)
    master["news_reasoning_summary"] = master.apply(_news_reasoning_summary, axis=1)
    master["explanation_summary_short"] = (
        master["what_was_abnormal_short"].fillna("No market summary.") + " " + master["xai_summary_human"].fillna("No model-signal summary.")
    ).str.strip()
    master["explanation_summary_long"] = master.apply(_explanation_summary_long, axis=1)
    master["top_event_user_summary_short"] = master.apply(_top_event_user_summary_short, axis=1)
    master["top_event_user_summary_long"] = master.apply(_top_event_user_summary_long, axis=1)
    master["chat_ready_numeric_summary"] = master.apply(_chat_ready_numeric_summary, axis=1)
    master["chat_ready_event_summary"] = master.apply(_chat_ready_event_summary, axis=1)
    master["chat_ready_full_summary"] = master.apply(_chat_ready_full_summary, axis=1)

    master["has_xai"] = (~master["xai_top_features"].isna()).astype(int)
    master["has_news"] = (~master["top_event_url"].isna()).astype(int)

    out_path = config.ANOMALIES_PATH
    ensure_dir(out_path.parent)
    write_csv(master, out_path)

    chat_path = results_dir / "chat_evidence.jsonl"
    xai_top_map = {
        (str(t), pd.to_datetime(d).strftime("%Y-%m-%d")): _make_top_feature_records(grp.copy(), top_k=5)
        for (t, d), grp in xai5.groupby(["ticker", "date"], sort=False)
    }
    news_top_map = {
        (str(t), pd.to_datetime(d).strftime("%Y-%m-%d")): _make_supporting_article_records(grp.copy(), top_k=3)
        for (t, d), grp in ea.groupby(["ticker", "peak_date"], sort=False)
    }
    cf_map = {
        (str(row["ticker"]), pd.to_datetime(row["date"]).strftime("%Y-%m-%d")): {k: _to_native(v) for k, v in row.to_dict().items()}
        for _, row in ec.iterrows()
    }
    chat_rows = _build_chat_evidence_rows(master, xai_top_map, news_top_map, cf_map)
    with open(chat_path, "w", encoding="utf-8") as f:
        for obj in chat_rows:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    manifest = {
        "step": "11_pack_for_web",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "anomalies_base": {"path": str(p_base), "sha256": sha256_file(p_base)},
            "raw_merged": {"path": str(p_raw), "sha256": sha256_file(p_raw)},
            "xai_feature": {"path": str(p_xai_feat), "sha256": sha256_file(p_xai_feat)},
            "xai_temporal": {"path": str(p_xai_temp), "bytes": int(p_xai_temp.stat().st_size)},
            "event_attribution": {"path": str(p_ea), "sha256": sha256_file(p_ea)},
            "event_counterfactual": {"path": str(p_ec), "sha256": sha256_file(p_ec)},
            "manifest_step06": {"present": step06_manifest is not None},
        },
        "outputs": {
            "anomalies_master": {"path": str(out_path), "sha256": sha256_file(out_path), "rows": int(len(master))},
            "chat_evidence": {"path": str(chat_path), "sha256": sha256_file(chat_path), "rows": int(len(chat_rows))},
        },
        "main_detector_name": main_detector_name,
        "main_score_col": main_score_col,
        "main_flag_col": main_flag_col,
        "phase3_fields": {
            "top_event_fields": [
                "top_event_title", "top_event_url", "top_event_domain", "top_event_contribution",
                "top_event_retrieval_score", "top_event_content_source_type", "top_event_evidence_excerpt",
                "top_event_sent_label",
            ],
            "counterfactual_fields": [
                "cf_strength", "cf_support_level", "cf_ambiguity_risk", "counterfactual_summary"
            ],
            "human_explain_fields": [
                "news_reasoning_summary", "news_evidence_limitations", "news_confidence_level",
                "explanation_summary_long", "top_event_user_summary_short", "top_event_user_summary_long",
                "chat_ready_numeric_summary", "chat_ready_event_summary", "chat_ready_full_summary",
            ],
        },
        "notes": {
            "artifacts_first": "detail UI should read anomalies.csv master; chatbot should prefer chat_evidence.jsonl for compact grounded context",
            "phase3_goal": "promote article-aware news reasoning + counterfactual confidence into anomalies.csv",
            "phase4_goal": "pack compact abnormality + model evidence + event evidence + counterfactual blocks for local chatbot consumption",
            "important_fix": "Step11 now resolves main_score/main_flag from canonical Step06 outputs instead of hard-coding fused detector columns.",
        },
    }
    write_json(manifest, _run_dir_latest() / "manifest_step11.json")

    ok(f"Saved MASTER -> {out_path}")
    ok(f"Saved CHAT EVIDENCE -> {chat_path}")
    ok("Step 11 DONE. anomalies.csv now contains article-aware event reasoning fields, and chat_evidence.jsonl is ready for UI/chatbot consumption.")


if __name__ == "__main__":
    main()

str