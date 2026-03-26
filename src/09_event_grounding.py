from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys
import json
import time
import math
import hashlib
import re
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv
from loguru import logger
import requests
import numpy as np
import pandas as pd

# --- allow: python src/09_event_grounding.py
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, warn, fail, require, require_columns, require_unique_key  # noqa: E402
from utils.io import read_csv, read_parquet, write_csv, write_parquet, sha256_file, write_json  # noqa: E402


TOPK_NEWS = 5
QUERY_TOP_FEATURES = 5
ENABLE_FINBERT = True
FINBERT_MODEL = "ProsusAI/finbert"
EMBED_TIMEOUT_SEC = 60
MAX_EVIDENCE_EMBED_CHARS = 1800

EMBED_CACHE_PATH = config.RESULTS_DIR / "news_embed_cache.json"
ATTR_PATH = config.RESULTS_DIR / "event_attribution.csv"
CF_PATH = config.RESULTS_DIR / "event_counterfactual.csv"
NEWS_ENRICHED_PATH = config.RESULTS_DIR / "news_enriched.parquet"

COMPANY_NAMES = {
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
}

FEATURE_PRETTY = {
    "log_return": "return",
    "rsi": "RSI",
    "macd": "MACD",
    "macd_signal": "MACD signal",
    "macd_diff": "MACD divergence",
    "volume_z_20": "volume z-score",
    "vol_5": "5-day volatility",
    "vol_10": "10-day volatility",
    "vol_20": "20-day volatility",
    "vol_60": "60-day volatility",
    "vix": "VIX",
    "dgs10": "10Y Treasury yield",
    "fedfunds": "Fed funds rate",
    "cpi": "CPI",
    "vix_diff": "VIX change",
    "dgs10_diff": "10Y yield change",
    "fedfunds_diff": "Fed funds change",
    "cpi_diff": "CPI change",
}

DOMAIN_SCORE_MAP = {
    "high": 0.95,
    "medium": 0.65,
    "low": 0.35,
    "unknown": 0.45,
}

DOMAIN_HOST_SCORE = {
    "reuters.com": 1.00,
    "apnews.com": 0.98,
    "bloomberg.com": 0.98,
    "wsj.com": 0.97,
    "ft.com": 0.96,
    "cnbc.com": 0.93,
    "marketwatch.com": 0.88,
    "finance.yahoo.com": 0.86,
    "investors.com": 0.82,
    "benzinga.com": 0.72,
    "seekingalpha.com": 0.69,
    "investopedia.com": 0.78,
    "fool.com": 0.58,
    "investorplace.com": 0.52,
    "insidermonkey.com": 0.45,
    "etftrends.com": 0.50,
    "defenseworld.net": 0.30,
}

EVENT_TYPE_KEYWORDS = {
    "EARNINGS": [
        "earnings", "revenue", "eps", "guidance", "quarterly", "q1", "q2", "q3", "q4",
        "results", "beat", "beats", "miss", "misses", "outlook", "forecast",
    ],
    "FOMC": [
        "fomc", "federal reserve", "fed", "powell", "rate cut", "rate hike", "interest rate", "monetary policy",
    ],
    "CPI": [
        "cpi", "inflation", "consumer price index", "core inflation", "price data",
    ],
    "ANALYST": [
        "analyst", "upgrade", "downgrade", "price target", "top pick", "rating", "initiated", "reiterated",
    ],
    "PRODUCT_AI": [
        "ai", "artificial intelligence", "chip", "chips", "gpu", "launch", "product", "gtc", "openai", "azure",
        "databricks", "cloud", "copilot", "model", "data center",
    ],
    "LEGAL": [
        "lawsuit", "antitrust", "probe", "investigation", "sec", "ftc", "court", "settlement", "fine", "fined",
    ],
    "LAYOFFS_DELIVERIES": [
        "layoff", "layoffs", "job cuts", "deliveries", "delivery", "vehicle", "factory", "plant", "production", "employees",
    ],
    "TECHNICAL_FLOW": [
        "options", "whales", "technical analysis", "buy point", "hedge fund", "position", "etf", "inflow", "outflow",
        "pullback", "breakout", "pattern",
    ],
    "MACRO": [
        "federal reserve", "fed", "rates", "rate", "inflation", "cpi", "treasury", "yield", "macro", "economy",
        "economic", "jobs report", "consumer prices",
    ],
}

MACRO_FEATURE_PHRASES = {
    "fedfunds": ["Federal Reserve", "interest rates"],
    "fedfunds_diff": ["Fed policy change", "rate path"],
    "cpi": ["inflation", "consumer prices"],
    "cpi_diff": ["inflation surprise", "price data"],
    "dgs10": ["Treasury yields", "bond market"],
    "dgs10_diff": ["yield move", "rates market"],
    "vix": ["market volatility", "risk sentiment"],
    "vix_diff": ["volatility spike", "risk-off move"],
}


def _run_dir_latest() -> Path:
    p = config.RUNS_DIR / "latest"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_seendate(s: str):
    if not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ")
    except Exception:
        return None


def _peak_end(peak_date: str) -> datetime:
    return datetime.strptime(peak_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate_text(text: str, max_chars: int = MAX_EVIDENCE_EMBED_CHARS) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def _tokenize(text: str) -> List[str]:
    text = _clean_text(text).lower()
    return re.findall(r"[a-z0-9]{2,}", text)


def _keyword_overlap(query: str, text: str) -> float:
    q = set(_tokenize(query))
    if not q:
        return 0.0
    t = set(_tokenize(text))
    if not t:
        return 0.0
    return float(len(q & t) / len(q))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(x) / len(x)


def _recency_weight(published_at: datetime, peak_end: datetime) -> float:
    d = (peak_end - published_at).total_seconds() / 86400.0
    return float(math.exp(-d / 3.0))


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("||".join([p or "" for p in parts]).encode("utf-8")).hexdigest()


def _feature_group(name: str) -> str:
    n = str(name)
    if n in {"log_return"}:
        return "price"
    if n in {"volume_z_20"}:
        return "volume"
    if n in {"vol_5", "vol_10", "vol_20", "vol_60"}:
        return "volatility"
    if n in {"rsi", "macd", "macd_signal", "macd_diff"}:
        return "technical"
    if n in {"vix", "dgs10", "fedfunds", "cpi", "vix_diff", "dgs10_diff", "fedfunds_diff", "cpi_diff"}:
        return "macro"
    return "other"


def _feature_pretty(name: str) -> str:
    return FEATURE_PRETTY.get(str(name), str(name).replace("_", " "))


def _body_support(content_source_type: str) -> float:
    c = str(content_source_type or "").strip().lower()
    if c == "full_text":
        return 1.0
    if c == "summary":
        return 0.65
    return 0.0


def _canonical_domain(domain: str) -> str:
    d = _clean_text(domain).lower()
    d = re.sub(r"^https?://", "", d)
    d = re.sub(r"^www\.", "", d)
    return d.split("/")[0]


def _canonical_url(url: str) -> str:
    u = _clean_text(url).lower()
    u = re.sub(r"^https?://", "", u)
    u = u.replace(":443", "")
    u = re.sub(r"^www\.", "", u)
    u = u.split("?")[0].split("#")[0].rstrip("/")
    return u


def _contains_any(text: str, patterns: List[str]) -> bool:
    txt = _clean_text(text).lower()
    return any(re.search(p, txt) for p in patterns)


def _is_generic_price_wrap(title: str, evidence_text: str, themes: List[str], event_window_match: float) -> bool:
    text = f"{_clean_text(title)} {_clean_text(evidence_text)}"
    generic_patterns = [
        r"\bstock\s+(rises|rose|falls|fell|gains|gained|drops|dropped|climbs|climbed|slides|slid|surges|surged)\b",
        r"\bshares?\s+(rose|fell|gained|dropped|climbed|slid|surged|up|down)\b",
        r"\boutperforms? market\b",
        r"\bunderperforms? market\b",
        r"\bon the day\b",
        r"\bmarket recap\b",
    ]
    has_generic = _contains_any(text, generic_patterns)
    if not has_generic:
        return False
    theme_set = set(themes)
    has_real_cause = bool(theme_set & {"EARNINGS", "FOMC", "CPI", "ANALYST", "PRODUCT_AI", "LEGAL", "LAYOFFS_DELIVERIES"})
    if has_real_cause or event_window_match >= 0.70:
        return False
    return True


def _is_holdings_boilerplate(title: str, evidence_text: str) -> bool:
    text = f"{_clean_text(title)} {_clean_text(evidence_text)}"
    pats = [
        r"\b13f\b",
        r"\binstitutional investors?\b",
        r"\bshares? sold by\b",
        r"\bshares? bought by\b",
        r"\btrimmed (its )?position\b",
        r"\bboosted (its )?position\b",
        r"\braised (its )?stake\b",
        r"\blowered (its )?stake\b",
        r"\bownership in\b",
        r"\bposition in\b",
        r"\bholdings? in\b",
        r"\badvisors?\b",
        r"\bwealth management\b",
        r"\blife insurance co\b",
    ]
    return _contains_any(text, pats)


def _direction_mismatch_penalty(title: str, evidence_text: str, direction: str) -> float:
    text = f"{_clean_text(title)} {_clean_text(evidence_text)}".lower()
    up_pats = [r"\brises?\b", r"\bclimbs?\b", r"\bgains?\b", r"\bsurges?\b", r"\brall(y|ied)\b", r"\bup\s+[0-9]", r"\bbeats?\b"]
    down_pats = [r"\bfalls?\b", r"\bdrops?\b", r"\bslides?\b", r"\bslumps?\b", r"\bselloff\b", r"\bdown\s+[0-9]", r"\bmiss(es|ed)?\b"]
    up_hit = _contains_any(text, up_pats)
    down_hit = _contains_any(text, down_pats)
    d = str(direction or "flat").lower()
    if d == "down" and up_hit and not down_hit:
        return 1.0
    if d == "up" and down_hit and not up_hit:
        return 1.0
    if d in {"up", "down"} and up_hit and down_hit:
        return 0.35
    return 0.0


def _dedup_candidates(cand: pd.DataFrame) -> pd.DataFrame:
    x = cand.copy()
    x["canonical_url"] = x["url"].map(_canonical_url)
    x["_body_q"] = x["content_source_type"].map(_body_support)
    x["_domain_q"] = [_domain_score(d, t) for d, t in zip(x.get("domain", ""), x.get("domain_tier", "unknown"))]
    x["_len_q"] = x["evidence_text"].map(_evidence_len_norm)
    x["_pub_q"] = x["published_at"].map(lambda z: z.timestamp() if pd.notna(z) and z is not None else -1)
    x["_dedup_q"] = 0.45 * x["_body_q"] + 0.30 * x["_domain_q"] + 0.15 * x["_len_q"] + 0.10 * (x["_pub_q"] > 0).astype(float)
    x = x.sort_values(["canonical_url", "_dedup_q", "_pub_q"], ascending=[True, False, False])
    x = x.drop_duplicates(subset=["canonical_url"], keep="first").copy()
    return x.drop(columns=[c for c in ["_body_q", "_domain_q", "_len_q", "_pub_q", "_dedup_q"] if c in x.columns])


def _cluster_consensus(topk: pd.DataFrame, top1_row: pd.Series, window_types: List[str]) -> Tuple[float, int, float]:
    if topk.empty:
        return 0.0, 0, 0.0
    top1_themes = set(str(top1_row.get("article_themes", "")).split("|")) - {"", "nan"}
    window_set = {str(w).upper() for w in window_types if str(w).strip()}

    def _is_supportive(row: pd.Series) -> bool:
        row_themes = set(str(row.get("article_themes", "")).split("|")) - {"", "nan"}
        if top1_themes and row_themes and len(top1_themes & row_themes) > 0:
            return True
        if window_set and row_themes and len(window_set & row_themes) > 0:
            return True
        if pd.to_numeric(row.get("event_window_match", 0.0), errors="coerce") >= 0.75 and pd.to_numeric(top1_row.get("event_window_match", 0.0), errors="coerce") >= 0.75:
            return True
        return False

    support = topk[topk.apply(_is_supportive, axis=1)].copy()
    support_sum = float(pd.to_numeric(support.get("contribution", 0.0), errors="coerce").fillna(0).sum())
    support_domains = int(support.get("domain", pd.Series(dtype=str)).astype(str).str.lower().nunique()) if len(support) else 0
    cluster_score = float(max(0.0, min(1.0, 0.70 * min(1.0, support_sum / 0.60) + 0.30 * min(1.0, support_domains / 3.0))))
    return cluster_score, support_domains, support_sum


def _domain_score(domain: str, domain_tier: str) -> float:
    host = _canonical_domain(domain)
    for known, score in DOMAIN_HOST_SCORE.items():
        if host == known or host.endswith("." + known):
            return float(score)
    return float(DOMAIN_SCORE_MAP.get(str(domain_tier or "unknown"), 0.45))


def _evidence_len_norm(text: str) -> float:
    n = len(_truncate_text(text or "", MAX_EVIDENCE_EMBED_CHARS))
    return float(max(0.0, min(1.0, n / 900.0)))


def _infer_article_themes(title: str, evidence_text: str) -> List[str]:
    text = (f"{_clean_text(title)} {_clean_text(evidence_text)}").lower()
    out = []
    for theme, kws in EVENT_TYPE_KEYWORDS.items():
        if any(kw in text for kw in kws):
            out.append(theme)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _sentiment_alignment(sent_label: str, direction: str, sent_score: float = np.nan) -> float:
    s = str(sent_label or "").strip().lower()
    d = str(direction or "").strip().lower()
    conf = 0.5 if pd.isna(sent_score) else float(max(0.0, min(1.0, sent_score)))
    if s in {"", "none"}:
        return 0.5
    if s == "neutral":
        return 0.58
    if d == "up" and s == "positive":
        return float(0.75 + 0.25 * conf)
    if d == "down" and s == "negative":
        return float(0.75 + 0.25 * conf)
    if d == "flat":
        return 0.55
    return float(0.10 + 0.20 * (1.0 - conf))


def _event_window_types_for_peak(ev: pd.DataFrame, ticker: str, dt: pd.Timestamp) -> List[str]:
    if ev.empty:
        return []
    sub = ev[((ev["ticker"] == ticker) | (ev["ticker"] == "ALL")) & (ev["start_date"] <= dt) & (ev["end_date"] >= dt)].copy()
    vals = sorted({str(x).upper().strip() for x in sub["event_type"].dropna().tolist() if str(x).strip()})
    return vals


def _build_peak_context(sub_x: pd.DataFrame, peak_row: pd.Series, window_types: List[str]) -> Dict[str, Any]:
    sx = sub_x.sort_values("rank").copy()
    if "feature_group" not in sx.columns:
        sx["feature_group"] = sx["feature"].map(_feature_group)
    if "feature_pretty" not in sx.columns:
        sx["feature_pretty"] = sx["feature"].map(_feature_pretty)
    if "importance_norm" in sx.columns:
        grp = sx.groupby("feature_group", as_index=False)["importance_norm"].sum().sort_values("importance_norm", ascending=False)
        primary_group = str(grp.iloc[0]["feature_group"]) if len(grp) else "other"
        macro_share = float(sx.loc[sx["feature_group"] == "macro", "importance_norm"].sum())
    else:
        primary_group = str(sx["feature_group"].iloc[0]) if len(sx) else "other"
        macro_share = float((sx["feature_group"] == "macro").mean()) if len(sx) else 0.0
    return {
        "primary_group": primary_group,
        "macro_share": float(max(0.0, min(1.0, macro_share))),
        "direction": str(peak_row.get("episode_direction", "flat") or "flat").lower(),
        "subtype": str(peak_row.get("episode_subtype", peak_row.get("subtype", "other")) or "other").lower(),
        "window_types": list(window_types),
    }


def _event_window_match(themes: List[str], peak_ctx: Dict[str, Any]) -> float:
    theme_set = set(themes)
    window_types = [str(x).upper() for x in peak_ctx.get("window_types", [])]
    if not window_types:
        if peak_ctx.get("primary_group") == "macro" and theme_set & {"FOMC", "CPI", "MACRO"}:
            return 0.75
        return 0.55

    hits = 0
    for w in window_types:
        if w == "EARNINGS" and "EARNINGS" in theme_set:
            hits += 1
        elif w == "FOMC" and (theme_set & {"FOMC", "MACRO"}):
            hits += 1
        elif w == "CPI" and (theme_set & {"CPI", "MACRO"}):
            hits += 1
    if hits == 0:
        return 0.18
    return float(min(1.0, 0.60 + 0.20 * hits))


def _xai_consistency(themes: List[str], peak_ctx: Dict[str, Any]) -> float:
    t = set(themes)
    primary = str(peak_ctx.get("primary_group", "other"))
    macro_share = float(peak_ctx.get("macro_share", 0.0))
    macro_like = bool(t & {"FOMC", "CPI", "MACRO"})
    company_like = bool(t & {"EARNINGS", "ANALYST", "PRODUCT_AI", "LEGAL", "LAYOFFS_DELIVERIES", "TECHNICAL_FLOW"})

    if primary == "macro" or macro_share >= 0.55:
        if macro_like:
            return 1.0
        if company_like:
            return 0.35
        return 0.55
    if primary == "technical":
        if company_like:
            return 0.88
        if macro_like:
            return 0.40
        return 0.62
    if primary in {"price", "volume", "volatility"}:
        if company_like:
            return 0.82
        if macro_like and macro_share < 0.35:
            return 0.45
        return 0.60
    return 0.60


def _market_consistency(sent_label: str, sent_score: float, direction: str, subtype: str, themes: List[str]) -> float:
    base = _sentiment_alignment(sent_label, direction, sent_score)
    t = set(themes)
    st = str(subtype or "other")

    if st == "regime-shift" and t & {"FOMC", "CPI", "MACRO"}:
        base = max(base, 0.78)
    if st == "volume-burst" and t & {"TECHNICAL_FLOW", "ANALYST", "EARNINGS", "PRODUCT_AI"}:
        base = min(1.0, base + 0.10)
    if direction == "down" and t & {"LEGAL", "LAYOFFS_DELIVERIES"}:
        base = min(1.0, base + 0.10)
    if direction == "up" and t & {"EARNINGS", "ANALYST", "PRODUCT_AI"}:
        base = min(1.0, base + 0.08)
    if direction == "up" and str(sent_label or "").lower() == "negative" and t & {"LEGAL", "LAYOFFS_DELIVERIES", "MACRO", "FOMC", "CPI"}:
        base = max(0.20, base - 0.08)
    return float(max(0.0, min(1.0, base)))


def _rank_support(rank: int) -> float:
    try:
        r = int(rank)
    except Exception:
        return 0.0
    if r <= 1:
        return 1.0
    if r == 2:
        return 0.85
    if r == 3:
        return 0.70
    if r <= 5:
        return 0.55
    return 0.20


def _gain01(x: float, scale: float = 0.12) -> float:
    return float(max(0.0, min(1.0, x / scale)))


def _support_level(cf_strength: float, ambiguity_risk: float, content_source_type: str, top1_confidence_score: float) -> str:
    c = str(content_source_type or "")
    if cf_strength >= 0.72 and ambiguity_risk <= 0.35 and c in {"full_text", "summary"} and top1_confidence_score >= 0.62:
        return "strong"
    if cf_strength >= 0.55 and ambiguity_risk <= 0.55 and c != "title_only" and top1_confidence_score >= 0.50:
        return "moderate"
    return "weak"


def _ollama_embed(text: str) -> np.ndarray:
    url = str(config.OLLAMA_BASE_URL).rstrip("/") + "/api/embeddings"
    payload = {"model": config.OLLAMA_EMBED_MODEL, "prompt": text}
    r = requests.post(url, json=payload, timeout=EMBED_TIMEOUT_SEC)
    if r.status_code != 200:
        fail(f"Ollama embeddings failed {r.status_code}: {r.text[:200]}")
    j = r.json()
    emb = j.get("embedding")
    require(isinstance(emb, list) and len(emb) > 0, "Ollama embedding response missing 'embedding'.")
    return np.asarray(emb, dtype=np.float32)


def _preflight_ollama() -> None:
    url = str(config.OLLAMA_BASE_URL).rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=15)
        require(r.status_code == 200, f"Ollama not reachable: GET /api/tags -> {r.status_code}")
    except Exception as e:
        fail(f"Ollama not reachable at {config.OLLAMA_BASE_URL}. Start Ollama or fix OLLAMA_BASE_URL. Err={e}")


def _load_finbert():
    try:
        from transformers import pipeline
    except Exception:
        fail("Missing transformers. Install: pip install transformers")

    try:
        pipe = pipeline("sentiment-analysis", model=FINBERT_MODEL)
        return pipe
    except Exception as e:
        fail(f"FinBERT load failed (need internet on first run or cached model). Err={e}")


def _choose_event_terms(sub_x: pd.DataFrame, peak_row: pd.Series) -> List[str]:
    event_terms: List[str] = []

    subtype = str(peak_row.get("episode_subtype", peak_row.get("subtype", ""))).replace("-", " ")
    if subtype and subtype not in {"other", "nan"}:
        event_terms.append(subtype)

    direction = str(peak_row.get("episode_direction", "")).strip().lower()
    if direction == "up":
        event_terms.extend(["upward move", "rally"])
    elif direction == "down":
        event_terms.extend(["selloff", "downward move"])

    sx = sub_x.sort_values("rank").copy()
    sx["feature_group"] = sx["feature"].map(_feature_group)

    picked = []
    for row in sx.itertuples(index=False):
        feat = str(row.feature)
        grp = _feature_group(feat)
        if grp == "macro":
            continue
        pretty = _feature_pretty(feat)
        if pretty not in picked:
            picked.append(pretty)
        if len(picked) >= 3:
            break
    if picked:
        event_terms.extend(picked)

    macro_feats = [str(x) for x in sx.loc[sx["feature_group"] == "macro", "feature"].head(2).tolist()]
    macro_share = float((sx["feature_group"] == "macro").mean()) if len(sx) else 0.0
    if macro_feats and (macro_share >= 0.50 or not picked):
        for mf in macro_feats:
            for phrase in MACRO_FEATURE_PHRASES.get(mf, []):
                event_terms.append(phrase)

    if not event_terms:
        event_terms.extend([_feature_pretty(str(x)) for x in sx.head(2)["feature"].tolist()])

    out = []
    seen = set()
    for t in event_terms:
        tt = _clean_text(t)
        key = tt.lower()
        if tt and key not in seen:
            seen.add(key)
            out.append(tt)
    return out


def _score_candidates(
    cand: pd.DataFrame,
    *,
    peak_end: datetime,
    q_full_emb: np.ndarray,
    q_company_emb: np.ndarray,
    q_event_emb: np.ndarray,
    query_full: str,
    query_event: str,
    embed_cache: Dict[str, List[float]],
    peak_ctx: Dict[str, Any],
    score_prefix: str = "",
) -> pd.DataFrame:
    rows = []
    for rr in cand.itertuples(index=False):
        title_id = rr.title_embedding_id
        evidence_id = rr.evidence_embedding_id
        title_emb = np.asarray(embed_cache[title_id], dtype=np.float32)
        evidence_emb = np.asarray(embed_cache[evidence_id], dtype=np.float32)

        sim_title = _cosine(q_full_emb, title_emb)
        sim_evidence = _cosine(q_full_emb, evidence_emb)
        sim_company = _cosine(q_company_emb, evidence_emb)
        sim_event = _cosine(q_event_emb, evidence_emb)

        title_text = getattr(rr, "title_clean", None) or rr.title
        evidence_text = getattr(rr, "evidence_text", None) or rr.title
        overlap_title = _keyword_overlap(query_event, title_text)
        overlap_evidence = _keyword_overlap(query_full, evidence_text)

        rec = 0.0 if rr.published_at is None else _recency_weight(rr.published_at, peak_end)
        domain_score = _domain_score(getattr(rr, "domain", ""), getattr(rr, "domain_tier", "unknown"))
        body_support = _body_support(getattr(rr, "content_source_type", "title_only"))
        evidence_len = _evidence_len_norm(evidence_text)
        sent_label = getattr(rr, "sent_label", "")
        sent_score = getattr(rr, "sent_score", np.nan)
        sent_align = _sentiment_alignment(sent_label, peak_ctx.get("direction", "flat"), sent_score)
        themes = _infer_article_themes(title_text, evidence_text)
        theme_join = "|".join(themes)
        event_window_match = _event_window_match(themes, peak_ctx)
        xai_consistency = _xai_consistency(themes, peak_ctx)
        market_consistency = _market_consistency(
            sent_label,
            sent_score,
            peak_ctx.get("direction", "flat"),
            peak_ctx.get("subtype", "other"),
            themes,
        )

        generic_wrap_flag = int(_is_generic_price_wrap(title_text, evidence_text, themes, event_window_match))
        holdings_flag = int(_is_holdings_boilerplate(title_text, evidence_text))
        direction_penalty = _direction_mismatch_penalty(title_text, evidence_text, peak_ctx.get("direction", "flat"))

        penalty_generic = 0.16 * generic_wrap_flag
        penalty_holdings = 0.30 * holdings_flag
        penalty_direction = 0.28 * direction_penalty
        retrieval_score = (
            0.13 * sim_title
            + 0.22 * sim_evidence
            + 0.09 * sim_company
            + 0.12 * sim_event
            + 0.05 * overlap_title
            + 0.07 * overlap_evidence
            + 0.04 * rec
            + 0.06 * domain_score
            + 0.06 * body_support
            + 0.07 * market_consistency
            + 0.05 * event_window_match
            + 0.04 * xai_consistency
            - penalty_generic
            - penalty_holdings
            - penalty_direction
        )
        confidence_score = (
            0.24 * body_support
            + 0.18 * domain_score
            + 0.18 * market_consistency
            + 0.14 * event_window_match
            + 0.12 * xai_consistency
            + 0.08 * sent_align
            + 0.06 * evidence_len
            - 0.08 * generic_wrap_flag
            - 0.18 * holdings_flag
            - 0.14 * direction_penalty
        )
        confidence_score = float(max(0.0, min(1.0, confidence_score)))

        rows.append({
            f"{score_prefix}sim_title": sim_title,
            f"{score_prefix}sim_evidence": sim_evidence,
            f"{score_prefix}sim_company": sim_company,
            f"{score_prefix}sim_event": sim_event,
            f"{score_prefix}overlap_title": overlap_title,
            f"{score_prefix}overlap_evidence": overlap_evidence,
            f"{score_prefix}recency_w": rec,
            f"{score_prefix}domain_score": domain_score,
            f"{score_prefix}body_support": body_support,
            f"{score_prefix}evidence_len_norm": evidence_len,
            f"{score_prefix}sentiment_align": sent_align,
            f"{score_prefix}market_consistency": market_consistency,
            f"{score_prefix}event_window_match": event_window_match,
            f"{score_prefix}xai_consistency": xai_consistency,
            f"{score_prefix}article_themes": theme_join,
            f"{score_prefix}retrieval_score": retrieval_score,
            f"{score_prefix}confidence_score": confidence_score,
            f"{score_prefix}generic_wrap_flag": generic_wrap_flag,
            f"{score_prefix}holdings_flag": holdings_flag,
            f"{score_prefix}direction_mismatch_penalty": direction_penalty,
            f"{score_prefix}canonical_url": getattr(rr, "canonical_url", _canonical_url(getattr(rr, "url", ""))),
        })

    out = cand.reset_index(drop=True).copy()
    extra = pd.DataFrame(rows)
    for c in extra.columns:
        out[c] = extra[c]
    return out


def main() -> None:
    run_dir = _run_dir_latest()
    log_path = run_dir / "logs_step09_event_grounding.txt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")

    logger.info("=== Step 09: Event grounding (article-aware, peaks only) ===")

    anomalies_path = config.RESULTS_DIR / "anomalies_base.csv"
    xai_path = config.XAI_FEATURE_PATH
    news_path = config.RESULTS_DIR / "news_enriched.parquet"
    m6_path = run_dir / "manifest_step06.json"

    require(anomalies_path.exists(), "Missing anomalies_base.csv (Step 06).")
    require(xai_path.exists(), "Missing xai_feature.csv (Step 07).")
    require(news_path.exists(), "Missing news_enriched.parquet (Step 08).")
    require(m6_path.exists(), "Missing manifest_step06.json.")

    m6 = json.loads(m6_path.read_text(encoding="utf-8"))
    main_flag_col = m6.get("main_flag_col")
    require(isinstance(main_flag_col, str) and main_flag_col, "manifest_step06.json missing main_flag_col.")

    _preflight_ollama()
    finbert = _load_finbert() if ENABLE_FINBERT else None

    anomalies = read_csv(anomalies_path)
    anomalies["date"] = pd.to_datetime(anomalies["date"])
    require_columns(anomalies, ["ticker", "date", "is_episode_peak", main_flag_col], where="anomalies_base")
    require_unique_key(anomalies, ["ticker", "date"], where="anomalies_base")

    a0 = pd.to_datetime(config.ANALYSIS_START)
    a1 = pd.to_datetime(config.ANALYSIS_END)
    peaks = anomalies[
        (anomalies["date"] >= a0)
        & (anomalies["date"] <= a1)
        & (anomalies["is_episode_peak"] == 1)
        & (anomalies[main_flag_col] == 1)
    ].copy()
    require(len(peaks) > 0, "No peaks in analysis window. Check Step 06 episode peaks.")

    xai = read_csv(xai_path)
    xai["date"] = pd.to_datetime(xai["date"])
    require_columns(xai, ["ticker", "date", "rank", "feature", "importance_norm"], where="xai_feature")

    ev_path = config.RESULTS_DIR / "event_windows.csv"
    ev = read_csv(ev_path) if ev_path.exists() else pd.DataFrame(columns=["event_type", "ticker", "start_date", "end_date"])
    if len(ev) > 0:
        require_columns(ev, ["event_type", "ticker", "start_date", "end_date"], where="event_windows")
        ev["start_date"] = pd.to_datetime(ev["start_date"]).dt.normalize()
        ev["end_date"] = pd.to_datetime(ev["end_date"]).dt.normalize()
        ev["ticker"] = ev["ticker"].astype(str).str.upper()
        ev["event_type"] = ev["event_type"].astype(str).str.upper()

    news = read_parquet(news_path).copy()
    require_columns(news, ["ticker", "peak_date", "seendate", "title", "url", "domain", "evidence_text", "content_source_type"], where="news_enriched")
    news["peak_date"] = pd.to_datetime(news["peak_date"])

    # Research-grade contract: final Step08 evidence must contain no title_only.
    allowed_types = {"full_text", "summary"}
    bad_types = sorted({str(x) for x in news["content_source_type"].dropna().astype(str).tolist() if str(x) not in allowed_types})
    require(len(bad_types) == 0, f"news_enriched.parquet contains disallowed content_source_type values: {bad_types}. Re-run Step08 clean evidence pipeline.")

    cov = news.groupby(["ticker", "peak_date"]).size()
    missing = []
    for r in peaks.itertuples(index=False):
        k = (r.ticker, pd.to_datetime(r.date))
        if k not in cov.index:
            missing.append(f"{r.ticker}_{pd.to_datetime(r.date).date().isoformat()}")
    require(len(missing) == 0, f"Missing news for peaks: {missing[:10]} (and {len(missing)} total). Fix Step08 first.")

    news["published_at"] = [_parse_seendate(x) for x in news["seendate"].tolist()]
    viol = 0
    for rr in news.itertuples(index=False):
        if rr.published_at is None:
            continue
        pe = _peak_end(pd.to_datetime(rr.peak_date).date().isoformat())
        if rr.published_at > pe:
            viol += 1
    require(viol == 0, f"Step09 no-leak check failed: {viol} rows have published_at > peak_end. Re-run Step08.")

    labels, scores = [], []
    if ENABLE_FINBERT:
        for rr in news.itertuples(index=False):
            sent_text = _truncate_text(getattr(rr, "evidence_text", "") or getattr(rr, "article_summary", "") or rr.title, 512)
            if not sent_text:
                labels.append("")
                scores.append(np.nan)
                continue
            out = finbert(sent_text)[0]
            labels.append(out.get("label", ""))
            scores.append(float(out.get("score", np.nan)))
    else:
        labels = [""] * len(news)
        scores = [np.nan] * len(news)
    news["sent_label"] = labels
    news["sent_score"] = scores

    embed_cache: Dict[str, List[float]] = {}
    if EMBED_CACHE_PATH.exists():
        try:
            embed_cache = json.loads(EMBED_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            warn("Embedding cache exists but failed to load. Rebuilding cache from scratch.")
            embed_cache = {}

    news["title_clean"] = news["title_clean"].fillna(news["title"]).astype(str) if "title_clean" in news.columns else news["title"].fillna("").astype(str)
    news["evidence_text"] = news["evidence_text"].fillna(news.get("article_summary", news["title"])).astype(str)
    news["title_embedding_id"] = [
        _hash_id(str(u), _clean_text(str(t)), "title") for u, t in zip(news["url"].tolist(), news["title_clean"].tolist())
    ]
    news["evidence_embedding_id"] = [
        _hash_id(str(u), _truncate_text(_clean_text(str(t))), "evidence") for u, t in zip(news["url"].tolist(), news["evidence_text"].tolist())
    ]

    missing_ids = []
    for rr in news.itertuples(index=False):
        if rr.title_embedding_id not in embed_cache:
            missing_ids.append((rr.title_embedding_id, _clean_text(rr.title_clean)))
        if rr.evidence_embedding_id not in embed_cache:
            missing_ids.append((rr.evidence_embedding_id, _truncate_text(rr.evidence_text)))

    dedup_missing: Dict[str, str] = {}
    for eid, text in missing_ids:
        if eid not in dedup_missing:
            dedup_missing[eid] = text

    if not EMBED_CACHE_PATH.exists():
        write_json({}, EMBED_CACHE_PATH)

    logger.info(f"Embedding cache size: {len(embed_cache)} | missing embeddings: {len(dedup_missing)}")
    for i, (eid, text) in enumerate(dedup_missing.items(), start=1):
        t0 = time.time()
        emb = _ollama_embed(text)
        dt = time.time() - t0
        embed_cache[eid] = emb.astype(float).tolist()
        if i <= 5 or i % 10 == 0 or i == len(dedup_missing):
            logger.info(f"embed {i}/{len(dedup_missing)} | last_sec={dt:.2f} | cache_size={len(embed_cache)}")
        if i % 5 == 0 or i == len(dedup_missing):
            write_json(embed_cache, EMBED_CACHE_PATH)
        time.sleep(0.10)

    out_attr = []
    out_cf = []

    for r in peaks.itertuples(index=False):
        tkr = r.ticker
        dt = pd.to_datetime(r.date)
        peak_date_iso = dt.date().isoformat()
        peak_end = _peak_end(peak_date_iso)
        peak_row = pd.Series(r._asdict())

        sub_x = xai[(xai["ticker"] == tkr) & (xai["date"] == dt)].sort_values("rank")
        require(len(sub_x) > 0, f"Missing xai rows for peak {tkr} {peak_date_iso}. Run Step07 again.")
        top_feats = sub_x.head(QUERY_TOP_FEATURES)["feature"].astype(str).tolist()
        top_feats_pretty = [_feature_pretty(x) for x in top_feats]
        event_terms = _choose_event_terms(sub_x, peak_row)

        company = COMPANY_NAMES.get(tkr, tkr)
        window_types = _event_window_types_for_peak(ev, tkr, dt.normalize())
        peak_ctx = _build_peak_context(sub_x, peak_row, window_types)

        query_company = f"{tkr} {company}".strip()
        query_event = " ".join([query_company] + event_terms).strip()
        non_macro_pretty = [_feature_pretty(x) for x in top_feats if _feature_group(x) != "macro"]
        macro_pretty = [_feature_pretty(x) for x in top_feats if _feature_group(x) == "macro"]
        query_nomacro = " ".join([query_company] + event_terms + non_macro_pretty).strip() or query_event
        query_xai = " ".join([query_company] + top_feats_pretty).strip()
        query_full = " ".join([query_company] + event_terms + non_macro_pretty + macro_pretty).strip()

        q_full_emb = _ollama_embed(query_full)
        q_company_emb = _ollama_embed(query_company)
        q_event_emb = _ollama_embed(query_event)
        q_nomacro_emb = _ollama_embed(query_nomacro)
        q_xai_emb = _ollama_embed(query_xai)

        cand = news[(news["ticker"] == tkr) & (news["peak_date"] == dt)].copy()
        require(len(cand) > 0, f"No news candidates for {tkr} {peak_date_iso}. Fix Step08 coverage.")
        cand["episode_direction"] = peak_row.get("episode_direction", "flat")
        cand = _dedup_candidates(cand)

        cand_full = _score_candidates(
            cand,
            peak_end=peak_end,
            q_full_emb=q_full_emb,
            q_company_emb=q_company_emb,
            q_event_emb=q_event_emb,
            query_full=query_full,
            query_event=query_event,
            embed_cache=embed_cache,
            peak_ctx=peak_ctx,
            score_prefix="",
        )
        scores_np = cand_full["retrieval_score"].to_numpy(dtype=np.float64)
        cand_full["contribution"] = _softmax(scores_np)
        cand_full = cand_full.sort_values("retrieval_score", ascending=False).reset_index(drop=True)
        cand_full["rank_news"] = np.arange(1, len(cand_full) + 1)

        q_zero = np.zeros_like(q_full_emb)
        cand_company = _score_candidates(
            cand,
            peak_end=peak_end,
            q_full_emb=q_company_emb,
            q_company_emb=q_company_emb,
            q_event_emb=q_zero,
            query_full=query_company,
            query_event=query_company,
            embed_cache=embed_cache,
            peak_ctx=peak_ctx,
            score_prefix="company_",
        ).sort_values("company_retrieval_score", ascending=False).reset_index(drop=True)
        cand_company["company_rank"] = np.arange(1, len(cand_company) + 1)

        cand_event = _score_candidates(
            cand,
            peak_end=peak_end,
            q_full_emb=q_event_emb,
            q_company_emb=q_company_emb,
            q_event_emb=q_event_emb,
            query_full=query_event,
            query_event=query_event,
            embed_cache=embed_cache,
            peak_ctx=peak_ctx,
            score_prefix="event_",
        ).sort_values("event_retrieval_score", ascending=False).reset_index(drop=True)
        cand_event["event_rank"] = np.arange(1, len(cand_event) + 1)

        cand_nomacro = _score_candidates(
            cand,
            peak_end=peak_end,
            q_full_emb=q_nomacro_emb,
            q_company_emb=q_company_emb,
            q_event_emb=q_event_emb,
            query_full=query_nomacro,
            query_event=query_event,
            embed_cache=embed_cache,
            peak_ctx=peak_ctx,
            score_prefix="nomacro_",
        ).sort_values("nomacro_retrieval_score", ascending=False).reset_index(drop=True)
        cand_nomacro["nomacro_rank"] = np.arange(1, len(cand_nomacro) + 1)

        cand_xai = _score_candidates(
            cand,
            peak_end=peak_end,
            q_full_emb=q_xai_emb,
            q_company_emb=q_company_emb,
            q_event_emb=q_zero,
            query_full=query_xai,
            query_event=query_xai,
            embed_cache=embed_cache,
            peak_ctx=peak_ctx,
            score_prefix="xai_",
        ).sort_values("xai_retrieval_score", ascending=False).reset_index(drop=True)
        cand_xai["xai_rank"] = np.arange(1, len(cand_xai) + 1)

        topk = cand_full.head(TOPK_NEWS).copy()
        top1_full = topk.iloc[0] if len(topk) > 0 else None
        require(top1_full is not None, f"No top1 article for {tkr} {peak_date_iso}")
        require(str(top1_full.get("content_source_type", "")) in allowed_types, f"Top1 evidence for {tkr} {peak_date_iso} is not summary/full_text.")
        cluster_consensus, cluster_support_domains, cluster_support_sum = _cluster_consensus(topk, top1_full, window_types)
        for row in topk.itertuples(index=False):
            out_attr.append(
                {
                    "ticker": tkr,
                    "peak_date": peak_date_iso,
                    "rank_news": int(row.rank_news),
                    "retrieval_score": float(row.retrieval_score),
                    "contribution": float(row.contribution),
                    "sim_title": float(row.sim_title),
                    "sim_evidence": float(row.sim_evidence),
                    "sim_company": float(row.sim_company),
                    "sim_event": float(row.sim_event),
                    "overlap_title": float(row.overlap_title),
                    "overlap_evidence": float(row.overlap_evidence),
                    "recency_w": float(row.recency_w),
                    "domain_score": float(row.domain_score),
                    "body_support": float(row.body_support),
                    "evidence_len_norm": float(row.evidence_len_norm),
                    "market_consistency": float(row.market_consistency),
                    "event_window_match": float(row.event_window_match),
                    "xai_consistency": float(row.xai_consistency),
                    "confidence_score": float(row.confidence_score),
                    "sentiment_align": float(row.sentiment_align),
                    "article_themes": getattr(row, "article_themes", ""),
                    "peak_event_window_types": "|".join(window_types),
                    "peak_primary_group": peak_ctx.get("primary_group", "other"),
                    "peak_macro_share": float(peak_ctx.get("macro_share", 0.0)),
                    "sent_label": getattr(row, "sent_label", ""),
                    "sent_score": float(getattr(row, "sent_score", np.nan)) if not pd.isna(getattr(row, "sent_score", np.nan)) else np.nan,
                    "domain": row.domain,
                    "domain_tier": getattr(row, "domain_tier", "unknown"),
                    "seendate": row.seendate,
                    "title": row.title,
                    "article_summary": getattr(row, "article_summary", ""),
                    "evidence_text": getattr(row, "evidence_text", ""),
                    "evidence_excerpt": _truncate_text(getattr(row, "evidence_text", ""), 320),
                    "content_source_type": getattr(row, "content_source_type", "title_only"),
                    "title_only": 0,
                    "url": row.url,
                    "final_url": getattr(row, "final_url", row.url),
                    "canonical_url": getattr(row, "canonical_url", _canonical_url(row.url)),
                    "generic_wrap_flag": int(getattr(row, "generic_wrap_flag", 0)),
                    "holdings_flag": int(getattr(row, "holdings_flag", 0)),
                    "direction_mismatch_penalty": float(getattr(row, "direction_mismatch_penalty", 0.0)),
                    "event_cluster_consensus": float(cluster_consensus),
                    "event_cluster_support_domains": int(cluster_support_domains),
                    "event_cluster_support_sum": float(cluster_support_sum),
                    "top_features_used": "|".join(top_feats),
                    "top_features_pretty": "|".join(top_feats_pretty),
                    "event_terms_used": "|".join(event_terms),
                    "query_company": query_company,
                    "query_event": query_event,
                    "query_nomacro": query_nomacro,
                    "query_xai": query_xai,
                    "query_full": query_full,
                }
            )

        top1_url_full = top1_full["url"]
        top1_score_full = float(top1_full["retrieval_score"])
        top2_score_full = float(cand_full.iloc[1]["retrieval_score"]) if len(cand_full) > 1 else top1_score_full
        top1_margin_full = max(0.0, top1_score_full - top2_score_full)

        row_company = cand_company[cand_company["url"] == top1_url_full]
        row_event = cand_event[cand_event["url"] == top1_url_full]
        row_nomacro = cand_nomacro[cand_nomacro["url"] == top1_url_full]
        row_xai = cand_xai[cand_xai["url"] == top1_url_full]
        top1_company = cand_company.iloc[0]["url"] if len(cand_company) > 0 else ""
        top1_event = cand_event.iloc[0]["url"] if len(cand_event) > 0 else ""
        top1_nomacro = cand_nomacro.iloc[0]["url"] if len(cand_nomacro) > 0 else ""
        top1_xai = cand_xai.iloc[0]["url"] if len(cand_xai) > 0 else ""

        company_rank = int(row_company.iloc[0]["company_rank"]) if len(row_company) > 0 else len(cand_company) + 1
        event_rank = int(row_event.iloc[0]["event_rank"]) if len(row_event) > 0 else len(cand_event) + 1
        nomacro_rank = int(row_nomacro.iloc[0]["nomacro_rank"]) if len(row_nomacro) > 0 else len(cand_nomacro) + 1
        xai_rank = int(row_xai.iloc[0]["xai_rank"]) if len(row_xai) > 0 else len(cand_xai) + 1

        company_score_top1 = float(row_company.iloc[0]["company_retrieval_score"]) if len(row_company) > 0 else 0.0
        event_score_top1 = float(row_event.iloc[0]["event_retrieval_score"]) if len(row_event) > 0 else 0.0
        nomacro_score_top1 = float(row_nomacro.iloc[0]["nomacro_retrieval_score"]) if len(row_nomacro) > 0 else 0.0
        xai_score_top1 = float(row_xai.iloc[0]["xai_retrieval_score"]) if len(row_xai) > 0 else 0.0

        drop_company = max(0.0, top1_score_full - company_score_top1)
        drop_event = max(0.0, top1_score_full - event_score_top1)
        drop_nomacro = max(0.0, top1_score_full - nomacro_score_top1)
        drop_xai = max(0.0, top1_score_full - xai_score_top1)

        stability_score = float(np.mean([
            _rank_support(company_rank),
            _rank_support(event_rank),
            _rank_support(nomacro_rank),
            _rank_support(xai_rank),
        ]))
        changed_any = float((top1_url_full != top1_company) or (top1_url_full != top1_event) or (top1_url_full != top1_xai))
        specificity_score = float(max(0.0, min(1.0,
            0.25 * _gain01(drop_company)
            + 0.12 * _gain01(drop_event)
            + 0.12 * _gain01(drop_xai)
            + 0.10 * changed_any
            + 0.41 * cluster_consensus
        )))
        macro_reliance = _gain01(drop_nomacro)
        margin_score = _gain01(top1_margin_full, scale=0.08)
        alignment_score = float(
            0.34 * float(top1_full.get("market_consistency", 0.5))
            + 0.22 * float(top1_full.get("event_window_match", 0.5))
            + 0.18 * float(top1_full.get("xai_consistency", 0.5))
            + 0.10 * float(top1_full.get("sentiment_align", 0.5))
            + 0.16 * (1.0 - float(top1_full.get("direction_mismatch_penalty", 0.0)))
        )
        evidence_score = float(
            0.35 * float(top1_full.get("body_support", 0.0))
            + 0.18 * float(top1_full.get("domain_score", 0.0))
            + 0.10 * float(top1_full.get("evidence_len_norm", 0.0))
            + 0.17 * float(top1_full.get("confidence_score", 0.0))
            + 0.20 * cluster_consensus
        )
        article_penalty = float(max(
            0.28 * float(top1_full.get("generic_wrap_flag", 0)),
            0.48 * float(top1_full.get("holdings_flag", 0)),
            0.42 * float(top1_full.get("direction_mismatch_penalty", 0.0)),
        ))
        ambiguity_risk = float(max(0.0, min(1.0,
            0.35 * (1.0 - margin_score)
            + 0.18 * (1.0 - stability_score)
            + 0.15 * macro_reliance
            + 0.17 * (1.0 - cluster_consensus)
            + 0.15 * article_penalty
        )))
        cf_strength = float(max(0.0, min(1.0,
            0.23 * stability_score
            + 0.18 * specificity_score
            + 0.12 * margin_score
            + 0.19 * alignment_score
            + 0.14 * evidence_score
            + 0.10 * cluster_consensus
            + 0.08 * (1.0 - macro_reliance)
            - 0.12 * article_penalty
        )))
        cf_support_level = _support_level(
            cf_strength,
            ambiguity_risk,
            str(top1_full.get("content_source_type", "title_only")),
            float(top1_full.get("confidence_score", 0.0)),
        )

        out_cf.append(
            {
                "ticker": tkr,
                "peak_date": peak_date_iso,
                "top_features_used": "|".join(top_feats),
                "event_terms_used": "|".join(event_terms),
                "query_company": query_company,
                "query_event": query_event,
                "query_nomacro": query_nomacro,
                "query_xai": query_xai,
                "query_full": query_full,
                "top1_url_full": top1_url_full,
                "top1_url_company_only": top1_company,
                "top1_url_event_only": top1_event,
                "top1_url_nomacro": top1_nomacro,
                "top1_url_xai_only": top1_xai,
                "top1_changed": int(top1_url_full != top1_company),
                "top1_changed_company": int(top1_url_full != top1_company),
                "top1_changed_event": int(top1_url_full != top1_event),
                "top1_changed_nomacro": int(top1_url_full != top1_nomacro),
                "top1_changed_xai": int(top1_url_full != top1_xai),
                "top1_rank_company_only": company_rank,
                "top1_rank_event_only": event_rank,
                "top1_rank_nomacro": nomacro_rank,
                "top1_rank_xai_only": xai_rank,
                "top1_score_full": top1_score_full,
                "top1_score_company_only": company_score_top1,
                "top1_score_event_only": event_score_top1,
                "top1_score_nomacro": nomacro_score_top1,
                "top1_score_xai_only": xai_score_top1,
                "top1_margin_full": top1_margin_full,
                "cf_top1_contribution_proxy": float(top1_full["contribution"]),
                "cf_drop_proxy": max(drop_company, drop_event, drop_xai),
                "cf_drop_company_only": drop_company,
                "cf_drop_event_only": drop_event,
                "cf_drop_nomacro": drop_nomacro,
                "cf_drop_xai_only": drop_xai,
                "cf_stability": stability_score,
                "cf_specificity": specificity_score,
                "cf_cluster_consensus": float(cluster_consensus),
                "cf_cluster_support_domains": int(cluster_support_domains),
                "cf_cluster_support_sum": float(cluster_support_sum),
                "cf_margin_score": margin_score,
                "cf_alignment": alignment_score,
                "cf_evidence_score": evidence_score,
                "cf_macro_reliance": macro_reliance,
                "cf_article_penalty": article_penalty,
                "cf_ambiguity_risk": ambiguity_risk,
                "cf_strength": cf_strength,
                "cf_support_level": cf_support_level,
                "topk": int(TOPK_NEWS),
            }
        )

    require(len(out_attr) > 0, "event_attribution empty. Check joins and Step08 outputs.")
    attr_df = pd.DataFrame(out_attr)
    cf_df = pd.DataFrame(out_cf)

    # Final safety: no title_only should appear in ranked outputs.
    if (attr_df["content_source_type"].astype(str) == "title_only").any():
        fail("event_attribution contains title_only rows. Step09 must rank only summary/full_text evidence.")

    write_parquet(news, NEWS_ENRICHED_PATH)
    write_csv(attr_df, ATTR_PATH)
    write_csv(cf_df, CF_PATH)

    manifest = {
        "step": "09_event_grounding",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "anomalies_base": str(anomalies_path),
            "anomalies_base_sha256": sha256_file(anomalies_path),
            "xai_feature": str(xai_path),
            "xai_feature_sha256": sha256_file(xai_path),
            "news_enriched": str(news_path),
            "news_enriched_sha256": sha256_file(news_path),
            "embed_cache": str(EMBED_CACHE_PATH),
            "manifest_step06": str(m6_path),
        },
        "params": {
            "topk_news": TOPK_NEWS,
            "query_top_features": QUERY_TOP_FEATURES,
            "embed_model": config.OLLAMA_EMBED_MODEL,
            "finbert_enabled": ENABLE_FINBERT,
            "finbert_model": FINBERT_MODEL,
            "evidence_policy": "final ranking allows only full_text or summary; title_only is disallowed",
            "scoring": {
                "sim_title": 0.13,
                "sim_evidence": 0.22,
                "sim_company": 0.09,
                "sim_event": 0.12,
                "overlap_title": 0.05,
                "overlap_evidence": 0.07,
                "recency": 0.04,
                "domain": 0.06,
                "body_support": 0.06,
                "market_consistency": 0.07,
                "event_window_match": 0.05,
                "xai_consistency": 0.04,
            },
            "counterfactual": "compare full query vs company-only, event-only, no-macro, and xai-only; cf_strength rewards robustness + event-cluster consensus + specificity, with penalties for generic price-wrap, holdings/13F boilerplate, and direction mismatch",
        },
        "outputs": {
            "news_enriched_updated": {"path": str(NEWS_ENRICHED_PATH), "sha256": sha256_file(NEWS_ENRICHED_PATH), "rows": int(len(news))},
            "event_attribution": {"path": str(ATTR_PATH), "sha256": sha256_file(ATTR_PATH), "rows": int(len(attr_df))},
            "event_counterfactual": {"path": str(CF_PATH), "sha256": sha256_file(CF_PATH), "rows": int(len(cf_df))},
        },
        "summary": {
            "peaks_explained": int(peaks[["ticker", "date"]].drop_duplicates().shape[0]),
            "top1_content_type_counts": attr_df.loc[attr_df["rank_news"] == 1, "content_source_type"].astype(str).value_counts().to_dict(),
            "all_ranked_content_type_counts": attr_df["content_source_type"].astype(str).value_counts().to_dict(),
        },
        "notes": {
            "no_leak": "enforces published_at <= peak_end; uses only peak-local news candidates",
            "article_aware": "retrieval uses title + evidence_text + event-window prior + xai/market consistency + domain/body support",
        },
    }
    (run_dir / "manifest_step09.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok(f"Saved enriched news -> {NEWS_ENRICHED_PATH}")
    ok(f"Saved event attribution -> {ATTR_PATH}")
    ok(f"Saved counterfactual -> {CF_PATH}")
    ok("Step 09 DONE. Next: python src/10_evaluate.py")


if __name__ == "__main__":
    main()

