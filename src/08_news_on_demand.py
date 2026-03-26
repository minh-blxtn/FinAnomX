
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse
import sys
import json
import time
import re
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger
import requests
import pandas as pd
from bs4 import BeautifulSoup

# --- allow: python src/08_news_on_demand.py
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, warn, fail, require, require_columns, require_unique_key  # noqa: E402
from utils.io import ensure_dir, read_csv, write_json, write_parquet, sha256_file  # noqa: E402


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_REQUEST_INTERVAL_SEC = 12.0
MAX_RETRIES_429 = 12
ARTICLE_TIMEOUT_SEC = 20
ARTICLE_MIN_INTERVAL_SEC = 0.5
ARTICLE_MIN_FULLTEXT_CHARS = 400
ARTICLE_MIN_SUMMARY_CHARS = 80
ARTICLE_MAX_CHARS = 6000
ARTICLE_MAX_PARAGRAPHS = 20

# --- stricter research defaults ---
FETCH_MAX_RESULTS = max(int(getattr(config, "NEWS_MAX_RESULTS", 7)), 15)
FINAL_ALLOWED_TYPES = {"full_text", "summary"}
STRICT_MIN_EVIDENCE_PER_PEAK = 1

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

COMPANY_TERMS = {
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
}

HIGH_TRUST_DOMAINS = {
    "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com",
    "marketwatch.com", "finance.yahoo.com", "nytimes.com", "theverge.com",
    "seekingalpha.com", "investopedia.com", "fool.com", "benzinga.com",
}

MEDIUM_TRUST_SUFFIXES = (
    ".com", ".org", ".co.uk", ".net",
)


def _run_dir_latest() -> Path:
    p = config.RUNS_DIR / "latest"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _cache_path(ticker: str, date_iso: str) -> Path:
    return config.NEWS_CACHE_DIR / f"{ticker}_{date_iso}.json"


def _parse_seendate(s: str):
    if not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ")
    except Exception:
        return None


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate_text(text: str, limit: int = ARTICLE_MAX_CHARS) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip()


def _extract_meta_content(soup: BeautifulSoup, *names: str) -> str:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            txt = _clean_text(tag.get("content"))
            if txt:
                return txt
    return ""


def _extract_jsonld_text(soup: BeautifulSoup) -> str:
    texts = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        stack = obj if isinstance(obj, list) else [obj]
        for item in stack:
            if not isinstance(item, dict):
                continue
            for key in ["articleBody", "description", "headline", "abstract"]:
                val = item.get(key)
                txt = _clean_text(val)
                if txt and len(txt) >= ARTICLE_MIN_SUMMARY_CHARS:
                    texts.append(txt)
    return _truncate_text(" ".join(texts), ARTICLE_MAX_CHARS)


def _extract_main_paragraphs(soup: BeautifulSoup) -> str:
    for tag_name in ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]:
        for t in soup.find_all(tag_name):
            t.decompose()

    containers = []
    for selector in [
        "article",
        "main",
        "div[itemprop='articleBody']",
        "section[itemprop='articleBody']",
        "div.article-body",
        "div.entry-content",
        "div.post-content",
    ]:
        found = soup.select(selector)
        if found:
            containers.extend(found)
            break

    if not containers:
        body = soup.body
        containers = [body] if body else []

    paras = []
    for container in containers:
        for p in container.find_all(["p", "li"]):
            txt = _clean_text(p.get_text(" ", strip=True))
            if len(txt) >= 40:
                paras.append(txt)
            if len(paras) >= ARTICLE_MAX_PARAGRAPHS:
                break
        if len(paras) >= ARTICLE_MAX_PARAGRAPHS:
            break

    return _truncate_text(" ".join(paras), ARTICLE_MAX_CHARS)


def _domain_tier(domain: str) -> str:
    d = (domain or "").lower().strip()
    if not d:
        return "unknown"
    if d in HIGH_TRUST_DOMAINS:
        return "high"
    if any(d.endswith(sfx) for sfx in MEDIUM_TRUST_SUFFIXES):
        return "medium"
    return "low"


def _try_trafilatura_extract(html: str, url: str) -> Tuple[str, str]:
    try:
        import trafilatura  # type: ignore
    except Exception:
        return "", "trafilatura_unavailable"

    try:
        txt = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            deduplicate=True,
            output_format="txt",
        )
        txt = _truncate_text(_clean_text(txt), ARTICLE_MAX_CHARS)
        if len(txt) >= ARTICLE_MIN_FULLTEXT_CHARS:
            return txt, "trafilatura_full_text"
        if len(txt) >= ARTICLE_MIN_SUMMARY_CHARS:
            return txt, "trafilatura_summary"
        return "", "trafilatura_too_short"
    except Exception as e:
        return "", f"trafilatura_error:{type(e).__name__}"


def _is_summary_candidate(text: str) -> bool:
    text = _clean_text(text)
    return len(text) >= ARTICLE_MIN_SUMMARY_CHARS and len(text.split()) >= 12


def _is_fulltext_candidate(text: str) -> bool:
    text = _clean_text(text)
    return len(text) >= ARTICLE_MIN_FULLTEXT_CHARS and len(text.split()) >= 80


def _extract_article_payload(url: str, title: str, session: requests.Session) -> Dict[str, Any]:
    url = _clean_text(url)
    title = _clean_text(title)
    if not url:
        return {
            "article_title": title,
            "article_text": "",
            "article_summary": "",
            "evidence_text": "",
            "content_source_type": "title_only",
            "content_length_chars": 0,
            "content_length_words": 0,
            "fetch_status": "missing_url",
            "extract_status": "title_only",
            "drop_reason": "missing_url",
            "final_url": "",
        }

    try:
        resp = session.get(url, headers=REQUEST_HEADERS, timeout=ARTICLE_TIMEOUT_SEC, allow_redirects=True)
    except Exception as e:
        return {
            "article_title": title,
            "article_text": "",
            "article_summary": "",
            "evidence_text": "",
            "content_source_type": "title_only",
            "content_length_chars": 0,
            "content_length_words": 0,
            "fetch_status": f"request_error:{type(e).__name__}",
            "extract_status": "title_only",
            "drop_reason": f"request_error:{type(e).__name__}",
            "final_url": url,
        }

    final_url = _clean_text(resp.url or url)
    fetch_status = f"http_{resp.status_code}"
    if resp.status_code != 200:
        return {
            "article_title": title,
            "article_text": "",
            "article_summary": "",
            "evidence_text": "",
            "content_source_type": "title_only",
            "content_length_chars": 0,
            "content_length_words": 0,
            "fetch_status": fetch_status,
            "extract_status": "title_only",
            "drop_reason": fetch_status,
            "final_url": final_url,
        }

    html = resp.text or ""
    if not html.strip():
        return {
            "article_title": title,
            "article_text": "",
            "article_summary": "",
            "evidence_text": "",
            "content_source_type": "title_only",
            "content_length_chars": 0,
            "content_length_words": 0,
            "fetch_status": "empty_html",
            "extract_status": "title_only",
            "drop_reason": "empty_html",
            "final_url": final_url,
        }

    soup = BeautifulSoup(html, "lxml")
    soup_title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    meta_desc = _extract_meta_content(soup, "description", "og:description", "twitter:description")
    jsonld_text = _extract_jsonld_text(soup)
    body_text = _extract_main_paragraphs(soup)
    trafi_text, trafi_status = _try_trafilatura_extract(html, final_url)

    article_title = soup_title or title
    summary_candidates = [meta_desc, jsonld_text]
    fulltext_candidates = [body_text, trafi_text]

    article_summary = ""
    for cand in summary_candidates:
        if _is_summary_candidate(cand):
            article_summary = _truncate_text(cand, min(1200, ARTICLE_MAX_CHARS))
            break
    if not article_summary:
        for cand in fulltext_candidates:
            if _is_summary_candidate(cand):
                article_summary = _truncate_text(cand, min(1200, ARTICLE_MAX_CHARS))
                break

    article_text = ""
    for cand in fulltext_candidates:
        if _is_fulltext_candidate(cand):
            article_text = _truncate_text(cand, ARTICLE_MAX_CHARS)
            break

    if article_text:
        evidence_text = article_text
        source_type = "full_text"
        extract_status = "full_text"
        drop_reason = ""
    elif article_summary:
        evidence_text = article_summary
        source_type = "summary"
        extract_status = "summary_only"
        drop_reason = ""
    else:
        evidence_text = ""
        source_type = "title_only"
        extract_status = "title_only"
        drop_reason = "body_too_short_and_no_summary"

    return {
        "article_title": article_title or title,
        "article_text": article_text,
        "article_summary": article_summary,
        "evidence_text": evidence_text,
        "content_source_type": source_type,
        "content_length_chars": len(evidence_text),
        "content_length_words": len(evidence_text.split()) if evidence_text else 0,
        "fetch_status": fetch_status,
        "extract_status": f"{extract_status}|{trafi_status}",
        "drop_reason": drop_reason,
        "final_url": final_url,
    }


def main() -> None:
    run_dir = _run_dir_latest()
    log_path = run_dir / "logs_step08_news_on_demand.txt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")

    logger.info("=== Step 08: News on demand (GDELT DOC + article evidence) | ANALYSIS PEAKS ONLY | FINAL EVIDENCE REQUIRES SUMMARY/FULL TEXT ===")

    anomalies_path = config.RESULTS_DIR / "anomalies_base.csv"
    require(anomalies_path.exists(), "Missing data/results/anomalies_base.csv. Run Step 06 first.")
    require(config.XAI_FEATURE_PATH.exists(), "Missing data/results/xai_feature.csv. Run Step 07 first.")

    m6_path = run_dir / "manifest_step06.json"
    require(m6_path.exists(), "Missing runs/latest/manifest_step06.json.")
    m6 = json.loads(m6_path.read_text(encoding="utf-8"))
    main_flag_col = m6.get("main_flag_col")
    require(isinstance(main_flag_col, str) and main_flag_col, "manifest_step06.json missing 'main_flag_col'.")

    anomalies = read_csv(anomalies_path)
    anomalies["date"] = pd.to_datetime(anomalies["date"])
    require_columns(anomalies, ["ticker", "date", "is_episode_peak", main_flag_col, "split"], where="anomalies_base")
    require_unique_key(anomalies, ["ticker", "date"], where="anomalies_base")

    a0 = pd.to_datetime(config.ANALYSIS_START)
    a1 = pd.to_datetime(config.ANALYSIS_END)

    peaks = anomalies[
        (anomalies["date"] >= a0)
        & (anomalies["date"] <= a1)
        & (anomalies["is_episode_peak"] == 1)
        & (anomalies[main_flag_col] == 1)
    ].copy()

    require(len(peaks) > 0, "No analysis peaks found. Check Step 06 outputs.")
    ensure_dir(config.NEWS_CACHE_DIR)

    last_request_ts = 0.0
    last_article_ts = 0.0

    def fetch_gdelt(params: dict) -> Optional[requests.Response]:
        nonlocal last_request_ts
        for attempt in range(1, MAX_RETRIES_429 + 1):
            now = time.monotonic()
            if last_request_ts > 0:
                gap = now - last_request_ts
                if gap < MIN_REQUEST_INTERVAL_SEC:
                    time.sleep(MIN_REQUEST_INTERVAL_SEC - gap)

            resp = requests.get(GDELT_DOC_URL, params=params, headers=REQUEST_HEADERS, timeout=30)
            last_request_ts = time.monotonic()

            if resp.status_code == 200:
                return resp

            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                try:
                    base = float(ra) if ra is not None else MIN_REQUEST_INTERVAL_SEC
                except Exception:
                    base = MIN_REQUEST_INTERVAL_SEC
                backoff = max(base, MIN_REQUEST_INTERVAL_SEC) * attempt
                logger.warning(f"429 rate limited. Backoff {backoff:.1f}s (attempt {attempt}/{MAX_RETRIES_429})")
                time.sleep(backoff)
                continue

            fail(f"GDELT request failed {resp.status_code}: {resp.text[:200]}")

        logger.error(f"GDELT 429 after {MAX_RETRIES_429} retries. Skip this peak.")
        return None

    session = requests.Session()
    stats = {
        "peaks": int(len(peaks)),
        "cache_hits": 0,
        "fetched": 0,
        "rows_raw": 0,
        "rows_after_dedup": 0,
        "rows_after_lang_country": 0,
        "rows_after_no_leak": 0,
        "rows_final_evidence": 0,
        "rows_dropped_title_only": 0,
        "rows_dropped_other": 0,
        "full_text": 0,
        "summary": 0,
        "title_only_raw": 0,
        "peaks_with_0_raw": 0,
        "coverage_raw": 0.0,
        "coverage_final": 0.0,
        "fetch_max_results": FETCH_MAX_RESULTS,
    }

    # --- Fetch GDELT cache if missing; then enrich each cached article with body/summary evidence ---
    for r in peaks.itertuples(index=False):
        ticker = r.ticker
        peak_date_iso = pd.to_datetime(r.date).date().isoformat()
        require(ticker in COMPANY_TERMS, f"Ticker {ticker} missing in COMPANY_TERMS mapping.")
        company = COMPANY_TERMS[ticker]

        end_dt = datetime.strptime(peak_date_iso, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        start_dt = end_dt - timedelta(days=int(config.NEWS_LOOKBACK_DAYS))
        query = f'({ticker} OR "{company}")'
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": _fmt_dt(start_dt),
            "enddatetime": _fmt_dt(end_dt),
            "maxrecords": FETCH_MAX_RESULTS,
        }

        cache_fp = _cache_path(ticker, peak_date_iso)
        payload: Dict[str, Any]
        if cache_fp.exists():
            stats["cache_hits"] += 1
            payload = json.loads(cache_fp.read_text(encoding="utf-8"))
        else:
            logger.info(f"[FETCH {ticker}] {peak_date_iso} window={start_dt.date()}..{end_dt.date()} max={FETCH_MAX_RESULTS}")
            resp = fetch_gdelt(params)
            if resp is None:
                continue
            payload = resp.json()
            payload["query"] = query
            stats["fetched"] += 1

        if not payload.get("query"):
            payload["query"] = query

        arts = payload.get("articles", [])
        if not isinstance(arts, list):
            arts = []
            payload["articles"] = arts

        payload_changed = False
        for a in arts[:FETCH_MAX_RESULTS]:
            a["sourceCountry"] = a.get("sourceCountry") or a.get("sourcecountry") or ""
            a["tone"] = a.get("tone") or ""
            a["domain_tier"] = _domain_tier(a.get("domain") or urlparse(a.get("url") or "").netloc)
            a["title_clean"] = _clean_text(a.get("title"))

            if a.get("content_source_type") and a.get("extract_status") and ("drop_reason" in a):
                continue

            now = time.monotonic()
            if last_article_ts > 0:
                gap = now - last_article_ts
                if gap < ARTICLE_MIN_INTERVAL_SEC:
                    time.sleep(ARTICLE_MIN_INTERVAL_SEC - gap)

            extracted = _extract_article_payload(a.get("url", ""), a.get("title", ""), session=session)
            last_article_ts = time.monotonic()
            a.update(extracted)
            a["domain_tier"] = _domain_tier(a.get("domain") or urlparse(a.get("final_url") or a.get("url") or "").netloc)
            payload_changed = True

        if payload_changed or not cache_fp.exists():
            write_json(payload, cache_fp)

    # --- Build dataframe from cache (authoritative) ---
    out_rows = []
    for r in peaks.itertuples(index=False):
        ticker = r.ticker
        split = str(r.split)
        peak_date_iso = pd.to_datetime(r.date).date().isoformat()
        company = COMPANY_TERMS[ticker]
        default_query = f'({ticker} OR "{company}")'
        cache_fp = _cache_path(ticker, peak_date_iso)
        if not cache_fp.exists():
            continue

        payload = json.loads(cache_fp.read_text(encoding="utf-8"))
        arts = payload.get("articles", [])
        if not isinstance(arts, list):
            continue

        if len(arts) == 0:
            stats["peaks_with_0_raw"] += 1

        query = payload.get("query") or default_query

        for a in arts[:FETCH_MAX_RESULTS]:
            evidence_text = _clean_text(a.get("evidence_text"))
            article_summary = _clean_text(a.get("article_summary"))
            article_text = _clean_text(a.get("article_text"))
            title = _clean_text(a.get("title"))
            content_source_type = _clean_text(a.get("content_source_type")) or "title_only"
            final_url = _clean_text(a.get("final_url")) or _clean_text(a.get("url"))
            domain = _clean_text(a.get("domain")) or urlparse(final_url).netloc
            drop_reason = _clean_text(a.get("drop_reason"))
            if content_source_type == "title_only":
                stats["title_only_raw"] += 1

            out_rows.append(
                {
                    "ticker": ticker,
                    "peak_date": peak_date_iso,
                    "split": split,
                    "query": query,
                    "seendate": a.get("seendate"),
                    "title": title,
                    "title_clean": _clean_text(a.get("title_clean")) or title,
                    "url": _clean_text(a.get("url")),
                    "final_url": final_url,
                    "domain": domain,
                    "domain_tier": _clean_text(a.get("domain_tier")) or _domain_tier(domain),
                    "language": a.get("language"),
                    "sourceCountry": a.get("sourceCountry") or a.get("sourcecountry"),
                    "tone": a.get("tone"),
                    "article_title": _clean_text(a.get("article_title")) or title,
                    "article_summary": article_summary,
                    "article_text": article_text,
                    "evidence_text": evidence_text,
                    "content_source_type": content_source_type,
                    "content_length_chars": int(a.get("content_length_chars") or len(evidence_text)),
                    "content_length_words": int(a.get("content_length_words") or len(evidence_text.split())) if evidence_text else 0,
                    "fetch_status": _clean_text(a.get("fetch_status")),
                    "extract_status": _clean_text(a.get("extract_status")),
                    "drop_reason": drop_reason,
                    "title_only": int(content_source_type == "title_only"),
                }
            )

    require(len(out_rows) > 0, "No news rows produced for peaks. Check cache + logs.")
    df_raw = pd.DataFrame(out_rows)
    stats["rows_raw"] = int(len(df_raw))

    if "url" in df_raw.columns:
        df_raw = df_raw.drop_duplicates(subset=["ticker", "peak_date", "url"], keep="first").copy()
    stats["rows_after_dedup"] = int(len(df_raw))

    df_filt = df_raw.copy()
    if getattr(config, "NEWS_LANG", ""):
        df_filt = df_filt[df_filt["language"].fillna("") == config.NEWS_LANG].copy()
    if getattr(config, "NEWS_COUNTRY", ""):
        df_filt = df_filt[df_filt["sourceCountry"].fillna("") == config.NEWS_COUNTRY].copy()
    stats["rows_after_lang_country"] = int(len(df_filt))
    df_used = df_filt if len(df_filt) > 0 else df_raw.copy()

    keep = []
    dropped_no_leak = 0
    for rr in df_used.itertuples(index=False):
        sd = _parse_seendate(rr.seendate)
        if sd is None:
            keep.append(True)
            continue
        peak_end = datetime.strptime(rr.peak_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        if sd > peak_end:
            dropped_no_leak += 1
            keep.append(False)
        else:
            keep.append(True)

    df_used = df_used.loc[keep].copy()
    require(len(df_used) > 0, "All rows dropped by no-leak filter (unexpected).")
    if dropped_no_leak > 0:
        warn(f"No-leak: dropped {dropped_no_leak} rows with seendate > peak_date_end.")
    stats["rows_after_no_leak"] = int(len(df_used))

    # raw coverage after no-leak
    raw_cov_keys = set(df_used.groupby(["ticker", "peak_date"]).size().index.tolist())
    need = {(r.ticker, pd.to_datetime(r.date).date().isoformat()) for r in peaks.itertuples(index=False)}
    stats["coverage_raw"] = float(sum((k in raw_cov_keys) for k in need) / len(need)) if need else 0.0

    # final evidence set: no title_only allowed
    valid_mask = df_used["content_source_type"].isin(FINAL_ALLOWED_TYPES) & (df_used["evidence_text"].fillna("").str.len() >= ARTICLE_MIN_SUMMARY_CHARS)
    df_final = df_used.loc[valid_mask].copy()
    df_drop = df_used.loc[~valid_mask].copy()

    if len(df_drop) > 0:
        df_drop["drop_reason_final"] = df_drop["drop_reason"].where(df_drop["drop_reason"].astype(str).str.len() > 0, "title_only_or_insufficient_evidence")
    else:
        df_drop["drop_reason_final"] = []

    stats["rows_dropped_title_only"] = int((df_drop["content_source_type"] == "title_only").sum()) if len(df_drop) else 0
    stats["rows_dropped_other"] = int(len(df_drop) - stats["rows_dropped_title_only"])
    stats["rows_final_evidence"] = int(len(df_final))
    stats["full_text"] = int((df_final["content_source_type"] == "full_text").sum()) if len(df_final) else 0
    stats["summary"] = int((df_final["content_source_type"] == "summary").sum()) if len(df_final) else 0

    final_cov_keys = set(df_final.groupby(["ticker", "peak_date"]).size().index.tolist()) if len(df_final) else set()
    stats["coverage_final"] = float(sum((k in final_cov_keys) for k in need) / len(need)) if need else 0.0

    # Save audit files first
    raw_path = config.RESULTS_DIR / "news_enriched_raw.parquet"
    dropped_path = config.RESULTS_DIR / "news_dropped_title_only.parquet"
    news_path = config.RESULTS_DIR / "news_enriched.parquet"

    write_parquet(df_used, raw_path)
    if len(df_drop) > 0:
        write_parquet(df_drop, dropped_path)
    else:
        if dropped_path.exists():
            dropped_path.unlink()

    missing_peaks = sorted(list(need - final_cov_keys))
    if missing_peaks:
        sample = ", ".join([f"{t}|{d}" for t, d in missing_peaks[:10]])
        fail(
            "Final evidence set still has peaks without summary/full_text rows. "
            f"Missing {len(missing_peaks)}/{len(need)} peaks. Examples: {sample}. "
            "Improve source extraction or fetch pool before proceeding."
        )

    require(len(df_final) > 0, "No final evidence rows survived after removing title_only.")
    write_parquet(df_final, news_path)

    manifest = {
        "step": "08_news_on_demand",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": "gdelt_doc_plus_article_fetch",
        "scope": "analysis_peaks_only",
        "inputs": {
            "anomalies_base": str(anomalies_path),
            "anomalies_base_sha256": sha256_file(anomalies_path),
            "manifest_step06": str(m6_path),
        },
        "params": {
            "news_lookback_days": int(config.NEWS_LOOKBACK_DAYS),
            "fetch_max_results": FETCH_MAX_RESULTS,
            "configured_news_max_results": int(config.NEWS_MAX_RESULTS),
            "news_lang": getattr(config, "NEWS_LANG", ""),
            "news_country": getattr(config, "NEWS_COUNTRY", ""),
            "final_allowed_types": sorted(FINAL_ALLOWED_TYPES),
            "strict_min_evidence_per_peak": STRICT_MIN_EVIDENCE_PER_PEAK,
            "article_min_fulltext_chars": ARTICLE_MIN_FULLTEXT_CHARS,
            "article_min_summary_chars": ARTICLE_MIN_SUMMARY_CHARS,
        },
        "outputs": {
            "news_enriched_raw": {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": int(len(df_used)),
            },
            "news_enriched": {
                "path": str(news_path),
                "sha256": sha256_file(news_path),
                "rows": int(len(df_final)),
                "coverage_peaks": stats["coverage_final"],
                "stats": stats,
            },
            "news_dropped_title_only": {
                "path": str(dropped_path),
                "sha256": sha256_file(dropped_path) if dropped_path.exists() else "",
                "rows": int(len(df_drop)),
            },
        },
        "notes": {
            "no_leak": "seendate must be <= peak_date_end; article fetch is only for already-discovered GDELT URLs",
            "evidence_text": "final artifact keeps only full_text or summary; title_only stays in audit outputs only",
        },
    }
    write_json(manifest, run_dir / "manifest_step08.json")

    ok(f"Saved raw enriched news -> {raw_path}")
    if len(df_drop) > 0:
        ok(f"Saved dropped title-only audit -> {dropped_path}")
    ok(f"Saved final evidence news -> {news_path}")
    ok(f"Coverage peaks with >=1 final evidence row: {stats['coverage_final']:.3f}")
    ok("Step 08 DONE. Next: python src/09_event_grounding.py")


if __name__ == "__main__":
    main()

