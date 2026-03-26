# src/03_build_event_windows.py
# Step 03 (AUTO: SEC/BLS/Fed)
# - Auto-generate:
#   data/external/event_seed_macro.csv   (CPI + FOMC)
#   data/external/event_seed_earnings.csv (SEC 8-K item 2.02 per ticker)
# - Then build:
#   data/results/event_windows.csv
#
# Notes:
# - No FMP.
# - Polite rate limit + SEC User-Agent required.
# - Windows: earnings [D-1, D+1], macro [D-1, D+1] (configurable below)

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, date
import os
import re
import sys
import time
import json
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# --- allow: python src/03_build_event_windows.py
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, warn, fail, require, require_unique_key  # noqa: E402
from utils.io import ensure_dir, write_csv, write_json, sha256_file  # noqa: E402
from utils.time_utils import parse_ymd  # noqa: E402


# ----------------- SETTINGS -----------------
# Window rules
EARN_LEFT, EARN_RIGHT = 1, 1
MACRO_LEFT, MACRO_RIGHT = 1, 1

# Paths
SEED_DIR = config.DATA_DIR / "external"
EARNINGS_SEED_PATH = SEED_DIR / "event_seed_earnings.csv"
MACRO_SEED_PATH = SEED_DIR / "event_seed_macro.csv"

# Sources
BLS_CPI_ARCHIVE_LIST = "https://www.bls.gov/bls/news-release/cpi.htm"
FED_FOMC_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# SEC endpoints
SEC_TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"

# Polite rate limit
SLEEP_SEC = 0.9

def _browser_headers() -> dict[str, str]:
    # Browser-like headers để tránh 403 kiểu bot
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
    }

def _run_dir_latest() -> Path:
    d = config.RUNS_DIR / "latest"
    ensure_dir(d)
    return d


def _analysis_range() -> tuple[date, date]:
    a0 = parse_ymd(config.ANALYSIS_START).date()
    a1 = parse_ymd(config.ANALYSIS_END).date()
    return a0, a1


def _fetch_text(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> str:
    import requests
    h = headers or _browser_headers()
    r = requests.get(url, headers=h, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
    return r.text


def _fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    import requests
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}: {r.text[:200]}")
    return r.json()


def _sec_headers() -> dict[str, str]:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    require(bool(ua), "Missing SEC_USER_AGENT in .env. Example: SEC_USER_AGENT=FinAnomX Ken your_email@example.com")

    return {
        "User-Agent": ua,
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }


# ----------------- MACRO: CPI -----------------
FRED_API_BASE = "https://api.stlouisfed.org/fred"

def _fred_get_json(endpoint: str, params: dict) -> Any:
    import requests
    url = f"{FRED_API_BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FRED HTTP {r.status_code}: {r.text[:200]}")
    return r.json()

def _auto_seed_cpi_from_fred() -> list[dict[str, str]]:
    """
    Fallback: lấy CPI release dates qua FRED API (không đụng BLS trực tiếp).
    Flow:
      - /fred/releases -> tìm release name chứa 'Consumer Price Index'
      - /fred/release/dates -> lấy dates 2023-2024
    """
    require(bool(config.FRED_API_KEY), "Missing FRED_API_KEY for CPI fallback via FRED.")

    a0, a1 = _analysis_range()

    # 1) get releases
    js = _fred_get_json("releases", {"api_key": config.FRED_API_KEY, "file_type": "json"})
    rels = js.get("releases", [])
    if not rels:
        fail("FRED releases empty. Check FRED_API_KEY or network.")

    # pick release that matches CPI
    cpi_rel = None
    for r in rels:
        name = str(r.get("name", "")).lower()
        if "consumer price index" in name:
            cpi_rel = r
            break

    if cpi_rel is None:
        # debug: write sample releases to runs/latest
        sample_path = _run_dir_latest() / "debug_fred_releases_sample.json"
        sample_path.write_text(json.dumps(rels[:50], indent=2), encoding="utf-8")
        fail("Cannot find CPI release in FRED releases. Dumped sample to runs/latest/debug_fred_releases_sample.json")

    release_id = cpi_rel.get("id")
    release_name = cpi_rel.get("name", "UNKNOWN")

    # 2) get release dates
    js2 = _fred_get_json(
        "release/dates",
        {
            "api_key": config.FRED_API_KEY,
            "file_type": "json",
            "release_id": release_id,
        },
    )
    dates = js2.get("release_dates", [])
    if not dates:
        fail(f"FRED release/dates empty for release_id={release_id}")

    out: dict[str, dict[str, str]] = {}
    for d in dates:
        ds = str(d.get("date", "")).strip()
        if not ds:
            continue
        try:
            ed = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        if not (a0 <= ed <= a1):
            continue
        out[ed.isoformat()] = {
            "event_type": "CPI",
            "event_date": ed.isoformat(),
            "source": f"FRED release_id={release_id} ({release_name})",
            "note": "CPI release date via FRED release calendar (fallback when BLS blocked)",
        }

    rows = [out[k] for k in sorted(out.keys())]
    ok(f"Auto-seeded CPI events via FRED fallback: {len(rows)}")
    return rows

def _auto_seed_cpi() -> list[dict[str, str]]:
    """
    Primary: parse BLS CPI archive URLs (date in URL)
    Fallback: use FRED release calendar if BLS blocks (403)
    """
    a0, a1 = _analysis_range()

    try:
        html = _fetch_text(BLS_CPI_ARCHIVE_LIST, headers=_browser_headers())

        urls = sorted(set(re.findall(r"https://www\.bls\.gov/news\.release/archives/cpi_\d{8}\.htm", html)))
        if not urls:
            raise RuntimeError("Cannot parse CPI archive URLs from BLS page (structure changed?)")

        out: dict[str, dict[str, str]] = {}
        for u in urls:
            m = re.search(r"cpi_(\d{2})(\d{2})(\d{4})\.htm", u)
            if not m:
                continue
            mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ed = date(yyyy, mm, dd)
            if not (a0 <= ed <= a1):
                continue
            out[ed.isoformat()] = {
                "event_type": "CPI",
                "event_date": ed.isoformat(),
                "source": u,
                "note": "BLS CPI archived news release (date parsed from URL)",
            }

        rows = [out[k] for k in sorted(out.keys())]
        ok(f"Auto-seeded CPI events via BLS: {len(rows)}")
        return rows

    except Exception as e:
        warn(f"BLS CPI blocked/failed ({e}). Switching to FRED fallback for CPI dates...")
        return _auto_seed_cpi_from_fred()

# ----------------- MACRO: FOMC -----------------
_MONTH_MAP = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6,
    "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9,
    "OCT": 10, "NOV": 11, "DEC": 12,
}


def _html_to_loose_text(html: str) -> str:
    html = re.sub(r"<(br|BR)\s*/?>", "\n", html)
    html = re.sub(r"</(p|div|li|tr|td|th|h\d)>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def _parse_fomc_year(text: str, year: int) -> list[dict[str, str]]:
    """
    Parse year block from Fed calendar page.
    event_date = last day of meeting range (decision day proxy).
    """
    key = f"{year} FOMC Meetings"
    start_idx = text.find(key)
    if start_idx < 0:
        warn(f"Cannot find '{key}' on Fed calendar page.")
        return []

    tail = text[start_idx:start_idx + 25000]
    lines = [ln.strip() for ln in tail.split("\n") if ln.strip()]

    rows: list[dict[str, str]] = []
    cur_month_label: str | None = None

    for ln in lines:
        up = ln.upper()

        # Stop when reaching another year header
        if re.search(r"\b20\d{2}\s+FOMC\s+MEETINGS\b", up) and (f"{year} FOMC MEETINGS" not in up):
            break

        # Month label (including cross-month like "Jan/Feb")
        if re.fullmatch(r"[A-Z]{3,9}(/?[A-Z]{3,9})?", up):
            cur_month_label = up
            continue

        if not cur_month_label:
            continue

        # Range like "30-31" or "30-1" or "19-20*"
        if re.match(r"^\d{1,2}\s*[-–]\s*\d{1,2}\*?$", up):
            s = up.replace("–", "-").replace("*", "").replace(" ", "")
            a, b = s.split("-")
            d_start, d_end = int(a), int(b)

            # month mapping
            if "/" in cur_month_label:
                m1, m2 = cur_month_label.split("/", 1)
                m_start = _MONTH_MAP.get(m1, None)
                m_end = _MONTH_MAP.get(m2, None)
            else:
                m_start = _MONTH_MAP.get(cur_month_label, None)
                m_end = m_start

            if m_start is None:
                continue

            # cross-month inference if d_end < d_start and m_end not provided
            if d_end < d_start and (m_end == m_start):
                m_end = 1 if m_start == 12 else (m_start + 1)

            ed = date(year, m_end, d_end).isoformat()
            rows.append(
                {
                    "event_type": "FOMC",
                    "event_date": ed,
                    "source": FED_FOMC_CALENDAR,
                    "note": f"Fed calendar (decision day inferred as range end from '{cur_month_label} {s}')",
                }
            )
            continue

        # Single day like "31"
        if re.fullmatch(r"\d{1,2}\*?", up):
            d_end = int(up.replace("*", ""))
            m = _MONTH_MAP.get(cur_month_label.replace("*", ""), None)
            if m is None:
                continue
            ed = date(year, m, d_end).isoformat()
            rows.append(
                {
                    "event_type": "FOMC",
                    "event_date": ed,
                    "source": FED_FOMC_CALENDAR,
                    "note": f"Fed calendar (single day '{cur_month_label} {d_end}')",
                }
            )
            continue

    # Unique by date
    uniq: dict[str, dict[str, str]] = {}
    for r in rows:
        uniq[r["event_date"]] = r
    out = [uniq[k] for k in sorted(uniq.keys())]
    ok(f"Auto-seeded FOMC {year} events: {len(out)}")
    return out


def _auto_seed_fomc() -> list[dict[str, str]]:
    html = _fetch_text(FED_FOMC_CALENDAR, headers=_browser_headers())
    text = _html_to_loose_text(html)

    rows = []
    rows.extend(_parse_fomc_year(text, 2023))
    rows.extend(_parse_fomc_year(text, 2024))

    a0, a1 = _analysis_range()
    rows = [
        r for r in rows
        if a0 <= datetime.strptime(r["event_date"], "%Y-%m-%d").date() <= a1
    ]
    ok(f"Auto-seeded FOMC (filtered) events: {len(rows)}")
    return rows


# ----------------- EARNINGS via SEC (8-K Item 2.02) -----------------
def _ticker_to_cik_map() -> dict[str, str]:
    """
    Fetch SEC company_tickers.json then build map ticker -> zero-padded 10-digit CIK.
    """
    headers = _sec_headers()
    data = _fetch_json(SEC_TICKER_MAP, headers=headers)
    time.sleep(SLEEP_SEC)

    # company_tickers.json is dict keyed by number: { "0": {"cik_str":..., "ticker":..., "title":...}, ... }
    out: dict[str, str] = {}
    for _, obj in data.items():
        t = str(obj.get("ticker", "")).strip().upper()
        cik_str = obj.get("cik_str", None)
        if not t or cik_str is None:
            continue
        cik10 = str(int(cik_str)).zfill(10)
        out[t] = cik10

    if not out:
        fail("SEC ticker->CIK mapping is empty. SEC may have blocked the request (check SEC_USER_AGENT).")

    return out


def _sec_earnings_dates_for_ticker(ticker: str, cik10: str) -> list[dict[str, str]]:
    """
    Pull submissions JSON, then filter:
      form == '8-K' AND items contains '2.02'
    Use filingDate as event_date.
    """
    headers = _sec_headers()
    url = SEC_SUBMISSIONS.format(cik=cik10)

    js = _fetch_json(url, headers=headers)
    time.sleep(SLEEP_SEC)

    recent = js.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    items = recent.get("items", [])

    if not (forms and filing_dates):
        warn(f"SEC submissions has no recent filings for {ticker} (CIK={cik10}).")
        return []

    a0, a1 = _analysis_range()

    rows = []
    n = min(len(forms), len(filing_dates))
    for i in range(n):
        if str(forms[i]).strip().upper() != "8-K":
            continue

        it = ""
        if i < len(items):
            it = str(items[i] or "")
        if "2.02" not in it:
            continue

        d_str = str(filing_dates[i]).strip()
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
        except Exception:
            continue

        if not (a0 <= d <= a1):
            continue

        rows.append(
            {
                "ticker": ticker,
                "event_date": d.isoformat(),
                "source": url,
                "note": "SEC EDGAR: 8-K with Item 2.02 (earnings proxy); event_date = filingDate",
            }
        )

    # unique by date
    uniq: dict[str, dict[str, str]] = {}
    for r in rows:
        uniq[r["event_date"]] = r
    out = [uniq[k] for k in sorted(uniq.keys())]
    ok(f"Auto-seeded earnings (SEC 8-K 2.02) for {ticker}: {len(out)}")
    return out


def _auto_seed_earnings_sec() -> list[dict[str, str]]:
    m = _ticker_to_cik_map()

    rows: list[dict[str, str]] = []
    for t in config.TICKERS:
        cik10 = m.get(t, "")
        if not cik10:
            warn(f"Ticker {t} not found in SEC ticker map. Skipping.")
            continue
        rows.extend(_sec_earnings_dates_for_ticker(t, cik10))

    if not rows:
        warn("SEC earnings rows are empty. Possible causes: missing SEC_USER_AGENT, network block, or items field not present.")
    return rows


# ----------------- BUILD WINDOWS + WRITE FILES -----------------
def _build_windows(earn_rows: list[dict[str, str]], macro_rows: list[dict[str, str]]):
    import pandas as pd

    a0, a1 = _analysis_range()
    rows = []

    # Earnings windows per ticker
    for r in earn_rows:
        t = r["ticker"]
        ed = datetime.strptime(r["event_date"], "%Y-%m-%d").date()
        s = max(ed - timedelta(days=EARN_LEFT), a0)
        e = min(ed + timedelta(days=EARN_RIGHT), a1)
        if e < a0 or s > a1:
            continue
        rows.append(
            {
                "event_type": "EARNINGS",
                "ticker": t,
                "event_date": ed.isoformat(),
                "start_date": s.isoformat(),
                "end_date": e.isoformat(),
                "source": r.get("source", ""),
            }
        )

    # Macro windows ticker=ALL
    for r in macro_rows:
        et = r["event_type"].upper()
        ed = datetime.strptime(r["event_date"], "%Y-%m-%d").date()
        s = max(ed - timedelta(days=MACRO_LEFT), a0)
        e = min(ed + timedelta(days=MACRO_RIGHT), a1)
        if e < a0 or s > a1:
            continue
        rows.append(
            {
                "event_type": et,
                "ticker": "ALL",
                "event_date": ed.isoformat(),
                "start_date": s.isoformat(),
                "end_date": e.isoformat(),
                "source": r.get("source", ""),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        fail("No event windows produced. Check seeds and analysis range.")

    out = out.sort_values(["event_date", "event_type", "ticker"]).reset_index(drop=True)
    require_unique_key(out, ["event_type", "ticker", "event_date"], where="event_windows")
    return out


def main() -> None:
    run_dir = _run_dir_latest()

    log_path = run_dir / "logs_step03_build_event_windows.txt"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")

    logger.info("=== Step 03: build event windows (AUTO SEC/BLS/Fed) ===")
    logger.info(f"Analysis range: {config.ANALYSIS_START} -> {config.ANALYSIS_END}")

    ensure_dir(SEED_DIR)

    # 1) Macro seeds (BLS + Fed)
    macro_rows = []
    macro_rows.extend(_auto_seed_cpi())
    macro_rows.extend(_auto_seed_fomc())

    # write macro seed
    import pandas as pd
    macro_df = pd.DataFrame(macro_rows).sort_values(["event_type", "event_date"]).reset_index(drop=True)
    macro_df.to_csv(MACRO_SEED_PATH, index=False, encoding="utf-8")
    ok(f"Wrote macro seed -> {MACRO_SEED_PATH} (rows={len(macro_df)})")

    # 2) Earnings seeds (SEC)
    earn_rows = _auto_seed_earnings_sec()
    earn_df = pd.DataFrame(earn_rows).sort_values(["ticker", "event_date"]).reset_index(drop=True)
    if earn_df.empty:
        warn("Earnings seed is empty. You can still continue (macro windows work), but earnings-window metrics may be weaker.")
    else:
        earn_df.to_csv(EARNINGS_SEED_PATH, index=False, encoding="utf-8")
        ok(f"Wrote earnings seed -> {EARNINGS_SEED_PATH} (rows={len(earn_df)})")

    # 3) Build event_windows.csv
    out = _build_windows(earn_rows=earn_rows, macro_rows=macro_rows)
    ensure_dir(config.RESULTS_DIR)
    write_csv(out, config.EVENT_WINDOWS_PATH)
    out_hash = sha256_file(config.EVENT_WINDOWS_PATH)

    # 4) Manifest
    step_manifest = {
        "step": "03_build_event_windows",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "macro_seed_path": str(MACRO_SEED_PATH),
            "earnings_seed_path": str(EARNINGS_SEED_PATH),
            "event_windows_path": str(config.EVENT_WINDOWS_PATH),
            "event_windows_sha256": out_hash,
            "rows": int(len(out)),
        },
        "window_rules": {"earn": [EARN_LEFT, EARN_RIGHT], "macro": [MACRO_LEFT, MACRO_RIGHT]},
        "sources": {"cpi": BLS_CPI_ARCHIVE_LIST, "fomc": FED_FOMC_CALENDAR, "sec_map": SEC_TICKER_MAP},
        "notes": {
            "earnings": "SEC 8-K Item 2.02, event_date=filingDate (proxy). Consider ±1 window due to after-hours.",
            "rate_limit": f"sleep={SLEEP_SEC}s per SEC request",
        },
    }
    write_json(step_manifest, run_dir / "manifest_step03.json")

    ok(f"Saved event_windows.csv -> {config.EVENT_WINDOWS_PATH}")
    ok("Step 03 DONE. Next: python src/04_baselines.py")


if __name__ == "__main__":
    main()