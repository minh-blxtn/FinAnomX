# src/01_ingest_raw.py
# (GỘP) tải market + macro + merge + fill(causal ffill-only) + save data/raw/raw_merged.parquet
# Version: NO RUN_ID (logs/manifest luôn ghi vào runs/latest)

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import sys
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# --- allow: python src/01_ingest_raw.py
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, warn, fail, require, require_unique_key, require_no_nan  # noqa: E402
from utils.io import ensure_dir, sha256_file, write_json, write_parquet  # noqa: E402
from utils.time_utils import parse_ymd  # noqa: E402


def _get_run_dir_latest() -> Path:
    """
    Không dùng RUN_ID nữa.
    Luôn ghi log/manifest vào runs/latest/ (đơn giản cho newbie).
    """
    run_dir = config.RUNS_DIR / "latest"
    ensure_dir(run_dir)
    return run_dir


def _download_market_yfinance() -> "pd.DataFrame":
    """
    Output chuẩn hoá:
    ticker, date, open, high, low, close, adj_close, volume
    """
    import pandas as pd
    import yfinance as yf

    start = config.DOWNLOAD_START
    end = config.DOWNLOAD_END

    logger.info(f"Downloading market data: {config.TICKERS} | {start} -> {end}")

    df = yf.download(
        tickers=config.TICKERS,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        fail("yfinance download returned empty dataframe. Check internet / ticker symbols / date range.")

    frames = []

    # Case 1: MultiIndex with top level = ticker
    if isinstance(df.columns, pd.MultiIndex) and set(config.TICKERS).issubset(set(df.columns.get_level_values(0))):
        for t in config.TICKERS:
            sub = df[t].copy().reset_index()
            sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
            sub["ticker"] = t
            frames.append(sub)

    # Case 2: MultiIndex with level 1 = ticker (Attr, Ticker)
    elif isinstance(df.columns, pd.MultiIndex):
        for t in config.TICKERS:
            sub = df.xs(t, axis=1, level=1).copy().reset_index()
            sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
            sub["ticker"] = t
            frames.append(sub)

    # Case 3: single ticker
    else:
        sub = df.reset_index()
        sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
        sub["ticker"] = config.TICKERS[0]
        frames.append(sub)

    import pandas as pd

    market = pd.concat(frames, ignore_index=True)

    # normalize column names
    if "adj_close" not in market.columns and "adjclose" in market.columns:
        market.rename(columns={"adjclose": "adj_close"}, inplace=True)

    needed = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    for c in needed:
        if c not in market.columns:
            fail(f"Missing market column from yfinance: {c}. Current columns={list(market.columns)}")

    # normalize date
    market["date"] = pd.to_datetime(market["date"]).dt.tz_localize(None).dt.normalize()

    # sort + basic cleaning
    market = market.sort_values(["ticker", "date"]).reset_index(drop=True)
    market = market.dropna(subset=["close", "volume"])

    # sanity checks
    require_unique_key(market, ["ticker", "date"], where="market")

    # validate values
    if (market["close"] <= 0).any():
        fail("Invalid close price (<=0) detected in market data.")
    if (market["volume"] < 0).any():
        fail("Invalid volume (<0) detected in market data.")

    ok(f"Market downloaded rows={len(market):,}")
    return market


def _download_macro_fred(dates: "pd.Series") -> "pd.DataFrame":
    """
    Macro output:
    date + macro cols (vix, dgs10, fedfunds, cpi)
    Merge theo market trading dates, fill ffill-only.
    """
    import pandas as pd
    from fredapi import Fred

    if not config.FRED_API_KEY:
        fail("FRED_API_KEY missing. Fix by setting it in .env then rerun.")

    fred = Fred(api_key=config.FRED_API_KEY)

    macro = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize().sort_values().unique()})
    macro = macro.sort_values("date").reset_index(drop=True)

    for series_id, alias in config.FRED_SERIES.items():
        logger.info(f"Downloading FRED series: {series_id} -> {alias}")
        try:
            s = fred.get_series(series_id)
        except Exception as e:
            fail(f"FRED get_series failed for {series_id}: {e}")

        s = s.rename(alias)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s_df = s.to_frame().reset_index().rename(columns={"index": "date"})
        s_df["date"] = pd.to_datetime(s_df["date"]).dt.normalize()

        macro = macro.merge(s_df, on="date", how="left")

    macro_cols = list(config.FRED_SERIES.values())

    # causal fill
    macro[macro_cols] = macro[macro_cols].ffill()

    ok("Macro downloaded + ffill-only done")
    return macro


def main() -> None:
    run_dir = _get_run_dir_latest()

    # log to file (overwrite each time is ok)
    log_path = run_dir / "logs_step01_ingest_raw.txt"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")

    logger.info("=== Step 01: ingest raw (NO RUN_ID) ===")
    logger.info(f"Run dir (fixed): {run_dir}")

    # quick config sanity
    d0 = parse_ymd(config.DOWNLOAD_START)
    d1 = parse_ymd(config.DOWNLOAD_END)
    require(d0 < d1, "DOWNLOAD_START must be < DOWNLOAD_END")

    # 1) market
    market = _download_market_yfinance()

    # 2) macro for market dates
    all_dates = market["date"].drop_duplicates().sort_values()
    macro = _download_macro_fred(all_dates)

    # 3) merge + ffill macro within each ticker (safety)
    import pandas as pd

    merged = market.merge(macro, on="date", how="left")
    macro_cols = list(config.FRED_SERIES.values())

    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged[macro_cols] = merged.groupby("ticker")[macro_cols].ffill()

    # drop initial segment still missing macro (expected)
    before = len(merged)
    merged = merged.dropna(subset=macro_cols).reset_index(drop=True)
    dropped = before - len(merged)
    if dropped > 0:
        warn(f"Dropped {dropped:,} rows due to initial macro NaNs (expected behavior).")

    # post-checks
    require_unique_key(merged, ["ticker", "date"], where="raw_merged")

    for t in config.TICKERS:
        sub = merged[merged["ticker"] == t]
        if not sub["date"].is_monotonic_increasing:
            fail(f"Dates not sorted for ticker={t} in raw_merged")

    require_no_nan(merged, macro_cols, where="raw_merged macro after ffill")

    # save
    ensure_dir(config.RAW_DIR)
    write_parquet(merged, config.RAW_MERGED_PATH)

    out_hash = sha256_file(config.RAW_MERGED_PATH)

    step_manifest = {
        "step": "01_ingest_raw",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "raw_merged_path": str(config.RAW_MERGED_PATH),
            "raw_merged_sha256": out_hash,
            "rows": int(len(merged)),
            "columns": list(merged.columns),
        },
        "inputs": {
            "tickers": config.TICKERS,
            "download_start": config.DOWNLOAD_START,
            "download_end": config.DOWNLOAD_END,
            "fred_series": config.FRED_SERIES,
        },
    }
    write_json(step_manifest, run_dir / "manifest_step01.json")

    ok(f"Saved raw_merged.parquet -> {config.RAW_MERGED_PATH}")
    ok("Step 01 DONE. Next: python src/02_process_data.py")


if __name__ == "__main__":
    main()