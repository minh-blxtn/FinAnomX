# src/04_baselines.py
# Step 04: Amazon Chronos benchmark (research-standard, no-leak)
#
# Output:
#   data/results/baseline_scores_chronos.csv
#   runs/latest/logs_step04_baselines.txt
#   runs/latest/manifest_step04.json

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys
from typing import List, Tuple

from dotenv import load_dotenv
from loguru import logger

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, fail, require, require_columns, require_unique_key  # noqa: E402
from utils.io import ensure_dir, read_parquet, write_csv, sha256_file, write_json  # noqa: E402

import numpy as np
import pandas as pd


REGULAR_START_TS = pd.Timestamp("2000-01-01")
REGULAR_FREQ = "D"


def _run_dir_latest() -> Path:
    d = config.RUNS_DIR / "latest"
    ensure_dir(d)
    return d


def _load_chronos_pipeline():
    try:
        from chronos import Chronos2Pipeline
    except Exception as e:
        fail(
            "Step 04 now uses Amazon Chronos as the only external benchmark. "
            "Package not found or failed to import. Install with:\n"
            "pip install chronos-forecasting\n"
            f"Original import error: {e}"
        )
    pipe = Chronos2Pipeline.from_pretrained(
        config.CHRONOS_MODEL_ID,
        device_map=config.CHRONOS_DEVICE_MAP,
        torch_dtype=config.CHRONOS_DTYPE,
    )
    return pipe


def _regular_timestamp_at(pos: int) -> pd.Timestamp:
    return REGULAR_START_TS + pd.to_timedelta(pos, unit=REGULAR_FREQ.lower())


def _build_snapshot_tables(
    feat: pd.DataFrame,
    target_col: str,
    cov_cols: List[str],
    context_max: int,
    min_context: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build one-step rolling forecast tasks for Chronos-2 using predict_df.

    Important design choice:
    Financial market dates are trading-day timestamps with weekend/holiday gaps.
    Chronos-2 tries to infer a regular frequency from the timestamp column and may fail
    on such irregular calendars. To avoid this, each snapshot uses a synthetic regular
    timestamp grid (step index -> equally spaced daily timestamps). Actual target dates
    are preserved separately in meta_df and used after prediction.
    """
    context_rows = []
    future_rows = []
    meta_rows = []

    for tkr in config.TICKERS:
        sub = feat[feat["ticker"] == tkr].sort_values("date").reset_index(drop=True)
        y = sub[target_col].astype(float).to_numpy()

        for i in range(len(sub)):
            if i < min_context:
                continue

            target_date = pd.to_datetime(sub.loc[i, "date"]).normalize()
            split_i = str(sub.loc[i, "split"])
            start = max(0, i - context_max)
            ctx = sub.iloc[start:i].copy().reset_index(drop=True)
            if len(ctx) < min_context:
                continue

            snapshot_id = f"{tkr}|{target_date.date().isoformat()}"

            for j, (_, r) in enumerate(ctx.iterrows()):
                row = {
                    "id": snapshot_id,
                    "timestamp": _regular_timestamp_at(j),
                    "target": float(r[target_col]),
                }
                for c in cov_cols:
                    row[c] = float(r[c])
                context_rows.append(row)

            future_ts = _regular_timestamp_at(len(ctx))
            frow = {
                "id": snapshot_id,
                "timestamp": future_ts,
            }
            for c in cov_cols:
                frow[c] = float(sub.loc[i, c])
            future_rows.append(frow)

            meta_rows.append(
                {
                    "ticker": tkr,
                    "date": target_date,
                    "split": split_i,
                    "id": snapshot_id,
                    "y_true": float(y[i]),
                    "context_len": int(len(ctx)),
                    "future_ts": future_ts,
                }
            )

    context_df = pd.DataFrame(context_rows)
    future_df = pd.DataFrame(future_rows)
    meta_df = pd.DataFrame(meta_rows)

    require(len(meta_df) > 0, "No Chronos snapshots created. Check context settings and features.")
    require_unique_key(meta_df, ["ticker", "date"], where="chronos meta")
    require_unique_key(meta_df, ["id"], where="chronos meta id")
    return context_df, future_df, meta_df


def main() -> None:
    run_dir = _run_dir_latest()
    log_path = run_dir / "logs_step04_baselines.txt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")
    logger.info("=== Step 04: Amazon Chronos benchmark ===")

    require(config.FEATURES_PATH.exists(), f"Missing input: {config.FEATURES_PATH}. Run Step 02 first.")
    feat = read_parquet(config.FEATURES_PATH).sort_values(["ticker", "date"]).reset_index(drop=True)

    need = ["ticker", "date", "split", config.CHRONOS_TARGET_COL] + list(config.CHRONOS_COVARIATE_COLS)
    require_columns(feat, need, where="features.parquet for Chronos")
    require_unique_key(feat, ["ticker", "date"], where="features.parquet")
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()

    pipe = _load_chronos_pipeline()
    logger.info(f"Chronos model: {config.CHRONOS_MODEL_ID}")

    context_df, future_df, meta_df = _build_snapshot_tables(
        feat=feat,
        target_col=config.CHRONOS_TARGET_COL,
        cov_cols=list(config.CHRONOS_COVARIATE_COLS) if config.CHRONOS_USE_COVARIATES else [],
        context_max=int(config.CHRONOS_CONTEXT_MAX),
        min_context=int(config.CHRONOS_MIN_CONTEXT),
    )

    logger.info(
        f"Chronos snapshots={len(meta_df)} context_rows={len(context_df)} future_rows={len(future_df)} regular_freq={REGULAR_FREQ}"
    )

    pred_df = pipe.predict_df(
        context_df,
        future_df=future_df if config.CHRONOS_USE_COVARIATES else None,
        prediction_length=int(config.CHRONOS_PREDICTION_LENGTH),
        quantile_levels=list(config.CHRONOS_QUANTILES),
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    expect_cols = {"id", "timestamp", "prediction"}
    if not expect_cols.issubset(set(pred_df.columns)):
        # Some versions may use 'predictions'; handle both.
        if {"id", "timestamp", "predictions"}.issubset(set(pred_df.columns)):
            pred_df = pred_df.rename(columns={"predictions": "prediction"})
        else:
            fail(f"Chronos output missing required columns. Need one of ['prediction','predictions']; got {list(pred_df.columns)}")

    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    # Keep only the one-step-ahead row and merge by synthetic future timestamp.
    pred_use = pred_df[["id", "timestamp", "prediction"]].rename(columns={"prediction": "pred_chronos"}).copy()
    out = meta_df.merge(pred_use, left_on=["id", "future_ts"], right_on=["id", "timestamp"], how="left")
    if out["pred_chronos"].isna().any():
        missing = int(out["pred_chronos"].isna().sum())
        fail(f"Chronos merge produced {missing} missing predictions. Check snapshot ids and prediction output.")

    out["score_chronos"] = (out["y_true"].astype(float) - out["pred_chronos"].astype(float)).abs()

    # align to full daily panel: neutral 0 before min context
    full = feat[["ticker", "date", "split"]].copy()
    full = full.merge(out[["ticker", "date", "pred_chronos", "score_chronos", "context_len"]], on=["ticker", "date"], how="left")
    full["pred_chronos"] = pd.to_numeric(full["pred_chronos"], errors="coerce")
    full["score_chronos"] = pd.to_numeric(full["score_chronos"], errors="coerce").fillna(0.0)
    full["context_len"] = pd.to_numeric(full["context_len"], errors="coerce").fillna(0).astype(int)

    out_path = config.RESULTS_DIR / "baseline_scores_chronos.csv"
    ensure_dir(config.RESULTS_DIR)
    write_csv(full, out_path)

    train_stats = (
        full[full["split"] == "train"]
        .groupby("ticker")["score_chronos"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .to_dict(orient="records")
    )

    manifest = {
        "step": "04_baselines",
        "mode": "chronos_only",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"features_path": str(config.FEATURES_PATH), "sha256": sha256_file(config.FEATURES_PATH)},
        "outputs": {"baseline_scores_chronos": {"path": str(out_path), "sha256": sha256_file(out_path), "rows": int(len(full))}},
        "chronos": {
            "model_id": config.CHRONOS_MODEL_ID,
            "use_covariates": bool(config.CHRONOS_USE_COVARIATES),
            "covariate_cols": list(config.CHRONOS_COVARIATE_COLS),
            "target_col": config.CHRONOS_TARGET_COL,
            "context_max": int(config.CHRONOS_CONTEXT_MAX),
            "min_context": int(config.CHRONOS_MIN_CONTEXT),
            "prediction_length": int(config.CHRONOS_PREDICTION_LENGTH),
            "regularized_timestamp_grid": True,
            "regular_freq": REGULAR_FREQ,
        },
        "train_score_stats": train_stats,
        "notes": {
            "score_definition": "abs(one-step forecast residual)",
            "no_leak": "each target date uses only past context; thresholds later use train-only or past-only statistics",
            "why_regular_timestamps": "Chronos frequency inference can fail on trading-day gaps (weekends/holidays); each snapshot uses a synthetic regular time grid",
        },
    }
    write_json(manifest, run_dir / "manifest_step04.json")

    ok(f"Saved Chronos benchmark -> {out_path}")
    ok("Step 04 DONE. Next: python src/05_train_models.py")


if __name__ == "__main__":
    main()

