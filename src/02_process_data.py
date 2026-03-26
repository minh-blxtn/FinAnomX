# src/02_process_data.py
# features + split + scale(train-only) + save processed (features.parquet + scaled.parquet) + save scalers
# Revised Step 02: robust train-only scaling, no clip saturation, preserve raw macro columns
# Version: NO RUN_ID (logs/manifest luôn ghi vào runs/latest)

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pickle
import sys

from dotenv import load_dotenv
from loguru import logger

# --- allow: python src/02_process_data.py
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, warn, fail, require, require_unique_key, require_no_nan  # noqa: E402
from utils.io import ensure_dir, read_parquet, write_parquet, sha256_file, write_json  # noqa: E402


def _get_run_dir_latest() -> Path:
    run_dir = config.RUNS_DIR / "latest"
    ensure_dir(run_dir)
    return run_dir


def _assign_split(date_series):
    import pandas as pd

    train_end = pd.to_datetime(config.TRAIN_END)
    val_end = pd.to_datetime(config.VAL_END)

    d = pd.to_datetime(date_series)
    split = pd.Series(["test"] * len(d), index=d.index)
    split[d <= val_end] = "val"
    split[d <= train_end] = "train"
    return split


def _compute_features(df):
    import numpy as np
    import pandas as pd

    def _rsi_manual(x: pd.Series, window: int) -> pd.Series:
        delta = x.astype(float).diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _macd_manual(x: pd.Series) -> pd.DataFrame:
        x = x.astype(float)
        ema_fast = x.ewm(span=config.MACD_FAST, adjust=False, min_periods=config.MACD_FAST).mean()
        ema_slow = x.ewm(span=config.MACD_SLOW, adjust=False, min_periods=config.MACD_SLOW).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=config.MACD_SIGNAL, adjust=False, min_periods=config.MACD_SIGNAL).mean()
        diff = macd - signal
        return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_diff": diff}, index=x.index)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["log_close"] = np.log(df["close"].astype(float))
    df["log_return"] = df.groupby("ticker")["log_close"].diff()

    for w in config.ROLL_WINDOWS:
        df[f"vol_{w}"] = (
            df.groupby("ticker")["log_return"]
            .rolling(window=w, min_periods=w)
            .std()
            .reset_index(level=0, drop=True)
        )

    vol_mean = (
        df.groupby("ticker")["volume"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    vol_std = (
        df.groupby("ticker")["volume"]
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["volume_z_20"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    df["rsi"] = df.groupby("ticker")["close"].transform(lambda s: _rsi_manual(s, config.RSI_PERIOD))

    macd_df = df.groupby("ticker")["close"].apply(_macd_manual).reset_index(level=0, drop=True)
    df[["macd", "macd_signal", "macd_diff"]] = macd_df[["macd", "macd_signal", "macd_diff"]]

    for c in list(config.FRED_SERIES.values()):
        df[f"{c}_diff"] = df.groupby("ticker")[c].diff()

    return df


def _select_feature_columns(df):
    id_cols = ["ticker", "date", "split"]
    market_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    extra_raw_cols = ["log_close"] if getattr(config, "KEEP_LOG_CLOSE", True) and "log_close" in df.columns else []

    macro_cols = [c for c in config.FRED_SERIES.values() if c in df.columns]
    macro_diff_cols = [f"{c}_diff" for c in macro_cols if f"{c}_diff" in df.columns]

    stable_model_features = [
        c for c in ["log_return", "rsi", "macd", "macd_signal", "macd_diff", "volume_z_20"] if c in df.columns
    ]
    stable_model_features += [f"vol_{w}" for w in config.ROLL_WINDOWS if f"vol_{w}" in df.columns]
    stable_model_features += macro_diff_cols

    model_feature_cols = list(dict.fromkeys(stable_model_features))
    if getattr(config, "SCALE_INCLUDE_MACRO_LEVELS", False):
        model_feature_cols += [c for c in macro_cols if c not in model_feature_cols]

    keep_cols = list(dict.fromkeys(id_cols + market_cols + macro_cols + extra_raw_cols + model_feature_cols))

    missing_keep = [c for c in keep_cols if c not in df.columns]
    if missing_keep:
        fail(f"Missing expected columns in features dataframe: {missing_keep}")

    if not model_feature_cols:
        fail("No model_feature_cols selected in Step 02.")

    return keep_cols, model_feature_cols


def _build_scaler():
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    method = str(getattr(config, "SCALE_METHOD", "robust")).lower()
    if method == "robust":
        q_low, q_high = getattr(config, "ROBUST_QUANTILE_RANGE", (10.0, 90.0))
        return method, RobustScaler(with_centering=True, with_scaling=True, quantile_range=(float(q_low), float(q_high)))
    if method == "standard":
        return method, StandardScaler()
    if method == "minmax":
        return method, MinMaxScaler(clip=False)
    fail(f"Unsupported SCALE_METHOD={method}")


def _fit_transform_scalers(df, feature_cols):
    import numpy as np
    import pandas as pd

    ensure_dir(config.SCALERS_DIR)

    df = df.copy()
    df_scaled = []
    scaler_name, scaler_template = _build_scaler()

    for t in config.TICKERS:
        sub = df[df["ticker"] == t].copy()
        train = sub[sub["split"] == "train"].copy()

        if train.empty:
            fail(f"No TRAIN rows for ticker={t}. Check TRAIN_END={config.TRAIN_END} vs data range.")

        scaler = scaler_template.__class__(**scaler_template.get_params())
        scaler.fit(train[feature_cols].astype(float).values)

        vals = scaler.transform(sub[feature_cols].astype(float).values)
        sub_scaled = sub.copy()
        sub_scaled[[f"{c}_s" for c in feature_cols]] = vals

        scaler_path = config.SCALERS_DIR / f"{t}_{scaler_name}.pkl"
        with scaler_path.open("wb") as f:
            pickle.dump(scaler, f)

        train_scaled = sub_scaled[sub_scaled["split"] == "train"][[f"{c}_s" for c in feature_cols]].to_numpy(dtype=float)
        if not np.isfinite(train_scaled).all():
            fail(f"Non-finite scaled values found in TRAIN for ticker={t}")

        extreme_share = float((np.abs(train_scaled) > 10).mean())
        if extreme_share > 0.01:
            warn(f"[{t}] TRAIN scaled values with |x|>10 share={extreme_share:.4f} (check feature tails).")

        df_scaled.append(sub_scaled)

    out = pd.concat(df_scaled, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    return scaler_name, out


def main() -> None:
    run_dir = _get_run_dir_latest()

    log_path = run_dir / "logs_step02_process_data.txt"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")

    logger.info("=== Step 02: process data (features + split + scale train-only) ===")
    logger.info(f"Run dir (fixed): {run_dir}")

    require(config.RAW_MERGED_PATH.exists(), f"Missing input: {config.RAW_MERGED_PATH}. Run Step 01 first.")

    df = read_parquet(config.RAW_MERGED_PATH)

    require_unique_key(df, ["ticker", "date"], where="raw_merged input")
    df["date"] = df["date"].astype("datetime64[ns]")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["split"] = _assign_split(df["date"])

    feat = _compute_features(df)

    keep_cols, feature_cols = _select_feature_columns(feat)
    feat = feat[keep_cols].copy()

    before = len(feat)
    feat = feat.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = before - len(feat)
    if dropped > 0:
        warn(f"Dropped {dropped:,} rows due to rolling/indicator NaNs (expected at beginning of series).")

    require_unique_key(feat, ["ticker", "date"], where="features")
    require_no_nan(feat, feature_cols, where="features (after dropna)")

    ensure_dir(config.PROCESSED_DIR)
    write_parquet(feat, config.FEATURES_PATH)
    feat_hash = sha256_file(config.FEATURES_PATH)
    ok(f"Saved features.parquet -> {config.FEATURES_PATH}")

    scaler_name, scaled = _fit_transform_scalers(feat, feature_cols)

    write_parquet(scaled, config.SCALED_PATH)
    scaled_hash = sha256_file(config.SCALED_PATH)
    ok(f"Saved scaled.parquet -> {config.SCALED_PATH}")

    step_manifest = {
        "step": "02_process_data",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "raw_merged_path": str(config.RAW_MERGED_PATH),
            "raw_merged_sha256": sha256_file(config.RAW_MERGED_PATH),
        },
        "outputs": {
            "features_path": str(config.FEATURES_PATH),
            "features_sha256": feat_hash,
            "scaled_path": str(config.SCALED_PATH),
            "scaled_sha256": scaled_hash,
            "scalers_dir": str(config.SCALERS_DIR),
            "rows_features": int(len(feat)),
            "rows_scaled": int(len(scaled)),
            "feature_cols": feature_cols,
            "scaled_feature_cols": [f"{c}_s" for c in feature_cols],
            "scaler": {
                "type": scaler_name,
                "include_macro_levels": bool(getattr(config, "SCALE_INCLUDE_MACRO_LEVELS", False)),
            },
        },
        "splits": {
            "train_end": config.TRAIN_END,
            "val_end": config.VAL_END,
            "counts": scaled["split"].value_counts().to_dict(),
        },
    }
    write_json(step_manifest, run_dir / "manifest_step02.json")

    ok("Step 02 DONE. Next: python src/03_build_event_windows.py")


if __name__ == "__main__":
    main()

