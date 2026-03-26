# src/07_xai_numeric.py
# Step 07: numeric XAI for episode peaks (analysis window only)
#
# Official-mode version aligned with Step 05/06 official implementations.
# - Loads OFFICIAL PatchTST + OFFICIAL TranAD from external repos
# - Uses separate branch windows (PatchTST window != TranAD window)
# - Rebuilds branch calibration from anomalies_base TRAIN rows (percentile calibration)
# - Uses Step 06 chosen fusion weights from manifest_step06.json
# - Fixes old stability bug by recomputing BOTH PatchTST and TranAD feature importance for jittered dates

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from types import ModuleType
import importlib
import importlib.util
import json
import math
import sys

import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import require, fail, ok, require_columns, require_unique_key  # noqa: E402
from utils.io import read_parquet, read_csv, write_csv, write_json, sha256_file  # noqa: E402

import torch
import torch.nn as nn

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
    "vix": "VIX level",
    "dgs10": "10Y Treasury yield",
    "fedfunds": "Fed funds rate",
    "cpi": "CPI level",
    "vix_diff": "VIX change",
    "dgs10_diff": "10Y yield change",
    "fedfunds_diff": "Fed funds change",
    "cpi_diff": "CPI change",
}


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
    if n in {"vix", "dgs10", "fedfunds", "cpi", "vix_diff", "dgs10_diff", "fedfunds_diff", "cpi_diff"}:
        return "macro"
    return "other"


def _feature_pretty(name: str) -> str:
    return FEATURE_PRETTY.get(str(name), str(name).replace("_", " "))


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


class _PatchTSTConfig:
    pass


def _run_dir_latest() -> Path:
    p = config.RUNS_DIR / "latest"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _device() -> torch.device:
    return torch.device(config.DEVICE if torch.cuda.is_available() and str(config.DEVICE).startswith("cuda") else "cpu")


def _require_official_repo_dirs() -> None:
    require(config.PATCHTST_SUPERVISED_DIR.exists(), f"Missing official PatchTST repo dir: {config.PATCHTST_SUPERVISED_DIR}")
    require(config.TRANAD_SRC_DIR.exists(), f"Missing official TranAD repo dir: {config.TRANAD_SRC_DIR}")



def _inject_dgl_stub_if_needed() -> bool:
    if importlib.util.find_spec("dgl") is not None:
        return False
    dgl_mod = ModuleType("dgl")
    dgl_nn_mod = ModuleType("dgl.nn")

    class _GATConvStub:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("dgl.GATConv stub should never be instantiated for TranAD-only runs.")

    dgl_nn_mod.GATConv = _GATConvStub
    dgl_mod.nn = dgl_nn_mod
    sys.modules.setdefault("dgl", dgl_mod)
    sys.modules.setdefault("dgl.nn", dgl_nn_mod)
    return True



def _apply_tranad_torch_compat_patch() -> Dict[str, bool]:
    sys.path.insert(0, str(config.TRANAD_REPO_DIR))
    tran_dl = importlib.import_module("src.dlutils")
    patched = {"encoder": False, "decoder": False}

    enc_cls = getattr(tran_dl, "TransformerEncoderLayer")
    if not getattr(enc_cls.forward, "_finanomx_compat", False):
        old_forward = enc_cls.forward

        def _compat_forward_encoder(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, _old_forward=old_forward, **kwargs):
            return _old_forward(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        _compat_forward_encoder._finanomx_compat = True  # type: ignore[attr-defined]
        enc_cls.forward = _compat_forward_encoder
        patched["encoder"] = True

    dec_cls = getattr(tran_dl, "TransformerDecoderLayer")
    if not getattr(dec_cls.forward, "_finanomx_compat", False):
        old_forward = dec_cls.forward

        def _compat_forward_decoder(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            tgt_is_causal=False,
            memory_is_causal=False,
            _old_forward=old_forward,
            **kwargs,
        ):
            return _old_forward(
                self,
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        _compat_forward_decoder._finanomx_compat = True  # type: ignore[attr-defined]
        dec_cls.forward = _compat_forward_decoder
        patched["decoder"] = True
    return patched



def _import_official_classes() -> Tuple[type, type, Dict[str, bool]]:
    _require_official_repo_dirs()
    sys.path.insert(0, str(config.PATCHTST_REPO_DIR))
    sys.path.insert(0, str(config.PATCHTST_SUPERVISED_DIR))
    sys.path.insert(0, str(config.TRANAD_REPO_DIR))
    dgl_stub_used = _inject_dgl_stub_if_needed()
    compat_patch = _apply_tranad_torch_compat_patch()

    PatchModule = importlib.import_module("models.PatchTST")
    TranModule = importlib.import_module("src.models")
    PatchTSTModel = getattr(PatchModule, "Model")
    TranADModel = getattr(TranModule, "TranAD")
    compat_patch["dgl_stub_used"] = dgl_stub_used
    return PatchTSTModel, TranADModel, compat_patch



def _load_ckpt(path: Path, device: torch.device) -> dict:
    require(path.exists(), f"Missing checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    require(isinstance(ckpt, dict) and "state_dict" in ckpt, f"Bad checkpoint format: {path}")
    return ckpt



def _build_patchtst_model(ckpt: dict, PatchTSTModel: type, device: torch.device) -> torch.nn.Module:
    cfg_dict = ckpt.get("patchtst_config")
    require(isinstance(cfg_dict, dict) and len(cfg_dict) > 0, "PatchTST checkpoint missing patchtst_config")
    cfg = _PatchTSTConfig()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    model = PatchTSTModel(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model



def _build_tranad_model(ckpt: dict, TranADModel: type, device: torch.device) -> torch.nn.Module:
    feat_cols = ckpt.get("feature_cols", [])
    require(len(feat_cols) > 0, "TranAD checkpoint missing feature_cols")
    model = TranADModel(len(feat_cols)).to(device).double()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model



def _patch_forward_scalar(model: nn.Module, x: torch.Tensor, target_idx: int) -> torch.Tensor:
    pred = model(x)
    if pred.ndim == 3:
        pred = pred[:, -1, target_idx]
    elif pred.ndim == 2:
        pred = pred[:, target_idx] if pred.shape[1] > 1 else pred[:, 0]
    else:
        pred = pred.view(-1)
    return pred.view(-1)



def _ig_for_patch_loss(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor, y_true: torch.Tensor, target_idx: int, steps: int = 32) -> torch.Tensor:
    model.eval()
    require(x.shape == baseline.shape, "IG: x and baseline shape mismatch.")
    require(x.dim() == 3 and x.size(0) == 1, "IG: x must be [1,W,F].")
    total_grad = torch.zeros_like(x)
    for s in range(1, steps + 1):
        alpha = float(s) / float(steps)
        xa = baseline + alpha * (x - baseline)
        xa.requires_grad_(True)
        pred = _patch_forward_scalar(model, xa, target_idx=target_idx)
        loss = (pred - y_true.view(-1)).pow(2).mean()
        model.zero_grad(set_to_none=True)
        if xa.grad is not None:
            xa.grad.zero_()
        loss.backward()
        total_grad += xa.grad.detach()
    avg_grad = total_grad / float(steps)
    return (x - baseline) * avg_grad



def _saliency_for_patch_loss(model: nn.Module, x: torch.Tensor, y_true: torch.Tensor, target_idx: int) -> torch.Tensor:
    model.eval()
    xb = x.clone().detach().requires_grad_(True)
    pred = _patch_forward_scalar(model, xb, target_idx=target_idx)
    loss = (pred - y_true.view(-1)).pow(2).mean()
    model.zero_grad(set_to_none=True)
    if xb.grad is not None:
        xb.grad.zero_()
    loss.backward()
    return xb.grad.detach()



def _rolling_z(series: pd.Series, lookback: int) -> pd.Series:
    m = series.rolling(lookback, min_periods=max(30, lookback // 4)).mean()
    s = series.rolling(lookback, min_periods=max(30, lookback // 4)).std(ddof=0)
    return (series - m) / s.replace(0.0, np.nan)



def _topk_mean_1d(x: np.ndarray, k: int) -> float:
    k = int(min(k, len(x)))
    if k <= 0:
        return float(np.mean(x))
    return float(np.mean(np.partition(x, -k)[-k:]))



def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    try:
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        c = np.corrcoef(ra, rb)[0, 1]
        return float(c) if np.isfinite(c) else float("nan")
    except Exception:
        return float("nan")



def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    try:
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if np.isfinite(c) else float("nan")
    except Exception:
        return float("nan")



def _select_train_reference(train_scores: np.ndarray, prefer_nonzero: bool = True) -> np.ndarray:
    x = np.asarray(train_scores, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([0.0], dtype=float)
    if prefer_nonzero:
        x2 = x[x > 0]
        if len(x2) >= max(20, int(0.1 * len(x))):
            x = x2
    return np.sort(x)



def _empirical_percentile(value: float, train_ref_sorted: np.ndarray) -> float:
    x = np.asarray(train_ref_sorted, dtype=float)
    if len(x) == 0:
        return 0.0
    pos = np.searchsorted(x, float(value), side="right")
    q = float(pos) / float(len(x))
    return float(np.clip(q, 0.0, 1.0))



def _build_calibration_refs(anom: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    refs: Dict[str, Dict[str, np.ndarray]] = {}
    train = anom[anom["split"] == "train"].copy()
    for tkr, sub in train.groupby("ticker"):
        refs[str(tkr)] = {
            "patch": _select_train_reference(sub["score_patchtst_raw"].astype(float).to_numpy(), prefer_nonzero=True),
            "tranad": _select_train_reference(sub["score_tranad_raw"].astype(float).to_numpy(), prefer_nonzero=True),
            "chronos": _select_train_reference(sub["score_chronos_raw"].astype(float).to_numpy(), prefer_nonzero=True),
        }
    return refs



def _tranad_forward(model: nn.Module, x_window: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    xb = torch.tensor(x_window[None, :, :], dtype=torch.float64, device=device)
    window = xb.permute(1, 0, 2)
    elem = window[-1:, :, :]
    with torch.no_grad():
        z = model(window, elem)
        if isinstance(z, tuple):
            z = z[1]
        err = ((z - elem) ** 2).detach().cpu().numpy()[0, 0]  # [F]
    return z.detach().cpu().numpy()[0, 0], err.astype(float)



def _patch_score_at_day(model: nn.Module, X: np.ndarray, y: np.ndarray, i: int, patch_window: int, tail_k: int, target_idx: int, device: torch.device, feat_del_idx=None, time_del_idx=None, baseline=None) -> Tuple[float, str]:
    mode = "rolling_topk"
    if i < patch_window:
        return 0.0, mode
    x_series = X.copy()
    if baseline is not None:
        if feat_del_idx:
            x_series[:, feat_del_idx] = baseline[feat_del_idx]
        if time_del_idx:
            valid = [t for t in time_del_idx if 0 <= t < patch_window]
            # local occlusion is handled per window below for rolling_topk
    if i < (2 * patch_window - 1):
        mode = "local_abs_error"
        xw = X[i - patch_window: i].copy()
        if baseline is not None:
            if feat_del_idx:
                xw[:, feat_del_idx] = baseline[feat_del_idx]
            if time_del_idx:
                valid = [t for t in time_del_idx if 0 <= t < patch_window]
                xw[valid, :] = baseline
        xt = torch.tensor(xw[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = _patch_forward_scalar(model, xt, target_idx=target_idx).detach().cpu().numpy().reshape(-1)[0]
        return float(abs(pred - float(y[i]))), mode

    start_day = i - patch_window + 1
    days = np.arange(start_day, i + 1)
    starts = days - patch_window
    batch = np.stack([X[s: s + patch_window].copy() for s in starts], axis=0)
    if baseline is not None:
        if feat_del_idx:
            batch[:, :, feat_del_idx] = baseline[feat_del_idx]
        if time_del_idx:
            valid = [t for t in time_del_idx if 0 <= t < patch_window]
            if valid:
                batch[:, valid, :] = baseline
    xb = torch.tensor(batch, dtype=torch.float32, device=device)
    y_true = torch.tensor(y[days], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = _patch_forward_scalar(model, xb, target_idx=target_idx).detach()
        err = torch.abs(pred - y_true).detach().cpu().numpy()
    return _topk_mean_1d(err, tail_k), mode



def _tranad_score_at_day(model: nn.Module, X: np.ndarray, i: int, tran_window: int, tail_k: int, device: torch.device, feat_del_idx=None, time_del_idx=None, baseline=None) -> Tuple[float, np.ndarray, np.ndarray]:
    require(i >= tran_window - 1, f"Not enough history for TranAD window at index={i}")
    xw = X[i - tran_window + 1: i + 1].copy()
    if baseline is not None:
        if feat_del_idx:
            xw[:, feat_del_idx] = baseline[feat_del_idx]
        if time_del_idx:
            valid = [t for t in time_del_idx if 0 <= t < tran_window]
            if valid:
                xw[valid, :] = baseline
    _, feat_err = _tranad_forward(model, xw, device=device)
    score = _topk_mean_1d(feat_err, tail_k)
    # temporal occlusion sensitivity within the local TranAD window
    time_imp = np.zeros(tran_window, dtype=float)
    if baseline is not None:
        for t in range(tran_window):
            x_alt = xw.copy()
            x_alt[t, :] = baseline
            _, feat_err_alt = _tranad_forward(model, x_alt, device=device)
            score_alt = _topk_mean_1d(feat_err_alt, tail_k)
            time_imp[t] = max(0.0, score - score_alt)
    return float(score), feat_err.astype(float), time_imp.astype(float)



def _align_tranad_time_to_patch(tran_time: np.ndarray, patch_window: int, tran_window: int) -> np.ndarray:
    out = np.zeros(patch_window, dtype=float)
    start = patch_window - tran_window
    out[start:start + tran_window] = tran_time
    return out



def main() -> None:
    run_dir = _run_dir_latest()
    log_path = run_dir / "logs_step07_xai_numeric.txt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")
    logger.info("=== Step 07: XAI numeric (episode peaks only, OFFICIAL PatchTST + OFFICIAL TranAD) ===")

    require((config.RESULTS_DIR / "anomalies_base.csv").exists(), "Missing anomalies_base.csv. Run Step 06 first.")
    require(config.SCALED_PATH.exists(), "Missing scaled.parquet. Run Step 02 first.")
    require(config.FEATURES_PATH.exists(), "Missing features.parquet. Run Step 02 first.")

    patch_path = config.DL_DIR / "patchtst_forecaster.pth"
    tranad_path = config.DL_DIR / "tranad_detector.pth"
    require(patch_path.exists(), f"Missing DL model: {patch_path}. Run Step 05 first.")
    require(tranad_path.exists(), f"Missing DL model: {tranad_path}. Run Step 05 first.")

    m6_path = run_dir / "manifest_step06.json"
    require(m6_path.exists(), "Missing runs/latest/manifest_step06.json.")
    m6 = json.loads(m6_path.read_text(encoding="utf-8"))
    main_flag_col = m6.get("main_flag_col")
    require(isinstance(main_flag_col, str) and main_flag_col, "manifest_step06.json missing 'main_flag_col'.")
    fusion = m6.get("fusion", {})
    PATCH_W = float(fusion.get("chosen_patch_weight", getattr(config, "MAIN_PATCH_WEIGHT", 0.5)))
    TRAN_W = float(fusion.get("chosen_tranad_weight", getattr(config, "MAIN_TRANAD_WEIGHT", 0.5)))
    require(abs((PATCH_W + TRAN_W) - 1.0) < 1e-9, "Chosen fusion weights must sum to 1.0")

    scaled = read_parquet(config.SCALED_PATH).sort_values(["ticker", "date"]).reset_index(drop=True)
    feats = read_parquet(config.FEATURES_PATH).sort_values(["ticker", "date"]).reset_index(drop=True)
    anomalies = read_csv(config.RESULTS_DIR / "anomalies_base.csv")
    anomalies["date"] = pd.to_datetime(anomalies["date"])

    for df, name in [(scaled, "scaled"), (feats, "features")]:
        require_columns(df, ["ticker", "date", "split"], where=name)
        require_unique_key(df, ["ticker", "date"], where=name)

    require_columns(
        anomalies,
        [
            "ticker", "date", "split", "is_episode_peak", main_flag_col, "score_patchtst_tranad",
            "score_patchtst_raw", "score_tranad_raw", "score_patchtst", "score_tranad"
        ],
        where="anomalies_base",
    )
    require_unique_key(anomalies, ["ticker", "date"], where="anomalies_base")

    calib_refs = _build_calibration_refs(anomalies)

    a0 = pd.to_datetime(config.ANALYSIS_START)
    a1 = pd.to_datetime(config.ANALYSIS_END)
    peaks = anomalies[
        (anomalies["date"] >= a0)
        & (anomalies["date"] <= a1)
        & (anomalies["is_episode_peak"] == 1)
        & (anomalies[main_flag_col] == 1)
    ].copy()
    require(len(peaks) > 0, "No peaks found in analysis window.")

    device = _device()
    logger.info(f"Device: {device}")

    PatchTSTModel, TranADModel, compat_info = _import_official_classes()
    ckpt_patch = _load_ckpt(patch_path, device)
    ckpt_tr = _load_ckpt(tranad_path, device)

    feat_cols_s = ckpt_patch.get("feature_cols", [])
    require(isinstance(feat_cols_s, list) and len(feat_cols_s) > 0, "patch checkpoint missing feature_cols")
    feat_cols_s_tr = ckpt_tr.get("feature_cols", [])
    require(feat_cols_s_tr == feat_cols_s, "patch and tranad checkpoints have different feature_cols ordering")
    for c in feat_cols_s:
        if c not in scaled.columns:
            fail(f"scaled.parquet missing feature col required by checkpoint: {c}")

    patch_window = int(ckpt_patch.get("window", config.WINDOW_SIZE))
    tran_window = int(ckpt_tr.get("window", getattr(config, "TRANAD_WINDOW", 10)))
    patch_target_idx = int(ckpt_patch.get("target_idx", 0))
    require(0 <= patch_target_idx < len(feat_cols_s), f"Invalid PatchTST target_idx={patch_target_idx}")

    patch_model = _build_patchtst_model(ckpt_patch, PatchTSTModel, device)
    tranad = _build_tranad_model(ckpt_tr, TranADModel, device)

    base_vec = {}
    for t in config.TICKERS:
        sub = scaled[scaled["ticker"] == t].copy()
        trn = sub[sub["split"] == "train"]
        require(len(trn) > 0, f"No TRAIN rows for ticker {t} in scaled.parquet.")
        base_vec[t] = trn[feat_cols_s].mean(axis=0).to_numpy(dtype=np.float32)

    LOOKBACK = 252
    feats_idx = feats.set_index(["ticker", "date"])
    raw_cols = set(feats.columns)
    patch_target_raw = feat_cols_s[patch_target_idx][:-2] if feat_cols_s[patch_target_idx].endswith("_s") else feat_cols_s[patch_target_idx]

    out_rows = []
    temporal = {}
    faith_rows = []
    stab_rows = []

    tail_k = int(config.TAIL_K)
    patch_tail_k = int(min(tail_k, patch_window))
    tran_tail_k = int(min(tail_k, len(feat_cols_s)))

    logger.info(f"Peaks to explain: {len(peaks)} | patch_window={patch_window} tran_window={tran_window}")

    def calibrate_branch(tkr: str, raw_score: float, branch: str) -> float:
        return _empirical_percentile(float(raw_score), calib_refs[tkr][branch])

    def combine_main_score(tkr: str, patch_raw: float, tran_raw: float) -> tuple[float, float, float]:
        patch_cal = calibrate_branch(tkr, patch_raw, "patch")
        tran_cal = calibrate_branch(tkr, tran_raw, "tranad")
        main = PATCH_W * patch_cal + TRAN_W * tran_cal
        return float(main), float(patch_cal), float(tran_cal)

    for r in peaks.itertuples(index=False):
        tkr = str(r.ticker)
        dt = pd.to_datetime(r.date)

        sub_s = scaled[scaled["ticker"] == tkr].sort_values("date").reset_index(drop=True)
        sub_f = feats[feats["ticker"] == tkr].sort_values("date").reset_index(drop=True)

        idx = sub_s.index[sub_s["date"] == dt]
        require(len(idx) == 1, f"Cannot locate peak date in scaled data: {tkr} {dt.date()}")
        i = int(idx[0])
        require(i >= patch_window, f"Not enough history for PatchTST window at {tkr} {dt.date()}")
        require(i >= tran_window - 1, f"Not enough history for TranAD window at {tkr} {dt.date()}")

        X = sub_s[feat_cols_s].to_numpy(dtype=np.float32)
        y_patch = sub_s[feat_cols_s[patch_target_idx]].to_numpy(dtype=np.float32)
        baseline = base_vec[tkr].astype(np.float32)

        # --- Official TranAD components ---
        tran_score_base, tran_feat_imp, tran_time_local = _tranad_score_at_day(
            tranad, X, i, tran_window=tran_window, tail_k=tran_tail_k, device=device, baseline=baseline
        )
        tran_time_imp = _align_tranad_time_to_patch(tran_time_local, patch_window=patch_window, tran_window=tran_window)

        # --- Official PatchTST components ---
        x_pa = torch.tensor(X[i - patch_window: i][None, :, :], dtype=torch.float32, device=device)
        y_true = torch.tensor([float(y_patch[i])], dtype=torch.float32, device=device)
        baseline_w = np.tile(baseline, (patch_window, 1)).astype(np.float32)
        b = torch.tensor(baseline_w[None, :, :], dtype=torch.float32, device=device)

        attr = _ig_for_patch_loss(patch_model, x_pa, b, y_true, target_idx=patch_target_idx, steps=32).squeeze(0).detach().cpu().numpy()
        patch_feat_imp = np.abs(attr).mean(axis=0)
        patch_time_imp = np.abs(attr).mean(axis=1)

        # combined feature/time importance on the official fusion weights from Step 06
        comb_feat = PATCH_W * patch_feat_imp + TRAN_W * tran_feat_imp
        comb_time = PATCH_W * patch_time_imp + TRAN_W * tran_time_imp
        denom = float(np.sum(np.abs(comb_feat))) if float(np.sum(np.abs(comb_feat))) > 0 else 1.0
        comb_norm = comb_feat / denom

        order = np.argsort(-comb_norm)[:10]
        temporal_key = f"{tkr}|{dt.date().isoformat()}"
        temporal[temporal_key] = {
            "ticker": tkr,
            "date": dt.date().isoformat(),
            "patch_window": int(patch_window),
            "tranad_window": int(tran_window),
            "time_importance": [float(x) for x in comb_time.tolist()],
        }

        for rank, j in enumerate(order, start=1):
            f_s = feat_cols_s[j]
            f_raw = f_s[:-2] if f_s.endswith("_s") else f_s
            value = np.nan
            z = np.nan
            if (tkr, dt) in feats_idx.index and f_raw in raw_cols:
                value = float(feats_idx.loc[(tkr, dt), f_raw])
                ser = sub_f.set_index("date")[f_raw].astype(float)
                ser_z = _rolling_z(ser, LOOKBACK)
                if dt in ser_z.index:
                    z = float(ser_z.loc[dt]) if pd.notna(ser_z.loc[dt]) else np.nan

            out_rows.append(
                {
                    "ticker": tkr,
                    "date": dt.date().isoformat(),
                    "split": str(r.split),
                    "main_score": float(r.score_patchtst_tranad),
                    "feature_scaled": f_s,
                    "feature": f_raw,
                    "feature_pretty": _feature_pretty(f_raw),
                    "feature_group": _feature_group(f_raw),
                    "rank": int(rank),
                    "importance": float(comb_feat[j]),
                    "importance_norm": float(comb_norm[j]),
                    "imp_patch": float(patch_feat_imp[j]),
                    "imp_tranad": float(tran_feat_imp[j]),
                    "value_raw": value,
                    "z_lookback": z,
                    "z_bucket": _z_bucket(z),
                }
            )

        patch_score_base, patch_mode = _patch_score_at_day(
            patch_model, X, y_patch, i, patch_window=patch_window, tail_k=patch_tail_k,
            target_idx=patch_target_idx, device=device, baseline=baseline
        )
        main_base, patch_cal_base, tran_cal_base = combine_main_score(tkr, patch_score_base, tran_score_base)
        ref_step06 = float(r.score_patchtst_tranad)
        score_diff = float(main_base - ref_step06)

        top1 = [int(order[0])]
        top3 = [int(x) for x in order[:3].tolist()]
        top_time_patch = [int(x) for x in np.argsort(-patch_time_imp)[:min(tail_k, patch_window)].tolist()]
        top_time_tran_global = [int(x) for x in np.argsort(-tran_time_imp)[:min(tail_k, patch_window)].tolist()]
        top_time_union = sorted(set(top_time_patch).union(set(top_time_tran_global)))

        variants = [
            ("baseline", None, None, None),
            ("delete_top1_feature", top1, None, None),
            ("delete_top3_features", top3, None, None),
            ("delete_topk_time_union", None, top_time_union, top_time_union),
        ]

        for variant, feat_del, tdel_patch_global, tdel_tran_global in variants:
            patch_score_v, _ = _patch_score_at_day(
                patch_model, X, y_patch, i, patch_window=patch_window, tail_k=patch_tail_k,
                target_idx=patch_target_idx, device=device, feat_del_idx=feat_del,
                time_del_idx=tdel_patch_global, baseline=baseline,
            )
            tdel_tran_local = None
            if tdel_tran_global is not None:
                start = patch_window - tran_window
                tdel_tran_local = [g - start for g in tdel_tran_global if start <= g < patch_window]
            tran_score_v, _, _ = _tranad_score_at_day(
                tranad, X, i, tran_window=tran_window, tail_k=tran_tail_k, device=device,
                feat_del_idx=feat_del, time_del_idx=tdel_tran_local, baseline=baseline,
            )
            main_v, patch_cal_v, tran_cal_v = combine_main_score(tkr, patch_score_v, tran_score_v)
            drop_abs = float(main_base - main_v)
            drop_pct = float(drop_abs / main_base) if main_base != 0 else float("nan")

            faith_rows.append(
                {
                    "ticker": tkr,
                    "date": dt.date().isoformat(),
                    "variant": variant,
                    "patch_mode": patch_mode,
                    "ref_step06_score": ref_step06,
                    "recomputed_baseline_score": float(main_base),
                    "recomputed_variant_score": float(main_v),
                    "drop_abs": drop_abs,
                    "drop_pct": drop_pct,
                    "patch_score_base": float(patch_score_base),
                    "tran_score_base": float(tran_score_base),
                    "patch_score_variant": float(patch_score_v),
                    "tran_score_variant": float(tran_score_v),
                    "patch_cal_base": float(patch_cal_base),
                    "tran_cal_base": float(tran_cal_base),
                    "patch_cal_variant": float(patch_cal_v),
                    "tran_cal_variant": float(tran_cal_v),
                    "deleted_feature_idx": "" if feat_del is None else "|".join([str(x) for x in feat_del]),
                    "deleted_feature_names": "" if feat_del is None else "|".join([feat_cols_s[x] for x in feat_del]),
                    "deleted_time_idx_union": "" if variant != "delete_topk_time_union" else "|".join([str(x) for x in top_time_union]),
                    "score_diff_to_step06": score_diff,
                    "faithfulness_pass": int(drop_abs > 0.0) if variant != "baseline" else 1,
                }
            )

        # stability: recompute BOTH PatchTST and TranAD importances for jittered dates
        sal0 = _saliency_for_patch_loss(patch_model, x_pa, y_true, target_idx=patch_target_idx).squeeze(0).detach().cpu().numpy()
        patch_feat0 = np.abs(sal0).mean(axis=0)
        patch_time0 = np.abs(sal0).mean(axis=1)
        comb_feat0 = PATCH_W * patch_feat0 + TRAN_W * tran_feat_imp
        comb_time0 = PATCH_W * patch_time0 + TRAN_W * tran_time_imp
        denom0 = float(np.sum(np.abs(comb_feat0))) if float(np.sum(np.abs(comb_feat0))) > 0 else 1.0
        comb_norm0 = comb_feat0 / denom0
        top5_0 = set(np.argsort(-comb_norm0)[:5].tolist())
        top10_0 = set(np.argsort(-comb_norm0)[:10].tolist())

        for delta in [-3, -2, -1, 1, 2, 3]:
            j = i + int(delta)
            if j < patch_window or j < (tran_window - 1) or j >= len(sub_s):
                continue
            dt_j = pd.to_datetime(sub_s.loc[j, "date"])
            if not (a0 <= dt_j <= a1):
                continue

            # recompute official TranAD feature+time at jittered date
            tran_score_j, tran_feat_j, tran_time_local_j = _tranad_score_at_day(
                tranad, X, j, tran_window=tran_window, tail_k=tran_tail_k, device=device, baseline=baseline
            )
            tran_time_j = _align_tranad_time_to_patch(tran_time_local_j, patch_window=patch_window, tran_window=tran_window)

            # recompute official PatchTST saliency at jittered date
            x_pa_j = torch.tensor(X[j - patch_window: j][None, :, :], dtype=torch.float32, device=device)
            y_true_j = torch.tensor([float(y_patch[j])], dtype=torch.float32, device=device)
            sal_j = _saliency_for_patch_loss(patch_model, x_pa_j, y_true_j, target_idx=patch_target_idx).squeeze(0).detach().cpu().numpy()
            patch_feat_j = np.abs(sal_j).mean(axis=0)
            patch_time_j = np.abs(sal_j).mean(axis=1)

            comb_feat_j = PATCH_W * patch_feat_j + TRAN_W * tran_feat_j
            comb_time_j = PATCH_W * patch_time_j + TRAN_W * tran_time_j
            denom_j = float(np.sum(np.abs(comb_feat_j))) if float(np.sum(np.abs(comb_feat_j))) > 0 else 1.0
            comb_norm_j = comb_feat_j / denom_j

            top5_j = set(np.argsort(-comb_norm_j)[:5].tolist())
            top10_j = set(np.argsort(-comb_norm_j)[:10].tolist())

            stab_rows.append(
                {
                    "ticker": tkr,
                    "date": dt.date().isoformat(),
                    "jitter_days": int(delta),
                    "jitter_date": dt_j.date().isoformat(),
                    "tran_score_j": float(tran_score_j),
                    "overlap_top5": float(len(top5_0.intersection(top5_j)) / 5.0),
                    "overlap_top10": float(len(top10_0.intersection(top10_j)) / 10.0),
                    "spearman_feat": _spearman(comb_norm0, comb_norm_j),
                    "pearson_time": _pearson(comb_time0, comb_time_j),
                }
            )

    out_df = pd.DataFrame(out_rows)
    require(len(out_df) > 0, "xai_feature empty (unexpected).")
    write_csv(out_df, config.XAI_FEATURE_PATH)
    write_json(temporal, config.XAI_TEMPORAL_PATH)

    faith_df = pd.DataFrame(faith_rows)
    require(len(faith_df) > 0, "xai_faithfulness empty (unexpected).")
    write_csv(faith_df, config.XAI_FAITHFULNESS_PATH)

    stab_df = pd.DataFrame(stab_rows)
    require(len(stab_df) > 0, "xai_stability empty (unexpected).")
    write_csv(stab_df, config.XAI_STABILITY_PATH)

    manifest = {
        "step": "07_xai_numeric",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "official_repo_only",
        "inputs": {
            "scaled_path": str(config.SCALED_PATH),
            "scaled_sha256": sha256_file(config.SCALED_PATH),
            "features_path": str(config.FEATURES_PATH),
            "features_sha256": sha256_file(config.FEATURES_PATH),
            "anomalies_base": str(config.RESULTS_DIR / "anomalies_base.csv"),
            "anomalies_base_sha256": sha256_file(config.RESULTS_DIR / "anomalies_base.csv"),
            "manifest_step06": str(m6_path),
            "ckpt_patch": str(patch_path),
            "ckpt_tranad": str(tranad_path),
        },
        "outputs": {
            "xai_feature": {"path": str(config.XAI_FEATURE_PATH), "sha256": sha256_file(config.XAI_FEATURE_PATH), "rows": int(len(out_df))},
            "xai_temporal": {"path": str(config.XAI_TEMPORAL_PATH), "bytes": int(config.XAI_TEMPORAL_PATH.stat().st_size)},
            "xai_faithfulness": {"path": str(config.XAI_FAITHFULNESS_PATH), "sha256": sha256_file(config.XAI_FAITHFULNESS_PATH), "rows": int(len(faith_df))},
            "xai_stability": {"path": str(config.XAI_STABILITY_PATH), "sha256": sha256_file(config.XAI_STABILITY_PATH), "rows": int(len(stab_df))},
        },
        "scope": "episode peaks only, analysis window only",
        "main_flag_col": main_flag_col,
        "patch_window": int(patch_window),
        "tranad_window": int(tran_window),
        "tail_k": int(config.TAIL_K),
        "fusion_weights_from_step06": {"patch": PATCH_W, "tranad": TRAN_W},
        "official_impl": {
            "patchtst": "yuqinie98/PatchTST",
            "tranad": "imperial-qore/TranAD",
            "compat_patch": compat_info,
        },
        "notes": {
            "no_leak": "analysis peaks from step06; branch calibration rebuilt from anomalies_base TRAIN rows only",
            "stability_fix": "recomputes BOTH PatchTST and TranAD feature importance for jittered dates",
        },
    }
    (run_dir / "manifest_step07.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok(f"Saved XAI -> {config.XAI_FEATURE_PATH}")
    ok(f"Saved temporal XAI -> {config.XAI_TEMPORAL_PATH}")
    ok(f"Saved faithfulness -> {config.XAI_FAITHFULNESS_PATH}")
    ok(f"Saved stability -> {config.XAI_STABILITY_PATH}")
    ok("Step 07 DONE. Next: python src/08_news_on_demand.py")


if __name__ == "__main__":
    main()

