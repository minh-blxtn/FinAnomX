# src/05_train_models.py
# Step 05: train OFFICIAL PatchTST + OFFICIAL TranAD for research-grade anomaly detection.
#
# What is fixed in this version:
# 1) Keep OFFICIAL repo-backed implementations for both PatchTST and TranAD.
# 2) Change PatchTST from single-target next-day return forecasting to
#    a 2-target next-day forecasting branch aligned to the point-wise GT.
# 3) Save explicit metadata so Step 06 can build multivariate forecast residual scores
#    instead of relying on a single-target residual that mismatches the evaluation target.
# 4) Keep compatibility shims only for environment/runtime compatibility with newer PyTorch,
#    without changing the official model architectures.

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import importlib
import importlib.util
import random
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple

from dotenv import load_dotenv
from loguru import logger
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, fail, require, require_unique_key  # noqa: E402
from utils.io import ensure_dir, read_parquet, sha256_file, write_json  # noqa: E402

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader


# ---------------- Runtime helpers ----------------
EPOCHS = int(getattr(config, "STEP05_EPOCHS", 20))
BATCH_SIZE = int(getattr(config, "STEP05_BATCH_SIZE", 64))
LR = float(getattr(config, "STEP05_LR", 1e-3))
WEIGHT_DECAY = float(getattr(config, "STEP05_WEIGHT_DECAY", 1e-4))
PATIENCE = int(getattr(config, "STEP05_PATIENCE", 5))
GRAD_CLIP = float(getattr(config, "STEP05_GRAD_CLIP", 1.0))
PATCH_WINDOW = int(getattr(config, "WINDOW_SIZE", 60))
TRANAD_WINDOW = int(getattr(config, "TRANAD_WINDOW", 10))
TRANAD_BATCH_SIZE = int(getattr(config, "TRANAD_BATCH_SIZE", 128))
TRANAD_LR = float(getattr(config, "TRANAD_LR", 1e-3))
DEVICE_NAME = str(getattr(config, "DEVICE", "cpu"))
SEED = int(getattr(config, "SEED", 42))

# PatchTST official repo config defaults. These mirror the repo surface, but are read from config when present.
PATCHTST_PRED_LEN = int(getattr(config, "PATCHTST_PRED_LEN", 1))
PATCHTST_E_LAYERS = int(getattr(config, "PATCHTST_E_LAYERS", 3))
PATCHTST_N_HEAD = int(getattr(config, "PATCHTST_N_HEAD", 8))
PATCHTST_D_MODEL = int(getattr(config, "PATCHTST_D_MODEL", 128))
PATCHTST_D_FF = int(getattr(config, "PATCHTST_D_FF", 256))
PATCHTST_DROPOUT = float(getattr(config, "PATCHTST_DROPOUT", 0.2))
PATCHTST_FC_DROPOUT = float(getattr(config, "PATCHTST_FC_DROPOUT", 0.2))
PATCHTST_HEAD_DROPOUT = float(getattr(config, "PATCHTST_HEAD_DROPOUT", 0.0))
PATCHTST_INDIVIDUAL = bool(getattr(config, "PATCHTST_INDIVIDUAL", False))
PATCHTST_PATCH_LEN = int(getattr(config, "PATCHTST_PATCH_LEN", 16))
PATCHTST_STRIDE = int(getattr(config, "PATCHTST_STRIDE", 8))
PATCHTST_PADDING_PATCH = str(getattr(config, "PATCHTST_PADDING_PATCH", "end"))
PATCHTST_REVIN = bool(getattr(config, "PATCHTST_REVIN", True))
PATCHTST_AFFINE = bool(getattr(config, "PATCHTST_AFFINE", True))
PATCHTST_SUBTRACT_LAST = bool(getattr(config, "PATCHTST_SUBTRACT_LAST", False))
PATCHTST_DECOMPOSITION = bool(getattr(config, "PATCHTST_DECOMPOSITION", False))
PATCHTST_KERNEL_SIZE = int(getattr(config, "PATCHTST_KERNEL_SIZE", 25))

PATCHTST_REPO_DIR = Path(getattr(config, "PATCHTST_REPO_DIR", ROOT_DIR / "external" / "PatchTST"))
PATCHTST_SUPERVISED_DIR = Path(getattr(config, "PATCHTST_SUPERVISED_DIR", PATCHTST_REPO_DIR / "PatchTST_supervised"))
TRANAD_REPO_DIR = Path(getattr(config, "TRANAD_REPO_DIR", ROOT_DIR / "external" / "TranAD"))
TRANAD_SRC_DIR = Path(getattr(config, "TRANAD_SRC_DIR", TRANAD_REPO_DIR / "src"))

SCALED_PATH = Path(getattr(config, "SCALED_PATH", ROOT_DIR / "data" / "processed" / "scaled.parquet"))
DL_DIR = Path(getattr(config, "DL_DIR", ROOT_DIR / "models" / "dl"))
RUNS_DIR = Path(getattr(config, "RUNS_DIR", ROOT_DIR / "runs"))
TICKERS = list(getattr(config, "TICKERS", ["TSLA", "AMZN", "NVDA", "MSFT"]))

# These are the anomaly-relevant targets the forecasting branch should learn jointly.
# They are chosen to align better with Step 10, where positives are driven by return and volume shocks.
DEFAULT_PATCHTST_SCORE_COLS = [
    "log_return_s",
    "volume_z_20_s",
]


def _resolve_patchtst_score_cols(df: pd.DataFrame) -> List[str]:
    cfg_cols = getattr(config, "PATCHTST_SCORE_COLS", None)
    if cfg_cols is None:
        cols = [c for c in DEFAULT_PATCHTST_SCORE_COLS if c in df.columns]
    else:
        if isinstance(cfg_cols, str):
            cols = [c.strip() for c in cfg_cols.split(",") if c.strip()]
        else:
            cols = [str(c) for c in cfg_cols]
        cols = [c for c in cols if c in df.columns]
    require(len(cols) == 2, "This Step 05 fix requires exactly 2 PatchTST targets: log_return_s and volume_z_20_s.")
    require(cols == ["log_return_s", "volume_z_20_s"], "PatchTST targets must be exactly [\"log_return_s\", \"volume_z_20_s\"].")
    return cols


@dataclass
class SeqBatch:
    x: torch.Tensor
    y_next: torch.Tensor | None
    end_date: List[str]
    ticker: List[str]


class ForecastWindowDataset:
    def __init__(
        self,
        df_ticker: pd.DataFrame,
        feature_cols: List[str],
        window: int,
        split: str,
        target_cols: List[str],
    ):
        self.X: List[np.ndarray] = []
        self.Y: List[np.ndarray] = []
        self.end_dates: List[str] = []
        self.tickers: List[str] = []

        sub = df_ticker.copy().reset_index(drop=True)
        sub["date"] = pd.to_datetime(sub["date"])
        x_all = sub[feature_cols].astype(float).to_numpy(dtype=np.float32)
        y_all = sub[target_cols].astype(float).to_numpy(dtype=np.float32)
        splits = sub["split"].astype(str).tolist()
        dates = sub["date"].dt.strftime("%Y-%m-%d").tolist()
        ticker = str(sub["ticker"].iloc[0])

        for i in range(window - 1, len(sub) - 1):
            # Keep strict no-leak: window end and realized next-day target must remain inside the same split.
            if splits[i] != split or splits[i + 1] != split:
                continue
            self.X.append(x_all[i - window + 1 : i + 1, :])
            self.Y.append(y_all[i + 1, :].copy())
            self.end_dates.append(dates[i + 1])
            self.tickers.append(ticker)

        if len(self.X) == 0:
            fail(f"[{ticker}] No forecast windows built for split={split}, window={window}.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.end_dates[idx], self.tickers[idx]


class ReconWindowDataset:
    """Official TranAD windows; target is the last step of each window (same pattern as the official repo)."""

    def __init__(self, df_ticker: pd.DataFrame, feature_cols: List[str], window: int, split: str):
        self.X: List[np.ndarray] = []
        self.end_dates: List[str] = []
        self.tickers: List[str] = []

        sub = df_ticker.copy().reset_index(drop=True)
        sub["date"] = pd.to_datetime(sub["date"])
        x_all = sub[feature_cols].astype(float).to_numpy(dtype=np.float64)
        splits = sub["split"].astype(str).tolist()
        dates = sub["date"].dt.strftime("%Y-%m-%d").tolist()
        ticker = str(sub["ticker"].iloc[0])

        for i in range(window - 1, len(sub)):
            if splits[i] != split:
                continue
            self.X.append(x_all[i - window + 1 : i + 1, :])
            self.end_dates.append(dates[i])
            self.tickers.append(ticker)

        if len(self.X) == 0:
            fail(f"[{ticker}] No TranAD windows built for split={split}, window={window}.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], None, self.end_dates[idx], self.tickers[idx]


# ---------------- Dataloaders ----------------
def _collate_forecast(batch) -> SeqBatch:
    xs = torch.tensor(np.stack([b[0] for b in batch], axis=0), dtype=torch.float32)
    ys = torch.tensor(np.stack([b[1] for b in batch], axis=0), dtype=torch.float32)
    return SeqBatch(x=xs, y_next=ys, end_date=[b[2] for b in batch], ticker=[b[3] for b in batch])


def _collate_recon(batch) -> SeqBatch:
    xs = torch.tensor(np.stack([b[0] for b in batch], axis=0), dtype=torch.float64)
    return SeqBatch(x=xs, y_next=None, end_date=[b[2] for b in batch], ticker=[b[3] for b in batch])


def _make_loader(dataset, batch_size: int, shuffle: bool, collate_fn) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0, collate_fn=collate_fn)


# ---------------- Common helpers ----------------
def _run_dir_latest() -> Path:
    d = RUNS_DIR / "latest"
    ensure_dir(d)
    return d


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if DEVICE_NAME.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _scaled_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_s")]
    if not cols:
        fail("No scaled feature columns (*_s) found. Ensure Step 02 created scaled.parquet correctly.")
    return cols


def _checkpoint_common(feature_cols: List[str], window: int) -> dict:
    return {
        "feature_cols": feature_cols,
        "window": int(window),
        "seed": SEED,
        "device": DEVICE_NAME,
        "implementation_mode": "official_repo",
    }


def _require_official_repo_dirs() -> None:
    if not PATCHTST_SUPERVISED_DIR.exists():
        fail(
            "Missing official PatchTST repo. Expected directory: "
            f"{PATCHTST_SUPERVISED_DIR}\n"
            "Clone or extract the official repo into external/PatchTST before running Step 05."
        )
    if not TRANAD_SRC_DIR.exists():
        fail(
            "Missing official TranAD repo. Expected directory: "
            f"{TRANAD_SRC_DIR}\n"
            "Clone or extract the official repo into external/TranAD before running Step 05."
        )


# ---------------- Official repo imports ----------------
def _install_dgl_stub_if_needed() -> bool:
    if importlib.util.find_spec("dgl") is not None:
        return False
    dgl_mod = types.ModuleType("dgl")
    dgl_nn = types.ModuleType("dgl.nn")

    class _UnusedGATConv:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("dgl is required to instantiate GAT-based models from the official TranAD repo.")

    dgl_nn.GATConv = _UnusedGATConv
    dgl_mod.nn = dgl_nn
    sys.modules.setdefault("dgl", dgl_mod)
    sys.modules.setdefault("dgl.nn", dgl_nn)
    return True


def _patch_official_tranad_torch_compat() -> dict:
    sys.path.insert(0, str(TRANAD_REPO_DIR))
    sys.path.insert(0, str(TRANAD_SRC_DIR.parent))
    dlutils = importlib.import_module("src.dlutils")
    patched = {"encoder": False, "decoder": False}

    enc_cls = getattr(dlutils, "TransformerEncoderLayer", None)
    dec_cls = getattr(dlutils, "TransformerDecoderLayer", None)

    if enc_cls is not None and getattr(enc_cls.forward, "__name__", "") != "_compat_forward_encoder":
        old_forward = enc_cls.forward

        def _compat_forward_encoder(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None, _old_forward=old_forward, **kwargs):
            return _old_forward(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        enc_cls.forward = _compat_forward_encoder
        patched["encoder"] = True

    if dec_cls is not None and getattr(dec_cls.forward, "__name__", "") != "_compat_forward_decoder":
        old_forward = dec_cls.forward

        def _compat_forward_decoder(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            tgt_is_causal=None,
            memory_is_causal=None,
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

        dec_cls.forward = _compat_forward_decoder
        patched["decoder"] = True

    return patched


def _import_official_patchtst_model():
    sys.path.insert(0, str(PATCHTST_SUPERVISED_DIR))
    module = importlib.import_module("models.PatchTST")
    require(hasattr(module, "Model"), "Official PatchTST module missing Model class")
    return module.Model


def _import_official_tranad_model() -> Tuple[type, bool, dict]:
    used_stub = _install_dgl_stub_if_needed()
    patch_info = _patch_official_tranad_torch_compat()
    module = importlib.import_module("src.models")
    require(hasattr(module, "TranAD"), "Official TranAD src.models missing TranAD class")
    return module.TranAD, used_stub, patch_info


# ---------------- Model builders ----------------
def _build_patchtst_configs(enc_in: int) -> SimpleNamespace:
    return SimpleNamespace(
        enc_in=enc_in,
        seq_len=PATCH_WINDOW,
        pred_len=PATCHTST_PRED_LEN,
        e_layers=PATCHTST_E_LAYERS,
        n_heads=PATCHTST_N_HEAD,
        d_model=PATCHTST_D_MODEL,
        d_ff=PATCHTST_D_FF,
        dropout=PATCHTST_DROPOUT,
        fc_dropout=PATCHTST_FC_DROPOUT,
        head_dropout=PATCHTST_HEAD_DROPOUT,
        individual=PATCHTST_INDIVIDUAL,
        patch_len=PATCHTST_PATCH_LEN,
        stride=PATCHTST_STRIDE,
        padding_patch=PATCHTST_PADDING_PATCH,
        revin=PATCHTST_REVIN,
        affine=PATCHTST_AFFINE,
        subtract_last=PATCHTST_SUBTRACT_LAST,
        decomposition=PATCHTST_DECOMPOSITION,
        kernel_size=PATCHTST_KERNEL_SIZE,
    )


# ---------------- Training ----------------
def _eval_patchtst_targets(model, data_loader, target_indices: List[int], target_cols: List[str], device: torch.device) -> dict:
    model.eval()
    idx = torch.tensor(target_indices, dtype=torch.long, device=device)
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch.x.to(device=device, dtype=torch.float32)
            y_true = batch.y_next.to(device=device, dtype=torch.float32)
            y_hat_full = model(x)[:, -1, :]
            y_hat = torch.index_select(y_hat_full, dim=1, index=idx)
            y_true_all.append(y_true.cpu().numpy())
            y_pred_all.append(y_hat.cpu().numpy())
    import numpy as _np
    yt = _np.concatenate(y_true_all, axis=0)
    yp = _np.concatenate(y_pred_all, axis=0)
    out = {}
    for j, col in enumerate(target_cols):
        err = yp[:, j] - yt[:, j]
        corr = float(_np.corrcoef(yt[:, j], yp[:, j])[0, 1]) if _np.std(yt[:, j]) > 1e-12 and _np.std(yp[:, j]) > 1e-12 else 0.0
        out[col] = {
            "mae": float(_np.mean(_np.abs(err))),
            "rmse": float(_np.sqrt(_np.mean(err ** 2))),
            "corr": corr,
            "true_std": float(_np.std(yt[:, j])),
            "pred_std": float(_np.std(yp[:, j])),
        }
    return out

def _train_patchtst_official(model, train_loader, val_loader, target_indices: List[int], target_cols: List[str], device: torch.device) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val = float("inf")
    best_state = None
    bad = 0
    last_epoch = 0
    crit = nn.MSELoss()
    target_idx_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)

    for epoch in range(1, EPOCHS + 1):
        last_epoch = epoch
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for batch in train_loader:
            x = batch.x.to(device=device, dtype=torch.float32)
            y_true = batch.y_next.to(device=device, dtype=torch.float32)  # [B, K]
            opt.zero_grad(set_to_none=True)
            y_hat_full = model(x)[:, -1, :]  # [B, C]
            y_hat = torch.index_select(y_hat_full, dim=1, index=target_idx_tensor)
            loss = crit(y_hat, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tr_loss += float(loss.item()) * x.size(0)
            n_tr += x.size(0)
        tr_loss /= max(n_tr, 1)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.x.to(device=device, dtype=torch.float32)
                y_true = batch.y_next.to(device=device, dtype=torch.float32)
                y_hat_full = model(x)[:, -1, :]
                y_hat = torch.index_select(y_hat_full, dim=1, index=target_idx_tensor)
                loss = crit(y_hat, y_true)
                va_loss += float(loss.item()) * x.size(0)
                n_va += x.size(0)
        va_loss /= max(n_va, 1)

        logger.info(f"[official_patchtst_multivar] epoch={epoch} train={tr_loss:.6f} val={va_loss:.6f}")
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                logger.info(f"[official_patchtst_multivar] early stop at epoch={epoch}")
                break

    require(best_state is not None, "No best PatchTST state captured")
    model.load_state_dict(best_state)
    val_diag = _eval_patchtst_targets(model, val_loader, target_indices=target_indices, target_cols=target_cols, device=device)
    logger.info(f"[official_patchtst_multivar] val diagnostics by target: {val_diag}")
    return {"best_val_loss": float(best_val), "epochs_ran": int(last_epoch), "val_diagnostics": val_diag}


def _train_tranad_official(model, train_loader, val_loader, device: torch.device) -> dict:
    lr = float(getattr(model, "lr", TRANAD_LR))
    batch = int(getattr(model, "batch", TRANAD_BATCH_SIZE))
    require(batch == TRANAD_BATCH_SIZE, "TranAD batch in config must match official model.batch for reproducibility")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best_val = float("inf")
    best_state = None
    bad = 0
    last_epoch = 0
    crit = nn.MSELoss(reduction="none")

    for epoch in range(1, EPOCHS + 1):
        last_epoch = epoch
        n_epoch = epoch + 1
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for batch_data in train_loader:
            x = batch_data.x.to(device=device, dtype=torch.float64)
            local_bs = x.size(0)
            window = x.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, x.size(2))
            opt.zero_grad(set_to_none=True)
            z = model(window, elem)
            loss_t = crit(z, elem) if not isinstance(z, tuple) else (1.0 / n_epoch) * crit(z[0], elem) + (1.0 - 1.0 / n_epoch) * crit(z[1], elem)
            loss = torch.mean(loss_t)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tr_loss += float(loss.item()) * local_bs
            n_tr += local_bs
        tr_loss /= max(n_tr, 1)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for batch_data in val_loader:
                x = batch_data.x.to(device=device, dtype=torch.float64)
                local_bs = x.size(0)
                window = x.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, x.size(2))
                z = model(window, elem)
                loss_t = crit(z, elem) if not isinstance(z, tuple) else (1.0 / n_epoch) * crit(z[0], elem) + (1.0 - 1.0 / n_epoch) * crit(z[1], elem)
                loss = torch.mean(loss_t)
                va_loss += float(loss.item()) * local_bs
                n_va += local_bs
        va_loss /= max(n_va, 1)

        logger.info(f"[official_tranad] epoch={epoch} train={tr_loss:.6f} val={va_loss:.6f}")
        if va_loss < best_val - 1e-9:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                logger.info(f"[official_tranad] early stop at epoch={epoch}")
                break

    require(best_state is not None, "No best TranAD state captured")
    model.load_state_dict(best_state)
    return {"best_val_loss": float(best_val), "epochs_ran": int(last_epoch), "train_dtype": "float64"}


# ---------------- Main ----------------
def main() -> None:
    run_dir = _run_dir_latest()
    log_path = run_dir / "logs_step05_train_models.txt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")
    logger.info("=== Step 05: train OFFICIAL PatchTST + OFFICIAL TranAD (research-grade multivariate PatchTST branch) ===")

    _require_official_repo_dirs()
    require(SCALED_PATH.exists(), f"Missing input: {SCALED_PATH}. Run Step 02 first.")

    df = read_parquet(SCALED_PATH)
    require_unique_key(df, ["ticker", "date"], where="scaled input")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for c in ["ticker", "date", "split"]:
        require(c in df.columns, f"Missing column '{c}' in scaled.parquet")

    feat_cols = _scaled_feature_cols(df)
    patch_score_cols = _resolve_patchtst_score_cols(df)
    patch_score_indices = [feat_cols.index(c) for c in patch_score_cols]
    require(all(i >= 0 for i in patch_score_indices), "Failed to resolve PatchTST score target indices")

    _set_seed(SEED)
    device = _device()
    ensure_dir(DL_DIR)

    PatchTSTOfficialModel = _import_official_patchtst_model()
    TranADOfficialModel, used_dgl_stub, tranad_patch_info = _import_official_tranad_model()

    train_fore_sets, val_fore_sets = [], []
    train_recon_sets, val_recon_sets = [], []
    for t in TICKERS:
        sub = df[df["ticker"] == t].copy()
        train_fore_sets.append(ForecastWindowDataset(sub, feat_cols, window=PATCH_WINDOW, split="train", target_cols=patch_score_cols))
        val_fore_sets.append(ForecastWindowDataset(sub, feat_cols, window=PATCH_WINDOW, split="val", target_cols=patch_score_cols))
        train_recon_sets.append(ReconWindowDataset(sub, feat_cols, window=TRANAD_WINDOW, split="train"))
        val_recon_sets.append(ReconWindowDataset(sub, feat_cols, window=TRANAD_WINDOW, split="val"))

    train_fore = ConcatDataset(train_fore_sets)
    val_fore = ConcatDataset(val_fore_sets)
    train_recon = ConcatDataset(train_recon_sets)
    val_recon = ConcatDataset(val_recon_sets)

    train_fore_loader = _make_loader(train_fore, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate_forecast)
    val_fore_loader = _make_loader(val_fore, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_forecast)
    train_recon_loader = _make_loader(train_recon, batch_size=TRANAD_BATCH_SIZE, shuffle=True, collate_fn=_collate_recon)
    val_recon_loader = _make_loader(val_recon, batch_size=TRANAD_BATCH_SIZE, shuffle=False, collate_fn=_collate_recon)

    logger.info(f"PatchTST forecast windows train={len(train_fore)} val={len(val_fore)} targets={patch_score_cols}")
    logger.info(f"TranAD recon windows train={len(train_recon)} val={len(val_recon)}")

    patch_cfg = _build_patchtst_configs(enc_in=len(feat_cols))
    patch_model = PatchTSTOfficialModel(patch_cfg).to(device=device, dtype=torch.float32)
    patch_stats = _train_patchtst_official(patch_model, train_fore_loader, val_fore_loader, target_indices=patch_score_indices, target_cols=patch_score_cols, device=device)
    patch_path = DL_DIR / "patchtst_forecaster.pth"
    patch_ckpt = {
        "state_dict": patch_model.state_dict(),
        **_checkpoint_common(feat_cols, PATCH_WINDOW),
        "model_name": "patchtst_forecaster",
        "official_repo": "yuqinie98/PatchTST",
        "official_module": "PatchTST_supervised.models.PatchTST.Model",
        "task": "forecast",
        "target_col": patch_score_cols[0],  # backward-compat field for old readers
        "target_idx": int(patch_score_indices[0]),  # backward-compat field for old readers
        "forecast_score_cols": patch_score_cols,
        "forecast_score_indices": patch_score_indices,
        "forecast_score_mode": "multivariate_next_step_residual",
        "pred_len": PATCHTST_PRED_LEN,
        "patchtst_config": patch_cfg.__dict__,
    }
    torch.save(patch_ckpt, patch_path)
    ok(f"Saved official PatchTST -> {patch_path}")

    tranad = TranADOfficialModel(len(feat_cols)).to(device)
    tranad = tranad.to(dtype=torch.float64)
    require(int(getattr(tranad, "n_window", TRANAD_WINDOW)) == TRANAD_WINDOW, "Official TranAD model.n_window must match config.TRANAD_WINDOW")
    require(int(getattr(tranad, "batch", TRANAD_BATCH_SIZE)) == TRANAD_BATCH_SIZE, "Official TranAD model.batch must match config.TRANAD_BATCH_SIZE")
    tran_stats = _train_tranad_official(tranad, train_recon_loader, val_recon_loader, device=device)
    tran_path = DL_DIR / "tranad_detector.pth"
    tran_ckpt = {
        "state_dict": tranad.state_dict(),
        **_checkpoint_common(feat_cols, TRANAD_WINDOW),
        "model_name": "tranad_detector",
        "official_repo": "imperial-qore/TranAD",
        "official_module": "src.models.TranAD",
        "task": "reconstruction",
        "target_col": str(getattr(config, "TRANAD_TARGET_COL", "log_return_s")),
        "train_dtype": "float64",
        "n_window": int(getattr(tranad, "n_window", TRANAD_WINDOW)),
        "batch": int(getattr(tranad, "batch", TRANAD_BATCH_SIZE)),
        "lr": float(getattr(tranad, "lr", TRANAD_LR)),
        "dgl_stub_used": bool(used_dgl_stub),
    }
    torch.save(tran_ckpt, tran_path)
    ok(f"Saved official TranAD -> {tran_path}")

    manifest = {
        "step": "05_train_models",
        "mode": "official_repo_only",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "scaled_path": str(SCALED_PATH),
            "scaled_sha256": sha256_file(SCALED_PATH),
            "patchtst_repo": str(PATCHTST_REPO_DIR),
            "tranad_repo": str(TRANAD_REPO_DIR),
        },
        "outputs": {
            "patchtst_forecaster": {"path": str(patch_path), "sha256": sha256_file(patch_path), "train": patch_stats},
            "tranad_detector": {"path": str(tran_path), "sha256": sha256_file(tran_path), "train": tran_stats},
        },
        "official_impl": {
            "patchtst": "yuqinie98/PatchTST -> PatchTST_supervised.models.PatchTST.Model",
            "tranad": "imperial-qore/TranAD -> src.models.TranAD",
            "tranad_dgl_stub_used": bool(used_dgl_stub),
            "tranad_torch_compat_patch": tranad_patch_info,
        },
        "design_notes": {
            "patchtst_branch": "official PatchTST, trained as 2-target next-step forecaster aligned to Step 10 point-wise GT",
            "patchtst_score_cols": patch_score_cols,
            "reason": "debug showed the previous 4-target PatchTST branch learned weakly and diluted anomaly relevance; keep only return + volume shock targets",
        },
        "no_leak": {
            "patchtst": "window end and realized next-day multivariate target must stay in same split",
            "tranad": "reconstruction windows selected by window end split only",
        },
        "hyperparams": {
            "epochs": EPOCHS,
            "batch_size_patchtst": BATCH_SIZE,
            "batch_size_tranad": TRANAD_BATCH_SIZE,
            "lr_patchtst": LR,
            "lr_tranad": float(getattr(tranad, 'lr', TRANAD_LR)),
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "grad_clip": GRAD_CLIP,
            "window_size_patchtst": PATCH_WINDOW,
            "window_size_tranad": TRANAD_WINDOW,
            "patchtst_config": patch_cfg.__dict__,
            "seed": SEED,
            "device": str(device),
        },
    }
    write_json(manifest, run_dir / "manifest_step05.json")
    ok("Step 05 DONE. Next step to update is Step 06 so it can score official multivariate PatchTST residuals correctly.")


if __name__ == "__main__":
    main()

