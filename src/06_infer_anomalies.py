# src/06_infer_anomalies.py
# Step 06: infer anomalies (OFFICIAL PatchTST + OFFICIAL TranAD + Chronos benchmark)

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import sys, json, math, types, importlib
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from loguru import logger

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent if (SRC_DIR / "utils").exists() else SRC_DIR
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config
from utils.checks import ok, warn, fail, require, require_unique_key
from utils.io import ensure_dir, read_parquet, read_csv, write_csv, sha256_file, write_json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

USE_ADAPTIVE_MAIN = bool(config.USE_ADAPTIVE_MAIN)
ADAPT_LOOKBACK = int(config.ADAPT_MAIN_LOOKBACK)
ADAPT_Q = float(config.ADAPT_MAIN_Q)
USE_ADAPTIVE_ALL = bool(config.USE_ADAPTIVE_ALL)
ADAPT_ALL_LOOKBACK = int(config.ADAPT_ALL_LOOKBACK)

def _run_dir_latest() -> Path:
    d = config.RUNS_DIR / "latest"
    ensure_dir(d)
    return d

def _device() -> torch.device:
    dev = str(config.DEVICE).lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _load_ckpt(path: Path, device: torch.device) -> dict:
    require(path.exists(), f"Missing model checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    require(isinstance(ckpt, dict) and "state_dict" in ckpt, f"Bad checkpoint format: {path}")
    return ckpt

def _require_official_repo_dirs() -> None:
    require(config.PATCHTST_SUPERVISED_DIR.exists(), f"Missing official PatchTST repo dir: {config.PATCHTST_SUPERVISED_DIR}")
    require(config.TRANAD_SRC_DIR.exists(), f"Missing official TranAD repo dir: {config.TRANAD_SRC_DIR}")

def _inject_dgl_stub_if_needed() -> bool:
    if "dgl" in sys.modules:
        return False
    dgl_mod = types.ModuleType("dgl")
    dgl_nn_mod = types.ModuleType("dgl.nn")
    class _GATConvStub:
        def __init__(self,*a,**k):
            raise RuntimeError("dgl.GATConv stub should never be instantiated for TranAD-only runs.")
    dgl_nn_mod.GATConv = _GATConvStub
    dgl_mod.nn = dgl_nn_mod
    sys.modules["dgl"] = dgl_mod
    sys.modules["dgl.nn"] = dgl_nn_mod
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
        _compat_forward_encoder._finanomx_compat = True
        enc_cls.forward = _compat_forward_encoder
        patched["encoder"] = True
    dec_cls = getattr(tran_dl, "TransformerDecoderLayer")
    if not getattr(dec_cls.forward, "_finanomx_compat", False):
        old_forward = dec_cls.forward
        def _compat_forward_decoder(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=False, memory_is_causal=False, _old_forward=old_forward, **kwargs):
            return _old_forward(self, tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        _compat_forward_decoder._finanomx_compat = True
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

class _PatchTSTConfig: ...

def _build_patchtst_model(ckpt: dict, PatchTSTModel: type, device: torch.device) -> torch.nn.Module:
    cfg_dict = ckpt.get("patchtst_config")
    require(isinstance(cfg_dict, dict) and len(cfg_dict) > 0, "PatchTST checkpoint missing patchtst_config")
    cfg = _PatchTSTConfig()
    for k,v in cfg_dict.items(): setattr(cfg,k,v)
    model = PatchTSTModel(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model

def _build_tranad_model(ckpt: dict, TranADModel: type, device: torch.device) -> torch.nn.Module:
    feat_cols = ckpt.get("feature_cols", [])
    require(len(feat_cols)>0, "TranAD checkpoint missing feature_cols")
    model = TranADModel(len(feat_cols)).to(device).double()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model

def _sliding_windows(X: np.ndarray, window: int) -> np.ndarray:
    n,_ = X.shape
    require(n>=window, f"Not enough rows ({n}) for window={window}")
    idx = np.arange(window)[None,:] + np.arange(n-window+1)[:,None]
    return X[idx]

def _topk_mean_rows(arr2d: np.ndarray, k:int) -> np.ndarray:
    k_eff = int(min(max(1,k), arr2d.shape[1]))
    part = np.partition(arr2d, -k_eff, axis=1)[:, -k_eff:]
    return part.mean(axis=1)

def _rolling_topk_mean(arr: np.ndarray, window:int, k:int) -> np.ndarray:
    n=len(arr); out=np.zeros(n,dtype=float); k=int(min(k,window))
    for i in range(n):
        start=max(0, i-window+1); seg=arr[start:i+1]
        out[i]=0.0 if len(seg)<k or k<=0 else float(np.mean(np.partition(seg,-k)[-k:]))
    return out

def _select_train_reference(train_scores: np.ndarray, prefer_nonzero=True) -> np.ndarray:
    x=np.asarray(train_scores, dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return np.array([0.0], dtype=float)
    if prefer_nonzero:
        x2=x[x>0]
        if len(x2)>=max(20,int(0.1*len(x))): x=x2
    return np.sort(x)

def _calibrate_train_percentile(train_scores: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    ref=_select_train_reference(train_scores, prefer_nonzero=True)
    ranks=np.searchsorted(ref, np.asarray(all_scores,dtype=float), side="right").astype(float)
    denom=float(max(1,len(ref))); out=ranks/denom
    return np.clip(out, float(config.CALIBRATION_CLIP_MIN), float(config.CALIBRATION_CLIP_MAX))

def _calibrate_scores(train_scores: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    return _calibrate_train_percentile(train_scores, all_scores)

def _pctl_threshold(train_scores: np.ndarray, p: float) -> float:
    x=np.asarray(train_scores,dtype=float); x=x[np.isfinite(x)]
    require(len(x)>=10, f"Too few TRAIN scores for percentile threshold: n={len(x)}")
    return float(np.percentile(x,p))

def _adaptive_threshold_series(s: pd.Series, static_thr: float, lookback: int, q: float) -> pd.Series:
    s=pd.to_numeric(s, errors="coerce").fillna(0.0)
    min_exp=max(60,int(config.WINDOW_SIZE))
    thr_expand=s.expanding(min_periods=min_exp).quantile(q).shift(1)
    thr_roll=s.rolling(lookback, min_periods=lookback).quantile(q).shift(1)
    thr=thr_roll.combine_first(thr_expand)
    return thr.fillna(float(static_thr)).astype(float)

def _compute_thresholds_and_flags(df: pd.DataFrame, score_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
    out=df.copy(); thr_map={}
    for c in score_cols:
        name=c.replace("score_","")
        out[f"thr_{name}"]=np.nan; out[f"flag_{name}"]=0
    for t in config.TICKERS:
        idx=out["ticker"]==t; sub=out.loc[idx].copy(); train_mask=sub["split"]=="train"
        thr_map[t]={}
        for c in score_cols:
            thr=_pctl_threshold(sub.loc[train_mask,c].to_numpy(dtype=float), float(config.THRESH_PCTL))
            name=c.replace("score_","")
            thr_map[t][c]=float(thr)
            out.loc[idx,f"thr_{name}"]=float(thr)
            out.loc[idx,f"flag_{name}"]=(sub[c].astype(float).to_numpy()>=thr).astype(int)
    return out, thr_map

def _apply_adaptive_thresholds(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    out=df.copy()
    for c in score_cols:
        name=c.replace("score_","")
        thr_static=f"thr_{name}"; thr_adapt=f"thr_{name}_adapt"; flag_adapt=f"flag_{name}_adapt"
        out[thr_adapt]=np.nan; out[flag_adapt]=0
        for t in config.TICKERS:
            idx=out["ticker"]==t; sub=out.loc[idx].sort_values("date").copy()
            lookback=ADAPT_LOOKBACK if name in ("patchtst_tranad","main") else ADAPT_ALL_LOOKBACK
            q=ADAPT_Q if name in ("patchtst_tranad","main") else (float(config.THRESH_PCTL)/100.0)
            thr=_adaptive_threshold_series(sub[c], float(sub[thr_static].iloc[0]), lookback, q)
            out.loc[idx,thr_adapt]=thr.to_numpy()
            out.loc[idx,flag_adapt]=(sub[c].astype(float).to_numpy()>=thr.to_numpy()).astype(int)
    return out

def _change_point_flags_from_returns(df: pd.DataFrame) -> pd.Series:
    import ruptures as rpt
    cp_flag=pd.Series(0,index=df.index,dtype=int)
    for t in config.TICKERS:
        sub=df[df["ticker"]==t].copy().reset_index()
        idx0=sub["index"].to_numpy(); s=pd.to_numeric(sub["log_return"], errors="coerce").fillna(0.0).to_numpy()
        if np.std(s)<1e-12: continue
        try:
            algo=rpt.Pelt(model="rbf", min_size=int(config.CP_MIN_SIZE)).fit(s)
            bkps=[b for b in algo.predict(pen=float(config.CP_PENALTY)) if b < len(s)]
        except Exception as e:
            warn(f"ruptures failed for {t}: {e}"); continue
        for b in bkps:
            for j in range(max(0,b-2), min(len(s), b+3)): cp_flag.iloc[idx0[j]]=1
    return cp_flag

def _compute_daily_context(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    out["log_return"]=pd.to_numeric(out["log_return"], errors="coerce").fillna(0.0)
    out["ret_mean_20"]=out.groupby("ticker")["log_return"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    out["ret_std_20"]=out.groupby("ticker")["log_return"].rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    out["ret_z_20"]=(out["log_return"]-out["ret_mean_20"])/out["ret_std_20"].replace(0,np.nan)
    out["abs_ret_z_20"]=out["ret_z_20"].abs().fillna(0.0)
    out["volume_z_20"]=pd.to_numeric(out["volume_z_20"], errors="coerce").fillna(0.0)
    out["vol_z_pos"]=out["volume_z_20"].clip(lower=0).fillna(0.0)
    out["cp_bonus_daily"]=(pd.to_numeric(out["cp_flag"], errors="coerce").fillna(0).astype(int)>0).astype(int)
    return out

def _episode_subtype(has_cp:int, max_abs_ret_z:float, max_vol_z:float, direction:str)->str:
    tags=[]
    if int(has_cp)==1: tags.append("regime-shift")
    if float(max_abs_ret_z)>=float(config.SUBTYPE_RET_Z): tags.append("price-spike-up" if direction=="up" else "price-spike-down")
    if float(max_vol_z)>=float(config.SUBTYPE_VOL_Z): tags.append("volume-burst")
    return "other" if not tags else (tags[0] if len(tags)==1 else "mixed")

def _build_episodes(out: pd.DataFrame, flag_col: str, score_col: str) -> pd.DataFrame:
    out=out.copy().sort_values(["ticker","date"]).reset_index(drop=True)
    out=_compute_daily_context(out)
    for c in ["episode_id","is_episode_peak","episode_len","episode_has_cp"]: out[c]=0
    out["episode_direction"]="flat"; out["episode_max_abs_ret_z"]=0.0; out["episode_max_vol_z"]=0.0; out["episode_peak_score"]=0.0; out["episode_subtype"]="other"
    for t in config.TICKERS:
        idx=out["ticker"]==t; sub=out.loc[idx].sort_values("date").copy()
        flags=sub[flag_col].astype(int).to_numpy(); scores=pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0).to_numpy()
        eid=-1; episode_ids=np.full(len(sub), -1, dtype=int); i=0
        while i < len(sub):
            if flags[i]==0: i+=1; continue
            eid+=1; j=i
            while j<len(sub) and flags[j]==1: episode_ids[j]=eid; j+=1
            i=j
        sub["episode_id"]=episode_ids
        score_rank=sub[score_col].rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=float)
        price_component=np.clip(sub["abs_ret_z_20"].astype(float).to_numpy()/5.0,0.0,1.0)
        vol_component=np.clip(sub["vol_z_pos"].astype(float).to_numpy()/5.0,0.0,1.0)
        cp_component=sub["cp_bonus_daily"].astype(float).to_numpy()
        composite=0.70*score_rank + 0.20*price_component + 0.08*vol_component + 0.02*cp_component
        peaks=np.zeros(len(sub),dtype=int)
        for e in np.unique(episode_ids):
            if e<0: continue
            inds=np.where(episode_ids==e)[0]
            ep_comp=composite[inds]; max_comp=float(np.nanmax(ep_comp)); cand=inds[np.isclose(ep_comp,max_comp)]
            if len(cand)==1: k=cand[0]
            else:
                cand_df=pd.DataFrame({"pos":cand,"score":scores[cand],"abs_ret_z":sub["abs_ret_z_20"].iloc[cand].astype(float).to_numpy(),"vol_z":sub["vol_z_pos"].iloc[cand].astype(float).to_numpy()}).sort_values(["score","abs_ret_z","vol_z","pos"], ascending=[False,False,False,True])
                k=int(cand_df.iloc[0]["pos"])
            peaks[k]=1
            ep_returns=pd.to_numeric(sub["log_return"].iloc[inds], errors="coerce").fillna(0.0); ep_cum=float(ep_returns.sum())
            direction="up" if ep_cum>0 else ("down" if ep_cum<0 else "flat")
            has_cp=int(pd.to_numeric(sub["cp_flag"].iloc[inds], errors="coerce").fillna(0).astype(int).max()>=1)
            max_abs=float(pd.to_numeric(sub["abs_ret_z_20"].iloc[inds], errors="coerce").fillna(0.0).max())
            max_vol=float(pd.to_numeric(sub["vol_z_pos"].iloc[inds], errors="coerce").fillna(0.0).max())
            subtype=_episode_subtype(has_cp,max_abs,max_vol,direction)
            sub.loc[sub.index[inds], "episode_len"]=int(len(inds))
            sub.loc[sub.index[inds], "episode_direction"]=direction
            sub.loc[sub.index[inds], "episode_has_cp"]=has_cp
            sub.loc[sub.index[inds], "episode_max_abs_ret_z"]=max_abs
            sub.loc[sub.index[inds], "episode_max_vol_z"]=max_vol
            sub.loc[sub.index[inds], "episode_peak_score"]=float(composite[k])
            sub.loc[sub.index[inds], "episode_subtype"]=subtype
        sub["is_episode_peak"]=peaks
        for c in ["episode_id","is_episode_peak","episode_len","episode_direction","episode_has_cp","episode_max_abs_ret_z","episode_max_vol_z","episode_peak_score","episode_subtype"]:
            out.loc[sub.index,c]=sub[c].to_numpy()
    out["subtype"]=out["episode_subtype"].astype(str)
    return out

def _make_point_gt_for_tuning(df: pd.DataFrame) -> pd.Series:
    stats=(df[df["split"]=="train"].groupby("ticker")["log_return"].agg(["mean","std"]).reset_index())
    stats["std"]=pd.to_numeric(stats["std"], errors="coerce").replace(0,np.nan).fillna(1.0)
    tmp=df[["ticker","date","split","log_return","volume_z_20"]].copy().merge(stats,on="ticker",how="left")
    z=(pd.to_numeric(tmp["log_return"], errors="coerce")-pd.to_numeric(tmp["mean"], errors="coerce"))/pd.to_numeric(tmp["std"], errors="coerce")
    gt_ret=(z.abs()>=float(config.WEAK_Z)).astype(int)
    gt_vol=(pd.to_numeric(tmp["volume_z_20"], errors="coerce").fillna(0.0) >= float(config.WEAK_VOL_Z)).astype(int)
    return ((gt_ret==1)|(gt_vol==1)).astype(int)

def _get_patch_score_indices(ckpt_patch: dict, feat_cols: List[str]) -> Tuple[List[int], List[str]]:
    cols=ckpt_patch.get("forecast_score_cols"); idxs=ckpt_patch.get("forecast_score_indices")
    if isinstance(cols,list) and len(cols)>0:
        cols=[str(c) for c in cols]
        if isinstance(idxs,list) and len(idxs)==len(cols): idxs=[int(i) for i in idxs]
        else: idxs=[feat_cols.index(c) for c in cols]
    else:
        target_idx=int(ckpt_patch.get("target_idx",0)); idxs=[target_idx]; cols=[str(feat_cols[target_idx])]
    return [int(i) for i in idxs], cols

def _infer_patch_residuals(model, X, score_indices, patch_window, batch, device):
    n=len(X); raw=np.zeros(n,dtype=float)
    if n < patch_window+1: return raw
    Xw=_sliding_windows(X, patch_window); Xw_fore=Xw[:-1]
    y_true_next=X[patch_window:, score_indices].astype(np.float32)
    errs=np.zeros(len(Xw_fore),dtype=float); k_targets=int(min(max(1,len(score_indices)), int(config.TAIL_K)))
    model.eval()
    with torch.no_grad():
        for i in range(0,len(Xw_fore),batch):
            xb=torch.tensor(Xw_fore[i:i+batch], dtype=torch.float32, device=device)
            yp=model(xb)
            if yp.ndim==3: yp=yp[:, -1, :]
            elif yp.ndim==1: yp=yp.view(-1,1)
            yp_sel=yp[:, score_indices]
            yt=torch.tensor(y_true_next[i:i+batch], dtype=torch.float32, device=device)
            err_mat=torch.abs(yt-yp_sel).detach().cpu().numpy().astype(float)
            if err_mat.ndim==1: err_mat=err_mat.reshape(-1,1)
            errs[i:i+batch]=_topk_mean_rows(err_mat, k=k_targets)
    raw[patch_window:]=errs
    return _rolling_topk_mean(raw, patch_window, int(min(config.TAIL_K, patch_window)))

def _infer_tranad_scores(model, X, tran_window, batch, device):
    n=len(X); raw=np.zeros(n,dtype=float)
    if n < tran_window: return raw
    Xw=_sliding_windows(X, tran_window); errs=np.zeros(len(Xw),dtype=float)
    model.eval()
    with torch.no_grad():
        for i in range(0,len(Xw),batch):
            xb=torch.tensor(Xw[i:i+batch], dtype=torch.float64, device=device)
            window=xb.permute(1,0,2); elem=window[-1:,:,:]
            z=model(window, elem)
            if isinstance(z, tuple): z=z[1]
            err=((z-elem)**2).detach().cpu().numpy()[0]
            errs[i:i+batch]=_topk_mean_rows(err, k=int(config.TAIL_K))
    raw[tran_window-1:]=errs
    return raw

def _eval_variant(sub: pd.DataFrame, score: np.ndarray, metric: str) -> dict:
    y=sub["gt_point"].astype(int).to_numpy()
    train_mask=sub["split"].astype(str).eq("train").to_numpy()
    val_mask=sub["split"].astype(str).eq("val").to_numpy()
    thr_static=float(np.percentile(score[train_mask], float(config.THRESH_PCTL)))
    pred_static=(score[val_mask] >= thr_static).astype(int)
    f1_static=float(f1_score(y[val_mask], pred_static, zero_division=0))
    tuned=sub[["ticker","date","split"]].copy(); tuned["s"]=score
    tuned["thr_ad"]=np.nan; tuned["flag_ad"]=0
    for t in config.TICKERS:
        idx=tuned["ticker"]==t; sub_t=tuned.loc[idx].sort_values("date").copy()
        thr=float(np.percentile(sub_t.loc[sub_t["split"]=="train","s"].to_numpy(dtype=float), float(config.THRESH_PCTL)))
        thr_ad=_adaptive_threshold_series(sub_t["s"], thr, ADAPT_LOOKBACK, ADAPT_Q)
        tuned.loc[idx,"thr_ad"]=thr_ad.to_numpy()
        tuned.loc[idx,"flag_ad"]=(sub_t["s"].astype(float).to_numpy() >= thr_ad.to_numpy()).astype(int)
    pred_ad=tuned.loc[val_mask, "flag_ad"].astype(int).to_numpy()
    f1_ad=float(f1_score(y[val_mask], pred_ad, zero_division=0))
    ap=float(average_precision_score(y[val_mask], score[val_mask])) if len(np.unique(y[val_mask])) > 1 else float("nan")
    return {"val_auprc":ap, "val_f1_static":f1_static, "val_f1_adapt":f1_ad, "train_thr_pctl":thr_static}

def _choose_main_detector(df: pd.DataFrame) -> tuple[str,float,list[dict],dict]:
    metric = "f1_adapt"
    # always include endpoints: if PatchTST hurts, allow pure TranAD as the main detector.
    grid = sorted(set([0.0, 1.0] + [float(w) for w in list(config.CALIBRATION_GRID_WEIGHTS)]))
    results=[]
    all_patch=df["score_patchtst_cal"].astype(float).to_numpy()
    all_tran=df["score_tranad_cal"].astype(float).to_numpy()
    all_chron=df["score_chronos_cal"].astype(float).to_numpy()
    for w in grid:
        fused = w*all_patch + (1.0-w)*all_tran
        res=_eval_variant(df, fused, metric)
        results.append({"candidate":"patchtst_tranad","w_patch":float(w),"w_tranad":float(1.0-w), **res})
    # also compare branch baselines as actual deploy candidates
    res_patch=_eval_variant(df, all_patch, metric); results.append({"candidate":"patchtst","w_patch":1.0,"w_tranad":0.0, **res_patch})
    res_tran=_eval_variant(df, all_tran, metric); results.append({"candidate":"tranad","w_patch":0.0,"w_tranad":1.0, **res_tran})
    # chronos is benchmark only, not eligible to be selected as main detector
    chron_metrics=_eval_variant(df, all_chron, metric)
    def key(r): return (r["val_f1_adapt"], r["val_auprc"], r["val_f1_static"], -abs(r["w_patch"]-0.5) if r["candidate"]=="patchtst_tranad" else 0.0)
    eligible=[r for r in results]
    best=max(eligible, key=key)
    if best["candidate"]=="patchtst_tranad":
        chosen_score = best["w_patch"]*all_patch + best["w_tranad"]*all_tran
    elif best["candidate"]=="patchtst":
        chosen_score = all_patch
    else:
        chosen_score = all_tran
    return best["candidate"], float(best["w_patch"]), results, chron_metrics, chosen_score

def main():
    run_dir=_run_dir_latest(); log_path=run_dir/"logs_step06_infer_anomalies.txt"
    logger.remove(); logger.add(sys.stderr, level="INFO"); logger.add(str(log_path), level="INFO", encoding="utf-8")
    logger.info("=== Step 06: infer anomalies (OFFICIAL PatchTST + OFFICIAL TranAD + Chronos) ===")
    chronos_path=config.RESULTS_DIR/"baseline_scores_chronos.csv"
    require(config.SCALED_PATH.exists(), f"Missing {config.SCALED_PATH}")
    require(chronos_path.exists(), f"Missing {chronos_path}")
    device=_device(); PatchTSTModel, TranADModel, compat_patch=_import_official_classes()
    df=read_parquet(config.SCALED_PATH); df["date"]=pd.to_datetime(df["date"]).dt.normalize(); require_unique_key(df, ["ticker","date"], where="scaled input")
    ckpt_patch=_load_ckpt(config.DL_DIR/"patchtst_forecaster.pth", device)
    ckpt_tran=_load_ckpt(config.DL_DIR/"tranad_detector.pth", device)
    feat_cols=ckpt_patch.get("feature_cols", []); require(feat_cols==ckpt_tran.get("feature_cols", []), "feature order mismatch")
    patch_window=int(ckpt_patch.get("window", config.WINDOW_SIZE)); tran_window=int(ckpt_tran.get("n_window", ckpt_tran.get("window", config.TRANAD_WINDOW)))
    keep=["ticker","date","split","close","volume","log_return","volume_z_20"]
    out=df[keep+feat_cols].copy().sort_values(["ticker","date"]).reset_index(drop=True)
    patch_model=_build_patchtst_model(ckpt_patch, PatchTSTModel, device)
    tran_model=_build_tranad_model(ckpt_tran, TranADModel, device)
    out["score_patchtst_raw"]=0.0; out["score_tranad_raw"]=0.0; out["score_patchtst_cal"]=0.0; out["score_tranad_cal"]=0.0
    calibration_stats={}
    patch_score_indices, patch_score_cols = _get_patch_score_indices(ckpt_patch, feat_cols)
    for t in config.TICKERS:
        idx=out["ticker"]==t; sub=out.loc[idx].sort_values("date").copy(); X=sub[feat_cols].astype(float).to_numpy()
        score_patch_raw=_infer_patch_residuals(patch_model, X, patch_score_indices, patch_window, int(config.STEP05_BATCH_SIZE), device)
        score_tran_raw=_infer_tranad_scores(tran_model, X, tran_window, int(config.TRANAD_BATCH_SIZE), device)
        train_mask=sub["split"].eq("train").to_numpy()
        patch_cal=_calibrate_scores(score_patch_raw[train_mask], score_patch_raw)
        tran_cal=_calibrate_scores(score_tran_raw[train_mask], score_tran_raw)
        out.loc[idx,"score_patchtst_raw"]=score_patch_raw; out.loc[idx,"score_tranad_raw"]=score_tran_raw
        out.loc[idx,"score_patchtst_cal"]=patch_cal; out.loc[idx,"score_tranad_cal"]=tran_cal
        calibration_stats[t]={"patch_window":patch_window,"tranad_window":tran_window,
                              "patch_raw_train_mean":float(np.mean(score_patch_raw[train_mask])),
                              "patch_raw_train_std":float(np.std(score_patch_raw[train_mask])),
                              "tranad_raw_train_mean":float(np.mean(score_tran_raw[train_mask])),
                              "tranad_raw_train_std":float(np.std(score_tran_raw[train_mask]))}
    chron=read_csv(chronos_path); chron["date"]=pd.to_datetime(chron["date"]).dt.normalize(); require_unique_key(chron,["ticker","date"],where="chronos baseline")
    chron=chron.rename(columns={"score_chronos":"score_chronos_raw"})
    out=out.merge(chron[["ticker","date","score_chronos_raw","pred_chronos","context_len"]], on=["ticker","date"], how="left")
    out["score_chronos_cal"]=0.0
    for t in config.TICKERS:
        idx=out["ticker"]==t; sub=out.loc[idx].copy(); train_mask=sub["split"].eq("train").to_numpy()
        out.loc[idx,"score_chronos_cal"]=_calibrate_scores(sub.loc[train_mask,"score_chronos_raw"].to_numpy(dtype=float), sub["score_chronos_raw"].to_numpy(dtype=float))
    out["gt_point"]=_make_point_gt_for_tuning(out)
    chosen_candidate, chosen_w_patch, tune_table, chron_metrics, chosen_main_score = _choose_main_detector(out)
    chosen_w_tran = 1.0 - chosen_w_patch
    out["score_patchtst_tranad"] = chosen_w_patch*out["score_patchtst_cal"].astype(float) + chosen_w_tran*out["score_tranad_cal"].astype(float)
    out["score_patchtst"]=out["score_patchtst_cal"].astype(float); out["score_tranad"]=out["score_tranad_cal"].astype(float); out["score_chronos"]=out["score_chronos_cal"].astype(float)
    out["score_main"]=chosen_main_score.astype(float)
    score_cols=["score_patchtst_tranad","score_patchtst","score_tranad","score_chronos","score_main"]
    out, thr_map = _compute_thresholds_and_flags(out, score_cols)
    if USE_ADAPTIVE_MAIN or USE_ADAPTIVE_ALL: out=_apply_adaptive_thresholds(out, score_cols)
    # alias canonical main columns for downstream steps
    out["thr_main"] = out["thr_main"].astype(float)
    out["flag_main"] = out["flag_main"].astype(int)
    if "thr_main_adapt" in out.columns:
        out["flag_main_adapt"]=out["flag_main_adapt"].astype(int)
        main_flag_col="flag_main_adapt" if USE_ADAPTIVE_MAIN else "flag_main"
    else:
        main_flag_col="flag_main"
    out["cp_flag"]=_change_point_flags_from_returns(out)
    out=_build_episodes(out, flag_col=main_flag_col, score_col="score_main")
    corr_rows=[]
    for t in config.TICKERS:
        sub=out[out["ticker"]==t].copy()
        c_patch=float(pd.to_numeric(sub["score_main"], errors="coerce").corr(pd.to_numeric(sub["score_patchtst"], errors="coerce")))
        c_tran=float(pd.to_numeric(sub["score_main"], errors="coerce").corr(pd.to_numeric(sub["score_tranad"], errors="coerce")))
        corr_rows.append({"ticker":t,"corr_main_patch":c_patch,"corr_main_tranad":c_tran})
    out_path=config.RESULTS_DIR/"anomalies_base.csv"; ensure_dir(config.RESULTS_DIR); write_csv(out,out_path)
    analysis_mask=(out["date"]>=pd.to_datetime(config.ANALYSIS_START))&(out["date"]<=pd.to_datetime(config.ANALYSIS_END))&(out["split"].isin(["val","test"]))
    analysis_out=out.loc[analysis_mask].copy()
    manifest={
        "step":"06_infer_anomalies","timestamp":datetime.now().isoformat(timespec="seconds"),"mode":"official_repo_only",
        "inputs":{"scaled_path":str(config.SCALED_PATH),"scaled_sha256":sha256_file(config.SCALED_PATH),"dl_ckpts":{"patchtst_forecaster":str(config.DL_DIR/"patchtst_forecaster.pth"),"tranad_detector":str(config.DL_DIR/"tranad_detector.pth")},"chronos":str(chronos_path)},
        "outputs":{"anomalies_base":{"path":str(out_path),"sha256":sha256_file(out_path),"rows":int(len(out))}},
        "official_impl":{"patchtst":ckpt_patch.get("official_repo",""),"tranad":ckpt_tran.get("official_repo",""),"tranad_torch_compat_patch":{"encoder":bool(compat_patch.get("encoder")),"decoder":bool(compat_patch.get("decoder"))},"tranad_dgl_stub_used":bool(compat_patch.get("dgl_stub_used",False))},
        "branch_windows":{"patchtst":int(patch_window),"tranad":int(tran_window)},
        "patchtst_scoring":{"score_cols":list(patch_score_cols),"score_indices":list(map(int, patch_score_indices)),"mode":str(ckpt_patch.get("forecast_score_mode","next_step_residual"))},
        "calibration":{"method":"train_percentile","ticker_stats":calibration_stats},
        "main_selection":{"selection_metric":"val_f1_adapt_primary","why":"debug showed PatchTST branch remained weak; main detector must be selected by deployment-aligned adaptive F1 on VAL, not forced fusion.","chosen_candidate":chosen_candidate,"chosen_patch_weight":float(chosen_w_patch),"chosen_tranad_weight":float(chosen_w_tran),"candidate_grid_results":tune_table,"chronos_val_metrics":chron_metrics,"corr_main_vs_branches":corr_rows},
        "thresholds_static_train_only":thr_map,
        "main_flag_col":main_flag_col,
        "adaptive_main":{"enabled":USE_ADAPTIVE_MAIN,"lookback":ADAPT_LOOKBACK,"quantile":ADAPT_Q},
        "adaptive_all":{"enabled":USE_ADAPTIVE_ALL,"lookback":ADAPT_ALL_LOOKBACK,"quantile":float(config.THRESH_PCTL)/100.0},
        "phase1_summary":{"main_flag_col":main_flag_col,"main_flag_counts_analysis":analysis_out.groupby("ticker")[main_flag_col].sum().to_dict(),"peak_counts_analysis":analysis_out.groupby("ticker")["is_episode_peak"].sum().to_dict()},
        "notes":{"no_leak":"branch calibration uses TRAIN only; main detector selected on VAL only; static thresholds from TRAIN only; adaptive thresholds use past-only quantile with shift(1)","cp_flag":"descriptive only; computed from log_return change-points, never from detector score","main_detector_policy":"do not force fusion if pure TranAD or pure PatchTST wins on VAL adaptive F1","score_columns":"score_main / flag_main(_adapt) are canonical downstream columns; score_patchtst_tranad retained for ablation"}
    }
    write_json(manifest, _run_dir_latest()/"manifest_step06.json")
    ok(f"Saved anomalies_base.csv -> {out_path}")
    ok("Step 06 DONE. Next: python src/10_evaluate.py")

if __name__=="__main__":
    main()

