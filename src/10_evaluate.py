# src/10_evaluate.py
# STEP 10 — Evaluate / Compare models (research-standard, cleaned)

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, List, Tuple

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

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support


def _run_dir_latest() -> Path:
    d = config.RUNS_DIR / "latest"
    ensure_dir(d)
    return d


def _parse_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.normalize()
    return df


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def _precision_at_k(df: pd.DataFrame, score_col: str, gt_col: str, k: int) -> float:
    if df.empty:
        return float("nan")
    top = df.sort_values(score_col, ascending=False).head(k)
    if len(top) == 0:
        return float("nan")
    return float(top[gt_col].mean())


def _expand_all_event_windows(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"]).dt.normalize()
    ev["start_date"] = pd.to_datetime(ev["start_date"]).dt.normalize()
    ev["end_date"] = pd.to_datetime(ev["end_date"]).dt.normalize()

    all_rows = ev[ev["ticker"].astype(str).str.upper() == "ALL"].copy()
    specific = ev[ev["ticker"].astype(str).str.upper() != "ALL"].copy()

    expanded = []
    for t in config.TICKERS:
        tmp = all_rows.copy()
        tmp["ticker"] = t
        tmp["source_ticker"] = "ALL"
        expanded.append(tmp)

    if expanded:
        all_exp = pd.concat(expanded, ignore_index=True)
        out = pd.concat([specific.assign(source_ticker=specific["ticker"]), all_exp], ignore_index=True)
    else:
        out = specific.assign(source_ticker=specific["ticker"])

    return out.reset_index(drop=True)


def _event_hit_delay(df: pd.DataFrame, ev: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    out_rows = []
    for _, r in ev.iterrows():
        t = r["ticker"]
        start = r["start_date"]
        end = r["end_date"]
        e_date = r["event_date"]

        sub = df[(df["ticker"] == t) & (df["date"] >= start) & (df["date"] <= end)]
        hits = sub[sub[flag_col] == 1]
        if hits.empty:
            out_rows.append(dict(ticker=t, event_type=r["event_type"], event_date=e_date, start_date=start, end_date=end, hit=0, delay_days=np.nan))
        else:
            first_hit = hits["date"].min()
            delay = int((first_hit - e_date).days)
            out_rows.append(dict(ticker=t, event_type=r["event_type"], event_date=e_date, start_date=start, end_date=end, hit=1, delay_days=delay))
    return pd.DataFrame(out_rows)


def main() -> None:
    run_dir = _run_dir_latest()
    log_path = run_dir / "logs_step10_evaluate.txt"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")
    logger.info("=== Step 10: Evaluate / Compare core models ===")

    results_dir = config.RESULTS_DIR
    paths = {
        "anomalies_base": results_dir / "anomalies_base.csv",
        "event_windows": results_dir / "event_windows.csv",
        "xai_faithfulness": results_dir / "xai_faithfulness.csv",
        "xai_stability": results_dir / "xai_stability.csv",
        "event_attribution": results_dir / "event_attribution.csv",
        "event_counterfactual": results_dir / "event_counterfactual.csv",
        "features": config.FEATURES_PATH,
    }
    for k, p in paths.items():
        if not p.exists():
            fail(f"Missing required file for step10: {k} -> {p}")

    an = read_csv(paths["anomalies_base"])
    an = _parse_dates(an, "date")
    require_unique_key(an, ["ticker", "date"], where="anomalies_base")

    ev = read_csv(paths["event_windows"])
    require_columns(ev, ["event_type", "ticker", "event_date", "start_date", "end_date"], where="event_windows")
    ev2 = _expand_all_event_windows(ev)

    feat = read_parquet(paths["features"])
    require_columns(feat, ["ticker", "date", "split", "log_return", "volume_z_20"], where="features")
    feat = _parse_dates(feat, "date")
    require_unique_key(feat, ["ticker", "date"], where="features")

    base = an.merge(feat[["ticker", "date", "split", "log_return", "volume_z_20"]], on=["ticker", "date"], how="left", suffixes=("", "_feat"))
    if base["split_feat"].isna().any():
        fail("Split missing after merge anomalies_base + features. Key mismatch ticker/date.")
    base["split"] = base["split_feat"]
    base["log_return"] = base["log_return_feat"]
    base["volume_z_20"] = base["volume_z_20_feat"]
    base.drop(columns=["split_feat", "log_return_feat", "volume_z_20_feat"], inplace=True)

    a0 = pd.to_datetime(config.ANALYSIS_START)
    a1 = pd.to_datetime(config.ANALYSIS_END)
    dfA = base[(base["date"] >= a0) & (base["date"] <= a1) & (base["split"].isin(["val", "test"]))].copy()
    if dfA.empty:
        fail("No rows in analysis window (val+test).")

    train = base[base["split"] == "train"].copy()
    if train.empty:
        fail("No TRAIN rows found to compute GT stats.")

    stats = train.groupby("ticker")["log_return"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].replace(0, 1.0)

    dfA = dfA.merge(stats, on="ticker", how="left")
    z = (dfA["log_return"].astype(float) - dfA["mean"].astype(float)) / dfA["std"].astype(float)
    dfA["gt_ret"] = (z.abs() >= float(config.WEAK_Z)).astype(int)
    dfA["gt_vol"] = (dfA["volume_z_20"].astype(float) >= float(config.WEAK_VOL_Z)).astype(int)
    dfA["gt_point"] = ((dfA["gt_ret"] == 1) | (dfA["gt_vol"] == 1)).astype(int)

    logger.info(f"gt_point positive rate (analysis window): {float(dfA['gt_point'].mean()):.2%}")

    model_defs = [
        {"name": "patchtst_tranad", "score": "score_patchtst_tranad", "flag": "flag_patchtst_tranad", "flag_adapt": "flag_patchtst_tranad_adapt"},
        {"name": "patchtst", "score": "score_patchtst", "flag": "flag_patchtst", "flag_adapt": "flag_patchtst_adapt"},
        {"name": "tranad", "score": "score_tranad", "flag": "flag_tranad", "flag_adapt": "flag_tranad_adapt"},
        {"name": "chronos", "score": "score_chronos", "flag": "flag_chronos", "flag_adapt": "flag_chronos_adapt"},
    ]

    det_rows = []
    for md in model_defs:
        for variant, flag_col in [("static", md["flag"]), ("adapt", md["flag_adapt"])]:
            if md["score"] not in dfA.columns or flag_col not in dfA.columns:
                fail(f"Missing score/flag columns for {md['name']} {variant}")
            y_true = dfA["gt_point"].astype(int).to_numpy()
            y_score = pd.to_numeric(dfA[md["score"]], errors="coerce").fillna(0.0).to_numpy()
            y_pred = dfA[flag_col].astype(int).to_numpy()

            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            tp, fp, fn, tn = _confusion(y_true, y_pred)

            det_rows.append(
                {
                    "model": md["name"] if variant == "static" else f"{md['name']}_adapt",
                    "base_model": md["name"],
                    "variant": variant,
                    "auroc": _safe_auc(y_true, y_score),
                    "auprc": _safe_auprc(y_true, y_score),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "precision_at_5": _precision_at_k(dfA, md["score"], "gt_point", 5),
                    "precision_at_10": _precision_at_k(dfA, md["score"], "gt_point", 10),
                    "precision_at_20": _precision_at_k(dfA, md["score"], "gt_point", 20),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "positive_rate": float(np.mean(y_pred)),
                    "gt_positive_rate": float(np.mean(y_true)),
                }
            )

    det_df = pd.DataFrame(det_rows).sort_values(["variant", "f1", "auprc"], ascending=[True, False, False]).reset_index(drop=True)

    static_df = det_df[det_df["variant"] == "static"].copy()
    adapt_df = det_df[det_df["variant"] == "adapt"].copy()

    p_static = results_dir / "eval_detection_static.csv"
    p_adapt = results_dir / "eval_detection_adapt.csv"
    write_csv(static_df, p_static)
    write_csv(adapt_df, p_adapt)

    p_cmp = results_dir / "model_comparison.csv"
    model_cmp = adapt_df.sort_values(["f1", "auprc"], ascending=False).reset_index(drop=True)
    write_csv(model_cmp, p_cmp)

    # Episode/event-level evaluation
    event_rows = []
    for md in model_defs:
        for variant, flag_col in [("static", md["flag"]), ("adapt", md["flag_adapt"])]:
            hit_df = _event_hit_delay(dfA, ev2, flag_col)
            hit_rate = float(hit_df["hit"].mean()) if len(hit_df) else float("nan")
            delay_med = float(hit_df.loc[hit_df["hit"] == 1, "delay_days"].median()) if (hit_df["hit"] == 1).any() else float("nan")
            event_rows.append({
                "model": md["name"] if variant == "static" else f"{md['name']}_adapt",
                "base_model": md["name"],
                "variant": variant,
                "event_hit_rate": hit_rate,
                "event_delay_median": delay_med,
                "event_count": int(len(hit_df)),
            })
    event_df = pd.DataFrame(event_rows)
    p_event = results_dir / "eval_detection_episode.csv"
    write_csv(event_df, p_event)

    # Explainability evaluation
    faith = read_csv(paths["xai_faithfulness"])
    stab = read_csv(paths["xai_stability"])
    ea = read_csv(paths["event_attribution"])
    cf = read_csv(paths["event_counterfactual"])

    top1 = ea[ea["rank_news"] == 1].copy() if len(ea) else pd.DataFrame()

    faith_score = float(pd.to_numeric(faith.loc[faith["variant"] != "baseline", "faithfulness_pass"], errors="coerce").mean()) if len(faith) else float("nan")
    faith_drop = float(pd.to_numeric(faith.loc[faith["variant"] != "baseline", "drop_abs"], errors="coerce").mean()) if len(faith) else float("nan")
    stab_top5 = float(pd.to_numeric(stab["overlap_top5"], errors="coerce").mean()) if len(stab) else float("nan")
    stab_spr = float(pd.to_numeric(stab["spearman_feat"], errors="coerce").mean()) if len(stab) else float("nan")
    grounding_conf = float(pd.to_numeric(ea["confidence_score"], errors="coerce").mean()) if len(ea) else float("nan")
    grounding_top1_mean_conf = float(pd.to_numeric(top1["confidence_score"], errors="coerce").mean()) if len(top1) else float("nan")
    grounding_top1_full_or_summary_share = float(top1["content_source_type"].astype(str).isin(["full_text", "summary"]).mean()) if len(top1) else float("nan")
    grounding_top1_full_text_share = float((top1["content_source_type"].astype(str) == "full_text").mean()) if len(top1) else float("nan")
    grounding_top1_cluster_consensus = float(pd.to_numeric(top1["event_cluster_consensus"], errors="coerce").mean()) if len(top1) and "event_cluster_consensus" in top1.columns else float("nan")
    cf_strength = float(pd.to_numeric(cf["cf_strength"], errors="coerce").mean()) if len(cf) else float("nan")
    cf_changed = float(pd.to_numeric(cf["top1_changed"], errors="coerce").mean()) if len(cf) else float("nan")
    cf_strong = float((cf["cf_support_level"].astype(str) == "strong").mean()) if len(cf) and "cf_support_level" in cf.columns else float("nan")

    eval_explain = pd.DataFrame(
        [
            {"metric": "faithfulness_pass_rate", "value": faith_score},
            {"metric": "faithfulness_mean_drop_abs", "value": faith_drop},
            {"metric": "stability_mean_overlap_top5", "value": stab_top5},
            {"metric": "stability_mean_spearman_feat", "value": stab_spr},
            {"metric": "grounding_mean_confidence_all_ranked", "value": grounding_conf},
            {"metric": "grounding_top1_mean_confidence", "value": grounding_top1_mean_conf},
            {"metric": "grounding_top1_full_or_summary_share", "value": grounding_top1_full_or_summary_share},
            {"metric": "grounding_top1_full_text_share", "value": grounding_top1_full_text_share},
            {"metric": "grounding_top1_cluster_consensus", "value": grounding_top1_cluster_consensus},
            {"metric": "counterfactual_mean_strength", "value": cf_strength},
            {"metric": "counterfactual_top1_changed_share", "value": cf_changed},
            {"metric": "counterfactual_strong_support_share", "value": cf_strong},
        ]
    )
    p_explain = results_dir / "eval_explain.csv"
    write_csv(eval_explain, p_explain)

    manifest = {
        "step": "10_evaluate",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in paths.items()},
        "outputs": {
            "eval_detection_static": {"path": str(p_static), "sha256": sha256_file(p_static)},
            "eval_detection_adapt": {"path": str(p_adapt), "sha256": sha256_file(p_adapt)},
            "eval_detection_episode": {"path": str(p_event), "sha256": sha256_file(p_event)},
            "model_comparison": {"path": str(p_cmp), "sha256": sha256_file(p_cmp)},
            "eval_explain": {"path": str(p_explain), "sha256": sha256_file(p_explain)},
        },
        "ground_truth": {
            "pointwise": "gt_point = (abs(train-zscore(log_return)) >= WEAK_Z) OR (volume_z_20 >= WEAK_VOL_Z)",
            "cp_in_gt": bool(config.EVAL_INCLUDE_CP_IN_GT),
            "notes": "cp_flag is excluded from point-wise GT to avoid circular evaluation",
        },
        "event_level_summary": event_df.to_dict(orient="records"),
        "explainability_summary": {
            "n_top1_articles": int(len(top1)),
            "top1_content_type_counts": top1["content_source_type"].astype(str).value_counts().to_dict() if len(top1) else {},
        },
    }
    write_json(manifest, _run_dir_latest() / "manifest_step10.json")

    ok(f"Saved point-wise evaluation -> {p_static} / {p_adapt}")
    ok(f"Saved episode evaluation -> {p_event}")
    ok(f"Saved explainability evaluation -> {p_explain}")
    ok("Step 10 DONE. Next: python src/11_pack_for_web.py")


if __name__ == "__main__":
    main()

