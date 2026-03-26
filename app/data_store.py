# =========================================================
# app/data_store.py
# Central data access layer for FinAnomX Streamlit app + chatbot
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import pandas as pd


# =========================================================
# 1. Project root + path helpers
# =========================================================
def get_project_root(start: Optional[Path] = None) -> Path:
    """
    Find the FinAnomX project root by walking upward until both
    'data' and 'app' directories are found.
    """
    start = (start or Path.cwd()).resolve()

    for p in [start] + list(start.parents):
        if (p / "data").exists() and (p / "app").exists():
            return p

    raise FileNotFoundError(
        f"Could not find project root from start path: {start}. "
        "Expected folders 'data' and 'app'."
    )


def _safe_read_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    df = pd.read_csv(path)
    parse_dates = parse_dates or []
    for c in parse_dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required parquet file: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet file: {path}\n"
            f"Original error: {e}\n"
            "If this is running on Windows, make sure pyarrow is installed:\n"
            "pip install pyarrow"
        ) from e


def _normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def _safe_read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)


def _safe_read_optional_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return _safe_read_csv(path, parse_dates=parse_dates)


def _fallback_raw_market(anomalies: pd.DataFrame, anomalies_base: pd.DataFrame) -> pd.DataFrame:
    src = anomalies.copy() if not anomalies.empty else anomalies_base.copy()
    keep = [c for c in ["ticker", "date", "open", "high", "low", "close", "volume"] if c in src.columns]
    if not keep or "ticker" not in keep or "date" not in keep or "close" not in keep:
        return pd.DataFrame(columns=["ticker", "date", "close", "volume"])
    out = src[keep].copy()
    out = _normalize_date_col(out, "date")
    out = out.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    return out.reset_index(drop=True)


# =========================================================
# 2. Model column mapping
# =========================================================
MODEL_MAP: Dict[str, Dict[str, str]] = {
    "main": {
        "score_col": "main_score",
        "flag_col": "main_flag",
        "threshold_col": None,
        "source": "anomalies",
    },
    "patchtst_tranad": {
        "score_col": "score_patchtst_tranad",
        "flag_col": "flag_patchtst_tranad",
        "threshold_col": "thr_patchtst_tranad",
        "source": "anomalies_base",
    },
    "patchtst_tranad_adapt": {
        "score_col": "score_patchtst_tranad",
        "flag_col": "flag_patchtst_tranad_adapt",
        "threshold_col": "thr_patchtst_tranad_adapt",
        "source": "anomalies_base",
    },
    "patchtst": {
        "score_col": "score_patchtst",
        "flag_col": "flag_patchtst",
        "threshold_col": "thr_patchtst",
        "source": "anomalies_base",
    },
    "patchtst_adapt": {
        "score_col": "score_patchtst",
        "flag_col": "flag_patchtst_adapt",
        "threshold_col": "thr_patchtst_adapt",
        "source": "anomalies_base",
    },
    "tranad": {
        "score_col": "score_tranad",
        "flag_col": "flag_tranad",
        "threshold_col": "thr_tranad",
        "source": "anomalies_base",
    },
    "tranad_adapt": {
        "score_col": "score_tranad",
        "flag_col": "flag_tranad_adapt",
        "threshold_col": "thr_tranad_adapt",
        "source": "anomalies_base",
    },
    "lstm_ae": {
        "score_col": "score_lstm_ae",
        "flag_col": "flag_lstm_ae",
        "threshold_col": "thr_lstm_ae",
        "source": "anomalies_base",
    },
    "lstm_ae_adapt": {
        "score_col": "score_lstm_ae",
        "flag_col": "flag_lstm_ae_adapt",
        "threshold_col": "thr_lstm_ae_adapt",
        "source": "anomalies_base",
    },
    "iforest": {
        "score_col": "score_iforest",
        "flag_col": "flag_iforest",
        "threshold_col": "thr_iforest",
        "source": "anomalies_base",
    },
    "iforest_adapt": {
        "score_col": "score_iforest",
        "flag_col": "flag_iforest_adapt",
        "threshold_col": "thr_iforest_adapt",
        "source": "anomalies_base",
    },
    "ocsvm": {
        "score_col": "score_ocsvm",
        "flag_col": "flag_ocsvm",
        "threshold_col": "thr_ocsvm",
        "source": "anomalies_base",
    },
    "ocsvm_adapt": {
        "score_col": "score_ocsvm",
        "flag_col": "flag_ocsvm_adapt",
        "threshold_col": "thr_ocsvm_adapt",
        "source": "anomalies_base",
    },
    "pca": {
        "score_col": "score_pca",
        "flag_col": "flag_pca",
        "threshold_col": "thr_pca",
        "source": "anomalies_base",
    },
    "pca_adapt": {
        "score_col": "score_pca",
        "flag_col": "flag_pca_adapt",
        "threshold_col": "thr_pca_adapt",
        "source": "anomalies_base",
    },
    "dlinear": {
        "score_col": "score_dlinear",
        "flag_col": "flag_dlinear",
        "threshold_col": "thr_dlinear",
        "source": "anomalies_base",
    },
    "dlinear_adapt": {
        "score_col": "score_dlinear",
        "flag_col": "flag_dlinear_adapt",
        "threshold_col": "thr_dlinear_adapt",
        "source": "anomalies_base",
    },
}


# =========================================================
# 3. ArtifactStore dataclass
# =========================================================
@dataclass
class ArtifactStore:
    root: Path
    raw_market: pd.DataFrame
    anomalies: pd.DataFrame
    anomalies_base: pd.DataFrame
    xai_feature: pd.DataFrame
    xai_temporal: Dict[str, Any]
    event_attribution: pd.DataFrame
    event_counterfactual: pd.DataFrame
    news_enriched: pd.DataFrame
    chat_evidence: pd.DataFrame
    eval_detection_static: pd.DataFrame
    eval_detection_adapt: pd.DataFrame
    eval_explain: pd.DataFrame
    model_comparison_static: pd.DataFrame
    model_comparison_adapt: pd.DataFrame

    # -----------------------------------------------------
    # Core helpers
    # -----------------------------------------------------
    def available_tickers(self) -> List[str]:
        vals = sorted(self.anomalies["ticker"].dropna().unique().tolist())
        return vals

    def available_subtypes(self) -> List[str]:
        vals = sorted([x for x in self.anomalies["subtype"].dropna().unique().tolist() if str(x).strip()])
        return vals

    def available_models(self) -> List[str]:
        return list(MODEL_MAP.keys())

    def get_model_spec(self, model_name: str) -> Dict[str, Optional[str]]:
        if model_name not in MODEL_MAP:
            raise KeyError(f"Unknown model name: {model_name}")
        return MODEL_MAP[model_name]

    def _model_frame(self, model_name: str) -> pd.DataFrame:
        spec = self.get_model_spec(model_name)
        return self.anomalies if spec["source"] == "anomalies" else self.anomalies_base

    # -----------------------------------------------------
    # Filter helpers
    # -----------------------------------------------------
    def filter_anomalies(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        subtype: Optional[str] = None,
        model_name: str = "main",
        flagged_only: bool = False,
    ) -> pd.DataFrame:
        spec = self.get_model_spec(model_name)
        df = self._model_frame(model_name).copy()

        if ticker:
            df = df[df["ticker"] == ticker]

        if start_date is not None:
            start_date = pd.to_datetime(start_date).normalize()
            df = df[df["date"] >= start_date]

        if end_date is not None:
            end_date = pd.to_datetime(end_date).normalize()
            df = df[df["date"] <= end_date]

        if subtype and subtype != "All" and "subtype" in df.columns:
            df = df[df["subtype"] == subtype]

        score_col = spec["score_col"]
        flag_col = spec["flag_col"]
        thr_col = spec["threshold_col"]

        # add normalized UI columns
        df = df.copy()
        df["ui_score"] = df[score_col] if score_col in df.columns else pd.NA
        df["ui_flag"] = df[flag_col] if flag_col in df.columns else 0
        df["ui_threshold"] = df[thr_col] if (thr_col and thr_col in df.columns) else pd.NA
        df["ui_model"] = model_name

        if flagged_only:
            df = df[df["ui_flag"] == 1]

        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def get_price_window(
        self,
        ticker: str,
        center_date: pd.Timestamp,
        window_days: int = 30,
    ) -> pd.DataFrame:
        center_date = pd.to_datetime(center_date).normalize()
        start_date = center_date - pd.Timedelta(days=window_days)
        end_date = center_date + pd.Timedelta(days=window_days)

        df = self.raw_market.copy()
        df = df[(df["ticker"] == ticker) & (df["date"] >= start_date) & (df["date"] <= end_date)].copy()
        return df.sort_values("date").reset_index(drop=True)

    def get_anomaly_window(
        self,
        ticker: str,
        center_date: pd.Timestamp,
        model_name: str = "main",
        window_days: int = 30,
    ) -> pd.DataFrame:
        center_date = pd.to_datetime(center_date).normalize()
        start_date = center_date - pd.Timedelta(days=window_days)
        end_date = center_date + pd.Timedelta(days=window_days)

        return self.filter_anomalies(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            model_name=model_name,
            flagged_only=False,
        )

    # -----------------------------------------------------
    # Row lookup helpers
    # -----------------------------------------------------
    def _nearest_previous_trading_row(
        self,
        ticker: str,
        target_date: pd.Timestamp,
        model_name: str = "main",
    ) -> Optional[pd.Series]:
        target_date = pd.to_datetime(target_date).normalize()
        df = self.filter_anomalies(ticker=ticker, model_name=model_name)
        df = df[df["date"] <= target_date].sort_values("date")
        if df.empty:
            return None
        return df.iloc[-1]

    def get_anomaly_row(
        self,
        ticker: str,
        date: pd.Timestamp,
        model_name: str = "main",
        nearest_previous: bool = True,
    ) -> Optional[pd.Series]:
        date = pd.to_datetime(date).normalize()
        df = self.filter_anomalies(ticker=ticker, model_name=model_name)
        exact = df[df["date"] == date]
        if not exact.empty:
            return exact.iloc[0]
        if nearest_previous:
            return self._nearest_previous_trading_row(ticker, date, model_name=model_name)
        return None

    # -----------------------------------------------------
    # XAI helpers
    # -----------------------------------------------------
    def get_xai_features(self, ticker: str, date: pd.Timestamp, top_k: int = 10) -> pd.DataFrame:
        date = pd.to_datetime(date).normalize()
        df = self.xai_feature.copy()
        df = df[(df["ticker"] == ticker) & (df["date"] == date)].copy()
        if "rank" in df.columns:
            df = df.sort_values("rank")
        elif "importance_norm" in df.columns:
            df = df.sort_values("importance_norm", ascending=False)
        return df.head(top_k).reset_index(drop=True)

    def get_temporal_importance(self, ticker: str, date: pd.Timestamp) -> Optional[Dict[str, Any]]:
        date_key = pd.to_datetime(date).strftime("%Y-%m-%d")
        key = f"{ticker}|{date_key}"
        return self.xai_temporal.get(key)

    def get_top_time_segment(self, ticker: str, date: pd.Timestamp) -> Optional[Dict[str, Any]]:
        obj = self.get_temporal_importance(ticker, date)
        if not obj:
            return None

        arr = obj.get("time_importance")
        if not arr:
            return None

        values = pd.Series(arr, dtype="float64")
        if values.empty:
            return None

        peak_idx = int(values.idxmax())
        peak_val = float(values.max())

        return {
            "window": int(obj.get("window", len(values))),
            "peak_idx": peak_idx,
            "peak_val": peak_val,
            "time_importance": values.tolist(),
        }

    # -----------------------------------------------------
    # News helpers
    # -----------------------------------------------------
    def get_top_news(self, ticker: str, date: pd.Timestamp, k: int = 5) -> pd.DataFrame:
        date = pd.to_datetime(date).normalize()
        df = self.event_attribution.copy()
        df = df[(df["ticker"] == ticker) & (df["peak_date"] == date)].copy()
        if "rank_news" in df.columns:
            df = df.sort_values("rank_news")
        return df.head(k).reset_index(drop=True)

    def get_counterfactual(self, ticker: str, date: pd.Timestamp) -> pd.DataFrame:
        date = pd.to_datetime(date).normalize()
        df = self.event_counterfactual.copy()
        df = df[(df["ticker"] == ticker) & (df["peak_date"] == date)].copy()
        return df.reset_index(drop=True)

    def get_chat_evidence(self, ticker: str, date: pd.Timestamp, nearest_previous: bool = True) -> Optional[Dict[str, Any]]:
        if self.chat_evidence.empty:
            return None
        date = pd.to_datetime(date).normalize()
        df = self.chat_evidence.copy()
        df = df[df["ticker"].astype(str) == str(ticker)].copy()
        if df.empty:
            return None
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        exact = df[df["date"] == date]
        if not exact.empty:
            return exact.iloc[0].to_dict()
        if nearest_previous:
            prev = df[df["date"] <= date].sort_values("date")
            if not prev.empty:
                return prev.iloc[-1].to_dict()
        return None

    def build_chat_context(
        self,
        ticker: str,
        date: pd.Timestamp,
        model_name: str = "main",
        nearest_previous: bool = True,
        news_k: int = 3,
    ) -> Dict[str, Any]:
        obj = self.get_chat_evidence(ticker=ticker, date=date, nearest_previous=nearest_previous)
        if obj is not None:
            actual_date = pd.to_datetime(obj.get("date")).normalize()
            requested_date = pd.to_datetime(date).normalize()
            return {
                "found": True,
                "ticker": str(obj.get("ticker", ticker)),
                "requested_date": requested_date.strftime("%Y-%m-%d"),
                "actual_date": actual_date.strftime("%Y-%m-%d"),
                "used_nearest_previous": actual_date != requested_date,
                "model_name": model_name,
                "abnormality": obj.get("abnormality", {}),
                "model_evidence": obj.get("model_evidence", {}),
                "event_evidence": obj.get("event_evidence", {}),
                "counterfactual": obj.get("counterfactual", {}),
                "chat_ready_numeric_summary": obj.get("chat_ready_numeric_summary"),
                "chat_ready_event_summary": obj.get("chat_ready_event_summary"),
                "chat_ready_full_summary": obj.get("chat_ready_full_summary"),
                "top_event_user_summary_short": obj.get("top_event_user_summary_short"),
                "top_event_user_summary_long": obj.get("top_event_user_summary_long"),
                "sources": [
                    {"file": "data/results/chat_evidence.jsonl", "key": f"ticker={obj.get('ticker', ticker)}, date={actual_date.date()}"},
                    {"file": "data/results/anomalies.csv", "key": f"ticker={obj.get('ticker', ticker)}, date={actual_date.date()}"},
                ],
            }

        legacy = self.build_anomaly_context(ticker=ticker, date=date, model_name=model_name, news_k=news_k, nearest_previous=nearest_previous)
        if not legacy.get("found", False):
            return legacy
        anomaly = legacy.get("anomaly", {})
        abnormality = {
            "what_was_abnormal_short": anomaly.get("what_was_abnormal_short"),
            "market_context_summary": anomaly.get("market_context_summary"),
            "ret_1d_pct": anomaly.get("ret_1d_pct"),
            "ret_5d_pct": anomaly.get("ret_5d_pct"),
            "ret_20d_pct": anomaly.get("ret_20d_pct"),
            "drawdown_20_pct": anomaly.get("drawdown_20_pct"),
            "drawup_20_pct": anomaly.get("drawup_20_pct"),
            "volume_ratio_20_pct": anomaly.get("volume_ratio_20_pct"),
            "volume_z_20": anomaly.get("volume_z_20"),
            "gap_pct": anomaly.get("gap_pct"),
            "range_pct": anomaly.get("range_pct"),
            "episode_peak_score": anomaly.get("episode_peak_score"),
            "score_main": anomaly.get("ui_score", anomaly.get("main_score")),
        }
        model_evidence = {
            "xai_primary_group": anomaly.get("xai_primary_group"),
            "xai_summary_human": anomaly.get("xai_summary_human"),
            "xai_top_features_pretty": anomaly.get("xai_top_features_pretty"),
            "top_features": legacy.get("xai_features", []),
            "temporal": legacy.get("temporal"),
        }
        top_news = legacy.get("top_news", [])
        top_event = top_news[0] if top_news else {}
        event_evidence = {
            "top_event_title": top_event.get("title", anomaly.get("top_event_title")),
            "top_event_url": top_event.get("url", anomaly.get("top_event_url")),
            "top_event_domain": top_event.get("domain", anomaly.get("top_event_domain")),
            "top_event_domain_tier": top_event.get("domain_tier", anomaly.get("top_event_domain_tier")),
            "top_event_content_source_type": top_event.get("content_source_type", anomaly.get("top_event_content_source_type")),
            "top_event_sent_label": top_event.get("sent_label", anomaly.get("top_event_sent_label")),
            "top_event_confidence_score": top_event.get("confidence_score", anomaly.get("top_event_confidence_score")),
            "top_event_market_consistency": top_event.get("market_consistency", anomaly.get("top_event_market_consistency")),
            "top_event_xai_consistency": top_event.get("xai_consistency", anomaly.get("top_event_xai_consistency")),
            "top_event_event_window_match": top_event.get("event_window_match", anomaly.get("top_event_event_window_match")),
            "top_event_article_themes": top_event.get("article_themes", anomaly.get("top_event_article_themes")),
            "top_event_evidence_excerpt": top_event.get("evidence_excerpt", anomaly.get("top_event_evidence_excerpt")),
            "top_event_article_summary": top_event.get("article_summary", anomaly.get("top_event_article_summary")),
            "news_confidence_level": anomaly.get("news_confidence_level"),
            "news_reasoning_summary": anomaly.get("news_reasoning_summary"),
            "news_evidence_limitations": anomaly.get("news_evidence_limitations"),
            "top_event_user_summary_short": anomaly.get("top_event_user_summary_short"),
            "top_event_user_summary_long": anomaly.get("top_event_user_summary_long"),
            "supporting_articles": top_news,
        }
        cf_rows = legacy.get("counterfactual", [])
        counterfactual = cf_rows[0] if cf_rows else {"counterfactual_summary": anomaly.get("counterfactual_summary")}
        return {
            "found": True,
            "ticker": legacy.get("ticker", ticker),
            "requested_date": legacy.get("requested_date"),
            "actual_date": legacy.get("actual_date"),
            "used_nearest_previous": legacy.get("used_nearest_previous", False),
            "model_name": model_name,
            "abnormality": abnormality,
            "model_evidence": model_evidence,
            "event_evidence": event_evidence,
            "counterfactual": counterfactual,
            "chat_ready_numeric_summary": anomaly.get("chat_ready_numeric_summary", anomaly.get("explanation_summary_short")),
            "chat_ready_event_summary": anomaly.get("chat_ready_event_summary", anomaly.get("news_reasoning_summary")),
            "chat_ready_full_summary": anomaly.get("chat_ready_full_summary", anomaly.get("explanation_summary_long")),
            "top_event_user_summary_short": anomaly.get("top_event_user_summary_short"),
            "top_event_user_summary_long": anomaly.get("top_event_user_summary_long"),
            "sources": legacy.get("sources", []),
            "anomaly": legacy.get("anomaly", {}),
        }

    # -----------------------------------------------------
    # Overview helpers
    # -----------------------------------------------------
    def get_overview_kpis(
        self,
        ticker: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        subtype: Optional[str] = None,
        model_name: str = "main",
        peaks_only: bool = False,
    ) -> Dict[str, Any]:
        df = self.filter_anomalies(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            subtype=subtype,
            model_name=model_name,
            flagged_only=False,
        )

        flagged = df[df["ui_flag"] == 1].copy()

        if peaks_only:
            peak_mask = pd.to_numeric(flagged.get("is_episode_peak", 0), errors="coerce").fillna(0).astype(int) == 1
            flagged = flagged[peak_mask].copy()

        if flagged.empty:
            return {
                "total_anomalies": 0,
                "top_subtype": "N/A",
                "max_score": None,
            }

        top_subtype = (
            flagged["subtype"].dropna().mode().iloc[0]
            if "subtype" in flagged.columns and not flagged["subtype"].dropna().empty
            else "N/A"
        )

        max_score = float(flagged["ui_score"].max()) if flagged["ui_score"].notna().any() else None

        return {
            "total_anomalies": int(len(flagged)),
            "top_subtype": top_subtype,
            "max_score": max_score,
        }

    # -----------------------------------------------------
    # Metrics helpers (for chatbot)
    # -----------------------------------------------------
    def get_detection_metrics(self, variant: str = "static") -> pd.DataFrame:
        df = self.eval_detection_static.copy() if variant == "static" else self.eval_detection_adapt.copy()
        return df.reset_index(drop=True)

    def get_model_comparison(self, variant: str = "static") -> pd.DataFrame:
        df = self.model_comparison_static.copy() if variant == "static" else self.model_comparison_adapt.copy()
        return df.reset_index(drop=True)

    def get_explain_metrics(self) -> pd.DataFrame:
        return self.eval_explain.copy().reset_index(drop=True)

    # -----------------------------------------------------
    # Chatbot-friendly packaged lookup
    # -----------------------------------------------------
    def build_anomaly_context(
        self,
        ticker: str,
        date: pd.Timestamp,
        model_name: str = "main",
        news_k: int = 3,
        nearest_previous: bool = True,
    ) -> Dict[str, Any]:
        row = self.get_anomaly_row(ticker=ticker, date=date, model_name=model_name, nearest_previous=nearest_previous)
        if row is None:
            return {
                "found": False,
                "message": "No anomaly row found for the requested ticker/date in current system."
            }

        actual_date = pd.to_datetime(row["date"]).normalize()

        xai_df = self.get_xai_features(ticker, actual_date, top_k=10)
        news_df = self.get_top_news(ticker, actual_date, k=news_k)
        cf_df = self.get_counterfactual(ticker, actual_date)
        temporal = self.get_top_time_segment(ticker, actual_date)

        return {
            "found": True,
            "ticker": ticker,
            "requested_date": pd.to_datetime(date).normalize().strftime("%Y-%m-%d"),
            "actual_date": actual_date.strftime("%Y-%m-%d"),
            "used_nearest_previous": actual_date != pd.to_datetime(date).normalize(),
            "model_name": model_name,
            "anomaly": row.to_dict(),
            "xai_features": xai_df.to_dict(orient="records"),
            "top_news": news_df.to_dict(orient="records"),
            "counterfactual": cf_df.to_dict(orient="records"),
            "temporal": temporal,
            "sources": self.build_sources_for_anomaly(ticker, actual_date),
        }

    def build_sources_for_anomaly(self, ticker: str, date: pd.Timestamp) -> List[Dict[str, Any]]:
        date = pd.to_datetime(date).normalize()
        sources = [
            {
                "file": "data/results/anomalies.csv",
                "key": f"ticker={ticker}, date={date.date()}",
            },
            {
                "file": "data/results/xai_feature.csv",
                "key": f"ticker={ticker}, date={date.date()}",
            },
            {
                "file": "data/results/xai_temporal.json",
                "key": f"{ticker}|{date.strftime('%Y-%m-%d')}",
            },
            {
                "file": "data/results/event_attribution.csv",
                "key": f"ticker={ticker}, peak_date={date.date()}",
            },
            {
                "file": "data/results/event_counterfactual.csv",
                "key": f"ticker={ticker}, peak_date={date.date()}",
            },
        ]
        return sources


# =========================================================
# 4. Main loader
# =========================================================
@lru_cache(maxsize=1)
def load_artifacts() -> ArtifactStore:
    root = get_project_root()

    data_raw = root / "data" / "raw"
    data_results = root / "data" / "results"

    # main outputs
    anomalies = _safe_read_csv(data_results / "anomalies.csv", parse_dates=["date", "news1_seendate"])
    anomalies = _normalize_date_col(anomalies, "date")

    anomalies_base = _safe_read_optional_csv(data_results / "anomalies_base.csv", parse_dates=["date"])
    anomalies_base = _normalize_date_col(anomalies_base, "date")

    raw_market_path = data_raw / "raw_merged.parquet"
    if raw_market_path.exists():
        raw_market = _safe_read_parquet(raw_market_path)
        raw_market = _normalize_date_col(raw_market, "date")
    else:
        raw_market = _fallback_raw_market(anomalies, anomalies_base)

    xai_feature = _safe_read_csv(data_results / "xai_feature.csv", parse_dates=["date"])
    xai_feature = _normalize_date_col(xai_feature, "date")

    with open(data_results / "xai_temporal.json", "r", encoding="utf-8") as f:
        xai_temporal = json.load(f)

    event_attribution = _safe_read_csv(data_results / "event_attribution.csv", parse_dates=["peak_date"])
    event_attribution = _normalize_date_col(event_attribution, "peak_date")

    event_counterfactual = _safe_read_csv(data_results / "event_counterfactual.csv", parse_dates=["peak_date"])
    event_counterfactual = _normalize_date_col(event_counterfactual, "peak_date")

    news_enriched = _safe_read_parquet(data_results / "news_enriched.parquet")
    for c in ["peak_date", "pub_date", "seendate", "date"]:
        if c in news_enriched.columns:
            news_enriched[c] = pd.to_datetime(news_enriched[c], errors="coerce")

    chat_evidence = _safe_read_jsonl(data_results / "chat_evidence.jsonl")
    if not chat_evidence.empty and "date" in chat_evidence.columns:
        chat_evidence["date"] = pd.to_datetime(chat_evidence["date"], errors="coerce").dt.normalize()

    # metrics files
    eval_detection_static = _safe_read_csv(data_results / "eval_detection_static.csv")
    eval_detection_adapt = _safe_read_csv(data_results / "eval_detection_adapt.csv")
    eval_explain = _safe_read_csv(data_results / "eval_explain.csv")
    model_comparison_static = _safe_read_csv(data_results / "model_comparison_static.csv")
    model_comparison_adapt = _safe_read_csv(data_results / "model_comparison_adapt.csv")

    return ArtifactStore(
        root=root,
        raw_market=raw_market,
        anomalies=anomalies,
        anomalies_base=anomalies_base,
        xai_feature=xai_feature,
        xai_temporal=xai_temporal,
        event_attribution=event_attribution,
        event_counterfactual=event_counterfactual,
        news_enriched=news_enriched,
        chat_evidence=chat_evidence,
        eval_detection_static=eval_detection_static,
        eval_detection_adapt=eval_detection_adapt,
        eval_explain=eval_explain,
        model_comparison_static=model_comparison_static,
        model_comparison_adapt=model_comparison_adapt,
    )


# =========================================================
# 5. Quick sanity helper
# =========================================================
def quick_sanity_check() -> Dict[str, Any]:
    store = load_artifacts()
    return {
        "root": str(store.root),
        "tickers": store.available_tickers(),
        "subtypes": store.available_subtypes(),
        "models": store.available_models()[:5] + ["..."],
        "anomalies_shape": store.anomalies.shape,
        "anomalies_base_shape": store.anomalies_base.shape,
        "xai_feature_shape": store.xai_feature.shape,
        "event_attribution_shape": store.event_attribution.shape,
        "event_counterfactual_shape": store.event_counterfactual.shape,
        "news_enriched_shape": store.news_enriched.shape,
        "chat_evidence_shape": store.chat_evidence.shape,
    }


if __name__ == "__main__":
    info = quick_sanity_check()
    for k, v in info.items():
        print(f"{k}: {v}")

