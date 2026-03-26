# =========================================================
# app/chatbot/intent.py
# Rule-based + alias-based intent parsing for FinAnomX chatbot
# =========================================================

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import re
import unicodedata
from calendar import monthrange

import pandas as pd


# =========================================================
# 1. Global config
# =========================================================
ANALYSIS_START = pd.Timestamp("2023-01-01")
ANALYSIS_END = pd.Timestamp("2024-12-31")


# =========================================================
# 2. Dataclass
# =========================================================
@dataclass
class IntentResult:
    raw_query: str
    normalized_query: str
    intent: str

    ticker: Optional[str] = None
    date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    model_name: Optional[str] = "main"
    metric_name: Optional[str] = None
    top_k: Optional[int] = None
    count_only: bool = False

    entities: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# 3. Alias maps
# =========================================================
TICKER_ALIAS_GROUPS: Dict[str, List[str]] = {
    "TSLA": ["tsla", "tesla", "tes"],
    "AMZN": ["amzn", "amazon", "amazone", "amz"],
    "NVDA": ["nvda", "nvida", "nvd", "nvidia"],
    "MSFT": ["msft", "microsoft", "micro", "ms"],
}

MODEL_ALIAS_GROUPS: Dict[str, List[str]] = {
    "main": ["main", "model chinh", "mô hình chính", "mo hinh chinh"],
    "patchtst_tranad": ["patchtst_tranad", "patchtst tranad", "main static"],
    "patchtst_tranad_adapt": ["patchtst_tranad_adapt", "patchtst tranad adapt", "main adapt", "adaptive main"],
    "patchtst": ["patchtst"],
    "patchtst_adapt": ["patchtst adapt", "patchtst_adapt"],
    "tranad": ["tranad"],
    "tranad_adapt": ["tranad adapt", "tranad_adapt"],
    "lstm_ae": ["lstm", "lstm ae", "lstm_ae", "autoencoder lstm"],
    "lstm_ae_adapt": ["lstm adapt", "lstm_ae_adapt"],
    "iforest": ["iforest", "isolation forest", "isolationforest"],
    "iforest_adapt": ["iforest adapt", "iforest_adapt"],
    "ocsvm": ["ocsvm", "one class svm", "one-class svm"],
    "ocsvm_adapt": ["ocsvm adapt", "ocsvm_adapt"],
    "pca": ["pca"],
    "pca_adapt": ["pca adapt", "pca_adapt"],
    "dlinear": ["dlinear"],
    "dlinear_adapt": ["dlinear adapt", "dlinear_adapt"],
    "baseline": ["baseline", "base line", "mô hình nền", "mo hinh nen"],
}

METRIC_ALIAS_GROUPS: Dict[str, List[str]] = {
    "auroc": ["auroc", "auc", "roc auc", "roc-auc"],
    "auprc": ["auprc", "pr auc", "pr-auc"],
    "precision": ["precision"],
    "recall": ["recall"],
    "f1": ["f1", "f1-score", "f1 score"],
    "precision_at_k": ["precision@k", "precision at k", "p@k", "precision@10", "precision@20", "precision@5"],
    "delay": ["delay", "lag", "độ trễ", "do tre"],
    "event_hit_rate": ["event hit rate", "hit rate", "event rate"],
    "faithfulness": ["faithfulness"],
    "stability": ["stability"],
    "grounding": ["grounding"],
    "counterfactual": ["counterfactual", "cf", "phản biện", "phan bien"],
}

TERM_ALIASES: Dict[str, str] = {
    "tt": "news",
    "tin tức": "news",
    "tin tuc": "news",
    "tin": "news",
    "bài báo": "news",
    "bai bao": "news",
    "nguyên nhân": "cause",
    "nguyen nhan": "cause",
    "vì sao": "cause",
    "vi sao": "cause",
    "xai": "xai",
    "ig": "integrated gradients",
    "cf": "counterfactual",
    "phản biện": "counterfactual",
    "phan bien": "counterfactual",
    "bất thường": "anomaly",
    "bat thuong": "anomaly",
    "đột biến": "anomaly",
    "dot bien": "anomaly",
    "spike": "anomaly",
    "abn": "anomaly",
    "auc": "auroc",
    "pr": "precision recall",
}


# =========================================================
# 4. Intent labels
# =========================================================
INTENT_PROJECT_OVERVIEW = "PROJECT_OVERVIEW"
INTENT_DATA_QUERY = "DATA_QUERY"
INTENT_ANOMALY_LOOKUP = "ANOMALY_LOOKUP"
INTENT_ANOMALY_TOPK = "ANOMALY_TOPK"
INTENT_ANOMALY_EXPLAIN_NUMERIC = "ANOMALY_EXPLAIN_NUMERIC"
INTENT_ANOMALY_EXPLAIN_NEWS = "ANOMALY_EXPLAIN_NEWS"
INTENT_COUNTERFACTUAL_VALIDATION = "COUNTERFACTUAL_VALIDATION"
INTENT_MODEL_COMPARE = "MODEL_COMPARE"
INTENT_METRICS_QUERY = "METRICS_QUERY"
INTENT_EVENT_WINDOW_QUERY = "EVENT_WINDOW_QUERY"
INTENT_DASHBOARD_HELP = "DASHBOARD_HELP"
INTENT_OUT_OF_SCOPE = "OUT_OF_SCOPE"


# =========================================================
# 5. Text normalization
# =========================================================
def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"[“”\"'`]", " ", text)
    text = re.sub(r"[(),!?]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text_no_acc = strip_accents(text)

    for src, dst in sorted(TERM_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        src_no_acc = strip_accents(src.lower())
        text_no_acc = re.sub(rf"\b{re.escape(src_no_acc)}\b", dst, text_no_acc)

    text_no_acc = re.sub(r"\s+", " ", text_no_acc).strip()
    return text_no_acc


# =========================================================
# 6. Helper extractors
# =========================================================
def _find_first_alias(query: str, groups: Dict[str, List[str]]) -> Optional[str]:
    for canonical, aliases in groups.items():
        for alias in sorted(aliases, key=len, reverse=True):
            alias_norm = normalize_text(alias)
            if re.search(rf"\b{re.escape(alias_norm)}\b", query):
                return canonical
    return None


def extract_ticker(query: str) -> Optional[str]:
    return _find_first_alias(query, TICKER_ALIAS_GROUPS)


def extract_model_name(query: str) -> Optional[str]:
    found = _find_first_alias(query, MODEL_ALIAS_GROUPS)
    return found if found else "main"


def extract_metric_name(query: str) -> Optional[str]:
    for canonical, aliases in METRIC_ALIAS_GROUPS.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm in query:
                return canonical

    if "p@5" in query or "precision@5" in query:
        return "precision_at_k"
    if "p@10" in query or "precision@10" in query:
        return "precision_at_k"
    if "p@20" in query or "precision@20" in query:
        return "precision_at_k"

    return None


def extract_top_k(query: str) -> Optional[int]:
    patterns = [
        r"\btop\s*(\d+)\b",
        r"\btop(\d+)\b",
        r"\blon nhat\s*(\d+)\b",
        r"\blớn nhất\s*(\d+)\b",
        r"\bnhieu nhat\s*(\d+)\b",
        r"\bnhiều nhất\s*(\d+)\b",
    ]

    for pat in patterns:
        m = re.search(pat, query)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

    return None


def extract_count_flag(query: str) -> bool:
    patterns = [
        r"\bbao nhieu\b",
        r"\bco may\b",
        r"\bmay\b",
        r"\bso luong\b",
        r"\bcount\b",
        r"\bhow many\b",
    ]
    return any(re.search(pat, query) for pat in patterns)


def _month_range(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    last_day = monthrange(year, month)[1]
    return pd.Timestamp(year=year, month=month, day=1), pd.Timestamp(year=year, month=month, day=last_day)


def _quarter_range(year: int, quarter: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    month_start = 3 * (quarter - 1) + 1
    month_end = month_start + 2
    start = pd.Timestamp(year=year, month=month_start, day=1)
    end_day = monthrange(year, month_end)[1]
    end = pd.Timestamp(year=year, month=month_end, day=end_day)
    return start, end


def _clip_to_analysis_range(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = max(start.normalize(), ANALYSIS_START)
    end = min(end.normalize(), ANALYSIS_END)
    return start, end


def extract_date_entities(query: str) -> Dict[str, Optional[pd.Timestamp]]:
    out = {
        "date": None,
        "start_date": None,
        "end_date": None,
        "is_relative_fallback": False,
    }

    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", query)
    if m:
        try:
            out["date"] = pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3))).normalize()
            return out
        except Exception:
            pass

    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", query)
    if m:
        try:
            out["date"] = pd.Timestamp(year=int(m.group(3)), month=int(m.group(2)), day=int(m.group(1))).normalize()
            return out
        except Exception:
            pass

    m = re.search(r"\bthang\s*(\d{1,2})\s*/\s*(20\d{2})\b", query)
    if not m:
        m = re.search(r"\b(\d{1,2})\s*/\s*(20\d{2})\b", query)

    if m:
        try:
            month = int(m.group(1))
            year = int(m.group(2))
            start, end = _month_range(year, month)
            out["start_date"], out["end_date"] = _clip_to_analysis_range(start, end)
            return out
        except Exception:
            pass

    m = re.search(r"\b(20\d{2})\s*q([1-4])\b", query)
    if not m:
        m = re.search(r"\bq([1-4])\s*(20\d{2})\b", query)
        if m:
            q = int(m.group(1))
            y = int(m.group(2))
            start, end = _quarter_range(y, q)
            out["start_date"], out["end_date"] = _clip_to_analysis_range(start, end)
            return out

    if m:
        try:
            y = int(m.group(1))
            q = int(m.group(2))
            start, end = _quarter_range(y, q)
            out["start_date"], out["end_date"] = _clip_to_analysis_range(start, end)
            return out
        except Exception:
            pass

    m = re.search(r"\b(20\d{2})\b", query)
    if m:
        year = int(m.group(1))
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        start, end = _clip_to_analysis_range(start, end)
        if start <= end:
            out["start_date"], out["end_date"] = start, end
            return out

    if "tuan nay" in query or "week this" in query or "this week" in query:
        out["start_date"] = ANALYSIS_END - pd.Timedelta(days=6)
        out["end_date"] = ANALYSIS_END
        out["is_relative_fallback"] = True
        return out

    if "thang nay" in query or "this month" in query:
        out["start_date"] = ANALYSIS_END - pd.Timedelta(days=29)
        out["end_date"] = ANALYSIS_END
        out["is_relative_fallback"] = True
        return out

    if "nam nay" in query or "this year" in query:
        out["start_date"] = pd.Timestamp("2024-01-01")
        out["end_date"] = ANALYSIS_END
        out["is_relative_fallback"] = True
        return out

    return out


# =========================================================
# 7. Intent keyword groups
# =========================================================
PROJECT_TERMS = [
    "du an", "project", "pipeline", "workflow", "he thong", "system", "framework",
    "dang phan tich", "co may ma", "bao nhieu ma", "ma nao", "ticker nao",
    "scope", "pham vi", "giai doan phan tich",
]

DATA_TERMS = [
    "du lieu", "data", "dataset", "source", "nguon", "macro", "ticker", "date range",
    "lay tu dau", "nguon du lieu", "market data", "macro data", "news source",
    "phan tich nam nao", "analysis window",
]

TOP_TERMS = [
    "top", "lon nhat", "lớn nhất", "nhieu nhat", "nhiều nhất", "max"
]

NEWS_TERMS = [
    "news", "bai bao", "bài báo", "article", "event", "cause", "nguyen nhan", "tin nao"
]

NUMERIC_EXPLAIN_TERMS = [
    "feature", "xai", "integrated gradients", "importance", "vi sao", "why", "nguyen nhan so"
]

COUNTERFACTUAL_TERMS = [
    "counterfactual", "phan bien", "phản biện", "co chac khong", "chac khong", "drop"
]

MODEL_COMPARE_TERMS = [
    "model nao tot hon", "compare model", "so sanh model", "baseline", "main", "compare"
]

DASHBOARD_HELP_TERMS = [
    "dashboard", "web", "xai web sao", "xai sao", "cach dung", "help", "filter", "tab"
]

EVENT_WINDOW_TERMS = [
    "event window", "earnings", "macro shock", "fomc", "cpi", "event hit rate", "window"
]

ANOMALY_TERMS = [
    "anomaly", "diem bat thuong", "diem anomaly", "bat thuong", "co gi bat thuong", "anomaly nao"
]


def has_any_term(query: str, terms: List[str]) -> bool:
    for t in terms:
        t_norm = normalize_text(t)
        if t_norm in query:
            return True
    return False


# =========================================================
# 8. Intent router
# =========================================================
def route_intent(
    query: str,
    ticker: Optional[str],
    exact_date: Optional[pd.Timestamp],
    date_start: Optional[pd.Timestamp],
    date_end: Optional[pd.Timestamp],
    metric_name: Optional[str],
    count_only: bool,
) -> str:
    if ticker and exact_date is not None:
        return INTENT_ANOMALY_LOOKUP

    if has_any_term(query, COUNTERFACTUAL_TERMS) and ticker:
        return INTENT_COUNTERFACTUAL_VALIDATION

    if has_any_term(query, NEWS_TERMS) and ticker:
        return INTENT_ANOMALY_EXPLAIN_NEWS

    if has_any_term(query, NUMERIC_EXPLAIN_TERMS) and ticker:
        return INTENT_ANOMALY_EXPLAIN_NUMERIC

    if metric_name is not None:
        return INTENT_METRICS_QUERY

    if has_any_term(query, MODEL_COMPARE_TERMS):
        return INTENT_MODEL_COMPARE

    if has_any_term(query, EVENT_WINDOW_TERMS):
        return INTENT_EVENT_WINDOW_QUERY

    if has_any_term(query, DASHBOARD_HELP_TERMS):
        return INTENT_DASHBOARD_HELP

    if has_any_term(query, PROJECT_TERMS):
        return INTENT_PROJECT_OVERVIEW

    if has_any_term(query, DATA_TERMS):
        return INTENT_DATA_QUERY

    if ticker and (date_start is not None or date_end is not None) and (
        count_only or has_any_term(query, ANOMALY_TERMS) or has_any_term(query, TOP_TERMS)
    ):
        return INTENT_ANOMALY_TOPK

    if ticker and (count_only or has_any_term(query, TOP_TERMS)):
        return INTENT_ANOMALY_TOPK

    if not ticker and count_only and ("ma" in query or "ticker" in query):
        return INTENT_PROJECT_OVERVIEW

    if not ticker and (count_only or has_any_term(query, TOP_TERMS)) and has_any_term(query, ANOMALY_TERMS):
        return INTENT_PROJECT_OVERVIEW

    if ticker:
        return INTENT_ANOMALY_LOOKUP

    return INTENT_OUT_OF_SCOPE


# =========================================================
# 9. Main parser
# =========================================================
def parse_user_query(raw_query: str) -> IntentResult:
    norm = normalize_text(raw_query)

    ticker = extract_ticker(norm)
    model_name = extract_model_name(norm)
    metric_name = extract_metric_name(norm)
    top_k = extract_top_k(norm)
    count_only = extract_count_flag(norm)

    date_info = extract_date_entities(norm)
    exact_date = date_info["date"]
    start_date = date_info["start_date"]
    end_date = date_info["end_date"]

    notes: List[str] = []
    if date_info.get("is_relative_fallback", False):
        notes.append(
            "Relative date phrase was mapped to the end of the analysis range (2024-12-31 based fallback)."
        )

    if count_only:
        notes.append("Count-style query detected.")

    intent = route_intent(
        query=norm,
        ticker=ticker,
        exact_date=exact_date,
        date_start=start_date,
        date_end=end_date,
        metric_name=metric_name,
        count_only=count_only,
    )

    if intent == INTENT_ANOMALY_TOPK and top_k is None and not count_only:
        top_k = 5

    entities = {
        "ticker": ticker,
        "date": exact_date.strftime("%Y-%m-%d") if exact_date is not None else None,
        "start_date": start_date.strftime("%Y-%m-%d") if start_date is not None else None,
        "end_date": end_date.strftime("%Y-%m-%d") if end_date is not None else None,
        "model_name": model_name,
        "metric_name": metric_name,
        "top_k": top_k,
        "count_only": count_only,
    }

    return IntentResult(
        raw_query=raw_query,
        normalized_query=norm,
        intent=intent,
        ticker=ticker,
        date=entities["date"],
        start_date=entities["start_date"],
        end_date=entities["end_date"],
        model_name=model_name,
        metric_name=metric_name,
        top_k=top_k,
        count_only=count_only,
        entities=entities,
        notes=notes,
    )


# # =========================================================
# # 10. Quick sanity tests
# # =========================================================
# if __name__ == "__main__":
#     examples = [
#         "NVDA ngày 2024-09-17 có gì bất thường không",
#         "top 5 anomaly của TSLA năm 2024",
#         "tin nào gây ra NVDA 2024-09-17",
#         "có chắc không, phản biện thử cho NVDA 2024-09-17",
#         "model nào tốt hơn giữa main và baseline",
#         "F1 của iforest là bao nhiêu",
#         "dữ liệu lấy từ đâu",
#         "web này xài sao",
#         "event hit rate là gì",
#         "MSFT tháng 5/2024 có gì nổi bật",
#         "NVDA q1 2024",
#         "tuần này NVDA có gì không",
#         "trong năm 2023 thì amazone có mấy điểm bất thường",
#         "có mấy mã đang phân tích",
#     ]

#     for q in examples:
#         print("=" * 100)
#         print("RAW:", q)
#         result = parse_user_query(q)
#         print(result.to_dict())