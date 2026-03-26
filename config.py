from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

# ---------- Paths ----------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

RAW_MERGED_PATH = RAW_DIR / "raw_merged.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
SCALED_PATH = PROCESSED_DIR / "scaled.parquet"

ANOMALIES_PATH = RESULTS_DIR / "anomalies.csv"
XAI_FEATURE_PATH = RESULTS_DIR / "xai_feature.csv"
XAI_TEMPORAL_PATH = RESULTS_DIR / "xai_temporal.json"
XAI_FAITHFULNESS_PATH = RESULTS_DIR / "xai_faithfulness.csv"
XAI_STABILITY_PATH = RESULTS_DIR / "xai_stability.csv"
EVENT_WINDOWS_PATH = RESULTS_DIR / "event_windows.csv"

NEWS_CACHE_DIR = RESULTS_DIR / "news_cache"

MODELS_DIR = ROOT / "models"
SCALERS_DIR = MODELS_DIR / "scalers"
DL_DIR = MODELS_DIR / "dl"
EVENT_FUSION_DIR = MODELS_DIR / "event_fusion"

RUNS_DIR = ROOT / "runs"
EXTERNAL_DIR = ROOT / "external"

# ---------- Official repos (REQUIRED for research-grade official implementation) ----------
USE_OFFICIAL_IMPLEMENTATIONS = True
REQUIRE_OFFICIAL_IMPLEMENTATIONS = True

PATCHTST_REPO_DIR = Path(os.getenv("PATCHTST_REPO_DIR", str(EXTERNAL_DIR / "PatchTST"))).resolve()
PATCHTST_SUPERVISED_DIR = PATCHTST_REPO_DIR / "PatchTST_supervised"
TRANAD_REPO_DIR = Path(os.getenv("TRANAD_REPO_DIR", str(EXTERNAL_DIR / "TranAD"))).resolve()
TRANAD_SRC_DIR = TRANAD_REPO_DIR / "src"

# ---------- Universe ----------
TICKERS = ["TSLA", "AMZN", "NVDA", "MSFT"]
FRED_SERIES = {
    "VIXCLS": "vix",
    "DGS10": "dgs10",
    "FEDFUNDS": "fedfunds",
    "CPIAUCSL": "cpi",
}

# ---------- Time blocks ----------
DOWNLOAD_START = "2018-01-01"
DOWNLOAD_END = "2024-12-31"

ANALYSIS_START = "2023-01-01"
ANALYSIS_END = "2024-12-31"

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"  # test = 2024

# ---------- Feature engineering ----------
RETURN_COL = "log_return"
ROLL_WINDOWS = [5, 10, 20, 60]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ---------- Step 02 representation / scaling ----------
SCALE_METHOD = "robust"   # robust | standard | minmax
ROBUST_QUANTILE_RANGE = (10.0, 90.0)
SCALE_INCLUDE_MACRO_LEVELS = False
KEEP_LOG_CLOSE = True

# ---------- Core modeling ----------
WINDOW_SIZE = 60
TAIL_K = 10

# Main detector = calibrated PatchTST residual + calibrated TranAD reconstruction
MAIN_PATCH_WEIGHT = 0.50
MAIN_TRANAD_WEIGHT = 0.50

# ---------- Score calibration / thresholding ----------
# Research-standard: do NOT fuse Gaussian z-scores from incomparable branches.
CALIBRATION_METHOD = "train_percentile"  # train_percentile | robust_z
CALIBRATION_CLIP_MIN = 0.0
CALIBRATION_CLIP_MAX = 1.0
CALIBRATION_GRID_WEIGHTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
FUSION_TUNE_METRIC = "auprc"

THRESH_METHOD = "pctl"
THRESH_PCTL = 95
USE_ADAPTIVE_MAIN = True
USE_ADAPTIVE_ALL = True
ADAPT_MAIN_LOOKBACK = 126
ADAPT_MAIN_Q = 0.995
ADAPT_ALL_LOOKBACK = 252

# ---------- Change-point / subtype ----------
# cp_flag is descriptive only. It must NEVER be part of point-wise GT.
CP_ALGO = "ruptures_pelt"
CP_SOURCE = "log_return"   # descriptive change-point source; no circular dependency
CP_MIN_SIZE = 20
CP_PENALTY = 8.0

SUBTYPE_RET_Z = 3.0
SUBTYPE_VOL_Z = 3.0

# ---------- Evaluation ----------
# Research-standard: point-wise GT is shock-based only.
WEAK_Z = 3.0
WEAK_VOL_Z = 3.0
EVAL_INCLUDE_CP_IN_GT = False

# ---------- Chronos benchmark ----------
# Keep Amazon Chronos as the external comparison benchmark.
# Requires: pip install chronos-forecasting
CHRONOS_MODEL_ID = os.getenv("CHRONOS_MODEL_ID", "autogluon/chronos-2-small")
CHRONOS_DEVICE_MAP = os.getenv("CHRONOS_DEVICE_MAP", "cpu")
CHRONOS_DTYPE = os.getenv("CHRONOS_DTYPE", "float32")
CHRONOS_USE_COVARIATES = True
CHRONOS_CONTEXT_MAX = int(os.getenv("CHRONOS_CONTEXT_MAX", "60"))
CHRONOS_MIN_CONTEXT = int(os.getenv("CHRONOS_MIN_CONTEXT", "30"))
CHRONOS_PREDICTION_LENGTH = 1
CHRONOS_QUANTILES = [0.1, 0.5, 0.9]
CHRONOS_TARGET_COL = "log_return"

CHRONOS_COVARIATE_COLS = [
    "volume_z_20",
    "vol_5", "vol_10", "vol_20", "vol_60",
    "rsi", "macd", "macd_signal", "macd_diff",
    "vix_diff", "dgs10_diff", "fedfunds_diff", "cpi_diff",
]

# ---------- News ----------
NEWS_LOOKBACK_DAYS = 7
NEWS_MAX_RESULTS = 7
NEWS_LANG = "English"
NEWS_COUNTRY = ""

# ---------- Keys / services ----------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
REQUIRE_FRED_KEY = True

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
REQUIRE_OLLAMA = False

# ---------- Repro / runtime ----------
SEED = 42
DEVICE = os.getenv("DEVICE", "cpu")
N_JOBS = int(os.getenv("N_JOBS", "4"))

# ---------- Step 05 official PatchTST / TranAD defaults ----------
STEP05_EPOCHS = int(os.getenv("STEP05_EPOCHS", "20"))
STEP05_BATCH_SIZE = int(os.getenv("STEP05_BATCH_SIZE", "64"))
STEP05_LR = float(os.getenv("STEP05_LR", "0.001"))
STEP05_WEIGHT_DECAY = float(os.getenv("STEP05_WEIGHT_DECAY", "0.0001"))
STEP05_PATIENCE = int(os.getenv("STEP05_PATIENCE", "5"))
STEP05_GRAD_CLIP = float(os.getenv("STEP05_GRAD_CLIP", "1.0"))

# PatchTST official supervised config surface
PATCHTST_PRED_LEN = 1
PATCHTST_E_LAYERS = int(os.getenv("PATCHTST_E_LAYERS", "2"))
PATCHTST_D_MODEL = int(os.getenv("PATCHTST_D_MODEL", "64"))
PATCHTST_N_HEAD = int(os.getenv("PATCHTST_N_HEAD", "4"))
PATCHTST_D_FF = int(os.getenv("PATCHTST_D_FF", "128"))
PATCHTST_DROPOUT = float(os.getenv("PATCHTST_DROPOUT", "0.1"))
PATCHTST_FC_DROPOUT = float(os.getenv("PATCHTST_FC_DROPOUT", "0.1"))
PATCHTST_HEAD_DROPOUT = float(os.getenv("PATCHTST_HEAD_DROPOUT", "0.0"))
PATCHTST_PATCH_LEN = int(os.getenv("PATCHTST_PATCH_LEN", "12"))
PATCHTST_STRIDE = int(os.getenv("PATCHTST_STRIDE", "6"))
PATCHTST_PADDING_PATCH = os.getenv("PATCHTST_PADDING_PATCH", "end")
PATCHTST_REVIN = os.getenv("PATCHTST_REVIN", "false").lower() == "true"
PATCHTST_AFFINE = os.getenv("PATCHTST_AFFINE", "false").lower() == "true"
PATCHTST_SUBTRACT_LAST = os.getenv("PATCHTST_SUBTRACT_LAST", "false").lower() == "true"
PATCHTST_DECOMPOSITION = os.getenv("PATCHTST_DECOMPOSITION", "false").lower() == "true"
PATCHTST_KERNEL_SIZE = int(os.getenv("PATCHTST_KERNEL_SIZE", "25"))
PATCHTST_INDIVIDUAL = os.getenv("PATCHTST_INDIVIDUAL", "false").lower() == "true"
PATCHTST_TARGET_COL = "log_return_s"  # official PatchTST branch forecasts scaled target

# TranAD official config surface
TRANAD_WINDOW = int(os.getenv("TRANAD_WINDOW", "10"))
TRANAD_BATCH_SIZE = int(os.getenv("TRANAD_BATCH_SIZE", "128"))
TRANAD_LR = float(os.getenv("TRANAD_LR", "0.001"))
TRANAD_DROPOUT = float(os.getenv("TRANAD_DROPOUT", "0.1"))
TRANAD_TARGET_COL = "log_return_s"

