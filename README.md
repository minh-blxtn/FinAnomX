# FinAnomX

FinAnomX is an offline-first, artifacts-first pipeline for explainable financial anomaly detection on U.S. equities. The project detects anomaly episodes from market and macro time series, explains them with numeric attribution, grounds them with external news evidence, and packages the results for a Streamlit dashboard and a grounded chatbot.

## Project scope

- Market tickers: TSLA, AMZN, NVDA, MSFT
- Frequency: daily
- Data sources:
  - Yahoo Finance for OHLCV market data
  - FRED for macroeconomic series
  - SEC / calendar seeds for earnings and macro event windows
  - GDELT and article fetching for news evidence
- Core outputs:
  - anomaly scores and flags
  - episode peaks
  - numeric feature attribution
  - temporal attribution
  - grounded event attribution
  - counterfactual evidence strength
  - packaged artifacts for the web app and chatbot

## Repository structure

```text
FinAnomX/
├── app/
│   ├── streamlit_app.py
│   ├── components.py
│   ├── data_store.py
│   └── chatbot/
├── data/
│   ├── external/
│   ├── raw/
│   ├── processed/
│   └── results/
├── models/
│   ├── baselines/
│   ├── dl/
│   └── scalers/
├── runs/
├── src/
│   ├── 00_preflight.py
│   ├── 01_ingest_raw.py
│   ├── 02_process_data.py
│   ├── 03_build_event_windows.py
│   ├── 04_baselines.py
│   ├── 05_train_models.py
│   ├── 06_infer_anomalies.py
│   ├── 07_xai_numeric.py
│   ├── 08_news_on_demand.py
│   ├── 09_event_grounding.py
│   ├── 10_evaluate.py
│   ├── 11_pack_for_web.py
│   └── utils/
├── config.py
├── requirements.txt
├── .env.example
└── README.md
```

## Environment setup

Create a local `.env` file in the project root from `.env.example`.

Required variables:

- `FRED_API_KEY`: API key for FRED macro data download
- `OLLAMA_BASE_URL`: local Ollama server URL
- `OLLAMA_LLM_MODEL`: local LLM used by the chatbot
- `OLLAMA_EMBED_MODEL`: local embedding model used in event grounding
- `SEC_USER_AGENT`: user agent string for SEC requests

Example workflow:

1. Copy `.env.example` to `.env`
2. Replace placeholder values with your real local values
3. Do not commit `.env` to GitHub

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline execution order

Run the pipeline in this order:

```bash
python src/00_preflight.py
python src/01_ingest_raw.py
python src/02_process_data.py
python src/03_build_event_windows.py
python src/04_baselines.py
python src/05_train_models.py
python src/06_infer_anomalies.py
python src/07_xai_numeric.py
python src/08_news_on_demand.py
python src/09_event_grounding.py
python src/10_evaluate.py
python src/11_pack_for_web.py
```

## Step summary

### 00_preflight
Checks environment, folders, dependencies, and runtime prerequisites.

### 01_ingest_raw
Downloads market data and macro series, aligns them on market dates, and saves `data/raw/raw_merged.parquet`.

### 02_process_data
Builds engineered features, assigns time splits, fits train-only scalers, and saves:

- `data/processed/features.parquet`
- `data/processed/scaled.parquet`

### 03_build_event_windows
Builds earnings and macro event windows and saves `data/results/event_windows.csv`.

### 04_baselines
Runs baseline methods and benchmark scoring, including Chronos-related baseline outputs.

### 05_train_models
Trains the deep learning models and saves model checkpoints.

### 06_infer_anomalies
Generates anomaly scores, static and adaptive flags, episode labels, and saves `data/results/anomalies_base.csv`.

### 07_xai_numeric
Computes feature-level and temporal-level numeric explanations and saves:

- `data/results/xai_feature.csv`
- `data/results/xai_temporal.json`
- `data/results/xai_faithfulness.csv`
- `data/results/xai_stability.csv`

### 08_news_on_demand
Collects and enriches article evidence around anomaly peaks and saves `data/results/news_enriched.parquet`.

### 09_event_grounding
Ranks candidate articles, computes grounded event attribution and counterfactual support, and saves:

- `data/results/event_attribution.csv`
- `data/results/event_counterfactual.csv`

### 10_evaluate
Evaluates detection and explanation outputs and saves metrics tables.

### 11_pack_for_web
Packages final web artifacts and saves:

- `data/results/anomalies.csv`
- `data/results/chat_evidence.jsonl`

## Files required by the web app and chatbot

The Streamlit app and chatbot directly read these artifacts:

- `data/raw/raw_merged.parquet`
- `data/results/anomalies.csv`
- `data/results/anomalies_base.csv`
- `data/results/xai_feature.csv`
- `data/results/xai_temporal.json`
- `data/results/event_attribution.csv`
- `data/results/event_counterfactual.csv`
- `data/results/news_enriched.parquet`
- `data/results/chat_evidence.jsonl`
- `data/results/eval_detection_static.csv`
- `data/results/eval_detection_adapt.csv`
- `data/results/model_comparison_static.csv`
- `data/results/model_comparison_adapt.csv`
- `data/results/event_windows.csv`

## Data files to keep

Keep these files if you want the full pipeline and the web app to work:

- `data/external/event_seed_earnings.csv`
- `data/external/event_seed_macro.csv`
- `data/raw/raw_merged.parquet`
- `data/processed/features.parquet`
- `data/processed/scaled.parquet`
- `data/results/anomalies.csv`
- `data/results/anomalies_base.csv`
- `data/results/baseline_scores_chronos.csv`
- `data/results/chat_evidence.jsonl`
- `data/results/eval_detection_static.csv`
- `data/results/eval_detection_adapt.csv`
- `data/results/eval_explain.csv`
- `data/results/event_attribution.csv`
- `data/results/event_counterfactual.csv`
- `data/results/event_windows.csv`
- `data/results/model_comparison_static.csv`
- `data/results/model_comparison_adapt.csv`
- `data/results/news_enriched.parquet`
- `data/results/xai_feature.csv`
- `data/results/xai_faithfulness.csv`
- `data/results/xai_stability.csv`
- `data/results/xai_temporal.json`

## Data files that can be removed from the repository

These files are not required by the Streamlit app or chatbot, and are safe to remove from the GitHub repository if you want a cleaner public repo:

- `data/results/debug_patchtst_per_target_metrics.csv`
- `data/results/debug_patchtst_prediction_samples.csv`
- `data/results/debug_tranad_feature_profile.csv`
- `data/results/debug_tranad_window_samples.csv`
- `data/results/news_enriched_raw.parquet`
- `data/results/news_dropped_title_only.parquet`
- `data/results/news_embed_cache.json`
- `data/results/news_cache/`
- `data/results/eval_detection.csv`
- `data/results/eval_detection_episode.csv`
- `data/results/eval_detection_peak.csv`
- `data/results/model_comparison.csv`
- `data/results/baseline_scores_dlinear.csv`
- `data/results/baseline_scores_iforest.csv`
- `data/results/baseline_scores_ocsvm.csv`
- `data/results/baseline_scores_pca.csv`

## Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

## Notes for GitHub

- Commit `.gitignore`
- Commit `.env.example`
- Do not commit `.env`
- Do not commit `.venv/`
- Do not commit `__pycache__/`
- Do not commit local debug caches or temporary logs

## Security note

If any API key or personal email has already been exposed in a tracked `.env` file, rotate the secret and replace the tracked file history before publishing the repository.
