from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import os
import platform
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# --- make "src/" importable when running: python src/00_preflight.py
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))     # so we can import config.py
sys.path.insert(0, str(SRC_DIR))      # so we can import utils/*

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

import config  # noqa: E402
from utils.checks import ok, warn, fail, require  # noqa: E402
from utils.io import ensure_dir, sha256_file, write_json  # noqa: E402
from utils.time_utils import parse_ymd  # noqa: E402


CORE_IMPORTS = [
    "numpy",
    "pandas",
    "pyarrow",
    "yfinance",
    "fredapi",
    "ta",
    "sklearn",
    "ruptures",
    "requests",
    "loguru",
]

OPTIONAL_IMPORTS = [
    "torch",
    "transformers",
    "sentence_transformers",
    "faiss",
    "streamlit",
    "plotly",
]


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_json(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (set, tuple)):
        return list(v)
    return v


def _collect_config_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for k, v in vars(config).items():
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        # keep only "simple" items
        try:
            json.dumps(_safe_json(v))
            snap[k] = _safe_json(v)
        except TypeError:
            snap[k] = str(v)
    return snap


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def _check_imports(pkgs: list[str], hard_fail: bool) -> None:
    import importlib

    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            missing.append(p)

    if missing and hard_fail:
        fail(f"Missing required packages: {missing}. Run: pip install -r requirements.txt")
    if missing and not hard_fail:
        warn(f"Missing optional packages: {missing} (ok for now)")


def _check_dates() -> None:
    # parse and validate order
    d0 = parse_ymd(config.DOWNLOAD_START)
    d1 = parse_ymd(config.DOWNLOAD_END)
    a0 = parse_ymd(config.ANALYSIS_START)
    a1 = parse_ymd(config.ANALYSIS_END)
    t_end = parse_ymd(config.TRAIN_END)
    v_end = parse_ymd(config.VAL_END)

    require(d0 < d1, "DOWNLOAD_START must be < DOWNLOAD_END")
    require(a0 < a1, "ANALYSIS_START must be < ANALYSIS_END")
    require(d0 <= a0 and a1 <= d1, "Analysis window must be inside download window")

    # no-leak split sanity
    require(t_end < v_end, "TRAIN_END must be < VAL_END")
    require(v_end <= a1, "VAL_END should be <= ANALYSIS_END (test is after VAL_END)")


def _check_env_keys() -> None:
    if getattr(config, "REQUIRE_FRED_KEY", False):
        if not config.FRED_API_KEY:
            fail(
                "Missing FRED_API_KEY in environment.\n"
                "Fix: create .env in project root and set FRED_API_KEY=...\n"
                "Then rerun: python src/00_preflight.py"
            )
        ok("FRED_API_KEY found")


def _check_folders_writable() -> None:
    required_dirs = [
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.RESULTS_DIR,
        config.NEWS_CACHE_DIR,
        config.MODELS_DIR,
        config.SCALERS_DIR,
        config.DL_DIR,
        config.EVENT_FUSION_DIR,
        config.RUNS_DIR,
    ]
    for d in required_dirs:
        ensure_dir(Path(d))
        test_file = Path(d) / ".write_test"
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
        except Exception as e:
            fail(f"Folder not writable: {d}. Error: {e}")
    ok("Folders exist & writable")


def _check_ollama(run_dir: Path) -> dict[str, Any]:
    # Only fail if REQUIRE_OLLAMA=True; otherwise warn.
    info: dict[str, Any] = {"reachable": False, "models": []}
    try:
        import requests

        url = config.OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        data = r.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        info["reachable"] = True
        info["models"] = models

        # Soft-check model availability
        llm_ok = config.OLLAMA_LLM_MODEL in models
        emb_ok = config.OLLAMA_EMBED_MODEL in models
        if not llm_ok:
            warn(f"Ollama reachable but LLM model not found: {config.OLLAMA_LLM_MODEL}")
        if not emb_ok:
            warn(f"Ollama reachable but EMBED model not found: {config.OLLAMA_EMBED_MODEL}")
        if llm_ok and emb_ok:
            ok("Ollama reachable & models found")

    except Exception as e:
        msg = f"Ollama check failed: {e}"
        if getattr(config, "REQUIRE_OLLAMA", False):
            fail(msg)
        else:
            warn(msg)

    # write debug snapshot
    (run_dir / "ollama_check.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    # --- run folder ---
    run_id = os.getenv("RUN_ID", "") or _now_run_id()
    run_dir = config.RUNS_DIR / run_id
    if run_dir.exists():
        run_id = run_id + "_dup"
        run_dir = config.RUNS_DIR / run_id
    ensure_dir(run_dir)

    # --- logger ---
    log_path = run_dir / "logs.txt"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", encoding="utf-8")

    logger.info("=== FinAnomX Preflight ===")
    logger.info(f"Project root: {ROOT_DIR}")
    logger.info(f"Run ID: {run_id}")

    # --- checks ---
    if sys.version_info < (3, 10):
        fail(f"Python>=3.10 required. Current: {sys.version}")

    _check_dates()
    ok("Config dates sanity OK")

    # normalize tickers
    t_upper = [t.strip().upper() for t in config.TICKERS]
    if t_upper != config.TICKERS:
        warn(f"TICKERS normalized to uppercase recommended. Current: {config.TICKERS}")

    _check_imports(CORE_IMPORTS, hard_fail=True)
    ok("Core imports OK")

    _check_imports(OPTIONAL_IMPORTS, hard_fail=False)

    _check_folders_writable()
    _check_env_keys()

    # Ollama (optional)
    ollama_info = _check_ollama(run_dir)

    # --- manifest ---
    files_to_hash = [
        ROOT_DIR / "config.py",
        ROOT_DIR / "requirements.txt",
    ]
    hashes = {}
    for fp in files_to_hash:
        if fp.exists():
            hashes[str(fp.relative_to(ROOT_DIR))] = sha256_file(fp)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": {
            "os": platform.platform(),
            "python": sys.version,
        },
        "config_snapshot": _collect_config_snapshot(),
        "pip_freeze": _pip_freeze(),
        "file_hashes": hashes,
        "ollama": ollama_info,
    }
    write_json(manifest, run_dir / "manifest.json")

    ok(f"Preflight DONE. Run folder: {run_dir}")
    logger.info("Next recommended step: python src/01_ingest_raw.py")


if __name__ == "__main__":
    main()