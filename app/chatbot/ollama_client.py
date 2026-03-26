# =========================================================
# app/chatbot/ollama_client.py
# Local Ollama client for FinAnomX chatbot
# =========================================================

from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json

import requests

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# 1. Config
# =========================================================
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "deepseek-r1:8b"
DEFAULT_TIMEOUT_SEC = 180


# =========================================================
# 2. Dataclasses
# =========================================================
@dataclass
class OllamaHealthResult:
    ok: bool
    base_url: str
    model_requested: str
    model_available: bool
    models: list
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OllamaGenerateResult:
    ok: bool
    model: str
    response_text: str
    thinking_text: str = ""
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# 3. Client
# =========================================================
class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_sec = timeout_sec

    # -----------------------------------------------------
    # low-level helpers
    # -----------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            self._url(path),
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_json(self, path: str) -> Dict[str, Any]:
        resp = requests.get(
            self._url(path),
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        return resp.json()

    # -----------------------------------------------------
    # health
    # -----------------------------------------------------
    def list_models(self) -> list:
        data = self._get_json("/api/tags")
        models = data.get("models", [])
        names = [m.get("name") for m in models if m.get("name")]
        return names

    def healthcheck(self, model_name: Optional[str] = None) -> OllamaHealthResult:
        target_model = model_name or self.model_name
        try:
            models = self.list_models()
            return OllamaHealthResult(
                ok=True,
                base_url=self.base_url,
                model_requested=target_model,
                model_available=target_model in models,
                models=models,
                error=None,
            )
        except Exception as e:
            return OllamaHealthResult(
                ok=False,
                base_url=self.base_url,
                model_requested=target_model,
                model_available=False,
                models=[],
                error=str(e),
            )

    # -----------------------------------------------------
    # generation
    # -----------------------------------------------------
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        num_predict: int = 900,
    ) -> OllamaGenerateResult:
        target_model = model_name or self.model_name

        payload = {
            "model": target_model,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            data = self._post_json("/api/generate", payload)

            response_text = (data.get("response") or "").strip()
            thinking_text = (data.get("thinking") or "").strip()
            done_reason = data.get("done_reason")

            # For reasoning models, if response is empty but thinking exists,
            # this usually means token budget ended before final answer.
            ok = True
            error = None

            if response_text == "" and thinking_text != "":
                ok = False
                error = (
                    "Model returned only reasoning/thinking but no final answer. "
                    "Increase num_predict or simplify prompt."
                )

            return OllamaGenerateResult(
                ok=ok,
                model=target_model,
                response_text=response_text,
                thinking_text=thinking_text,
                prompt_eval_count=data.get("prompt_eval_count"),
                eval_count=data.get("eval_count"),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_duration=data.get("prompt_eval_duration"),
                eval_duration=data.get("eval_duration"),
                done_reason=done_reason,
                raw=data,
                error=error,
            )

        except Exception as e:
            return OllamaGenerateResult(
                ok=False,
                model=target_model,
                response_text="",
                thinking_text="",
                raw=None,
                error=str(e),
            )

    def generate_grounded_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: Optional[str] = None,
    ) -> OllamaGenerateResult:
        """
        Preferred method for chatbot inference.
        Uses a larger token budget because DeepSeek-R1 often spends tokens on reasoning.
        """
        return self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=0.1,
            top_p=0.9,
            num_predict=1200,
        )


# =========================================================
# 4. Convenience helpers
# =========================================================
def get_default_client() -> OllamaClient:
    return OllamaClient()


def quick_healthcheck() -> Dict[str, Any]:
    client = get_default_client()
    return client.healthcheck().to_dict()


# =========================================================
# 5. Quick sanity test
# =========================================================
if __name__ == "__main__":
    client = OllamaClient()

    health = client.healthcheck()
    print("=" * 100)
    print("HEALTHCHECK")
    print("=" * 100)
    print(json.dumps(health.to_dict(), ensure_ascii=False, indent=2))

    if not health.ok:
        print("\nOllama server is not reachable.")
    elif not health.model_available:
        print(f"\nModel '{client.model_name}' is not available in local Ollama.")
    else:
        print("\nRunning a generation test with larger token budget...")
        result = client.generate(
            user_prompt="Say exactly: OLLAMA_OK",
            system_prompt="You are a precise assistant. Output only the requested text.",
            temperature=0.0,
            num_predict=200,
        )
        print("=" * 100)
        print("GENERATE TEST")
        print("=" * 100)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))