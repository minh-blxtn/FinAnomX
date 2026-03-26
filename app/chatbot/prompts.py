# =========================================================
# app/chatbot/prompts.py
# Prompt builder for grounded FinAnomX chatbot
# =========================================================

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict
import json

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chatbot.rag_engine import RetrievalResult  # noqa: E402
from app.chatbot.responders import build_grounded_response  # noqa: E402


SYSTEM_PROMPT = """
You are the end-user facing assistant inside the FinAnomX dashboard.

Rules:
1. Answer in clear, natural English.
2. Use only the information provided in the retrieved context.
3. If the current artifacts do not provide enough evidence, say that clearly instead of guessing.
4. Never invent dates, metrics, prices, signals, events, causes, or confidence levels.
5. Treat event explanations as plausible evidence-backed explanations, not definitive proof of causality.
6. Do not mention internal files, raw file paths, source tables, hidden reasoning, retrieval notes, or implementation details.
7. Do not include sections such as "Sources used" or "How this query was interpreted".
8. Keep the answer readable for a non-technical dashboard user.
9. For anomaly questions, explain three things when available:
   - what was unusual in the market,
   - which model signals stood out,
   - which external event is the strongest candidate explanation.
10. When evidence is weak, say so plainly and avoid overclaiming.
""".strip()


def _safe_json_dumps(obj: Any, max_len: int = 10000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [TRUNCATED FOR PROMPT SIZE]"


def build_user_prompt(result: RetrievalResult) -> str:
    context_json = _safe_json_dumps(result.context)
    parsed_json = _safe_json_dumps(result.parsed, max_len=2500)
    notes_json = _safe_json_dumps(result.notes, max_len=1500)
    grounded_draft = build_grounded_response(result)

    return f"""
User question:
{result.raw_query}

Resolved intent:
{result.effective_intent}

Parsed query fields:
{parsed_json}

Answerable from current artifacts:
{result.answerable}

Retrieval notes:
{notes_json}

Retrieved context:
{context_json}

Grounded draft answer (preserve these facts unless the retrieved context clearly says otherwise):
{grounded_draft}

If the query is a follow-up that refers to "that event", "that anomaly", or "it", use the resolved target already present in the retrieved context instead of asking the user to repeat the ticker or date.

Write a polished end-user answer in English.
Requirements:
- Do not mention internal files, paths, JSON keys, or source bookkeeping.
- Preserve the factual content of the grounded draft answer; improve wording only.
- Do not output a source section.
- Explain the anomaly in plain language.
- When event evidence exists, briefly summarize the event content before explaining why it may matter.
- If an event explanation is available, explain what the article is about and why that content plausibly relates to the observed market behaviour.
- Do not claim that no event is available if the grounded draft already identifies a specific related event.
- If evidence is weak or missing, say that clearly.
- Prefer short paragraphs or compact bullet points over raw data dumps.
""".strip()


def build_prompt_package(result: RetrievalResult) -> Dict[str, str]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(result),
    }




