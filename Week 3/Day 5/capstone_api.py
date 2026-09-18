
"""AFL Week 3 Day 5 Capstone API + hardening wrapper.

Run:
    uvicorn capstone_api:app --host 0.0.0.0 --port 8000

The existing Week-3 Day-4 LangGraph application remains the source of truth
for retrieval/prediction behavior. This layer adds:
- AFL-only pre-filter
- prompt-injection / scope-abuse detection
- per-conversation rate limiting
- bounded execution timeout
- consistent prediction disclaimer
- structured JSONL monitoring logs
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from W3T4 import ask
except Exception as exc:
    ask = None
    IMPORT_ERROR = str(exc)

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "afl_outputs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "capstone_api.jsonl"

MAX_QUERY_CHARS = 1500
TOOL_TIMEOUT_SECONDS = 15
RATE_LIMIT_COUNT = 20
RATE_WINDOW_SECONDS = 60

# Deliberately conservative. These are scope-abuse patterns, not a claim that
# every matching phrase is malicious.
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bignore\s+the\s+system\s+prompt\b",
    r"\boverride\s+(the\s+)?system\b",
    r"\bdisregard\s+(all\s+)?instructions\b",
    r"\byou\s+are\s+no\s+longer\s+an?\s+afl\b",
    r"\bpretend\s+you\s+are\b",
    r"\breveal\s+(the\s+)?system\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\bjailbreak\b",
]

OFF_TOPIC_TERMS = [
    "cricket", "soccer", "nba", "nfl", "tennis", "bitcoin",
    "cryptocurrency", "python programming", "javascript", "weather",
    "politics", "recipe", "stock market",
]

PREDICTION_DISCLAIMER = (
    "This is a model-based probabilistic forecast, not a certainty."
)

app = FastAPI(title="AFL Capstone Assistant", version="1.0.0")
executor = ThreadPoolExecutor(max_workers=8)
requests_by_conversation: dict[str, deque[float]] = defaultdict(deque)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    conversation_id: str = Field(default="default", min_length=1, max_length=128)


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    intent: str | None = None
    tools_called: list[str] = []
    latency_ms: float
    prediction: dict[str, Any] | None = None
    blocked: bool = False
    error: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: dict[str, Any]) -> None:
    event = {"timestamp": now_iso(), **event}
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def is_injection_or_scope_abuse(message: str) -> tuple[bool, str | None]:
    lowered = message.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return True, "prompt-injection/scope-override pattern"
    if any(term in lowered for term in OFF_TOPIC_TERMS):
        return True, "explicitly non-AFL topic"
    return False, None


def rate_allowed(conversation_id: str) -> bool:
    now = time.monotonic()
    q = requests_by_conversation[conversation_id]
    while q and now - q[0] > RATE_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= RATE_LIMIT_COUNT:
        return False
    q.append(now)
    return True


def normalize_prediction_disclaimer(response: str, intent: str | None) -> str:
    if not intent or intent != "prediction":
        return response
    if PREDICTION_DISCLAIMER.lower() in response.lower():
        return response
    return response.rstrip() + "\n\n" + PREDICTION_DISCLAIMER


def extract_prediction_metadata(result: dict[str, Any]) -> dict[str, Any] | None:
    tool = result.get("tool_name")
    raw = result.get("tool_result")
    if tool != "predict_match_winner" or not isinstance(raw, dict):
        return None
    return {
        "winner": raw.get("winner"),
        "probability": raw.get("probability"),
        "probabilities": raw.get("probabilities"),
        "date": raw.get("date"),
        "home_team": raw.get("home_team"),
        "away_team": raw.get("away_team"),
        "disclaimer": PREDICTION_DISCLAIMER,
    }


def safe_call(message: str, conversation_id: str) -> dict[str, Any]:
    if ask is None:
        raise RuntimeError(
            "Could not import afl_week3_day4_langgraph.ask: " + IMPORT_ERROR
        )
    future = executor.submit(ask, message, conversation_id)
    try:
        return future.result(timeout=TOOL_TIMEOUT_SECONDS)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutError(
            f"AFL graph exceeded {TOOL_TIMEOUT_SECONDS}s timeout"
        ) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "afl-capstone", "timestamp": now_iso()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()

    blocked, reason = is_injection_or_scope_abuse(req.message)
    if blocked:
        response = (
            "I’m focused on AFL, so I can’t help with that request. "
            "I can help with AFL teams, players, matches, statistics, "
            "history, rules, retrieval, or predictions."
        )
        latency = (time.perf_counter() - start) * 1000
        log_event({
            "event": "blocked_request",
            "query": req.message,
            "conversation_id": req.conversation_id,
            "reason": reason,
            "intent": "off-topic",
            "tools_called": [],
            "latency_ms": latency,
        })
        return ChatResponse(
            response=response,
            conversation_id=req.conversation_id,
            intent="off-topic",
            tools_called=[],
            latency_ms=round(latency, 2),
            blocked=True,
        )

    if not rate_allowed(req.conversation_id):
        latency = (time.perf_counter() - start) * 1000
        log_event({
            "event": "rate_limited",
            "query": req.message,
            "conversation_id": req.conversation_id,
            "latency_ms": latency,
        })
        raise HTTPException(status_code=429, detail="Rate limit exceeded; retry shortly.")

    try:
        result = safe_call(req.message, req.conversation_id)
        intent = result.get("intent")
        response = normalize_prediction_disclaimer(
            str(result.get("final_response", "I could not produce a response.")),
            intent,
        )
        trace = result.get("trace") or []
        tools = [
            str(step.get("details", {}).get("tool"))
            for step in trace
            if step.get("details", {}).get("tool")
        ]
        prediction = extract_prediction_metadata(result)
        latency = (time.perf_counter() - start) * 1000

        log_event({
            "event": "chat",
            "query": req.message,
            "conversation_id": req.conversation_id,
            "intent": intent,
            "tools_called": tools,
            "latency_ms": latency,
            # Token usage is provider/model dependent. Keep field explicit so
            # telemetry can be populated later without changing the schema.
            "token_usage": result.get("token_usage"),
            "prediction": prediction,
            "error": None,
        })

        return ChatResponse(
            response=response,
            conversation_id=req.conversation_id,
            intent=intent,
            tools_called=tools,
            latency_ms=round(latency, 2),
            prediction=prediction,
        )

    except TimeoutError as exc:
        latency = (time.perf_counter() - start) * 1000
        log_event({
            "event": "timeout",
            "query": req.message,
            "conversation_id": req.conversation_id,
            "latency_ms": latency,
            "error": str(exc),
        })
        raise HTTPException(status_code=504, detail="AFL assistant timed out; retry shortly.")

    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        log_event({
            "event": "error",
            "query": req.message,
            "conversation_id": req.conversation_id,
            "latency_ms": latency,
            "error": str(exc),
        })
        raise HTTPException(status_code=500, detail="AFL assistant failed safely; see server logs.")
