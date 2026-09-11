"""
FastAPI wrapper around the LangGraph ticket-triage agent.

Endpoints:
  POST /triage             submit a new ticket, runs the graph to completion
                            or to the human_checkpoint interrupt.
  POST /triage/{ticket_id}/resume
                            resume a paused (interrupted) run with a human
                            decision: approve / reject / edit.
  GET  /health              liveness check.
  GET  /logs/recent         last N structured log lines (for a monitoring
                            dashboard to poll).

Run:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent))
from agent.graph import compile_graph  # noqa: E402

LOG_PATH = Path(__file__).parent.parent / "logs" / "agent.log.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)

logger = logging.getLogger("triage_agent")
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))


def log_event(event: str, **fields: Any) -> None:
    record = {"ts": time.time(), "event": event, **fields}
    logger.info(json.dumps(record, default=str))


checkpointer = MemorySaver()
_graph = compile_graph(checkpointer=checkpointer)

app = FastAPI(title="Support Ticket Triage Agent", version="1.0")


class TriageRequest(BaseModel):
    client_name: str = Field(..., min_length=1, max_length=200)
    raw_text: str = Field(..., min_length=0, max_length=6000)
    ticket_id: Optional[str] = None


class TriageResponse(BaseModel):
    ticket_id: str
    status: Literal["completed", "awaiting_human_approval", "rejected_invalid_input"]
    category: Optional[str] = None
    severity: Optional[str] = None
    final_response: Optional[str] = None
    action_taken: Optional[str] = None
    pending_review: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: dict[str, float] = {}


class ResumeRequest(BaseModel):
    decision: Literal["approved", "rejected", "edited"]
    edited_text: Optional[str] = None


def _to_response(ticket_id: str, result: dict[str, Any]) -> TriageResponse:
    if result.get("__interrupt__"):
        return TriageResponse(
            ticket_id=ticket_id,
            status="awaiting_human_approval",
            category=result.get("category"),
            severity=result.get("severity"),
            pending_review=result["__interrupt__"],
            latency_ms=result.get("latency_ms", {}),
        )
    if result.get("validation_error"):
        return TriageResponse(
            ticket_id=ticket_id,
            status="rejected_invalid_input",
            error=result.get("validation_error"),
            latency_ms=result.get("latency_ms", {}),
        )
    return TriageResponse(
        ticket_id=ticket_id,
        status="completed",
        category=result.get("category"),
        severity=result.get("severity"),
        final_response=result.get("final_response"),
        action_taken=result.get("action_taken"),
        error=result.get("error"),
        latency_ms=result.get("latency_ms", {}),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    ticket_id = req.ticket_id or f"T-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    config = {"configurable": {"thread_id": ticket_id}}

    log_event("request_received", ticket_id=ticket_id, client_name=req.client_name, chars=len(req.raw_text))

    try:
        state = _graph.invoke(
            {"ticket_id": ticket_id, "client_name": req.client_name, "raw_text": req.raw_text},
            config=config,
        )
    except Exception as e:  # noqa: BLE001 - top-level safety net for the API boundary
        log_event("unhandled_error", ticket_id=ticket_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal agent error. Ticket logged for review.") from e

    snapshot = _graph.get_state(config)
    interrupt_payload = None
    if snapshot.next:  # graph is paused at an interrupt
        for task in snapshot.tasks:
            if task.interrupts:
                interrupt_payload = task.interrupts[0].value

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    log_event(
        "request_completed",
        ticket_id=ticket_id,
        status="awaiting_human_approval" if interrupt_payload else "completed",
        category=state.get("category"),
        severity=state.get("severity"),
        action_taken=state.get("action_taken"),
        error=state.get("error"),
        total_latency_ms=total_ms,
        node_latency_ms=state.get("latency_ms", {}),
        tool_calls=state.get("tool_calls", []),
    )

    result_payload = dict(state)
    if interrupt_payload:
        result_payload["__interrupt__"] = interrupt_payload
    return _to_response(ticket_id, result_payload)


@app.post("/triage/{ticket_id}/resume", response_model=TriageResponse)
def resume(ticket_id: str, req: ResumeRequest) -> TriageResponse:
    from langgraph.types import Command

    config = {"configurable": {"thread_id": ticket_id}}
    snapshot = _graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=404, detail="No paused run found for this ticket_id.")

    t0 = time.perf_counter()
    decision_payload = {"decision": req.decision, "edited_text": req.edited_text}
    log_event("human_decision_received", ticket_id=ticket_id, **decision_payload)

    state = _graph.invoke(Command(resume=decision_payload), config=config)
    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    log_event(
        "resume_completed",
        ticket_id=ticket_id,
        action_taken=state.get("action_taken"),
        error=state.get("error"),
        total_latency_ms=total_ms,
    )
    return _to_response(ticket_id, dict(state))


@app.get("/logs/recent")
def recent_logs(n: int = 50) -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().splitlines()[-n:]
    return [json.loads(line) for line in lines]
