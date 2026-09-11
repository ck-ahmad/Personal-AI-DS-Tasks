"""
State schema for the Client Support Ticket Triage & Response Agent.

Framework: LangGraph (see README.md / executive report for rationale).
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


Category = Literal["billing", "bug", "feature_request", "onboarding", "abuse", "unknown"]
Severity = Literal["low", "medium", "high", "critical"]


class TicketState(TypedDict, total=False):
    # --- input -----------------------------------------------------------
    ticket_id: str
    client_name: str
    raw_text: str

    # --- validation --------------------------------------------------------
    is_valid: bool
    validation_error: Optional[str]

    # --- classification ----------------------------------------------------
    category: Category
    severity: Severity
    escalation_required: bool  # true for high/critical or refund/legal language

    # --- retrieval -----------------------------------------------------------
    kb_hits: list[dict[str, Any]]

    # --- drafting / self-correction ----------------------------------------
    draft_response: str
    self_check_passed: bool
    self_check_feedback: Optional[str]
    revision_count: int

    # --- human-in-the-loop ---------------------------------------------------
    requires_human_approval: bool
    human_decision: Optional[Literal["approved", "rejected", "edited"]]
    human_edited_response: Optional[str]

    # --- output / bookkeeping -----------------------------------------------
    final_response: Optional[str]
    action_taken: Optional[str]
    error: Optional[str]
    tool_calls: list[dict[str, Any]]
    latency_ms: dict[str, float]
