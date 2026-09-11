"""
LangGraph StateGraph for the Client Support Ticket Triage & Response Agent.

Node flow:

    validate_input --> classify --> retrieve_kb --> lookup_history
        --> draft --> self_check --(fail, retries left)--> draft
                          |--(pass)--> route_decision
    route_decision --(needs human)--> human_checkpoint --> finalize
                     --(auto-ok)-----------------------------> finalize

`human_checkpoint` uses LangGraph's `interrupt()` so a real deployment can
pause the graph and resume once a human approves/edits/rejects via the API
-- this is the human-in-the-loop checkpoint required for any consequential
action (sending a reply, and implicitly, any refund/escalation language).

Failure handling (Task 2, "at least 2 failure scenarios"):
  1. Bad input            -> validate_input short-circuits to `finalize`
                              with a validation error, no LLM/tool calls made.
  2. Tool timeout/error    -> `send` node catches ToolTimeoutError /
                              ToolExecutionError and returns a graceful
                              degraded result instead of crashing.
  3. Model refusal          -> `classify` catches ModelRefusalError and
                              routes straight to human_checkpoint.
"""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from agent.state import TicketState
from agent import llm, tools


MAX_REVISIONS = 2


def _timed(state: TicketState, key: str, start: float) -> None:
    state.setdefault("latency_ms", {})
    state["latency_ms"][key] = round((time.perf_counter() - start) * 1000, 2)


def validate_input(state: TicketState) -> TicketState:
    t0 = time.perf_counter()
    text = (state.get("raw_text") or "").strip()
    state.setdefault("tool_calls", [])

    if not text:
        state["is_valid"] = False
        state["validation_error"] = "Empty ticket body."
    elif len(text) > 6000:
        state["is_valid"] = False
        state["validation_error"] = "Ticket exceeds max length (6000 chars) -- possible payload abuse."
    elif not state.get("client_name"):
        state["is_valid"] = False
        state["validation_error"] = "Missing client_name."
    else:
        state["is_valid"] = True
        state["validation_error"] = None

    _timed(state, "validate_input", t0)
    return state


def classify(state: TicketState) -> TicketState:
    t0 = time.perf_counter()
    try:
        result = llm.classify_ticket(state["raw_text"])
        state["category"] = result["category"]
        state["severity"] = result["severity"]
        state["escalation_required"] = result["escalation_required"]
    except llm.ModelRefusalError as e:
        state["error"] = f"model_refusal: {e}"
        state["category"] = "unknown"
        state["severity"] = "medium"
        state["escalation_required"] = True  # fail safe: route to human
    _timed(state, "classify", t0)
    return state


def retrieve_kb(state: TicketState) -> TicketState:
    t0 = time.perf_counter()
    query = f"{state.get('category', '')} {state['raw_text']}"
    hits = tools.search_knowledge_base(query, top_k=2)
    state["kb_hits"] = hits
    state.setdefault("tool_calls", []).append({"tool": "search_knowledge_base", "n_hits": len(hits)})
    _timed(state, "retrieve_kb", t0)
    return state


def lookup_history(state: TicketState) -> TicketState:
    t0 = time.perf_counter()
    try:
        history = tools.lookup_client_history(state.get("client_name", ""))
        # Only an *unresolved* prior escalation should force human review on a
        # new, otherwise-routine ticket -- a past escalation that was already
        # resolved shouldn't permanently flag every future ticket from that
        # client. (Earlier version counted ANY past escalation, which
        # incorrectly flagged routine requests -- see eval TC1 / report.)
        unresolved_escalations = sum(1 for h in history if h.get("was_escalated") and not h.get("resolved"))
        if unresolved_escalations >= 1:
            state["escalation_required"] = True
        state.setdefault("tool_calls", []).append(
            {
                "tool": "lookup_client_history",
                "n_prior_tickets": len(history),
                "unresolved_escalations": unresolved_escalations,
            }
        )
    except tools.ToolExecutionError as e:
        # Non-fatal: proceed without history rather than failing the ticket.
        state.setdefault("tool_calls", []).append({"tool": "lookup_client_history", "error": str(e)})
    _timed(state, "lookup_history", t0)
    return state


def draft(state: TicketState) -> TicketState:
    t0 = time.perf_counter()
    state["draft_response"] = llm.draft_response(
        raw_text=state["raw_text"],
        category=state.get("category", "unknown"),
        severity=state.get("severity", "low"),
        kb_hits=state.get("kb_hits", []),
        client_name=state.get("client_name", ""),
    )
    _timed(state, f"draft_r{state.get('revision_count', 0)}", t0)
    return state


def self_check(state: TicketState) -> TicketState:
    t0 = time.perf_counter()
    result = llm.self_check_response(state["draft_response"], state["raw_text"], state.get("category", "unknown"))
    state["self_check_passed"] = result["passed"]
    state["self_check_feedback"] = result["feedback"]
    state["revision_count"] = state.get("revision_count", 0) + 1
    _timed(state, f"self_check_r{state['revision_count']}", t0)
    return state


def human_checkpoint(state: TicketState) -> TicketState:
    """Pauses the graph for a human approval decision on any escalation-
    required ticket before a reply can be sent. Uses LangGraph's
    `interrupt()` so this can be resumed from the API layer with a real
    human decision instead of auto-approving."""
    decision = interrupt(
        {
            "reason": "human_approval_required",
            "ticket_id": state.get("ticket_id"),
            "category": state.get("category"),
            "severity": state.get("severity"),
            "draft_response": state.get("draft_response"),
        }
    )
    state["requires_human_approval"] = True
    state["human_decision"] = decision.get("decision", "approved")
    state["human_edited_response"] = decision.get("edited_text")
    return state


def send(state: TicketState) -> TicketState:
    t0 = time.perf_counter()

    if state.get("human_decision") == "rejected":
        state["final_response"] = None
        state["action_taken"] = "not_sent_rejected_by_human"
        _timed(state, "send", t0)
        return state

    text = state.get("human_edited_response") or state.get("draft_response")
    state["final_response"] = text

    try:
        result = tools.send_client_response(state["ticket_id"], text)
        state["action_taken"] = f"sent (chars={result['chars_sent']})"
        state.setdefault("tool_calls", []).append({"tool": "send_client_response", **result})
    except tools.ToolTimeoutError as e:
        state["action_taken"] = "queued_for_retry_after_timeout"
        state["error"] = str(e)
        state.setdefault("tool_calls", []).append({"tool": "send_client_response", "error": str(e), "recoverable": True})
    except tools.ToolExecutionError as e:
        state["action_taken"] = "failed_flagged_for_manual_send"
        state["error"] = str(e)
        state.setdefault("tool_calls", []).append({"tool": "send_client_response", "error": str(e), "recoverable": False})

    _timed(state, "send", t0)
    return state


def finalize_invalid(state: TicketState) -> TicketState:
    state["final_response"] = None
    state["action_taken"] = f"rejected_invalid_input: {state.get('validation_error')}"
    return state


def route_decision_node(state: TicketState) -> TicketState:
    """Pass-through node: exists purely as a named point in the graph that
    the conditional edge `decide_route` branches from."""
    return state


# --- routing functions (return next node name, do not mutate meaningfully) --

def route_after_validate(state: TicketState) -> str:
    return "classify" if state.get("is_valid") else "finalize_invalid"


def route_after_self_check(state: TicketState) -> str:
    if state.get("self_check_passed"):
        return "route_decision"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        # give up revising, force human review rather than looping forever
        state["escalation_required"] = True
        return "route_decision"
    return "draft"


def decide_route(state: TicketState) -> str:
    return "human_checkpoint" if state.get("escalation_required") else "send"


def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("classify", classify)
    graph.add_node("retrieve_kb", retrieve_kb)
    graph.add_node("lookup_history", lookup_history)
    graph.add_node("draft", draft)
    graph.add_node("self_check", self_check)
    graph.add_node("human_checkpoint", human_checkpoint)
    graph.add_node("send", send)
    graph.add_node("finalize_invalid", finalize_invalid)
    graph.add_node("route_decision", route_decision_node)

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges("validate_input", route_after_validate, ["classify", "finalize_invalid"])
    graph.add_edge("classify", "retrieve_kb")
    graph.add_edge("retrieve_kb", "lookup_history")
    graph.add_edge("lookup_history", "draft")
    graph.add_edge("draft", "self_check")
    graph.add_conditional_edges("self_check", route_after_self_check, ["draft", "route_decision"])
    graph.add_conditional_edges("route_decision", decide_route, ["human_checkpoint", "send"])
    graph.add_edge("human_checkpoint", "send")
    graph.add_edge("send", END)
    graph.add_edge("finalize_invalid", END)

    return graph


def compile_graph(checkpointer=None):
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
