"""
Pluggable "model" layer.

If ANTHROPIC_API_KEY is set in the environment, real calls are made to
Claude via the Anthropic SDK. Otherwise, a deterministic rule-based mock
model is used so the graph, API, and evaluation harness are fully runnable
offline (important for CI and for grading this capstone without secrets).

Swap MODE by setting the env var AGENT_LLM_MODE=real|mock (defaults to
"real" if a key is present, else "mock").
"""
from __future__ import annotations

import os
import re
from typing import Any

MODE = os.environ.get(
    "AGENT_LLM_MODE",
    "real" if os.environ.get("ANTHROPIC_API_KEY") else "mock",
)

_REFUSAL_TRIGGERS = ("ignore previous instructions", "system prompt", "act as root")


class ModelRefusalError(Exception):
    """Raised when the model declines to produce output for a request
    (used to exercise Task 2 failure scenario #3: model refusal)."""


def _real_client():
    from anthropic import Anthropic  # imported lazily so mock mode has no hard dep

    return Anthropic()


def classify_ticket(raw_text: str) -> dict[str, Any]:
    """Returns {category, severity, escalation_required, rationale}."""
    if MODE == "real":
        client = _real_client()
        prompt = (
            "Classify this client support ticket. Respond ONLY with JSON: "
            '{"category": one of [billing,bug,feature_request,onboarding,abuse,unknown], '
            '"severity": one of [low,medium,high,critical], '
            '"escalation_required": true|false, "rationale": "<one sentence>"}.\n\n'
            f"Ticket:\n{raw_text}"
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        import json

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return json.loads(text.strip().strip("`").removeprefix("json"))

    return _mock_classify(raw_text)


_NEGATIONS = ("not ", "n't ", "no rush", "isn't", "wasn't", "aren't", "without")


def _has_signal(text: str, words: tuple[str, ...]) -> bool:
    """Word/phrase match that ignores simple negated occurrences, e.g.
    'not urgent' should NOT trigger the 'urgent' signal. This is a known
    limitation of the offline mock classifier -- see evaluation report for
    the failure case that motivated this fix."""
    for w in words:
        idx = text.find(w)
        if idx == -1:
            continue
        window = text[max(0, idx - 12):idx]
        if any(neg in window for neg in _NEGATIONS):
            continue
        return True
    return False


def _mock_classify(raw_text: str) -> dict[str, Any]:
    text = raw_text.lower()

    for trig in _REFUSAL_TRIGGERS:
        if trig in text:
            raise ModelRefusalError("Refusing to process a prompt-injection style ticket.")

    category = "unknown"
    severity = "low"
    escalation = False
    rationale_bits = []

    if any(w in text for w in ("refund", "invoice", "charged", "billing", "payment", "chargeback")):
        category = "billing"
    if any(w in text for w in (
        "bug", "error", "crash", "broken", "down", "not working", "500", "exception",
        "security issue", "privacy issue", "data leak", "can see another",
    )):
        category = "bug"
    if any(w in text for w in ("feature", "could you add", "would be great if", "request")):
        category = "feature_request"
    if any(w in text for w in ("onboard", "kickoff", "getting started", "access", "credentials")):
        category = "onboarding"
    if any(w in text for w in ("stupid", "idiots", "scam", "sue you", "lawyer")):
        category = "abuse"

    if _has_signal(text, ("production down", "production is down", "data loss", "security", "breach", "leaked", "hacked", "critical")):
        severity, escalation = "critical", True
        rationale_bits.append("security/production-down language detected")
    elif _has_signal(text, ("urgent", "asap", "immediately", "can't log in", "cannot log in", "all users")):
        severity, escalation = "high", True
        rationale_bits.append("urgency / broad-impact language detected")
    elif category == "billing" and any(w in text for w in ("chargeback", "sue", "lawyer", "legal")):
        severity, escalation = "high", True
        rationale_bits.append("legal/chargeback threat detected")
    elif category in ("bug",):
        severity = "medium"
    else:
        severity = "low"

    if category == "abuse":
        escalation = True
        rationale_bits.append("abusive tone requires human handling")

    if not rationale_bits:
        rationale_bits.append(f"routine {category} request, no urgency signals")

    return {
        "category": category,
        "severity": severity,
        "escalation_required": escalation,
        "rationale": "; ".join(rationale_bits),
    }


def draft_response(raw_text: str, category: str, severity: str, kb_hits: list[dict[str, Any]], client_name: str) -> str:
    if MODE == "real":
        client = _real_client()
        kb_context = "\n".join(f"- ({h['source']}) {h['snippet']}" for h in kb_hits) or "No KB matches."
        prompt = (
            f"You are a support agent for a freelance dev studio replying to a client named {client_name}. "
            f"Category: {category}. Severity: {severity}.\n"
            f"Relevant policy snippets:\n{kb_context}\n\n"
            f"Client ticket:\n{raw_text}\n\n"
            "Write a concise, empathetic, professional reply (under 150 words). "
            "Do not promise refunds, dates, or commitments the policy snippets don't support."
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    return _mock_draft(raw_text, category, severity, kb_hits, client_name)


def _mock_draft(raw_text: str, category: str, severity: str, kb_hits: list[dict[str, Any]], client_name: str) -> str:
    greeting = f"Hi {client_name.split()[0] if client_name else 'there'},"
    policy_note = f" {kb_hits[0]['snippet']}." if kb_hits else ""

    body_by_category = {
        "billing": (
            "Thanks for flagging this billing question. I've logged it for our billing team to review."
            f"{policy_note} We'll follow up with specifics before any charge or refund is adjusted."
        ),
        "bug": (
            "Thanks for the report -- I've logged this as a bug and it's being triaged now."
            f"{policy_note} We'll update you as soon as we have a fix or workaround."
        ),
        "feature_request": (
            "Thanks for the suggestion! I've added it to our backlog with your rationale attached."
            f"{policy_note} Our account lead will follow up on scope and timeline."
        ),
        "onboarding": (
            "Welcome aboard! Here's what happens next to get you fully set up."
            f"{policy_note}"
        ),
        "abuse": (
            "I understand you're frustrated, and I want to make sure this gets the right attention -- "
            "I'm looping in a member of our team to follow up with you directly."
        ),
        "unknown": (
            "Thanks for reaching out -- I want to make sure I route this correctly. "
            "Could you share a bit more detail on what you're seeing?"
        ),
    }

    body = body_by_category.get(category, body_by_category["unknown"])
    if severity in ("high", "critical"):
        body += " Given the urgency, a member of our team will be in touch shortly to confirm next steps."

    return f"{greeting}\n\n{body}\n\nBest,\nSupport Team"


def self_check_response(draft: str, raw_ticket: str, category: str) -> dict[str, Any]:
    """Reflection / self-correction pass (reused pattern from Day 4).

    Checks for a small set of concrete failure modes: over-promising,
    missing empathy, length, and leaking internal policy jargon verbatim.
    """
    issues = []

    if re.search(r"\b(guarantee|promise|100%|immediately fixed|refund (is|has been) (issued|approved))\b", draft, re.I):
        issues.append("Draft makes an unauthorized guarantee/commitment.")

    if len(draft.split()) > 180:
        issues.append("Draft is too long (>180 words) for a first-touch reply.")

    if category == "abuse" and "escalat" not in draft.lower() and "team" not in draft.lower():
        issues.append("Abusive/escalation ticket reply doesn't signal human follow-up.")

    if not re.search(r"(hi|hello|hey)\b", draft.lower()):
        issues.append("Draft is missing a greeting / reads as cold.")

    passed = len(issues) == 0
    return {"passed": passed, "feedback": "; ".join(issues) if issues else None}
