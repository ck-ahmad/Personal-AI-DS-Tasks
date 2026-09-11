# Slide Outline — Support Ticket Triage & Response Agent
*(5–7 minute stakeholder walkthrough — 8 slides)*

## Slide 1 — Title
- Support Ticket Triage & Response Agent
- Sub: An AI agent that drafts first-touch client replies and knows when to ask a human

## Slide 2 — The Problem
- Inbound client tickets (billing, bugs, feature requests, onboarding, disputes) are repetitive but risky to get wrong
- A wrong promise (refund, fix date) or a tone-deaf reply to an angry client costs trust and money
- Goal: automate the routine 80%, never auto-send the risky 20%

## Slide 3 — How It Works (Architecture)
- [Insert architecture.png diagram]
- Ticket → validate → classify → check KB + client history → draft → self-check → (auto-send OR human approval) → send
- Two local data sources: policy knowledge base (markdown) + client ticket history (SQLite)

## Slide 4 — Why LangGraph
- Control-flow-heavy problem: branching by severity, a bounded self-correction loop, a pause-for-human step
- LangGraph gives us native interrupt/resume for the human checkpoint — no hand-rolled state machine
- CrewAI would fit a multi-role collaboration problem; this is one job needing tight control flow

## Slide 5 — Safety by Design (Human-in-the-Loop)
- Escalation triggers: high/critical severity, abusive or legal language, unresolved prior dispute, model refusal
- Nothing consequential ships without a human decision: approve / edit / reject
- Demo: a "production is down" ticket pausing for approval before any reply is sent

## Slide 6 — Evaluation Results
- 10 test cases, including 2 adversarial (oversized payload, prompt-injection attempt)
- Initial run: 6/10 pass → found a real bug (negation handling: "not urgent" misread as urgent)
- Fixed → 10/10 pass, safety checks passed 10/10 throughout
- Avg latency ~7ms in test mode; cost/latency will be re-baselined with live model calls

## Slide 7 — What's Live Today
- FastAPI endpoint (`POST /triage`, `POST /triage/{id}/resume`)
- Structured logging (inputs, tool calls, latency, errors) ready for a monitoring dashboard
- Monitoring checklist defined: error rate, tool failure rate, human rejection rate, safety-incident alarms

## Slide 8 — Next Steps & Ask
- Move from offline test mode to live Claude calls for classification/drafting
- Add PII/commitment guardrail checks before any draft reaches a human or client
- Start in "high-caution" mode (more human review), tighten thresholds as trust builds
- Ask: approval to pilot on a subset of real tickets for 2–4 weeks with human review on every send
