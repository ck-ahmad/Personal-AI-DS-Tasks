# Support Ticket Triage & Response Agent — Week 2 Day 5 Capstone

An end-to-end LangGraph agent that triages inbound client support tickets
(freelance dev-shop / Web3Geeks-style client base), drafts a first-touch
reply grounded in a local policy knowledge base, self-corrects the draft,
and routes any consequential action (refunds, security bugs, abusive/legal
tickets, low-confidence classifications) through a human approval
checkpoint before anything is sent.

## Project layout

```
agent/
  state.py     # LangGraph state schema (TypedDict)
  tools.py     # search_knowledge_base (file RAG), lookup_client_history (SQLite),
               # send_client_response (simulated flaky external API)
  llm.py       # pluggable model layer: real Anthropic client if ANTHROPIC_API_KEY
               # is set, else a deterministic offline mock (for CI / grading)
  graph.py     # the StateGraph: nodes, conditional routing, human interrupt
api/
  main.py      # FastAPI wrapper: POST /triage, POST /triage/{id}/resume,
               # GET /health, GET /logs/recent
data/
  kb/*.md      # local knowledge base (refund policy, bug SLA, onboarding, etc.)
  init_db.py   # bootstraps tickets.db (client ticket history)
eval/
  test_cases.py  # 10 test cases incl. 2 edge/adversarial
  run_eval.py    # scoring harness -> results.csv / results.md
diagrams/
  architecture.png / .dot
reports/
  executive_report.pdf
  slide_outline.md
  monitoring_checklist.md
logs/
  agent.log.jsonl   # structured run logs (written at runtime)
```

## Setup

```bash
pip install fastapi uvicorn "langgraph>=0.2" langchain-core pydantic anthropic
python data/init_db.py          # one-time: create tickets.db
```

By default the agent runs in **mock mode** (`AGENT_LLM_MODE=mock`) if no
`ANTHROPIC_API_KEY` is set, so the whole system — API, graph, eval — is
runnable offline. Set `ANTHROPIC_API_KEY` (and optionally
`AGENT_LLM_MODE=real`) to use live Claude calls for classification and
drafting.

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
curl -X POST localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"client_name": "Acme Corp", "raw_text": "Production is down for all users, please help ASAP."}'
# -> {"status": "awaiting_human_approval", "pending_review": {...}, ...}

curl -X POST localhost:8000/triage/<ticket_id>/resume \
  -H "Content-Type: application/json" \
  -d '{"decision": "approved"}'
```

## Run the evaluation harness

```bash
AGENT_LLM_MODE=mock python3 eval/run_eval.py
```

Writes `eval/results.csv` and `eval/results.md`.

## Design notes

- **Framework: LangGraph.** The workflow has explicit branching (severity/
  category routing), a bounded retry cycle (self-correction), and a
  mid-run pause-and-resume requirement (human approval) — this is
  control-flow-heavy, stateful orchestration, which is what LangGraph is
  built for. See `reports/executive_report.pdf` for the full rationale.
- **Human-in-the-loop:** implemented with LangGraph's `interrupt()` in
  `human_checkpoint`, resumable via `POST /triage/{id}/resume` using
  `Command(resume=...)` and a `MemorySaver` checkpointer (swap for a
  persistent checkpointer, e.g. Postgres, in production).
- **Failure handling (Task 2):**
  1. *Bad input* — `validate_input` rejects empty/oversized/malformed
     tickets before any LLM or tool call.
  2. *Tool timeout/error* — `send_client_response` simulates a flaky
     downstream helpdesk API; `send()` catches `ToolTimeoutError` /
     `ToolExecutionError` and degrades gracefully (queues for retry /
     flags for manual send) instead of crashing.
  3. *Model refusal* — `classify()` catches `ModelRefusalError` (e.g. a
     prompt-injection-style ticket) and fails safe to human review rather
     than guessing.
