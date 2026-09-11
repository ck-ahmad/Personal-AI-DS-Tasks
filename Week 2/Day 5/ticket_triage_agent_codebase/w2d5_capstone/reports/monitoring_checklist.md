# Production Monitoring Checklist — Ticket Triage Agent

## What to track

| Metric | Source | Why it matters |
|---|---|---|
| Error rate (5xx / unhandled exceptions) | `/logs/recent`, `unhandled_error` events | Detects graph crashes, upstream API breakage |
| Tool failure rate (`send_client_response` timeouts/500s) | `tool_calls` field in `request_completed` logs | Flags a flaky/degraded downstream helpdesk API |
| Human-approval rate & turnaround time | `human_decision_received` vs `request_completed` timestamps | Rising escalation rate = drift in ticket mix or over-cautious routing |
| Human rejection/edit rate | `resume_completed.action_taken` | Rising rejection rate = drafts are getting worse; retrain/re-prompt signal |
| Latency (p50/p95) per node and end-to-end | `latency_ms` in logs | Catches slow KB search, slow model calls, slow DB lookups |
| Cost per run (token usage in real mode) | Anthropic API response usage field (add to `log_event`) | Cost drift, runaway retry loops |
| Self-check revision rate | `revision_count` in state | Rising rate = drafts failing quality bar more often — model or prompt regression |
| Classification distribution (category/severity mix) | aggregate `request_completed.category/severity` | Output drift vs. the mix the system was evaluated on |
| Safety incidents: any auto-send on abuse/critical/injection-flagged ticket | cross-check `category`/`escalation_required` against `action_taken` | Should be **zero**, ever — treat as a P0 alert |

## Suggested alert thresholds

- Error rate > 2% of requests over a rolling 1-hour window → page on-call.
- Tool failure rate (send) > 10% over 1 hour → page on-call (matches the
  simulated flakiness rate used in eval; anything sustained above that in
  prod indicates a real outage, not noise).
- Any safety incident (auto-send bypassing a required human checkpoint) →
  immediate P0 page, no batching.
- p95 end-to-end latency > 3x the evaluation baseline (~9 ms mock / set a
  real-mode baseline after first production week) → investigate.
- Human rejection/edit rate > 25% over a rolling 3-day window → pull a
  sample for prompt/quality review.
- Cost per run > 1.5x the 7-day rolling average → investigate (retry
  loops, prompt bloat, model upgrade).

## Re-evaluation cadence

- **Weekly:** re-run `eval/run_eval.py` against the fixed test suite as a
  regression check after any prompt/model/graph change.
- **Monthly:** sample 20–30 real production tickets (stratified across
  category/severity), score them against the same rubric, and refresh the
  test suite with any new edge cases found in production.
- **On every model version bump** (e.g. Claude model upgrade): full eval
  suite re-run before rollout, compare pass rate and latency against the
  previous model's baseline.
- **Immediately after any safety incident:** root-cause, add a regression
  test case that reproduces it, re-run full suite before re-enabling
  auto-send for the affected category.
