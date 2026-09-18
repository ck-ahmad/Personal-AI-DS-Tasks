
# AFL Assistant — Monitoring & Maintenance Checklist

## Daily / continuous
- **API availability:** health endpoint, HTTP 5xx/504 rate.
  - Alert: >2% 5xx over 10 minutes.
- **Latency:** p50/p95/p99.
  - Alert: p95 > 5 s for 10 minutes; investigate any >15 s timeout.
- **Tool errors:** retrieval/prediction failures.
  - Alert: >5% of tool calls fail over 30 minutes.
- **Guardrail leak rate:** sampled off-topic/injection tests that receive an AFL answer.
  - Alert: any confirmed leak in production; immediately reproduce and patch.
- **Abuse/rate limiting:** 429 volume and repeated blocked probes.
  - Alert: sustained >100 blocked requests/minute or abnormal per-client bursts.
- **Prediction observability:** log model version, feature version, fixture date,
  probabilities, winner, and eventual outcome when available.

## After each new AFL round
1. Ingest newly completed match results.
2. Re-run the Day-1 leakage-safe feature pipeline.
3. Append the new round to the evaluation set.
4. Score the current production model on the new round.
5. Compare accuracy, macro-F1, Brier score, calibration, and class-specific recall
   with the prior rolling window.
6. Run the 30+ conversational/guardrail regression suite.
7. Run the ladder-position baseline comparison.
8. Publish a short model/data health report.

## Drift / retraining triggers
Retrain when:
- a new round has completed (weekly refresh), OR
- rolling 4-round accuracy drops ≥5 percentage points from the prior 8-round
  reference window, OR
- Brier score worsens by ≥0.05, OR
- calibration error shows a sustained deterioration, OR
- feature schema/data-quality checks fail after ingestion.

Do not automatically promote a retrained model. Require:
- leakage audit,
- hold-out evaluation,
- baseline comparison,
- guardrail regression pass,
- model artifact/version review.

## Weekly refresh loop
**Monday:** ingest completed round -> validate schema -> rebuild feature table.

**Tuesday:** retrain candidate model on all eligible historical data using the
same time-aware evaluation policy; retain a recent chronological hold-out.

**Wednesday:** run 30+ behavioral tests + injection tests + benchmark.

**Thursday:** inspect drift/calibration and review errors.

**Friday:** promote only if acceptance checks pass; otherwise retain current model
and open an investigation.

## Ownership
- Data pipeline: validates source files, dates, duplicates, missingness.
- ML owner: model performance, calibration, leakage, retraining.
- Application owner: API latency, errors, rate limiting.
- Product/stakeholder owner: scope, answer quality, demo acceptance.
