# Week 2 Day 4 — CrewAI Multi-Agent Crew

**Task:** Review a Q1 sales dataset, generate insights, write a stakeholder-ready summary.
**Code:** `crew_lab.py` (agents, tools, tasks, both crew topologies, cost logging)

---

## Task 1 — Multi-Agent Design

| Agent | Role | Goal | Backstory (persona) |
|---|---|---|---|
| Data Analyst | Numeric profiling | Produce accurate raw stats, no interpretation | Meticulous analytics engineer, correctness over narrative |
| Insight Strategist | Interpretation | Turn stats into 3-5 prioritized business insights/risks | Ex sales-ops lead, knows what actually moves decisions |
| Report Writer | Communication | Write a <300-word executive summary + actions | Comms specialist, translates analyst-speak for a VP |

**Why specialization helps here:** each stage needs a genuinely different skill — precise computation, business judgment, and persuasive concise writing — and a single generalist prompt tends to blend them, producing a summary that either drowns in raw numbers or skips the "why it matters" framing. Splitting them also lets each agent's prompt (and expected_output) stay narrow and easy to verify at each hand-off, which made the "wrong format" bug in Task 3 fast to catch and fix.

**Where it isn't true:** for a task this small (~9 data rows, one report), a single well-prompted agent with a scratchpad/chain-of-thought could plausibly do all three steps in one call, faster and cheaper, with no inter-agent hand-off risk. The multi-agent split earns its keep more clearly at larger scale (bigger datasets, longer reports, need for auditability at each stage) than on this toy example — see Task 5.

---

## Task 2 — Tools

- **Data Analyst:** `FileReadTool` (reads `sales_sample.csv`) + custom `CsvStatsTool` (aggregates revenue/units/churn by region). Needs both because raw file access alone would force it to do arithmetic in-context, which is error-prone.
- **Insight Strategist:** *no tools.* Deliberately restricted to reasoning over the Analyst's JSON output — giving it file access risked it re-deriving (and possibly contradicting) the Analyst's numbers instead of trusting the hand-off.
- **Report Writer:** *no tools.* Pure synthesis/prose task; tool access here would be unused surface area and a place for the agent to wander off-task.

Keeping tool access role-appropriate also made the sequential log easier to audit — any numeric claim in the final report could be traced back to exactly one tool call.

---

## Task 3 — Sequential Process, format-mismatch note

Running `Process.sequential`, the first pass surfaced a real hand-off problem:

- **Problem:** the Data Analyst's first draft returned its JSON wrapped in a sentence ("Here are the Q1 stats: ```json ... ```"), and the Insight Strategist's task, which expected clean JSON in `context`, ended up quoting the markdown fences back verbatim in its bullet list.
- **Fix:** tightened `analyze_task.expected_output` to explicitly say *"Valid JSON only, no prose wrapper"* and added *"do not add commentary or recommendations"* to the description, so the model had an unambiguous stop condition. Also added `csv_stats_tool` (instead of relying on the LLM to eyeball `FileReadTool`'s raw CSV text) so the numbers themselves were tool-computed rather than model-guessed, removing a second source of formatting drift.
- **Result:** on the re-run, the Insight Strategist consumed clean JSON and produced the expected numbered markdown list with no re-prompting.

---

## Task 4 — Sequential vs Hierarchical

| | Sequential | Hierarchical |
|---|---|---|
| **Pros** | Predictable order, cheapest, easiest to debug/log, deterministic context hand-offs | Manager can catch a bad hand-off and re-delegate before it propagates; more resilient to one agent's off-spec output |
| **Cons** | No self-correction — a bad Task 1 output silently poisons Tasks 2 & 3 | Extra manager LLM calls (delegation + review) roughly doubled total tokens; slower; manager's own judgment can itself be wrong |
| **When to use** | Task order is fixed and well-specified, budget/latency matters, outputs are easy to validate downstream | Sub-tasks are more open-ended, quality control matters more than cost, or you expect occasional bad hand-offs |

On this task specifically, hierarchical caught nothing sequential's tightened prompts didn't already catch — the manager mostly re-stated the same delegation sequential effectively did with more overhead.

---

## Task 5 — Evaluation & Cost

**Success criteria (manually scored 1-5 across 3 runs):**

| Criterion | Definition |
|---|---|
| Factual grounding | Every number in the final report traces to the Analyst's tool-computed stats |
| Completeness | Report covers highest-revenue AND highest-churn regions, plus 3 actions |
| Tone | Under 300 words, no raw JSON/jargon, reads like something a VP would actually read |

| Run | Factual grounding | Completeness | Tone | Notes |
|---|---|---|---|---|
| Sequential #1 (pre-fix) | 3/5 | 3/5 | 3/5 | JSON leaked into strategist's bullets |
| Sequential #2 (post-fix) | 5/5 | 5/5 | 4/5 | Clean hand-offs; one bullet slightly verbose |
| Hierarchical #1 | 5/5 | 5/5 | 4/5 | Same quality as fixed sequential, ~2x tokens |

**Was multi-agent worth it here?** For this specific task — one small CSV, one short report — a single well-designed agent with a clear step-by-step prompt (compute → interpret → write) would very likely match the fixed sequential crew's quality at a fraction of the token cost and with no inter-agent formatting risk to debug. The crew's value showed up not in output quality but in *process legibility*: each agent's output was independently checkable, and the format bug in Task 3 was trivial to locate and fix specifically because responsibilities were separated. That legibility, not raw output quality, is the actual case for multi-agent on small tasks — the cost/complexity trade-off tips further toward "worth it" as the dataset, report length, or need for auditable intermediate steps grows.
