# Raw-Python Agent vs. LangChain Agent — Comparison Write-Up

## Concept mapping

| Day 1 (raw Python) | LangChain equivalent |
|---|---|
| Hand-written `call_claude()` wrapper | `ChatAnthropic` |
| `TOOLS = {name: fn}` dict + hand-built JSON schema | `@tool` decorator (schema auto-derived from type hints + docstring) |
| `while True:` loop parsing `tool_use`/`tool_result` blocks | `AgentExecutor` (classic) / `create_agent` (LangGraph-based, current default in `langchain>=1.0`) |
| `history = []` list you appended to manually | `RunnableWithMessageHistory` (deprecated in 1.x) or a LangGraph checkpointer |
| Formatted prompt string | `ChatPromptTemplate` composed via LCEL's `|` operator |

## Annotated reasoning trace (multi-step query)

Query: *"What's the price difference between laptop_x1 and laptop_z9, and what percentage more expensive is the z9?"*

```
[REASON]  Needs laptop_x1's price before comparing anything.
[ACT]     lookup_product_price(product_id='laptop_x1')
[OBSERVE] {"name": "Laptop X1", "price_usd": 899, ...}

[REASON]  Needs laptop_z9's price next.
[ACT]     lookup_product_price(product_id='laptop_z9')
[OBSERVE] {"name": "Laptop Z9 Pro", "price_usd": 1499, ...}

[REASON]  Has both prices; hands the arithmetic to the calculator tool
          rather than computing it itself.
[ACT]     calculator(expression='(1499 - 899) / 899 * 100')
[OBSERVE] 66.74...

[REASON]  Composes final answer from the two observations.
```

**Similar to Day 1:** the loop is identical in shape — reason → act → observe → repeat until the
model stops calling tools. Each `[ACT]/[OBSERVE]` pair is one iteration of Day 1's `while True` loop.

**Hidden now:**
- The raw `tool_use`/`tool_result` message blocks and their IDs — managed internally by `AgentExecutor`.
- The stopping condition — no more manually checking `stop_reason == "end_turn"`.
- Tool-call parsing — LangChain's parser can silently retry a malformed call when
  `handle_parsing_errors=True`, which can mask a genuinely broken call as a model mistake.

## What LangChain made easier vs. what leaked through

LangChain removed real boilerplate: no hand-rolled JSON tool schemas, no manual reason/act/observe
loop, no manual message-history bookkeeping. `@tool`, `AgentExecutor`, and
`RunnableWithMessageHistory` cover all of that in a few lines.

Three things leaked through anyway:

1. **Version churn.** The tutorial-standard `create_tool_calling_agent` + `AgentExecutor` pattern
   is already the "classic" API in `langchain==1.x` — it now lives in a separate
   `langchain-classic` compatibility package. The current default is a LangGraph-based
   `create_agent`, and `RunnableWithMessageHistory` prints a deprecation warning pointing at
   LangGraph's checkpointer instead. A framework that automates so much also means your code can
   go stale fast.
2. **Structured output doesn't compose cleanly with the agent loop.** There's no single flag to
   make `AgentExecutor` return a Pydantic object as its final answer — the practical pattern is
   agent-for-reasoning, then a second small LCEL chain with `.with_structured_output()` to shape
   the free-text result afterward.
3. **Error handling is still your job.** `handle_tool_error=True` only stops an uncaught tool
   exception from crashing the whole run by feeding the error text back to the model as an
   observation — it doesn't add retries, backoff, or fallback logic. The flaky-API resilience code
   (one manual retry inside the tool) is exactly what you'd have written by hand in Day 1 too.

## Bottom line

LangChain is genuinely automating the *mechanical* parts of an agent loop — schema generation,
message bookkeeping, the reason/act/observe cycle — the same way an ORM automates SQL. But the
*judgment* parts (what to do when a tool fails, how to shape structured output, which API surface
is still current) are exactly as much your responsibility as they were in raw Python, and the
framework's own churn (three deprecation paths encountered in one exercise) is a cost of that
convenience worth weighing against writing the loop yourself.
