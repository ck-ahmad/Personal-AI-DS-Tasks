"""
Week 2 Day 1 — Agent Foundations
Raw Python Agent built on a FREE, LOCAL, OPEN-SOURCE model via Ollama
(no LangChain / LangGraph, no paid API)

Setup:
    1. Install Ollama:            https://ollama.com/download
    2. Pull a tool-calling model: ollama pull llama3.1
       (alternatives that also support tool calling: qwen2.5, mistral-nemo,
       firefunction-v2 — swap the MODEL constant below)
    3. pip install ollama
    4. python agent_foundations.py

Ollama runs the model entirely on your own machine — no API key, no cost,
fully open-source weights. The tool-calling wire format is OpenAI-style
({"type": "function", "function": {...}}), which is what's used below.

This single file walks through Tasks 1-5 of the assignment. Read it top to
bottom like a lab notebook — each section is runnable on its own via the
`if __name__ == "__main__"` block at the bottom.
"""

import os
import json
import ollama

MODEL = "llama3.1"  # any local Ollama model that supports tool calling

# =============================================================================
# TASK 1 — AGENT CONCEPTS & MENTAL MODEL   (see writeup.md for the prose)
# =============================================================================
#
#   Chatbot  : single turn-by-turn text response. No tools, no state beyond
#              the chat transcript. Cannot act on the world.
#   Workflow : a FIXED sequence of steps (possibly calling an LLM at one or
#              more steps) wired together by a human in advance. The control
#              flow is deterministic — the LLM fills in content, not "what
#              happens next".
#   Agent    : the LLM itself decides, at runtime, which tool to call, in
#              what order, how many times, and when it's done. Control flow
#              lives INSIDE the loop, not outside it.
#
#   "Agentic" = autonomy over next-step decisions + tool use + the ability to
#   observe a tool's result and change plan (self-correction) across
#   multiple steps, without a human re-prompting between each step.
#
#   ReAct pattern:
#       Reason  -> model thinks about what to do next (visible as text)
#       Act     -> model emits a tool call
#       Observe -> we execute the tool and feed the result back in
#       (repeat until the model responds with plain text = done)
#
#   Pseudocode:
#       messages = [user_task]
#       loop:
#           response = llm(messages)
#           if response has tool_call:
#               result = run_tool(response.tool_call)
#               messages.append(response)
#               messages.append(tool_result(result))
#           else:
#               return response.text   # final answer, stop looping
#
#   When an agent is overkill: if the task is a single well-defined
#   transformation (summarize this, translate that, one lookup you could
#   hardcode) a plain prompt or a linear script is faster, cheaper, and far
#   more predictable than looping an LLM that might call tools in a
#   surprising order. Reach for an agent only when the number/order of steps
#   genuinely can't be known ahead of time.


# =============================================================================
# TASK 2 — TOOL CALLING FUNDAMENTALS
# =============================================================================
#
# Ollama (and most open-source tool-calling models) use the OpenAI-style
# wrapper: {"type": "function", "function": {name, description, parameters}}.
# `parameters` is a JSON Schema object — same idea as Anthropic's
# `input_schema`, just nested one level deeper and named differently.
#
# The description is not documentation for humans — it's the ONLY signal the
# model has for (a) whether to call this tool at all and (b) how to fill in
# its arguments. Vague descriptions ("gets weather") lead to the model
# calling the wrong tool, calling it at the wrong time, or guessing at
# argument formats. Specific descriptions (units, valid ranges, what happens
# on invalid input) measurably improve calling reliability — this matters
# even more for smaller open-source models, which are generally less robust
# at tool calling than large hosted models.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a basic arithmetic expression (+, -, *, /, parentheses, "
                "decimals). Use this whenever the user's request requires a "
                "numeric computation rather than an estimate. Input must be a "
                "valid arithmetic expression, e.g. '(3 + 4) * 2'. Does not "
                "support variables, functions, or non-numeric input."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "An arithmetic expression to evaluate, e.g. '12.5 * 3 - 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Look up the current weather for a named city. Only supports a "
                "small fixed set of demo cities: Lahore, Karachi, Faisalabad, "
                "Toronto, London. Returns temperature in Celsius and a short "
                "condition string. If the city is not in the supported set, "
                "returns an error — do not guess a temperature yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Lahore'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text_file",
            "description": (
                "Read the full contents of a small local .txt file given its "
                "path. Use this only when the user refers to a specific local "
                "file. Returns an error if the file does not exist or is not a "
                ".txt file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filesystem path to a .txt file",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        # This tool exists purely to make Task 4 (working memory) concrete:
        # it lets the model explicitly write a fact to a scratchpad that is
        # DIFFERENT from the conversation transcript.
        "type": "function",
        "function": {
            "name": "record_finding",
            "description": (
                "Save a short intermediate finding to the agent's working-memory "
                "scratchpad so it can be referenced later in the task without "
                "re-deriving it. Use this after each tool call that produces a "
                "fact you'll need again (e.g. a looked-up temperature)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "A short factual note, e.g. 'Lahore: 34C, sunny'",
                    }
                },
                "required": ["note"],
            },
        },
    },
]

# --- Fake backend data for the weather stub -------------------------------
_WEATHER_DB = {
    "lahore": {"temp_c": 34, "condition": "sunny"},
    "karachi": {"temp_c": 30, "condition": "humid"},
    "faisalabad": {"temp_c": 33, "condition": "hazy"},
    "toronto": {"temp_c": 18, "condition": "cloudy"},
    "london": {"temp_c": 16, "condition": "rainy"},
}


# =============================================================================
# TOOL EXECUTION — the "Act" -> "Observe" wiring
# =============================================================================
# Working memory scratchpad: state the AGENT explicitly curates mid-task.
# This is intentionally separate from `messages` (conversation memory below).
WORKING_MEMORY = []


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """
    Runs the requested tool and returns a plain dict:
        {"ok": True,  "result": <value>}   or
        {"ok": False, "error": <message>}
    Never raises — tool failures must become a tool-result message, not
    a Python exception, or the agent loop crashes instead of self-correcting.
    """
    try:
        if tool_name == "calculator":
            expr = tool_input["expression"]
            allowed_chars = set("0123456789+-*/(). ")
            if not set(expr) <= allowed_chars:
                return {"ok": False, "error": f"Unsupported characters in expression: {expr!r}"}
            value = eval(expr, {"__builtins__": {}}, {})
            return {"ok": True, "result": value}

        elif tool_name == "get_weather":
            city = tool_input["city"].strip().lower()
            if city not in _WEATHER_DB:
                return {
                    "ok": False,
                    "error": f"No weather data for '{tool_input['city']}'. "
                             f"Supported cities: {list(_WEATHER_DB.keys())}",
                }
            return {"ok": True, "result": _WEATHER_DB[city]}

        elif tool_name == "read_text_file":
            path = tool_input["path"]
            if not path.endswith(".txt"):
                return {"ok": False, "error": "Only .txt files are supported."}
            if not os.path.exists(path):
                return {"ok": False, "error": f"File not found: {path}"}
            with open(path, "r") as f:
                return {"ok": True, "result": f.read()}

        elif tool_name == "record_finding":
            note = tool_input["note"]
            WORKING_MEMORY.append(note)
            return {"ok": True, "result": f"Recorded. Scratchpad now has {len(WORKING_MEMORY)} note(s)."}

        else:
            # The model asked for a tool we never defined. This is one of
            # the failure modes in Task 5 — a "hallucinated tool call".
            return {"ok": False, "error": f"Unknown tool: '{tool_name}'"}

    except Exception as e:
        # Any unexpected crash inside a tool becomes a normal error
        # observation instead of taking down the whole agent loop.
        return {"ok": False, "error": f"Tool '{tool_name}' raised an exception: {e}"}


def _parse_tool_call_args(raw_args) -> dict:
    """
    Ollama's Python client usually hands back tool call arguments already
    as a dict, but some models/versions return a JSON string instead.
    Normalize both cases so execute_tool() always gets a dict.
    """
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args)
    except (TypeError, json.JSONDecodeError):
        return {}


# =============================================================================
# TASK 3 — THE MINIMAL AGENT LOOP
# =============================================================================

def run_agent(user_task: str, max_iterations: int = 6, verbose: bool = True) -> str:
    """
    Conversation memory = `messages` list below: the full transcript the
    model sees on every call, including every tool-call / tool-result pair.
    This is what gives the model continuity between iterations.

    Working memory = WORKING_MEMORY (declared above): a curated scratchpad
    the agent writes to explicitly, only when important. It's not the same
    thing — conversation memory grows automatically and unboundedly with
    every turn; working memory is small, deliberate, and survives even if
    you later truncate/summarize the transcript to save tokens.
    """
    WORKING_MEMORY.clear()
    messages = [{"role": "user", "content": user_task}]

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        message = response["message"]

        # Log the model's reasoning text (if any) for this step
        reasoning_text = (message.get("content") or "").strip()
        if verbose and reasoning_text:
            print(f"[reason] {reasoning_text}")

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            # No tool call -> the model considers the task done.
            if verbose:
                print(f"[final]  {reasoning_text}")
            return reasoning_text

        # Append the assistant's turn (including its tool_calls) to
        # conversation memory before we can reply with tool results.
        messages.append(message)

        for call in tool_calls:
            tool_name = call["function"]["name"]
            tool_input = _parse_tool_call_args(call["function"]["arguments"])

            if verbose:
                print(f"[act]    calling `{tool_name}` with input {tool_input}")

            outcome = execute_tool(tool_name, tool_input)

            if verbose:
                print(f"[observe] {outcome}")

            # Ollama's chat protocol takes tool results back as role="tool"
            # messages (no separate tool_use_id bookkeeping needed for the
            # single-call-at-a-time models used here).
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(outcome),
                }
            )

        # Loop continues -> model sees the tool result(s) on the next call

    # ---- TASK 3 safeguard: max_iterations reached without a final answer ----
    if verbose:
        print(f"\n[guardrail] Hit max_iterations={max_iterations} without a final answer.")
    return (
        f"[Agent stopped after {max_iterations} iterations without a definitive answer. "
        f"Partial working memory: {WORKING_MEMORY}]"
    )


# =============================================================================
# TASK 5 — FAILURE MODES & GUARDRAILS (demo triggers)
# =============================================================================
# These three calls are designed to break the agent on purpose so you can
# read the printed trace and see exactly how it fails. See writeup.md for
# the documented list of failure modes + mitigations. Open-source models
# tend to trigger these MORE readily than large hosted models, especially
# hallucinated tool calls and malformed arguments — that's a useful, honest
# data point for the write-up, not a bug in this script.

def demo_ambiguous_request():
    # Ambiguous: no city named at all. Watch whether the model asks for
    # clarification, guesses a city, or calls the tool with garbage input.
    return run_agent("What's the weather like today?")


def demo_unsupported_tool_input():
    # Valid tool, but an argument the tool can't satisfy -> tests whether
    # the agent reads the error and recovers or just repeats the same call.
    return run_agent("What's the weather in Islamabad?")


def demo_task_needing_undefined_tool():
    # No email-sending tool exists. Watch for a hallucinated tool call to a
    # tool name we never registered, or a decline in plain text.
    return run_agent("Email the Faisalabad weather to my manager.")


# =============================================================================
# MAIN — exercises Task 2 (single tool call) and Task 3 (multi-step task)
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TASK 2 DEMO — single tool call, manual execution")
    print("=" * 70)
    single_msgs = [{"role": "user", "content": "What's 12.5% of 480?"}]
    resp = ollama.chat(model=MODEL, messages=single_msgs, tools=TOOLS)
    msg = resp["message"]
    call = msg["tool_calls"][0]
    tool_name = call["function"]["name"]
    tool_args = _parse_tool_call_args(call["function"]["arguments"])
    print("Model chose tool:", tool_name, tool_args)
    result = execute_tool(tool_name, tool_args)
    print("Manually executed result:", result)
    # Manually building + returning the tool-result message, as required:
    single_msgs.append(msg)
    single_msgs.append({"role": "tool", "name": tool_name, "content": json.dumps(result)})
    followup = ollama.chat(model=MODEL, messages=single_msgs, tools=TOOLS)
    print("Model's final answer:", followup["message"].get("content", "").strip())

    print("\n" + "=" * 70)
    print("TASK 3 DEMO — multi-step agent loop (2+ tool calls required)")
    print("=" * 70)
    answer = run_agent(
        "Look up the weather in Faisalabad and Toronto, record each finding, "
        "then tell me which city is warmer right now and by how much."
    )
    print("\nFINAL ANSWER:", answer)
    print("WORKING MEMORY AT END:", WORKING_MEMORY)

    print("\n" + "=" * 70)
    print("TASK 5 DEMOS — deliberately breaking the agent")
    print("=" * 70)
    print("\n>>> Ambiguous request:")
    print(demo_ambiguous_request())
    print("\n>>> Unsupported city:")
    print(demo_unsupported_tool_input())
    print("\n>>> Task needing an undefined tool:")
    print(demo_task_needing_undefined_tool())
