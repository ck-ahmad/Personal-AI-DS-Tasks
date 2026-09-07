# Week 2 Day 1: Agent Foundations — Reasoning Loops, Tool Calling & Raw Python Agents

## Write-Up: Agent Concepts, Architecture, and Failure Modes

### 1. Agent vs. Chatbot vs. Workflow

* **Chatbot:** Operates on a single-turn conversational model. It takes the message transcript, predicts the most plausible next text completion, and returns it. It has no access to external tools, cannot perform actions on external systems, and holds no persistent state beyond the conversation context.
* **Workflow:** A hardcoded, deterministic pipeline designed by a engineer. While individual nodes in the pipeline may leverage an LLM to transform, reformat, or generate text, the execution path, sequence of steps, and branching logic are entirely fixed and controlled outside the LLM.
* **Agent:** An LLM loop where control flow lives inside the model's reasoning process. Given a high-level user goal, the agent dynamically decides at runtime which tools to invoke, what parameters to pass, how to interpret tool results, and when the task is complete.

**What makes a system "agentic":**

* **Autonomy:** Makes real-time decisions on next steps without explicit step-by-step human prompts.
* **Tool Use:** Queries or modifies external environments (databases, APIs, web scrapers).
* **Multi-Step Planning:** Deconstructs complex goals into smaller sub-tasks and executes them sequentially.
* **Self-Correction:** Reads error messages or unexpected outputs from tools and adjusts its strategy dynamically.

---

### 2. The ReAct Pattern

The **ReAct** (Reasoning + Acting) pattern decouples cognitive planning from external execution:

```
 ┌──────────────┐      ┌─────────────┐      ┌────────────────┐
 │    Reason    │ ───> │     Act     │ ───> │    Observe     │ ──┐
 │ (Think/Plan) │      │ (Call Tool) │      │ (Tool Result)  │   │
 └──────────────┘      └─────────────┘      └────────────────┘   │
        ▲                                                        │
        └────────────────────────────────────────────────────────┘
                       Repeat until task completion

```

**Pseudocode Core Loop:**

```python
messages = [{"role": "user", "content": user_task}]

while iteration < max_iterations:
    response = llm(messages=messages, tools=tools)

    if response.stop_reason == "tool_use":
        for tool_call in response.tool_calls:
            # 1. ACT
            result = execute_tool(tool_call.name, tool_call.input)
            # 2. OBSERVE
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": str(result),
                        }
                    ],
                }
            )
    else:
        # Final Answer Reached
        return response.text

```

---

### 3. When an Agent Is Overkill

An agent is overkill when a task follows a static, predictable sequence or a single transformation—such as parsing a fixed JSON schema, summarizing a text document, or executing a known SQL query. In these scenarios, a deterministic script or a single LLM prompt is significantly faster, cheaper, and more reliable. Reserve agents exclusively for ambiguous tasks where the required sequence of steps cannot be known in advance and depends on intermediate tool returns.

---

### 4. Tool Schemas and API Design

Tools are exposed to the LLM via JSON Schemas that specify the tool's name, purpose, and argument structure:

```json
{
  "name": "calculator",
  "description": "Evaluates basic mathematical expressions (+, -, *, /, %, **). Use this strictly for exact numerical computations rather than performing mental math.",
  "input_schema": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "The math expression to evaluate, e.g. '(12.5 / 100) * 480'"
      }
    },
    "required": ["expression"]
  }
}

```

**Why Tool Descriptions Are Critical:**
The tool description serves as the model's primary instruction set for tool selection. Unlike code documentation intended for humans, the model uses these descriptions to evaluate tool applicability and parameter formats. A vague description (e.g., *"Does calculations"*) leads to hallucinated parameter names, inappropriate tool calls, or failure to trigger the tool when needed.

---

### 5. Conversation Memory vs. Working Memory

* **Conversation Memory:** The complete context history (system prompt, user messages, assistant responses, and tool_use / tool_result pairs) passed to the API on every turn. It grows monotonically and provides overall task continuity.
* **Working Memory:** An isolated, structured scratchpad state (e.g., key-value store, temporary table) maintained by the application. The agent explicitly reads from or writes to working memory via specialized tools. This state survives transcript truncation, context window limits, and message summarization.

---

### 6. Failure Modes & Mitigations

| Failure Mode | Description | Mitigation Strategy |
| --- | --- | --- |
| **Infinite Looping** | Agent continuously calls tools without reaching a terminal response state. | Enforce a strict `max_iterations` counter cap and terminate with a partial result. |
| **Hallucinated Tool Calls** | Model invokes non-existent tools or invents unauthorized schema fields. | Validate tool names against a registered tool dispatch dict; return an explicit error string in `tool_result` to prompt self-correction. |
| **Malformed Arguments** | Model passes invalid types (e.g., passing a string where a float is expected). | Enforce strict pydantic/JSON-schema validation on input dicts before execution; capture validation errors and feed them back to the model. |
| **Silent Tool Execution Crashes** | Unhandled runtime exceptions in tool functions crash the host process. | Wrap tool execution logic in `try-except` blocks; package runtime errors as clean `tool_result` output blocks (`is_error: true`). |
| **Premature Assumptions on Ambiguous Inputs** | Model guesses missing parameters instead of requesting user clarification. | Include explicit instructions in system prompts and tool descriptions to flag missing parameter dependencies and prompt for user input. |

---

### 7. Why Agent Frameworks Exist

While building a raw Python agent loop clarifies core mechanics (message management, dispatching, exception catching, and iteration tracking), managing these primitives by hand rapidly scales in complexity. Production agents require state persistence, streaming responses, parallel tool execution, human-in-the-loop approvals, graph-based routing, and observability traces. Frameworks like LangChain, LangGraph, and CrewAI eliminate boilerplate by providing standard abstractions for these complex requirements.

---

## Python Implementation Script

Here is the Python implementation using the Anthropic API (via `anthropic`).

```python
"""
Week 2 Day 1 — Agent Foundations using Anthropic Claude API

Install:
    pip install anthropic pydantic

Set API key:
    export ANTHROPIC_API_KEY="your-api-key"

Run:
    python agent_foundations_anthropic.py
"""

import ast
import json
import operator
import os
import sys
from typing import Any, Dict, List
import anthropic

# Initialize Client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-3-5-sonnet-20241022"

# Global Working Memory (Scratchpad State)
WORKING_MEMORY: List[str] = []

# =====================================================================
# Task 2: Tool Schemas & Definitions
# =====================================================================

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate basic arithmetic expressions using numbers, +, -, *, /, %, ** and parentheses. Use this for numeric calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '(3 + 4) * 2'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get demo weather for Lahore, Karachi, Faisalabad, Toronto, or London. Do not guess unsupported cities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The target city name.",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "read_text_file",
        "description": "Read a small local .txt file when explicitly requested.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the text file.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "record_finding",
        "description": "Save a short factual finding in working memory scratchpad for later use across turns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The fact or result to store.",
                }
            },
            "required": ["note"],
        },
    },
]

_WEATHER_DB = {
    "lahore": {"temp_c": 34, "condition": "sunny"},
    "karachi": {"temp_c": 30, "condition": "humid"},
    "faisalabad": {"temp_c": 33, "condition": "hazy"},
    "toronto": {"temp_c": 18, "condition": "cloudy"},
    "london": {"temp_c": 16, "condition": "rainy"},
}

# =====================================================================
# Execution & Guardrail Dispatcher
# =====================================================================


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Executes registered tools with defensive parsing and error catching."""
    try:
        if name == "calculator":
            expr = args.get("expression", "")
            allowed = set("0123456789+-*/%(). ")
            if not set(expr) <= allowed:
                return {
                    "ok": False,
                    "error": "Unsupported characters in expression.",
                }

            tree = ast.parse(expr, mode="eval")
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            def eval_node(node):
                if isinstance(node, ast.Constant) and isinstance(
                    node.value, (int, float)
                ):
                    return node.value
                if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                    return ops[type(node.op)](eval_node(node.operand))
                if isinstance(node, ast.BinOp) and type(node.op) in ops:
                    return ops[type(node.op)](
                        eval_node(node.left), eval_node(node.right)
                    )
                raise ValueError("Unsupported AST node.")

            return {"ok": True, "result": eval_node(tree.body)}

        elif name == "get_weather":
            city = str(args.get("city", "")).strip().lower()
            if city not in _WEATHER_DB:
                return {
                    "ok": False,
                    "error": f"No weather data for '{args.get('city')}'. Supported cities: {list(_WEATHER_DB.keys())}",
                }
            return {"ok": True, "result": _WEATHER_DB[city]}

        elif name == "read_text_file":
            path = str(args.get("path", ""))
            if not path.endswith(".txt"):
                return {
                    "ok": False,
                    "error": "Security restriction: Only .txt files are accessible.",
                }
            if not os.path.exists(path):
                return {"ok": False, "error": f"File not found: {path}"}
            with open(path, encoding="utf-8") as f:
                return {"ok": True, "result": f.read()}

        elif name == "record_finding":
            note = str(args.get("note", ""))
            WORKING_MEMORY.append(note)
            return {
                "ok": True,
                "result": f"Recorded. Scratchpad size: {len(WORKING_MEMORY)} entry/entries.",
            }

        else:
            return {"ok": False, "error": f"Tool '{name}' is not registered."}

    except Exception as e:
        return {"ok": False, "error": f"Execution error inside {name}: {str(e)}"}


# =====================================================================
# Task 3 & 4: Minimal Agent Loop & Logging
# =====================================================================


def run_agent(
    user_task: str, max_iterations: int = 6, verbose: bool = True
) -> str:
    """Core ReAct while-loop executing Anthropic tool calls."""
    WORKING_MEMORY.clear()
    messages = [{"role": "user", "content": user_task}]

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # API Request
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Log assistant text blocks (Reasoning)
        assistant_content = response.content
        for block in assistant_content:
            if block.type == "text" and verbose:
                print(f"[Reason] {block.text}")

        # If no tool calls requested, model reached final answer
        if response.stop_reason != "tool_use":
            final_text = "".join(
                [b.text for b in assistant_content if b.type == "text"]
            )
            if verbose:
                print(f"[Final Answer] {final_text}")
            return final_text

        # Append assistant response to transcript
        messages.append({"role": "assistant", "content": assistant_content})

        # Process Tool Calls (Acts & Observations)
        tool_results_content = []
        for block in assistant_content:
            if block.type == "tool_use":
                tool_id = block.id
                tool_name = block.name
                tool_input = block.input

                if verbose:
                    print(
                        f"[Act] Calling `{tool_name}` with input: {json.dumps(tool_input)}"
                    )

                execution_response = execute_tool(tool_name, tool_input)

                if verbose:
                    print(f"[Observe] {json.dumps(execution_response)}")

                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(execution_response),
                    }
                )

        # Append execution results back to context
        messages.append({"role": "user", "content": tool_results_content})

    return f"[Agent halted: Maximum iteration threshold ({max_iterations}) reached without convergence.]"


# =====================================================================
# Task 5: Demonstrations & Failure Scenarios
# =====================================================================

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    print("=" * 70)
    print("TASK 2 DEMO — Single Calculation Tool Call")
    print("=" * 70)
    run_agent("What is 12.5% of 480?")

    print("\n" + "=" * 70)
    print("TASK 3 & 4 DEMO — Multi-Step ReAct Loop & Memory Tracing")
    print("=" * 70)
    run_agent(
        "Look up the weather in Faisalabad and Toronto, record each finding, "
        "then calculate which city is warmer right now and by how much."
    )
    print(f"\nFinal Working Memory Scratchpad: {WORKING_MEMORY}")

    print("\n" + "=" * 70)
    print("TASK 5 DEMOS — Failure Modes & Edge Case Recovery")
    print("=" * 70)

    print("\n[Scenario A: Ambiguous Input]")
    run_agent("What's the weather like today?")

    print("\n[Scenario B: Unsupported Input]")
    run_agent("What's the weather in Islamabad?")

    print("\n[Scenario C: Undefined Tool Request]")
    run_agent("Email the Faisalabad weather report to my manager.")

```
