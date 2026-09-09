from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ============================================================
# 1. STATE DESIGN
# ============================================================

class ResearchState(TypedDict, total=False):
    question: str
    plan: List[str]
    evidence: List[str]
    draft: str
    critique: str
    quality_score: float
    retries: int
    max_retries: int
    approval: str
    final_answer: str
    history: List[str]


# ============================================================
# 2. LOCAL KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = {
    "langgraph": (
        "LangGraph models workflows as graphs of stateful "
        "nodes and transitions."
    ),
    "state": (
        "Shared state carries information between nodes "
        "and can be checkpointed."
    ),
    "human": (
        "Human approval is useful before irreversible, "
        "expensive, or externally visible actions."
    ),
}


def local_search(query: str) -> list[str]:
    """Simple retrieval tool."""
    query = query.lower()

    matches = [
        text
        for key, text in KNOWLEDGE_BASE.items()
        if key in query
    ]

    return matches or list(KNOWLEDGE_BASE.values())[:2]


# ============================================================
# 3. LINEAR GRAPH NODES
# ============================================================

def plan_node(state: ResearchState):
    """Create a plan."""
    question = state["question"]

    return {
        "plan": [
            f"Understand the question: {question}",
            "Retrieve supporting information",
            "Write a concise answer",
        ],
        "history": state.get("history", []) + ["plan"],
    }


def retrieve_node(state: ResearchState):
    """Retrieve evidence."""
    evidence = local_search(state["question"])

    return {
        "evidence": evidence,
        "history": state.get("history", []) + ["retrieve"],
    }


def generate_node(state: ResearchState):
    """Generate an answer from the evidence."""
    evidence = " ".join(state.get("evidence", []))

    draft = (
        f"Question: {state['question']}\n\n"
        f"Answer: {evidence}\n\n"
        f"Revision pass: {state.get('retries', 0)}"
    )

    return {
        "draft": draft,
        "history": state.get("history", []) + ["generate"],
    }


def format_node(state: ResearchState):
    """Prepare the final answer."""
    return {
        "final_answer": state["draft"].strip(),
        "history": state.get("history", []) + ["format"],
    }


# ============================================================
# 4. CONDITIONAL EDGE + SELF-CORRECTION
# ============================================================

def critique_node(state: ResearchState):
    """Evaluate the draft."""
    retries = state.get("retries", 0)

    # Demonstration rule:
    # First pass fails, second pass succeeds.
    score = 0.65 if retries == 0 else 0.92

    if score < 0.8:
        critique = "Add more detail and improve completeness."
    else:
        critique = "Draft is acceptable."

    print(
        f"Critique pass {retries + 1}: "
        f"score={score} | {critique}"
    )

    return {
        "quality_score": score,
        "critique": critique,
        "retries": retries + 1,
        "history": state.get("history", []) + ["critique"],
    }


def route_after_critique(state: ResearchState):
    """Decide whether to revise or continue."""
    if (
        state["quality_score"] < 0.8
        and state["retries"] <= state.get("max_retries", 2)
    ):
        return "generate"

    return "approval"


# ============================================================
# 5. HUMAN-IN-THE-LOOP
# ============================================================

def approval_node(state: ResearchState):
    """Pause before releasing the answer."""
    decision = interrupt({
        "message": "Approve releasing this answer?",
        "draft": state["draft"],
        "quality_score": state["quality_score"],
    })

    return {
        "approval": decision,
        "history": state.get("history", []) + ["approval"],
    }


def route_after_approval(state: ResearchState):
    """Approved -> format, rejected -> revise."""
    if state.get("approval") == "approved":
        return "format"

    return "revise"


def revise_node(state: ResearchState):
    """Handle human rejection."""
    return {
        "draft": (
            state["draft"]
            + "\n\nRevision: The answer was reviewed "
              "and clarified."
        ),
        "history": state.get("history", []) + ["revise"],
    }


# ============================================================
# 6. BUILD THE GRAPH
# ============================================================

builder = StateGraph(ResearchState)

builder.add_node("plan", plan_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("critique", critique_node)
builder.add_node("approval", approval_node)
builder.add_node("revise", revise_node)
builder.add_node("format", format_node)

# Linear edges
builder.add_edge(START, "plan")
builder.add_edge("plan", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "critique")

# Conditional edges
builder.add_conditional_edges(
    "critique",
    route_after_critique
)

builder.add_conditional_edges(
    "approval",
    route_after_approval
)

# Rejection creates another cycle
builder.add_edge("revise", "generate")

# End
builder.add_edge("format", END)


# ============================================================
# 7. COMPILE WITH PERSISTENCE
# ============================================================

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# ============================================================
# 8. RUN THE GRAPH
# ============================================================

config = {
    "configurable": {
        "thread_id": "research-demo-1"
    }
}

initial_state = {
    "question": "Explain why LangGraph uses shared state.",
    "retries": 0,
    "max_retries": 2,
    "history": [],
}

print("\n=== RUNNING GRAPH ===\n")

# First invocation pauses at approval
paused = graph.invoke(
    initial_state,
    config
)

print("\n=== GRAPH PAUSED ===")
print("State before approval:")
print(paused)

# Simulated human approval
print("\n=== HUMAN APPROVAL ===")

resumed = graph.invoke(
    Command(resume="approved"),
    config
)

print("\n=== FINAL ANSWER ===")
print(resumed["final_answer"])


# ============================================================
# 9. INSPECT STATE HISTORY
# ============================================================

print("\n=== STATE HISTORY ===")

for i, snapshot in enumerate(
    graph.get_state_history(config)
):
    print(f"\n--- Snapshot {i} ---")
    print("Next:", snapshot.next)
    print("State:", snapshot.values)
