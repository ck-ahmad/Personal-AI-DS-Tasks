"""
Tools available to the triage agent:

1. search_knowledge_base -- lightweight local RAG over data/kb/*.md
   (reused pattern from Day 3's retrieval tool). This is the
   "real ... file ... data source" required by the capstone.
2. lookup_client_history -- queries the local SQLite ticket_history table
   (the "small local database" data source).
3. send_client_response -- the *consequential* external action. Gated by
   a human-in-the-loop approval checkpoint in the graph; this function
   itself simulates a flaky downstream mail API so we can exercise the
   tool-timeout/error failure path required by Task 2.

All tools raise a small set of typed exceptions so the graph can catch and
handle failures gracefully instead of crashing.
"""
from __future__ import annotations

import random
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).parent.parent / "data" / "kb"
DB_PATH = Path(__file__).parent.parent / "data" / "tickets.db"


class ToolTimeoutError(Exception):
    """Raised when a simulated external call exceeds its timeout budget."""


class ToolExecutionError(Exception):
    """Raised for any other tool-side failure (bad payload, 5xx, etc.)."""


@dataclass
class KBHit:
    source: str
    score: float
    snippet: str


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and",
    "for", "on", "in", "it", "this", "that", "my", "our", "your", "i",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def search_knowledge_base(query: str, top_k: int = 2) -> list[dict[str, Any]]:
    """Very small local TF-overlap retriever over the markdown KB.

    Kept dependency-free (no embeddings/network) so the tool is fast,
    deterministic, and testable offline -- appropriate for a triage agent
    whose KB is a few dozen short policy docs, not a large corpus.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    hits: list[KBHit] = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_tokens = _tokenize(text)
        overlap = query_tokens & doc_tokens
        if not overlap:
            continue
        score = len(overlap) / len(query_tokens)
        # grab the most relevant bullet line as the snippet
        best_line = ""
        best_line_score = -1
        for line in text.splitlines():
            line_tokens = _tokenize(line)
            line_overlap = len(query_tokens & line_tokens)
            if line_overlap > best_line_score:
                best_line_score = line_overlap
                best_line = line.strip("-* \n")
        hits.append(KBHit(source=path.name, score=round(score, 3), snippet=best_line))

    hits.sort(key=lambda h: h.score, reverse=True)
    return [h.__dict__ for h in hits[:top_k]]


def lookup_client_history(client_name: str) -> list[dict[str, Any]]:
    """Reads prior tickets for a client from the local SQLite DB."""
    if not DB_PATH.exists():
        raise ToolExecutionError(f"Ticket DB not found at {DB_PATH}. Run data/init_db.py first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM ticket_history WHERE client_name = ? ORDER BY ticket_id",
            (client_name,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def send_client_response(ticket_id: str, response_text: str, *, simulate_flakiness: bool = True) -> dict[str, Any]:
    """Simulates the final consequential action: sending the approved reply
    through a downstream email/helpdesk API.

    `simulate_flakiness=True` randomly injects a timeout or 500 error so
    the graph's failure-handling path (Task 2, failure scenario #2) is
    exercised during evaluation runs. In production this would be a real
    HTTP call (e.g., to Zendesk/Intercom) guarded by a real timeout.
    """
    if not response_text or not response_text.strip():
        raise ToolExecutionError("Refusing to send an empty response body.")

    if simulate_flakiness:
        roll = random.random()
        if roll < 0.10:
            time.sleep(0.05)
            raise ToolTimeoutError(f"send_client_response timed out for {ticket_id}")
        if roll < 0.15:
            raise ToolExecutionError(f"Downstream helpdesk API returned 500 for {ticket_id}")

    return {"status": "sent", "ticket_id": ticket_id, "chars_sent": len(response_text)}
