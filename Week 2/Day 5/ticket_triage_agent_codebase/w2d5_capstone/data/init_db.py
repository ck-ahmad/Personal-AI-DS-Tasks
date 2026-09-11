"""
Bootstraps a small local SQLite database of ticket history.
This is the "small local database" external data source used by the
`lookup_client_history` tool -- lets the agent check whether a client has
prior escalations before deciding how to route a new ticket.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tickets.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ticket_history")
    cur.execute(
        """
        CREATE TABLE ticket_history (
            ticket_id TEXT PRIMARY KEY,
            client_name TEXT NOT NULL,
            category TEXT,
            severity TEXT,
            was_escalated INTEGER,
            resolved INTEGER
        )
        """
    )
    seed = [
        ("T-1001", "Acme Corp", "bug", "high", 1, 1),
        ("T-1002", "Acme Corp", "billing", "medium", 0, 1),
        ("T-1003", "Nimbus Labs", "onboarding", "low", 0, 1),
        ("T-1004", "Nimbus Labs", "bug", "critical", 1, 1),
        ("T-1005", "Vantage Retail", "billing", "high", 1, 0),  # unresolved refund dispute
        ("T-1006", "Vantage Retail", "feature_request", "low", 0, 1),
    ]
    cur.executemany(
        "INSERT INTO ticket_history VALUES (?, ?, ?, ?, ?, ?)", seed
    )
    conn.commit()
    conn.close()
    print(f"Initialized {DB_PATH} with {len(seed)} historical tickets.")


if __name__ == "__main__":
    init_db()
