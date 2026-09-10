"""
Week 2 Day 4 — CrewAI: Multi-Agent Collaboration, Roles & Task Delegation
==========================================================================

Task chosen: "Review a sales dataset, generate insights, and write a
stakeholder-ready summary."

Three specialized agents:
    1. Data Analyst      -> profiles the dataset, surfaces raw stats/trends
    2. Insight Strategist -> turns raw stats into business insights & risks
    3. Report Writer      -> turns insights into a polished stakeholder memo

Two crew topologies are built from the SAME agents/tasks:
    - Process.sequential   (crew_sequential)
    - Process.hierarchical (crew_hierarchical, adds a manager agent)

Run:
    pip install crewai crewai-tools --break-system-packages
    export OPENAI_API_KEY=...        # or configure another LLM below
    python crew_lab.py --mode sequential
    python crew_lab.py --mode hierarchical
    python crew_lab.py --mode both     # runs both, prints cost comparison
"""

import argparse
import json
import os
import time
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import FileReadTool

# ---------------------------------------------------------------------------
# 0. Sample dataset (small, synthetic — swap for a real CSV via FileReadTool)
# ---------------------------------------------------------------------------

DATA_PATH = Path("sales_sample.csv")

def _write_sample_csv():
    DATA_PATH.write_text(
        "month,region,revenue,units_sold,new_customers,churned_customers\n"
        "Jan,North,42000,210,18,6\n"
        "Feb,North,39500,198,14,9\n"
        "Mar,North,51000,240,22,5\n"
        "Jan,South,31000,155,10,3\n"
        "Feb,South,33500,168,12,4\n"
        "Mar,South,29800,149,8,7\n"
        "Jan,West,58000,290,25,4\n"
        "Feb,West,61200,305,28,3\n"
        "Mar,West,47500,238,15,11\n"
    )

if not DATA_PATH.exists():
    _write_sample_csv()


# ---------------------------------------------------------------------------
# 1. Custom tool: a tiny stats calculator (beyond just reading the file)
# ---------------------------------------------------------------------------

class CsvStatsTool(BaseTool):
    """Reused/extended from Day 3's LangGraph tool node — computes quick
    aggregate stats so the Data Analyst doesn't have to eyeball raw rows."""

    name: str = "csv_stats_tool"
    description: str = (
        "Computes summary statistics (totals, averages, month-over-month "
        "deltas) from the sales CSV at sales_sample.csv. Input is ignored; "
        "always reads the same file."
    )

    def _run(self, *args, **kwargs) -> str:
        import csv
        from collections import defaultdict

        rows = list(csv.DictReader(DATA_PATH.open()))
        by_region = defaultdict(lambda: {"revenue": 0, "units": 0, "new": 0, "churn": 0})
        for r in rows:
            reg = r["region"]
            by_region[reg]["revenue"] += int(r["revenue"])
            by_region[reg]["units"] += int(r["units_sold"])
            by_region[reg]["new"] += int(r["new_customers"])
            by_region[reg]["churn"] += int(r["churned_customers"])

        total_revenue = sum(v["revenue"] for v in by_region.values())
        summary = {
            "total_revenue_q1": total_revenue,
            "by_region": dict(by_region),
            "highest_churn_region": max(by_region, key=lambda k: by_region[k]["churn"]),
            "highest_revenue_region": max(by_region, key=lambda k: by_region[k]["revenue"]),
        }
        return json.dumps(summary, indent=2)


file_read_tool = FileReadTool(file_path=str(DATA_PATH))
csv_stats_tool = CsvStatsTool()


# ---------------------------------------------------------------------------
# 2. LLM configs — each agent gets its own config (could be different
#    models/temperatures per role; here we vary temperature by task type)
# ---------------------------------------------------------------------------

analyst_llm = LLM(model="gpt-4o-mini", temperature=0.1)   # deterministic, numeric work
strategist_llm = LLM(model="gpt-4o-mini", temperature=0.4)  # needs some interpretive judgment
writer_llm = LLM(model="gpt-4o-mini", temperature=0.6)      # prose/tone work
manager_llm = LLM(model="gpt-4o", temperature=0.2)           # hierarchical manager only


# ---------------------------------------------------------------------------
# 3. Agents — Task 1 & Task 2
# ---------------------------------------------------------------------------

data_analyst = Agent(
    role="Data Analyst",
    goal=(
        "Profile the Q1 sales dataset and produce accurate, well-organized "
        "raw statistics (totals, regional breakdowns, trends, anomalies) "
        "with no interpretation or business framing."
    ),
    backstory=(
        "A meticulous analytics engineer who has spent years turning messy "
        "CSVs into trustworthy numbers. Cares about correctness over "
        "narrative — leaves interpretation to others."
    ),
    tools=[file_read_tool, csv_stats_tool],  # needs to read + compute; no writing tools
    llm=analyst_llm,
    verbose=True,
    allow_delegation=False,
)

insight_strategist = Agent(
    role="Insight Strategist",
    goal=(
        "Turn raw statistics into 3-5 prioritized business insights and "
        "risks (e.g., churn hotspots, regional under-performance), each "
        "with a one-line 'why it matters'."
    ),
    backstory=(
        "A former sales-ops lead who has sat through a hundred QBRs and "
        "knows which numbers actually change decisions versus which are "
        "just noise. Does not touch raw data directly — works from the "
        "Analyst's output."
    ),
    tools=[],  # deliberately no tools: works purely from Task 1's output
    llm=strategist_llm,
    verbose=True,
    allow_delegation=False,
)

report_writer = Agent(
    role="Stakeholder Report Writer",
    goal=(
        "Write a concise, executive-ready summary (under 300 words) that "
        "communicates the prioritized insights clearly, with a short "
        "recommended-actions section."
    ),
    backstory=(
        "A comms specialist who translates analyst-speak into language a "
        "VP will actually read. Obsessed with clarity, brevity, and "
        "avoiding jargon."
    ),
    tools=[],  # pure writing task, no data/compute tools needed
    llm=writer_llm,
    verbose=True,
    allow_delegation=False,
)

# Manager agent — only used in the hierarchical crew (Task 4)
manager_agent = Agent(
    role="Delegating Manager",
    goal=(
        "Coordinate the Data Analyst, Insight Strategist, and Report Writer "
        "to produce a final stakeholder summary, reviewing each hand-off "
        "for quality and sending work back if it's incomplete or wrong."
    ),
    backstory=(
        "An experienced analytics manager who doesn't do the work herself "
        "but knows enough about each specialty to catch bad output and "
        "ask for revisions before passing work downstream."
    ),
    llm=manager_llm,
    verbose=True,
    allow_delegation=True,
)


# ---------------------------------------------------------------------------
# 4. Tasks — Task 3 (with context dependencies)
# ---------------------------------------------------------------------------

analyze_task = Task(
    description=(
        "Read sales_sample.csv and use csv_stats_tool to compute Q1 totals, "
        "per-region revenue/units/new-customers/churn, and flag the "
        "highest-revenue and highest-churn regions. Output RAW NUMBERS ONLY "
        "as clean, labeled JSON — do not add commentary or recommendations."
    ),
    expected_output=(
        "A JSON object with keys: total_revenue_q1, by_region "
        "(region -> {revenue, units, new, churn}), highest_revenue_region, "
        "highest_churn_region. Valid JSON only, no prose wrapper."
    ),
    agent=data_analyst,
)

strategize_task = Task(
    description=(
        "Using the Data Analyst's JSON output (see context), identify 3-5 "
        "prioritized business insights or risks. For each: a short title, "
        "the supporting number, and one sentence on why it matters to "
        "leadership. Order by importance, most urgent first."
    ),
    expected_output=(
        "A numbered list (markdown) of 3-5 insights, each formatted as: "
        "'**Title** — supporting stat. Why it matters: ...'"
    ),
    agent=insight_strategist,
    context=[analyze_task],
)

write_task = Task(
    description=(
        "Using the Insight Strategist's prioritized list (see context), "
        "write a stakeholder-ready summary under 300 words with two "
        "sections: 'Key Findings' (prose, 2-3 short paragraphs) and "
        "'Recommended Actions' (3 bullet points). No jargon, no raw JSON, "
        "no restating every number — synthesize."
    ),
    expected_output=(
        "A markdown document with headers '## Key Findings' and "
        "'## Recommended Actions', total length under 300 words."
    ),
    agent=report_writer,
    context=[strategize_task],
)


# ---------------------------------------------------------------------------
# 5. Crews — sequential (Task 3) and hierarchical (Task 4)
# ---------------------------------------------------------------------------

crew_sequential = Crew(
    agents=[data_analyst, insight_strategist, report_writer],
    tasks=[analyze_task, strategize_task, write_task],
    process=Process.sequential,
    verbose=True,
)

# Hierarchical: give the manager the SAME tasks; it decides who does what.
# Note: in hierarchical mode CrewAI's manager auto-delegates, so we do not
# assign `agent=` on tasks meant to be delegated, and we don't include the
# manager itself in `agents=`.
analyze_task_h = Task(
    description=analyze_task.description,
    expected_output=analyze_task.expected_output,
)
strategize_task_h = Task(
    description=strategize_task.description,
    expected_output=strategize_task.expected_output,
    context=[analyze_task_h],
)
write_task_h = Task(
    description=write_task.description,
    expected_output=write_task.expected_output,
    context=[strategize_task_h],
)

crew_hierarchical = Crew(
    agents=[data_analyst, insight_strategist, report_writer],
    tasks=[analyze_task_h, strategize_task_h, write_task_h],
    process=Process.hierarchical,
    manager_agent=manager_agent,
    verbose=True,
)


# ---------------------------------------------------------------------------
# 6. Runner with basic cost/token logging — Task 5
# ---------------------------------------------------------------------------

def run_and_log(crew: Crew, label: str) -> dict:
    start = time.time()
    result = crew.kickoff()
    elapsed = time.time() - start

    usage = getattr(result, "token_usage", None)
    log = {
        "label": label,
        "elapsed_sec": round(elapsed, 2),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "output_preview": str(result)[:500],
    }
    print(f"\n=== {label} run complete in {log['elapsed_sec']}s ===")
    print(json.dumps({k: v for k, v in log.items() if k != "output_preview"}, indent=2))
    print("\n--- Output ---\n", result)
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequential", "hierarchical", "both"], default="both")
    args = parser.parse_args()

    logs = []
    if args.mode in ("sequential", "both"):
        logs.append(run_and_log(crew_sequential, "sequential"))
    if args.mode in ("hierarchical", "both"):
        logs.append(run_and_log(crew_hierarchical, "hierarchical"))

    if len(logs) == 2:
        print("\n=== Cost/Latency Comparison ===")
        print(json.dumps(logs, indent=2, default=str))
