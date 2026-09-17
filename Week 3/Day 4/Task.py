"""
AFL Week 3 Day 4
LangGraph Integration — Chat + Retrieval + Prediction

Requirements:
    pip install -U langgraph langchain-core langchain-ollama pandas numpy joblib

Project structure:
    afl_datasets/
        afl_players_info_raw.csv
        afl_players_round_by_round_stats_raw.csv
        afl_players_seasonal_stats_raw.csv
        team_matches_home_away_raw.csv

    afl_outputs/
        player_feature_table.csv
        day2_feature_importance.csv       # optional, for match explanations

    afl_models/
        match_winner_pipeline.joblib
        top_player_disposals_pipeline.joblib
        top_player_goals_pipeline.joblib # optional
        model_metadata.joblib

Run:
    python afl_week3_day4_langgraph.py

The app intentionally routes prediction requests through a dedicated
prediction node instead of letting a generic agent decide whether to
predict. This keeps probability framing and validation deterministic.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


# ============================================================
# 1. CONFIG
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "afl_datasets"
OUTPUT_DIR = BASE / "afl_outputs"
MODEL_DIR = BASE / "afl_models"

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

ROUND_FILE = DATA_DIR / "afl_players_round_by_round_stats_raw.csv"
SEASON_FILE = DATA_DIR / "afl_players_seasonal_stats_raw.csv"
TEAM_FILE = DATA_DIR / "team_matches_home_away_raw.csv"
PLAYER_FILE = DATA_DIR / "afl_players_info_raw.csv"

ROUTING_REPORT = OUTPUT_DIR / "day4_routing_accuracy.csv"
TRACE_FILE = OUTPUT_DIR / "day4_state_traces.json"
CONVERSATION_LOG = OUTPUT_DIR / "day4_conversation_log.json"


# ============================================================
# 2. STATE SCHEMA
# ============================================================

Intent = Literal["factual", "retrieval", "prediction", "off-topic"]


class AFLState(TypedDict, total=False):
    user_query: str
    conversation_history: list[dict[str, str]]
    intent: Intent
    route_reason: str

    tool_name: str | None
    tool_input: dict[str, Any] | None
    tool_result: Any
    validation_ok: bool
    validation_message: str

    resolved_entities: dict[str, Any]
    final_response: str

    needs_clarification: bool
    out_of_scope: bool

    trace: list[dict[str, Any]]


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


round_df = load_csv(ROUND_FILE, dtype={"player_id": str})
season_df = load_csv(SEASON_FILE, dtype={"player_id": str})
team_df = load_csv(TEAM_FILE)
player_df = load_csv(PLAYER_FILE, dtype=str)

for _df in (round_df, season_df, team_df, player_df):
    _df.columns = [
        str(c).strip()
        for c in _df.columns
    ]


# ============================================================
# 4. DATA HELPERS
# ============================================================

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def clean(value: Any) -> str:
    return str(value).strip().lower()


def first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return find_column(df, candidates)


def team_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "home": first_column(
            df, ["home_team", "home_team_name", "home", "home_club"]
        ),
        "away": first_column(
            df, ["away_team", "away_team_name", "away", "away_club"]
        ),
        "date": first_column(
            df, ["match_date", "date", "game_date", "round_date"]
        ),
        "season": first_column(
            df, ["season", "year", "season_year"]
        ),
        "id": first_column(
            df, ["match_id", "game_id", "matchid", "gameid", "id"]
        ),
        "home_score": first_column(
            df, ["home_score", "home_points", "home_total_score", "home_total"]
        ),
        "away_score": first_column(
            df, ["away_score", "away_points", "away_total_score", "away_total"]
        ),
    }


TC = team_columns(team_df)


def all_dataset_teams() -> list[str]:
    teams = set()
    for key in ("home", "away"):
        col = TC[key]
        if col:
            teams.update(
                team_df[col].dropna().astype(str).str.strip().tolist()
            )
    return sorted(teams)


# Common AFL nicknames/aliases. Resolution is always checked against
# the exact team names present in the user's dataset.
TEAM_ALIASES = {
    "pies": "Collingwood",
    "magpies": "Collingwood",
    "collingwood": "Collingwood",
    "cats": "Geelong",
    "geelong": "Geelong",
    "blues": "Carlton",
    "carlton": "Carlton",
    "bombers": "Essendon",
    "essendon": "Essendon",
    "tigers": "Richmond",
    "richmond": "Richmond",
    "hawks": "Hawthorn",
    "hawthorn": "Hawthorn",
    "dogs": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
    "western bulldogs": "Western Bulldogs",
    "swans": "Sydney",
    "sydney": "Sydney",
    "giants": "Greater Western Sydney",
    "gws": "Greater Western Sydney",
    "greater western sydney": "Greater Western Sydney",
    "lions": "Brisbane Lions",
    "brisbane": "Brisbane Lions",
    "brisbane lions": "Brisbane Lions",
    "dockers": "Fremantle",
    "fremantle": "Fremantle",
    "eagles": "West Coast",
    "west coast": "West Coast",
    "saints": "St Kilda",
    "st kilda": "St Kilda",
    "crows": "Adelaide",
    "adelaide": "Adelaide",
    "power": "Port Adelaide",
    "port adelaide": "Port Adelaide",
    "suns": "Gold Coast",
    "gold coast": "Gold Coast",
    "demons": "Melbourne",
    "melbourne": "Melbourne",
    "kangaroos": "North Melbourne",
    "north melbourne": "North Melbourne",
    "roos": "North Melbourne",
    "swans": "Sydney",
}


def resolve_team(value: str) -> tuple[str | None, str | None]:
    """Resolve an alias only when it maps to exactly one dataset team."""
    raw = clean(value)
    teams = all_dataset_teams()
    exact = {clean(t): t for t in teams}

    if raw in exact:
        return exact[raw], None

    if raw in TEAM_ALIASES:
        candidate = TEAM_ALIASES[raw]
        candidate_clean = clean(candidate)
        matches = [t for t in teams if clean(t) == candidate_clean]
        if len(matches) == 1:
            return matches[0], None

        # Dataset may use a slightly different spelling. Only accept
        # a unique substring match; never guess between multiple teams.
        fuzzy = [t for t in teams if candidate_clean in clean(t)]
        if len(fuzzy) == 1:
            return fuzzy[0], None

    # Unique partial match from the dataset.
    partial = [t for t in teams if raw in clean(t)]
    if len(partial) == 1:
        return partial[0], None
    if len(partial) > 1:
        return None, f"Ambiguous team '{value}'. Candidates: {partial}"

    return None, f"Could not resolve AFL team '{value}' from the dataset."


def resolve_player_name(value: str) -> tuple[str | None, str | None]:
    name_col = find_column(
        player_df, ["player_name", "player_full_name", "name", "playerName"]
    )
    if name_col is None:
        return None, "Player-name column is missing from player info data."

    names = player_df[name_col].fillna("").astype(str).str.strip()
    raw = clean(value)

    exact = player_df[names.str.lower() == raw]
    if len(exact) == 1:
        return str(exact.iloc[0][name_col]), None

    partial = player_df[names.str.lower().str.contains(raw, regex=False)]
    if len(partial) == 1:
        return str(partial.iloc[0][name_col]), None
    if len(partial) > 1:
        candidates = partial[name_col].astype(str).tolist()[:10]
        return None, f"Ambiguous player '{value}'. Candidates: {candidates}"

    return None, f"Could not resolve AFL player '{value}' from the dataset."


def parse_round(text: str) -> int | None:
    match = re.search(r"\bround\s*(\d+)\b", text, re.I)
    return int(match.group(1)) if match else None


def parse_year(text: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def parse_stat_type(text: str) -> str | None:
    lowered = text.lower()
    if any(x in lowered for x in ["disposals", "disposal", "possessions"]):
        return "disposals"
    if any(x in lowered for x in ["goals", "goal", "top-score", "top score", "topscorer"]):
        return "goals"
    return None


def parse_two_team_mentions(text: str) -> tuple[str, str] | None:
    """Try to identify two teams around common matchup phrases."""
    teams = all_dataset_teams()
    aliases = list(TEAM_ALIASES.keys())

    # Exact dataset names first, longest first.
    candidates = sorted(
        [(t, clean(t)) for t in teams],
        key=lambda x: len(x[1]),
        reverse=True,
    )

    found: list[tuple[int, int, str]] = []
    lowered = text.lower()
    for display, token in candidates:
        pos = lowered.find(token)
        if pos >= 0:
            found.append((pos, pos + len(token), display))

    # Alias scan.
    for alias in sorted(aliases, key=len, reverse=True):
        pos = lowered.find(alias)
        if pos >= 0:
            resolved, _ = resolve_team(alias)
            if resolved:
                found.append((pos, pos + len(alias), resolved))

    unique = {}
    for start, end, team in found:
        unique[clean(team)] = (start, end, team)

    ordered = sorted(unique.values(), key=lambda x: x[0])
    if len(ordered) >= 2:
        return ordered[0][2], ordered[1][2]
    return None


# ============================================================
# 5. DAY-3 RETRIEVAL TOOLS
# ============================================================

@tool
def get_player_round_stats(
    player_name: str,
    round_number: int,
    year: int | None = None,
) -> str:
    """Retrieve exact AFL player statistics for one round."""
    resolved, error = resolve_player_name(player_name)
    if error:
        return json.dumps({"ok": False, "error": error})

    player_col = find_column(
        round_df, ["player_name", "player", "name", "playerName"]
    )
    round_col = find_column(round_df, ["round", "round_number", "round_no"])

    if not player_col or not round_col:
        return json.dumps({
            "ok": False,
            "error": "Required player/round columns are missing.",
        })

    mask = (
        round_df[player_col].astype(str).str.lower().eq(clean(resolved))
        & round_df[round_col].astype(str).str.strip().eq(str(round_number))
    )

    year_col = find_column(round_df, ["year", "season"])
    if year is not None and year_col:
        mask &= pd.to_numeric(
            round_df[year_col], errors="coerce"
        ).eq(year)

    result = round_df.loc[mask]
    if result.empty:
        return json.dumps({
            "ok": False,
            "error": f"No AFL dataset record found for {resolved}, Round {round_number}"
                    + (f", {year}." if year else "."),
        })

    return json.dumps({
        "ok": True,
        "type": "player_round",
        "player": resolved,
        "round": round_number,
        "year": year,
        "records": result.to_dict(orient="records"),
    }, default=str)


@tool
def get_player_season_stats(
    player_name: str,
    year: int | None = None,
) -> str:
    """Retrieve exact AFL player season statistics."""
    resolved, error = resolve_player_name(player_name)
    if error:
        return json.dumps({"ok": False, "error": error})

    player_col = find_column(
        season_df, ["player_name", "player", "name", "playerName"]
    )
    if not player_col:
        return json.dumps({
            "ok": False,
            "error": "Player-name column is missing from season data.",
        })

    mask = season_df[player_col].astype(str).str.lower().eq(clean(resolved))

    year_col = find_column(season_df, ["year", "season"])
    if year is not None and year_col:
        mask &= pd.to_numeric(
            season_df[year_col], errors="coerce"
        ).eq(year)

    result = season_df.loc[mask]
    if result.empty:
        return json.dumps({
            "ok": False,
            "error": f"No AFL season record found for {resolved}"
                    + (f" in {year}." if year else "."),
        })

    return json.dumps({
        "ok": True,
        "type": "player_season",
        "player": resolved,
        "year": year,
        "records": result.to_dict(orient="records"),
    }, default=str)


@tool
def get_team_head_to_head(team_a: str, team_b: str) -> str:
    """Retrieve historical AFL head-to-head records."""
    a, err_a = resolve_team(team_a)
    b, err_b = resolve_team(team_b)

    if err_a or err_b:
        return json.dumps({
            "ok": False,
            "error": err_a or err_b,
        })

    home_col = TC["home"]
    away_col = TC["away"]
    if not home_col or not away_col:
        return json.dumps({
            "ok": False,
            "error": "Home/away team columns are missing.",
        })

    mask = (
        (
            team_df[home_col].astype(str).str.lower().eq(clean(a))
            & team_df[away_col].astype(str).str.lower().eq(clean(b))
        )
        |
        (
            team_df[home_col].astype(str).str.lower().eq(clean(b))
            & team_df[away_col].astype(str).str.lower().eq(clean(a))
        )
    )

    result = team_df.loc[mask]
    if result.empty:
        return json.dumps({
            "ok": False,
            "error": f"No AFL head-to-head records found between {a} and {b}.",
        })

    return json.dumps({
        "ok": True,
        "type": "head_to_head",
        "team_a": a,
        "team_b": b,
        "matches_found": int(len(result)),
        "records": result.to_dict(orient="records"),
    }, default=str)


RETRIEVAL_TOOLS = {
    get_player_round_stats.name: get_player_round_stats,
    get_player_season_stats.name: get_player_season_stats,
    get_team_head_to_head.name: get_team_head_to_head,
}


# ============================================================
# 6. PREDICTION FUNCTIONS — DAY 2
# ============================================================

try:
    from predict import predict_match_winner, predict_top_player
except Exception as exc:
    predict_match_winner = None
    predict_top_player = None
    PREDICT_IMPORT_ERROR = str(exc)
else:
    PREDICT_IMPORT_ERROR = None


def resolve_upcoming_fixture(
    team_a: str,
    team_b: str,
    reference_date: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """
    Resolve a future fixture from the local match dataset.

    Important: if the local dataset has no future fixture, this function
    returns None instead of inventing a fixture/date.
    """
    if not TC["home"] or not TC["away"] or not TC["date"]:
        return None

    reference_date = reference_date or pd.Timestamp.now().normalize()
    dates = pd.to_datetime(team_df[TC["date"]], errors="coerce")

    a = clean(team_a)
    b = clean(team_b)

    mask = (
        (
            team_df[TC["home"]].astype(str).str.lower().eq(a)
            & team_df[TC["away"]].astype(str).str.lower().eq(b)
        )
        |
        (
            team_df[TC["home"]].astype(str).str.lower().eq(b)
            & team_df[TC["away"]].astype(str).str.lower().eq(a)
        )
    )

    future = team_df.loc[mask & dates.ge(reference_date)].copy()
    if future.empty:
        return None

    future["_parsed_date"] = pd.to_datetime(
        future[TC["date"]], errors="coerce"
    )
    row = future.sort_values("_parsed_date").iloc[0]

    home = str(row[TC["home"]])
    away = str(row[TC["away"]])
    date = pd.to_datetime(row[TC["date"]]).date()

    match_id = None
    if TC["id"]:
        match_id = str(row[TC["id"]])
    else:
        match_id = str(row.name)

    return {
        "home_team": home,
        "away_team": away,
        "date": str(date),
        "match_id": match_id,
    }


def feature_explanation(
    home: str,
    away: str,
    date: str,
) -> list[str]:
    """
    Build a short deterministic explanation from Day-2 pre-match features.

    If the feature-importance CSV exists, use its ranking. Otherwise use
    the Day-1 feature semantics and show the strongest available signals.
    """
    try:
        from predict import _build_match_features

        X = _build_match_features(home, away, date)
    except Exception:
        return [
            "The model uses leakage-safe pre-match features.",
            "Relevant feature families include recent form, scoring form, rest and H2H.",
        ]

    importance_path = OUTPUT_DIR / "day2_feature_importance.csv"
    if importance_path.exists():
        try:
            fi = pd.read_csv(importance_path)
            feature_col = find_column(fi, ["feature"])
            importance_col = find_column(fi, ["importance"])
            if feature_col and importance_col:
                top = fi.sort_values(
                    importance_col, ascending=False
                )[feature_col].head(3).tolist()
            else:
                top = []
        except Exception:
            top = []
    else:
        top = []

    # Prefer requested, interpretable features if they are available.
    preferred = [
        "form_diff_l5",
        "form_diff_l3",
        "score_form_diff_l5",
        "score_form_diff_l3",
        "h2h_home_win_rate",
        "home_days_rest",
        "away_days_rest",
    ]
    ordered = []
    for f in preferred + top:
        if f in X.columns and f not in ordered:
            ordered.append(f)

    explanations = []
    labels = {
        "form_diff_l5": "5-match form difference",
        "form_diff_l3": "3-match form difference",
        "score_form_diff_l5": "5-match scoring-form difference",
        "score_form_diff_l3": "3-match scoring-form difference",
        "h2h_home_win_rate": "historical H2H home-team win rate",
        "home_days_rest": "home-team rest",
        "away_days_rest": "away-team rest",
    }

    for f in ordered[:3]:
        value = X.iloc[0][f]
        if pd.isna(value):
            continue
        explanations.append(f"{labels.get(f, f)} = {float(value):.3f}")

    return explanations or [
        "The model uses leakage-safe pre-match form, scoring, rest and H2H features."
    ]


# ============================================================
# 7. ROUTER
# ============================================================

ROUTER_SYSTEM = """
Classify an AFL user query into exactly one intent:

prediction:
- asks who will win, beat, defeat, top-score, lead, or be predicted
- asks for a forecast/probability about an upcoming AFL match/player

retrieval:
- asks for exact statistics, records, results, season stats, round stats,
  player stats, team H2H, or other dataset-backed numerical facts

factual:
- asks AFL rules, history, concepts, explanations, or non-numerical factual
  AFL information

off-topic:
- anything unrelated to AFL

Important:
- "who will win" and "who will top-score" are prediction.
- "what were X's stats" is retrieval.
- Do not treat a prediction as retrieval merely because statistics are needed.
"""


def classify_intent(query: str) -> tuple[Intent, str]:
    """
    Deterministic first-pass router.

    A lightweight rule router is deliberately used here because the assignment
    requires predictable routing. It is easier to audit than a free-form LLM
    tool chooser.
    """
    q = query.lower().strip()

    off_topic_markers = [
        "python", "javascript", "programming", "crypto", "weather",
        "recipe", "nba", "nfl", "cricket", "soccer", "tennis",
        "politics", "stock market", "bitcoin",
    ]
    if any(marker in q for marker in off_topic_markers):
        return "off-topic", "Detected an explicitly non-AFL topic."

    prediction_markers = [
        "who will win",
        "who wins",
        "will win",
        "will the ",
        "beat ",
        "defeat ",
        "winner prediction",
        "predict",
        "prediction",
        "forecast",
        "probability",
        "chance of winning",
        "top-score",
        "top score",
        "topscorer",
        "top scorer",
        "who will score the most",
        "who scores the most",
        "likely to score the most",
        "score the most goals",
        "scores the most goals",
        "most goals this week",
        "leading scorer",
    ]
    if any(marker in q for marker in prediction_markers):
        return "prediction", "Detected forecast/prediction language."

    retrieval_markers = [
        "stats", "statistics", "disposals", "goals", "marks",
        "tackles", "possessions", "averages", "record",
        "head to head", "h2h", "last round", "previous round",
        "season", "round ", "how many", "how much",
    ]
    if any(marker in q for marker in retrieval_markers):
        return "retrieval", "Detected dataset/statistics request."

    afl_markers = [
        "afl", "australian rules", "football", "team", "player",
        "round", "premiership", "grand final", "fixture", "match",
    ]
    if any(marker in q for marker in afl_markers):
        return "factual", "Detected AFL factual/explanatory request."

    return "off-topic", "No supported AFL intent detected."


# ============================================================
# 8. GRAPH NODES
# ============================================================

def add_trace(
    state: AFLState,
    node: str,
    details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append({
        "node": node,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
    })
    return trace


def router_node(state: AFLState) -> AFLState:
    intent, reason = classify_intent(state["user_query"])
    return {
        "intent": intent,
        "route_reason": reason,
        "trace": add_trace(
            state,
            "router",
            {"intent": intent, "reason": reason},
        ),
    }


def direct_answer_node(state: AFLState) -> AFLState:
    history = state.get("conversation_history", [])
    messages = [
        SystemMessage(
            content=(
                "You are an AFL-only assistant. Answer concise factual AFL "
                "questions. Do not invent numerical statistics. If a question "
                "requires exact dataset statistics, say that the retrieval "
                "route is required rather than making numbers up."
            )
        )
    ]

    for item in history[-8:]:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(HumanMessage(content=f"Previous assistant answer: {content}"))

    messages.append(HumanMessage(content=state["user_query"]))

    try:
        llm = ChatOllama(model=MODEL_NAME, temperature=0)
        response = llm.invoke(messages)
        answer = str(response.content).strip()
    except Exception as exc:
        answer = (
            "I can answer AFL factual questions, but the local LLM is not "
            f"available right now: {exc}"
        )

    return {
        "final_response": answer,
        "tool_name": None,
        "tool_result": None,
        "validation_ok": True,
        "trace": add_trace(
            state,
            "direct_answer",
            {"llm": MODEL_NAME},
        ),
    }


def refusal_node(state: AFLState) -> AFLState:
    answer = (
        "I’m focused on AFL, so I can’t help with that. "
        "I can help with AFL teams, players, matches, statistics, history, "
        "rules, retrieval, or predictions."
    )
    return {
        "final_response": answer,
        "out_of_scope": True,
        "validation_ok": True,
        "trace": add_trace(
            state,
            "refusal",
            {"reason": "off-topic"},
        ),
    }


def retrieval_node(state: AFLState) -> AFLState:
    q = state["user_query"]
    q_lower = q.lower()

    # H2H
    if "head to head" in q_lower or "h2h" in q_lower or "record against" in q_lower:
        pair = parse_two_team_mentions(q)
        if not pair:
            return {
                "needs_clarification": True,
                "validation_ok": False,
                "validation_message": "Could not resolve two teams for H2H.",
                "final_response": (
                    "Which two AFL teams should I compare for the head-to-head record?"
                ),
                "trace": add_trace(
                    state,
                    "retrieval",
                    {"status": "clarification", "reason": "two teams required"},
                ),
            }

        result = get_team_head_to_head.invoke({
            "team_a": pair[0],
            "team_b": pair[1],
        })
        return {
            "tool_name": get_team_head_to_head.name,
            "tool_input": {"team_a": pair[0], "team_b": pair[1]},
            "tool_result": result,
            "resolved_entities": {"team_a": pair[0], "team_b": pair[1]},
            "trace": add_trace(
                state,
                "retrieval",
                {"tool": get_team_head_to_head.name},
            ),
        }

    # Player name is extracted by comparing dataset names against query.
    name_col = find_column(
        player_df, ["player_name", "player_full_name", "name", "playerName"]
    )
    player_candidate = None
    if name_col:
        for name in sorted(
            player_df[name_col].dropna().astype(str).unique(),
            key=len,
            reverse=True,
        ):
            if clean(name) in q_lower:
                player_candidate = name
                break

    # If no player is explicitly named, reuse the player from the most
    # recent conversation turn. This supports follow-ups such as
    # "What about last round?" without guessing a new player.
    if not player_candidate:
        for previous in reversed(state.get("conversation_history", [])):
            previous_text = previous.get("content", "")
            if name_col:
                for name in sorted(
                    player_df[name_col].dropna().astype(str).unique(),
                    key=len,
                    reverse=True,
                ):
                    if clean(name) in previous_text.lower():
                        player_candidate = name
                        break
            if player_candidate:
                break

    # If still unresolved, try a likely capitalized phrase after
    # "player"/"for".
    if not player_candidate:
        match = re.search(
            r"(?:for|player)\s+([A-Z][A-Za-z.\'-]+(?:\s+[A-Z][A-Za-z.\'-]+){1,2})",
            q,
        )
        if match:
            player_candidate = match.group(1)

    if not player_candidate:
        return {
            "needs_clarification": True,
            "validation_ok": False,
            "validation_message": "Could not resolve a player.",
            "final_response": (
                "Which AFL player do you mean? Please provide the player's name."
            ),
            "trace": add_trace(
                state,
                "retrieval",
                {"status": "clarification", "reason": "player unresolved"},
            ),
        }

    if "last round" in q_lower or "previous round" in q_lower:
        round_col = find_column(round_df, ["round", "round_number", "round_no"])
        if not round_col:
            return {
                "needs_clarification": True,
                "validation_ok": False,
                "final_response": "The round column is unavailable in the dataset.",
                "trace": add_trace(state, "retrieval", {"status": "error"}),
            }

        numeric_rounds = pd.to_numeric(round_df[round_col], errors="coerce").dropna()
        if numeric_rounds.empty:
            return {
                "needs_clarification": True,
                "validation_ok": False,
                "final_response": "I could not determine the latest round from the dataset.",
                "trace": add_trace(state, "retrieval", {"status": "error"}),
            }

        round_number = int(numeric_rounds.max())
    else:
        round_number = parse_round(q)

    year = parse_year(q)

    if round_number is not None:
        result = get_player_round_stats.invoke({
            "player_name": player_candidate,
            "round_number": round_number,
            "year": year,
        })
        tool_name = get_player_round_stats.name
        tool_input = {
            "player_name": player_candidate,
            "round_number": round_number,
            "year": year,
        }
    else:
        result = get_player_season_stats.invoke({
            "player_name": player_candidate,
            "year": year,
        })
        tool_name = get_player_season_stats.name
        tool_input = {
            "player_name": player_candidate,
            "year": year,
        }

    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_result": result,
        "resolved_entities": {"player": player_candidate},
        "trace": add_trace(
            state,
            "retrieval",
            {"tool": tool_name, "input": tool_input},
        ),
    }


def prediction_node(state: AFLState) -> AFLState:
    q = state["user_query"]

    if predict_match_winner is None or predict_top_player is None:
        return {
            "validation_ok": False,
            "validation_message": "Day-2 predict.py could not be imported.",
            "final_response": (
                "Prediction tools are unavailable. Check that predict.py and "
                f"the Day-2 model artifacts are present. Import error: "
                f"{PREDICT_IMPORT_ERROR}"
            ),
            "trace": add_trace(
                state,
                "prediction",
                {"status": "import_error"},
            ),
        }

    stat_type = parse_stat_type(q)

    # Top-player prediction.
    if any(
        x in q.lower()
        for x in [
            "top-score", "top score", "topscorer", "top scorer",
            "who will score the most", "who scores the most",
            "leading scorer", "top player",
        ]
    ):
        if stat_type not in {"goals", "disposals"}:
            stat_type = "goals"

        pair = parse_two_team_mentions(q)
        if not pair:
            return {
                "needs_clarification": True,
                "validation_ok": False,
                "validation_message": "Could not resolve both teams.",
                "final_response": (
                    "Which two AFL teams are you asking about for the prediction?"
                ),
                "trace": add_trace(
                    state,
                    "prediction",
                    {"status": "clarification", "reason": "teams unresolved"},
                ),
            }

        home, home_err = resolve_team(pair[0])
        away, away_err = resolve_team(pair[1])
        if home_err or away_err:
            return {
                "needs_clarification": True,
                "validation_ok": False,
                "final_response": home_err or away_err,
                "trace": add_trace(
                    state,
                    "prediction",
                    {"status": "clarification", "reason": "team resolution"},
                ),
            }

        fixture = resolve_upcoming_fixture(home, away)
        if not fixture:
            return {
                "needs_clarification": True,
                "validation_ok": False,
                "validation_message": "No future fixture exists in local dataset.",
                "final_response": (
                    f"I resolved the teams as {home} and {away}, but the local "
                    "dataset does not contain an upcoming fixture/date for this "
                    "pair. Please provide the match ID or exact date available "
                    "in your dataset."
                ),
                "resolved_entities": {"home": home, "away": away},
                "trace": add_trace(
                    state,
                    "prediction",
                    {"status": "clarification", "reason": "fixture unavailable"},
                ),
            }

        try:
            result = predict_top_player(
                match_id=fixture["match_id"],
                stat_type=stat_type,
                top_k=5,
            )
        except Exception as exc:
            return {
                "tool_name": "predict_top_player",
                "tool_input": {
                    "match_id": fixture["match_id"],
                    "stat_type": stat_type,
                },
                "tool_result": {"ok": False, "error": str(exc)},
                "validation_ok": False,
                "validation_message": str(exc),
                "trace": add_trace(
                    state,
                    "prediction",
                    {"status": "tool_error", "error": str(exc)},
                ),
            }

        return {
            "tool_name": "predict_top_player",
            "tool_input": {
                "match_id": fixture["match_id"],
                "stat_type": stat_type,
                "top_k": 5,
            },
            "tool_result": {"ok": True, "ranking": result},
            "resolved_entities": fixture,
            "trace": add_trace(
                state,
                "prediction",
                {"tool": "predict_top_player", "fixture": fixture},
            ),
        }

    # Match-winner prediction.
    pair = parse_two_team_mentions(q)
    if not pair:
        return {
            "needs_clarification": True,
            "validation_ok": False,
            "validation_message": "Could not resolve two teams.",
            "final_response": (
                "Which two AFL teams are you asking about? "
                "Please provide both team names or nicknames."
            ),
            "trace": add_trace(
                state,
                "prediction",
                {"status": "clarification", "reason": "teams unresolved"},
            ),
        }

    home, home_err = resolve_team(pair[0])
    away, away_err = resolve_team(pair[1])
    if home_err or away_err:
        return {
            "needs_clarification": True,
            "validation_ok": False,
            "final_response": home_err or away_err,
            "trace": add_trace(
                state,
                "prediction",
                {"status": "clarification", "reason": "team resolution"},
            ),
        }

    fixture = resolve_upcoming_fixture(home, away)

    # Explicit date can be used if the user supplied one.
    year = parse_year(q)
    explicit_date = None
    date_match = re.search(
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        q,
    )
    if date_match:
        explicit_date = (
            f"{date_match.group(1)}-{int(date_match.group(2)):02d}-"
            f"{int(date_match.group(3)):02d}"
        )

    if explicit_date:
        match_date = explicit_date
    elif fixture:
        match_date = fixture["date"]
    else:
        return {
            "needs_clarification": True,
            "validation_ok": False,
            "validation_message": "No resolvable fixture/date.",
            "final_response": (
                f"I resolved the teams as {home} and {away}, but I cannot "
                "determine a fixture date from the local dataset. Please give "
                "the exact match date."
            ),
            "resolved_entities": {"home": home, "away": away},
            "trace": add_trace(
                state,
                "prediction",
                {"status": "clarification", "reason": "date unavailable"},
            ),
        }

    try:
        result = predict_match_winner(home, away, match_date)
    except Exception as exc:
        return {
            "tool_name": "predict_match_winner",
            "tool_input": {
                "team_a": home,
                "team_b": away,
                "date": match_date,
            },
            "tool_result": {"ok": False, "error": str(exc)},
            "validation_ok": False,
            "validation_message": str(exc),
            "trace": add_trace(
                state,
                "prediction",
                {"status": "tool_error", "error": str(exc)},
            ),
        }

    if not isinstance(result, dict) or "winner" not in result:
        return {
            "tool_name": "predict_match_winner",
            "tool_input": {
                "team_a": home,
                "team_b": away,
                "date": match_date,
            },
            "tool_result": result,
            "validation_ok": False,
            "validation_message": "Prediction tool returned no usable winner.",
            "trace": add_trace(
                state,
                "prediction",
                {"status": "invalid_result"},
            ),
        }

    result["explanation_features"] = feature_explanation(
        home, away, match_date
    )

    return {
        "tool_name": "predict_match_winner",
        "tool_input": {
            "team_a": home,
            "team_b": away,
            "date": match_date,
        },
        "tool_result": {"ok": True, **result},
        "resolved_entities": {
            "home": home,
            "away": away,
            "date": match_date,
        },
        "trace": add_trace(
            state,
            "prediction",
            {
                "tool": "predict_match_winner",
                "input": {
                    "team_a": home,
                    "team_b": away,
                    "date": match_date,
                },
            },
        ),
    }


def validation_node(state: AFLState) -> AFLState:
    if state.get("needs_clarification"):
        return {
            "validation_ok": False,
            "validation_message": state.get(
                "validation_message",
                "Clarification required.",
            ),
            "trace": add_trace(
                state,
                "validation",
                {"status": "clarification"},
            ),
        }

    result = state.get("tool_result")

    if result is None:
        # Direct/refusal paths intentionally have no tool result.
        return {
            "validation_ok": True,
            "trace": add_trace(
                state,
                "validation",
                {"status": "no_tool_required"},
            ),
        }

    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = {"ok": False, "error": result}
    else:
        parsed = result

    ok = bool(isinstance(parsed, dict) and parsed.get("ok"))

    if not ok:
        error = (
            parsed.get("error", "Tool returned no usable result.")
            if isinstance(parsed, dict)
            else "Tool returned no usable result."
        )
        return {
            "validation_ok": False,
            "validation_message": str(error),
            "needs_clarification": True,
            "final_response": (
                f"I couldn't complete that request from the available AFL "
                f"data: {error} "
                "Please provide a more specific player, team, match ID, or date."
            ),
            "trace": add_trace(
                state,
                "validation",
                {"status": "failed", "error": str(error)},
            ),
        }

    return {
        "validation_ok": True,
        "validation_message": "Validated tool result.",
        "trace": add_trace(
            state,
            "validation",
            {"status": "passed", "tool": state.get("tool_name")},
        ),
    }


def format_response_node(state: AFLState) -> AFLState:
    # Clarification/fallback response already produced.
    if state.get("final_response") and not state.get("validation_ok"):
        return {
            "trace": add_trace(
                state,
                "response_formatter",
                {"status": "pass_through_clarification"},
            ),
        }

    result = state.get("tool_result")
    tool_name = state.get("tool_name")

    # Defensive fallback: some clarification/error branches intentionally
    # have no tool result. Never call .get() on None and never hallucinate.
    if result is None:
        existing = state.get("final_response")
        if existing:
            return {
                "final_response": existing,
                "trace": add_trace(
                    state,
                    "response_formatter",
                    {"status": "pass_through_no_tool_result"},
                ),
            }
        return {
            "final_response": (
                "I could not obtain a usable result from the AFL tools. "
                "Please provide a more specific team, player, match, or date."
            ),
            "trace": add_trace(
                state,
                "response_formatter",
                {"status": "fallback_no_tool_result"},
            ),
        }

    if tool_name == "predict_match_winner":
        data = result
        winner = data["winner"]
        probability = float(data["probability"]) * 100
        probs = data.get("probabilities", {})
        features = data.get("explanation_features", [])

        prob_text = ", ".join(
            [
                f"{data['home_team']}: {probs.get('home_win', 0) * 100:.1f}%",
                f"{data['away_team']}: {probs.get('away_win', 0) * 100:.1f}%",
                f"draw: {probs.get('draw', 0) * 100:.1f}%",
            ]
        )

        explanation = "; ".join(features[:3])

        answer = (
            f"Model prediction for {data['home_team']} vs "
            f"{data['away_team']} on {data['date']}: **{winner}** "
            f"at an estimated probability of **{probability:.1f}%**.\n\n"
            f"Class probabilities: {prob_text}.\n"
            f"Grounding: {explanation}.\n\n"
            "This is a model-based probabilistic forecast, not a certainty."
        )

    elif tool_name == "predict_top_player":
        ranking = result["ranking"]
        fixture = state.get("resolved_entities", {})
        stat_type = state.get("tool_input", {}).get("stat_type", "disposals")

        lines = []
        for row in ranking[:5]:
            lines.append(
                f"{row['rank']}. {row['player']} — "
                f"predicted {stat_type}: {float(row['predicted_value']):.2f}"
            )

        answer = (
            f"Probabilistic top-{stat_type} ranking for "
            f"{fixture.get('home_team', 'the match')} vs "
            f"{fixture.get('away_team', 'the opponent')} "
            f"({fixture.get('date', 'date unavailable')}):\n"
            + "\n".join(lines)
            + "\n\nThese are model predictions/rankings, not guaranteed outcomes."
        )

    else:
        # Retrieval result.
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

        if data.get("type") == "player_round":
            records = data["records"]
            answer = (
                f"Dataset result for {data['player']} — Round {data['round']}"
                + (f", {data['year']}" if data.get("year") else "")
                + ":\n"
                + json.dumps(records, indent=2, default=str)
            )
        elif data.get("type") == "player_season":
            answer = (
                f"Season statistics for {data['player']}"
                + (f" ({data['year']})" if data.get("year") else "")
                + ":\n"
                + json.dumps(data["records"], indent=2, default=str)
            )
        elif data.get("type") == "head_to_head":
            answer = (
                f"Head-to-head: {data['team_a']} vs {data['team_b']} — "
                f"{data['matches_found']} matches found.\n"
                + json.dumps(data["records"], indent=2, default=str)
            )
        else:
            answer = json.dumps(data, indent=2, default=str)

    return {
        "final_response": answer,
        "trace": add_trace(
            state,
            "response_formatter",
            {"status": "formatted", "tool": tool_name},
        ),
    }


# ============================================================
# 9. GRAPH ROUTING
# ============================================================

def route_after_router(state: AFLState) -> str:
    return state["intent"]


def route_to_validation(state: AFLState) -> str:
    # Failed prediction/retrieval goes through the clarification response.
    return "format"


builder = StateGraph(AFLState)

builder.add_node("router", router_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("prediction", prediction_node)
builder.add_node("refusal", refusal_node)
builder.add_node("validation", validation_node)
builder.add_node("format", format_response_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "factual": "direct_answer",
        "retrieval": "retrieval",
        "prediction": "prediction",
        "off-topic": "refusal",
    },
)

builder.add_edge("retrieval", "validation")
builder.add_edge("prediction", "validation")

builder.add_conditional_edges(
    "validation",
    route_to_validation,
    {
        "format": "format",
    },
)

builder.add_edge("direct_answer", END)
builder.add_edge("refusal", END)
builder.add_edge("format", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ============================================================
# 10. PUBLIC CHAT API
# ============================================================

SESSION_HISTORY: dict[str, list[dict[str, str]]] = {}


def ask(
    query: str,
    thread_id: str = "default",
) -> dict[str, Any]:
    history = SESSION_HISTORY.setdefault(thread_id, [])

    state: AFLState = {
        "user_query": query,
        "conversation_history": history[-10:],
        "trace": [],
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = graph.invoke(state, config=config)

    answer = result.get(
        "final_response",
        "I could not produce a response.",
    )

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    SESSION_HISTORY[thread_id] = history[-20:]

    return result


# ============================================================
# 11. ROUTING ACCURACY TEST — 20 CASES
# ============================================================

ROUTING_TESTS = [
    ("Who will win Collingwood vs Geelong?", "prediction"),
    ("Will the Pies beat the Cats?", "prediction"),
    ("Who will top-score in the Carlton vs Essendon match?", "prediction"),
    ("Predict the winner of Sydney vs Brisbane.", "prediction"),
    ("What is the probability that Richmond wins?", "prediction"),
    ("Who is likely to score the most goals this week?", "prediction"),

    ("What were Nick Daicos's stats last round?", "retrieval"),
    ("How many disposals did a player have in Round 10?", "retrieval"),
    ("What are this player's season statistics?", "retrieval"),
    ("What is Collingwood's record against Geelong?", "retrieval"),
    ("Show the H2H between Carlton and Essendon.", "retrieval"),
    ("How many goals did the player kick in Round 5?", "retrieval"),

    ("What are the basic rules of AFL?", "factual"),
    ("Explain what a behind is in AFL.", "factual"),
    ("When was the AFL founded?", "factual"),
    ("What does a mark mean in Australian football?", "factual"),

    ("How do I write a Python loop?", "off-topic"),
    ("What's the weather in Lahore?", "off-topic"),
    ("Who won the last Pakistan cricket match?", "off-topic"),
    ("Explain Bitcoin mining.", "off-topic"),
]


def run_routing_tests() -> pd.DataFrame:
    rows = []
    for query, expected in ROUTING_TESTS:
        predicted, reason = classify_intent(query)
        rows.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "reason": reason,
        })

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(ROUTING_REPORT, index=False)
    return df


# ============================================================
# 12. REPRESENTATIVE STATE TRACES
# ============================================================

def run_trace_demos() -> list[dict[str, Any]]:
    """
    These traces are generated against the local application.

    The prediction demo may end at validation/clarification when the local
    dataset has no future fixture. That is intentional: the application must
    not invent a date or fixture.
    """
    demos = [
        (
            "trace-retrieval",
            "What were Nick Daicos's stats last round?",
        ),
        (
            "trace-prediction",
            "Will the Pies beat the Cats this week?",
        ),
        (
            "trace-offtopic",
            "How do I write a Python loop?",
        ),
    ]

    traces = []
    for thread_id, query in demos:
        result = ask(query, thread_id=thread_id)
        traces.append({
            "thread_id": thread_id,
            "query": query,
            "intent": result.get("intent"),
            "route_reason": result.get("route_reason"),
            "tool_name": result.get("tool_name"),
            "tool_input": result.get("tool_input"),
            "validation_ok": result.get("validation_ok"),
            "validation_message": result.get("validation_message"),
            "final_response": result.get("final_response"),
            "state_trace": result.get("trace", []),
        })

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(TRACE_FILE, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False, default=str)

    return traces


# ============================================================
# 13. END-TO-END TEST SCRIPT
# ============================================================

END_TO_END_TESTS = [
    ("factual", "What are the basic rules of AFL?"),
    ("retrieval-round", "What were Nick Daicos's stats last round?"),
    ("retrieval-season", "What are Nick Daicos's season statistics?"),
    ("retrieval-h2h", "What is Collingwood's record against Geelong?"),
    ("prediction-match", "Will the Pies beat the Cats this week?"),
    ("prediction-player", "Who will top-score in Collingwood vs Geelong?"),
    ("off-topic", "How do I write a Python loop?"),
    ("ambiguous", "Who will win?"),
    ("unsupported", "Predict the top ruck hitouts for Collingwood vs Geelong."),
    ("multiturn", "What were Nick Daicos's stats last round?"),
]


def run_end_to_end_tests() -> list[dict[str, Any]]:
    outputs = []

    for label, query in END_TO_END_TESTS:
        thread_id = f"e2e-{label}"
        try:
            result = ask(query, thread_id=thread_id)
            outputs.append({
                "test": label,
                "query": query,
                "intent": result.get("intent"),
                "tool": result.get("tool_name"),
                "validation_ok": result.get("validation_ok"),
                "answer": result.get("final_response"),
            })
        except Exception as exc:
            outputs.append({
                "test": label,
                "query": query,
                "intent": "ERROR",
                "tool": None,
                "validation_ok": False,
                "answer": str(exc),
            })

    # Multi-turn follow-up.
    try:
        first = ask(
            "What are Nick Daicos's season statistics?",
            thread_id="e2e-followup",
        )
        second = ask(
            "What about last round?",
            thread_id="e2e-followup",
        )
        outputs.append({
            "test": "multiturn-followup",
            "query": "What about last round?",
            "intent": second.get("intent"),
            "tool": second.get("tool_name"),
            "validation_ok": second.get("validation_ok"),
            "answer": second.get("final_response"),
            "previous_intent": first.get("intent"),
        })
    except Exception as exc:
        outputs.append({
            "test": "multiturn-followup",
            "query": "What about last round?",
            "intent": "ERROR",
            "tool": None,
            "validation_ok": False,
            "answer": str(exc),
        })

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(CONVERSATION_LOG, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False, default=str)

    return outputs


# ============================================================
# 14. CLI
# ============================================================

def print_routing_report(df: pd.DataFrame) -> None:
    correct = int(df["correct"].sum())
    total = len(df)
    accuracy = correct / total if total else 0

    print("\n" + "=" * 80)
    print("DAY 4 — ROUTING ACCURACY")
    print("=" * 80)
    print(df[["expected", "predicted", "correct", "query"]].to_string(index=False))
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Saved: {ROUTING_REPORT}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    routing_df = run_routing_tests()
    print_routing_report(routing_df)

    print("\n" + "=" * 80)
    print("DAY 4 — REPRESENTATIVE STATE TRACES")
    print("=" * 80)

    traces = run_trace_demos()
    for item in traces:
        print(f"\n[{item['thread_id']}] {item['query']}")
        print(f"Intent: {item['intent']}")
        print(f"Tool: {item['tool_name']}")
        print(f"Validation: {item['validation_ok']}")
        print("Trace:")
        for step in item["state_trace"]:
            print(
                f"  -> {step['node']}: "
                f"{json.dumps(step.get('details', {}), default=str)}"
            )

    print(f"\nSaved traces: {TRACE_FILE}")

    print("\n" + "=" * 80)
    print("AFL DAY 4 LANGGRAPH CHAT")
    print("=" * 80)
    print("Type 'exit' to stop.")
    print("Example: What are the basic rules of AFL?")
    print("Example: What were Nick Daicos's stats last round?")
    print("Example: Will the Pies beat the Cats this week?")
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            break

        try:
            result = ask(query, thread_id="cli")
            print("\nAgent:", result.get("final_response"))
            print(
                f"\n[trace] intent={result.get('intent')} "
                f"tool={result.get('tool_name')} "
                f"validation={result.get('validation_ok')}"
            )
        except Exception as exc:
            print(f"\n[ERROR] {exc}")


if __name__ == "__main__":
    main()
