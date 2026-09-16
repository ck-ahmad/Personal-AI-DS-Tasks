"""
AFL Week 3 Day 3
Domain-Scoped AFL Chat Agent
Retrieval, Guardrails, Grounding & Memory

Features:
- AFL-only system prompt
- Off-topic guardrails
- Structured Pandas retrieval
- Player round statistics
- Player season statistics
- Team head-to-head records
- LangChain tools
- Ollama + Qwen local LLM
- Multi-turn conversation memory
- Tool-result grounding
- Tool-call logging
"""

import os
import json
import pandas as pd

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = "afl_datasets"

ROUND_FILE = os.path.join(
    DATA_DIR,
    "afl_players_round_by_round_stats_raw.csv"
)

SEASON_FILE = os.path.join(
    DATA_DIR,
    "afl_players_seasonal_stats_raw.csv"
)

TEAM_FILE = os.path.join(
    DATA_DIR,
    "team_matches_home_away_raw.csv"
)

PLAYER_FILE = os.path.join(
    DATA_DIR,
    "afl_players_info_raw.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("AFL WEEK 3 DAY 3 - CHAT AGENT")
print("=" * 70)

print("\nLoading AFL datasets...")

try:
    round_df = pd.read_csv(ROUND_FILE, low_memory=False, dtype={"player_id": str})
    print(f"[OK] Round data:   {round_df.shape}")

except Exception as e:
    print(f"[ERROR] Could not load round data: {e}")
    raise


try:
    season_df = pd.read_csv(SEASON_FILE, low_memory=False, dtype={"player_id": str})
    print(f"[OK] Season data:  {season_df.shape}")

except Exception as e:
    print(f"[ERROR] Could not load season data: {e}")
    raise


try:
    team_df = pd.read_csv(TEAM_FILE, low_memory=False)
    print(f"[OK] Team data:    {team_df.shape}")

except Exception as e:
    print(f"[ERROR] Could not load team data: {e}")
    raise

try:
    player_df = pd.read_csv(PLAYER_FILE, low_memory=False, dtype=str)
    print(f"[OK] Player data:  {player_df.shape}")

except Exception as e:
    print(f"[ERROR] Could not load player data: {e}")
    raise


# ============================================================
# 3. DISPLAY DATASET COLUMNS
# ============================================================

print("\nDataset columns:")

print("\nRound dataset:")
print(round_df.columns.tolist())

print("\nSeason dataset:")
print(season_df.columns.tolist())

print("\nTeam dataset:")
print(team_df.columns.tolist())

print("\nPlayer dataset:")
print(player_df.columns.tolist())


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):
    """
    Find the first matching column from a list of possible names.
    """

    lower_columns = {
        str(col).lower(): col
        for col in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_columns:
            return lower_columns[name.lower()]

    return None


def safe_value(row, column):
    """
    Safely retrieve a value from a dataframe row.
    """

    if column is None:
        return None

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return value


def resolve_player_ids(player_name):
    """Resolve a player name to one or more player IDs using player-info data."""
    search_name = str(player_name).strip().lower()

    name_col = find_column(
        player_df,
        ["player_name", "player_full_name", "name", "playerName"]
    )

    if name_col is not None:
        names = player_df[name_col].fillna("").astype(str).str.strip()
        matches = player_df[
            names.str.lower().str.contains(search_name, na=False, regex=False)
        ]

        exact = player_df[names.str.lower() == search_name]
        if not exact.empty:
            matches = exact
    else:
        first_col = find_column(player_df, ["first_name", "firstName"])
        last_col = find_column(player_df, ["last_name", "lastName"])

        if first_col is None or last_col is None:
            return [], "ERROR: Could not find player-name columns in player-info dataset."

        full_names = (
            player_df[first_col].fillna("").astype(str).str.strip()
            + " "
            + player_df[last_col].fillna("").astype(str).str.strip()
        ).str.strip()

        matches = player_df[
            full_names.str.lower().str.contains(search_name, na=False, regex=False)
        ]

    if matches.empty:
        return [], f"No AFL player found with name '{player_name}'."

    id_col = find_column(player_df, ["player_id", "id"])
    if id_col is None:
        return [], "ERROR: Could not find player ID column in player-info dataset."

    ids = (
        matches[id_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return ids, None


# ============================================================
# 5. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AFL-only conversational assistant.

SCOPE
You may discuss:
- Australian Football League (AFL)
- AFL teams
- AFL players
- AFL matches
- AFL rounds
- AFL player statistics
- AFL team statistics
- AFL history
- AFL seasons
- AFL rules
- AFL match performance

OUT OF SCOPE
Do not answer questions about:
- Cricket
- Football/soccer
- NBA
- NFL
- Tennis
- Other sports
- Politics
- Programming
- General trivia
- Cryptocurrency
- Weather
- Unrelated personal questions
- General chit-chat unrelated to AFL

OFF-TOPIC BEHAVIOR
If the user asks an unrelated question, politely decline and redirect
the conversation toward AFL.

For example:
"I’m focused on AFL, so I can’t help with that. I can help with AFL
teams, players, matches, or statistics instead."

DATA GROUNDING
For numerical AFL statistics, records, match results, player
performance, or season statistics, ALWAYS use the provided retrieval
tools.

Never invent or guess statistics.

If the requested information is not available in the dataset,
clearly say that the information is not available.

GROUNDING RULE
Every numerical statistic in your final answer must come from a
retrieval tool result.

Do not use your own memory for numerical AFL statistics.

MULTI-TURN MEMORY
Use previous conversation context.

For example:
- "he" may refer to the previously discussed player.
- "his team" may refer to the player's team.
- "previous round" refers to the relevant previously discussed round.
- "that team" refers to the team mentioned earlier.

AMBIGUOUS QUESTIONS
If "football" is ambiguous, ask whether the user means AFL.

STYLE
Keep answers concise, clear, and factual.
"""


# ============================================================
# 6. TOOL 1 - PLAYER ROUND STATISTICS
# ============================================================

@tool
def get_player_round_stats(
    player_name: str,
    round_number: int,
    year: int | None = None
) -> str:
    """Retrieve exact statistics for an AFL player in a specific round."""
    player_ids, error = resolve_player_ids(player_name)
    if error:
        return error

    player_id_col = find_column(round_df, ["player_id", "id"])
    round_col = find_column(round_df, ["round", "round_number", "round_no"])

    if player_id_col is None:
        return "ERROR: Could not find player_id column in round-by-round dataset."
    if round_col is None:
        return "ERROR: Could not find round column in round-by-round dataset."

    mask = (
        round_df[player_id_col].fillna("").astype(str).str.strip().isin(player_ids)
        & (round_df[round_col].astype(str).str.strip() == str(round_number))
    )

    year_col = find_column(round_df, ["year", "season"])
    if year is not None and year_col is not None:
        mask &= pd.to_numeric(round_df[year_col], errors="coerce") == int(year)

    result = round_df[mask]

    if result.empty:
        return (
            f"No AFL dataset record found for {player_name} "
            f"in Round {round_number}"
            + (f", {year}." if year is not None else ".")
        )

    return json.dumps(result.to_dict(orient="records"), default=str, indent=2)


# ============================================================
# 7. TOOL 2 - PLAYER SEASON STATISTICS
# ============================================================

@tool
def get_player_season_stats(
    player_name: str,
    year: int | None = None
) -> str:
    """Retrieve exact AFL player season statistics."""

    player_ids, error = resolve_player_ids(player_name)
    if error:
        return error

    player_id_col = find_column(season_df, ["player_id", "id"])
    if player_id_col is None:
        return "ERROR: Could not find player_id column in seasonal dataset."

    mask = season_df[player_id_col].fillna("").astype(str).str.strip().isin(player_ids)

    year_col = find_column(season_df, ["year", "season"])
    if year is not None and year_col is not None:
        mask &= pd.to_numeric(season_df[year_col], errors="coerce") == int(year)

    result = season_df[mask]

    if result.empty:
        return (
            f"No AFL season record found for {player_name}"
            + (f" in {year}." if year is not None else ".")
        )

    return json.dumps(result.to_dict(orient="records"), default=str, indent=2)


# ============================================================
# 8. TOOL 3 - TEAM HEAD-TO-HEAD
# ============================================================
# ============================================================

@tool
def get_team_head_to_head(
    team_a: str,
    team_b: str
) -> str:
    """
    Retrieve AFL match records between two teams.

    Use this for questions about historical head-to-head results
    between AFL teams.
    """

    df = team_df.copy()

    team_col = find_column(
        df,
        [
            "team_name",
            "team",
            "teamname"
        ]
    )

    opponent_col = find_column(
        df,
        [
            "opponent",
            "opponent_team",
            "opponent_name"
        ]
    )

    result_col = find_column(
        df,
        [
            "result",
            "match_result",
            "outcome"
        ]
    )

    if team_col is None:
        return "ERROR: Team column not found."

    if opponent_col is None:
        return "ERROR: Opponent column not found."

    if result_col is None:
        return "ERROR: Result column not found."

    mask = (
        df[team_col]
        .astype(str)
        .str.lower()
        .str.strip()
        == team_a.lower().strip()
    )

    mask &= (
        df[opponent_col]
        .astype(str)
        .str.lower()
        .str.strip()
        == team_b.lower().strip()
    )

    result = df[mask]

    if result.empty:

        return (
            f"No AFL head-to-head records found between "
            f"{team_a} and {team_b}."
        )

    wins = (
        result[result_col]
        .astype(str)
        .str.lower()
        .str.contains(
            "win",
            na=False
        )
        .sum()
    )

    losses = (
        result[result_col]
        .astype(str)
        .str.lower()
        .str.contains(
            "loss",
            na=False
        )
        .sum()
    )

    draws = (
        result[result_col]
        .astype(str)
        .str.lower()
        .str.contains(
            "draw",
            na=False
        )
        .sum()
    )

    output = {
        "team_a": team_a,
        "team_b": team_b,
        "matches_found": len(result),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "records": result.to_dict(
            orient="records"
        )
    }

    return json.dumps(
        output,
        default=str,
        indent=2
    )


# ============================================================
# 9. REGISTER TOOLS
# ============================================================

TOOLS = [
    get_player_round_stats,
    get_player_season_stats,
    get_team_head_to_head,
]

TOOL_MAP = {
    tool.name: tool
    for tool in TOOLS
}


# ============================================================
# 10. OLLAMA MODEL
# ============================================================

print("\nConnecting to Ollama...")

llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)

llm_with_tools = llm.bind_tools(TOOLS)

print("[OK] Ollama model loaded.")


# ============================================================
# 11. MEMORY
# ============================================================

messages = [
    SystemMessage(
        content=SYSTEM_PROMPT
    )
]


# ============================================================
# 12. TOOL LOG
# ============================================================

tool_log = []


# ============================================================
# 13. CHAT FUNCTION
# ============================================================

def ask_agent(user_input):

    messages.append(
        HumanMessage(
            content=user_input
        )
    )

    response = llm_with_tools.invoke(
        messages
    )

    # --------------------------------------------------------
    # No tool required
    # --------------------------------------------------------

    if not response.tool_calls:

        messages.append(response)

        return response.content

    # --------------------------------------------------------
    # Tool calls
    # --------------------------------------------------------

    messages.append(response)

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        print(
            f"\n[TOOL CALL] {tool_name}"
        )

        print(
            f"[ARGUMENTS] {tool_args}"
        )

        if tool_name not in TOOL_MAP:

            tool_output = (
                f"Unknown tool: {tool_name}"
            )

        else:

            selected_tool = TOOL_MAP[
                tool_name
            ]

            try:

                tool_output = selected_tool.invoke(
                    tool_args
                )

            except Exception as e:

                tool_output = (
                    f"Tool error: {str(e)}"
                )

        # ----------------------------------------------------
        # Log grounding information
        # ----------------------------------------------------

        tool_log.append(
            {
                "user_question": user_input,
                "tool": tool_name,
                "arguments": tool_args,
                "tool_result": tool_output,
            }
        )

        print(
            "\n[TOOL RESULT]"
        )

        print(tool_output)

        messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            )
        )

    # --------------------------------------------------------
    # Final grounded answer
    # --------------------------------------------------------

    final_response = llm.invoke(
        messages
    )

    messages.append(
        final_response
    )

    return final_response.content


# ============================================================
# 14. SAVE TOOL LOG
# ============================================================

def save_tool_log():

    output_file = "afl_retrieval_logs.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            tool_log,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    print(
        f"\n[LOG SAVED] {output_file}"
    )


# ============================================================
# 15. MAIN CHAT LOOP
# ============================================================

print("\n" + "=" * 70)
print("AFL CHAT AGENT READY")
print("=" * 70)

print(
    """
Ask questions about:
- AFL players
- AFL teams
- AFL matches
- AFL statistics
- AFL history

Examples:

  How many disposals did Player X have in Round 10?
  What are Player X's season statistics?
  What is Team A's record against Team B?

Type 'exit' to stop.
"""
)


while True:

    try:

        user_input = input("\nYou: ").strip()

    except KeyboardInterrupt:

        print("\n\nExiting...")
        break

    if not user_input:
        continue

    if user_input.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        break

    try:

        answer = ask_agent(
            user_input
        )

        print(
            "\nAgent:",
            answer
        )

    except Exception as e:

        print(
            "\n[ERROR]",
            e
        )


# ============================================================
# 16. SAVE LOGS
# ============================================================

save_tool_log()

print("\nAFL Chat Agent stopped.")
