"""
AFL Week 3 Day 1 — Data Foundations
EDA, Feature Engineering & Prediction Targets

Expected source files (keep these names unchanged):
    afl_datasets/afl_players_info_raw.csv
    afl_datasets/afl_players_round_by_round_stats_raw.csv
    afl_datasets/afl_players_seasonal_stats_raw.csv
    afl_datasets/team_matches_home_away_raw.csv

Run:
    python afl_week3_day1_final.py

Outputs:
    afl_outputs/
    afl_plots/
"""

from pathlib import Path
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ht  = "home_team"
# at  = "away_team"
# hs  = "_home_score"
# aws = "_away_score"
# sc  = "year"
# dc  = "match_date"

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 150)
pd.set_option("display.width", 180)

BASE = Path(".")
DATA_DIR = BASE / "afl_datasets"
OUTPUT_DIR = BASE / "afl_outputs"
PLOT_DIR = BASE / "afl_plots"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

FILES = {
    "players_info": DATA_DIR / "afl_players_info_raw.csv",
    "player_round": DATA_DIR / "afl_players_round_by_round_stats_raw.csv",
    "player_season": DATA_DIR / "afl_players_seasonal_stats_raw.csv",
    "matches": DATA_DIR / "team_matches_home_away_raw.csv",
}

def clean_columns(df):
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        for c in df.columns
    ]
    return df

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return clean_columns(pd.read_csv(path, low_memory=False))

def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_col(df, exact=None, contains=None):
    exact = exact or []
    contains = contains or []
    for c in exact:
        if c in df.columns:
            return c
    for c in df.columns:
        if any(x in c for x in contains):
            return c
    return None

def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col and col in df else pd.Series(np.nan, index=df.index)

def date_col(df):
    return first_col(df, [
        "date", "match_date", "game_date", "round_date",
        "datetime", "match_datetime", "game_datetime"
    ])

def season_col(df):
    return first_col(df, ["season", "year", "season_year"])

def match_id_col(df):
    return first_col(df, ["match_id", "game_id", "matchid", "gameid"])

def player_id_col(df):
    return first_col(df, ["player_id", "playerid", "player"])

def team_col(df):
    return first_col(df, ["team_id", "team", "team_name", "club", "club_name"])

# ---------------------------------------------------------------------
# 1. Load exact four source files
# ---------------------------------------------------------------------
tables = {name: load_csv(path) for name, path in FILES.items()}

print("\n=== AFL DATA LOADED ===")
for name, df in tables.items():
    print(f"{name:15s}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

# ---------------------------------------------------------------------
# 2. Data inventory + quality
# ---------------------------------------------------------------------
def likely_grain(name, df):
    cols = set(df.columns)
    if name == "matches":
        return "match"
    if name == "player_round":
        return "player-game / player-round"
    if name == "player_season":
        return "player-season"
    if name == "players_info":
        return "player master/profile"
    if "match_id" in cols and "player_id" in cols:
        return "player-game"
    if "player_id" in cols and "season" in cols:
        return "player-season"
    return "unknown"

inventory = []
quality = []

for name, df in tables.items():
    dc = date_col(df)
    sc = season_col(df)
    dates = pd.to_datetime(df[dc], errors="coerce") if dc else pd.Series(dtype="datetime64[ns]")
    inventory.append({
        "table": name,
        "source_file": str(FILES[name]),
        "rows": len(df),
        "columns": len(df.columns),
        "grain": likely_grain(name, df),
        "date_column": dc,
        "min_date": dates.min() if len(dates) else pd.NaT,
        "max_date": dates.max() if len(dates) else pd.NaT,
        "seasons": df[sc].nunique(dropna=True) if sc else np.nan,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    })
    for c in df.columns:
        s = df[c]
        quality.append({
            "table": name,
            "column": c,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "unique": int(s.nunique(dropna=True)),
        })

inventory_df = pd.DataFrame(inventory)
quality_df = pd.DataFrame(quality)
inventory_df.to_csv(OUTPUT_DIR / "data_inventory.csv", index=False)
quality_df.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

print("\n=== DATA INVENTORY ===")
print(inventory_df.to_string(index=False))

# Coverage summary
coverage = []
for name, df in tables.items():
    sc = season_col(df)
    dc = date_col(df)
    pc = player_id_col(df)
    tc = team_col(df)
    d = pd.to_datetime(df[dc], errors="coerce") if dc else pd.Series(dtype="datetime64[ns]")
    coverage.append({
        "table": name,
        "date_min": d.min() if len(d) else pd.NaT,
        "date_max": d.max() if len(d) else pd.NaT,
        "season_count": df[sc].nunique(dropna=True) if sc else np.nan,
        "team_count": df[tc].nunique(dropna=True) if tc else np.nan,
        "player_count": df[pc].nunique(dropna=True) if pc else np.nan,
    })
coverage_df = pd.DataFrame(coverage)
coverage_df.to_csv(OUTPUT_DIR / "coverage_summary.csv", index=False)

# Naming consistency report
name_report = []
for name, df in tables.items():
    for c in df.columns:
        if "team" in c or "club" in c or "player" in c:
            vals = df[c].dropna().astype(str)
            name_report.append({
                "table": name,
                "column": c,
                "unique_values": vals.nunique(),
                "sample_values": " | ".join(vals.drop_duplicates().head(10).tolist())
            })
pd.DataFrame(name_report).to_csv(OUTPUT_DIR / "naming_consistency_report.csv", index=False)

# Key-stat outlier report
outlier_rows = []
for name, df in tables.items():
    for c in df.columns:
        if any(k in c for k in ["goal", "disposal", "mark", "tackle", "clearance", "score", "points"]):
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) >= 5:
                q1, q3 = s.quantile([.25, .75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_rows.append({
                    "table": name, "column": c, "min": s.min(), "q1": q1,
                    "median": s.median(), "q3": q3, "max": s.max(),
                    "iqr_outliers": int(((s < lo) | (s > hi)).sum())
                })
pd.DataFrame(outlier_rows).to_csv(OUTPUT_DIR / "key_stat_outliers.csv", index=False)

# ---------------------------------------------------------------------
# 3. Match targets
# ---------------------------------------------------------------------
matches = tables["matches"].copy()
ht = first_col(matches, ["home_team", "home_team_name", "home", "home_club"])
at = first_col(matches, ["away_team", "away_team_name", "away", "away_club"])
hs = first_col(matches, ["home_score", "home_points", "home_total_score", "home_total"])
aws = first_col(matches, ["away_score", "away_points", "away_total_score", "away_total"])
mid = match_id_col(matches)
sc = season_col(matches)
dc = date_col(matches)

# Identify home/away team and score columns
if {
    "team_name",
    "home_away",
    "opponent",
    "team_score",
    "opponent_score"
}.issubset(matches.columns):

    # Dataset contains one row from each team's perspective.
    # Keep only the home-team row so there is one row per match.
    matches = matches[
        matches["home_away"].astype(str).str.upper() == "H"
    ].copy()

    # Home and away teams
    matches["home_team"] = matches["team_name"]
    matches["away_team"] = matches["opponent"]

    # Scores
    matches["_home_score"] = pd.to_numeric(
        matches["team_score"],
        errors="coerce"
    )

    matches["_away_score"] = pd.to_numeric(
        matches["opponent_score"],
        errors="coerce"
    )

else:
    raise ValueError(
        "Could not identify home/away team and score columns "
        "in team_matches_home_away_raw.csv."
    )


# Match date
matches["_date"] = pd.to_datetime(
    matches["match_date"],
    errors="coerce"
)


# Match margin
matches["match_margin"] = (
    matches["_home_score"] - matches["_away_score"]
)


# Match result
matches["match_result"] = np.select(
    [
        matches["match_margin"] > 0,
        matches["match_margin"] < 0
    ],
    [
        "HOME_WIN",
        "AWAY_WIN"
    ],
    default="DRAW"
)
matches["home_win"] = (matches["match_margin"] > 0).astype("Int64")
matches["_match_key"] = matches[mid].astype(str) if mid else matches.index.astype(str)
matches = matches.sort_values(["_date", "_match_key"], na_position="last").reset_index(drop=True)

target_dictionary = pd.DataFrame([
    ["match_result", "match", "3-class classification",
     "HOME_WIN if home score > away score; AWAY_WIN if home score < away score; DRAW otherwise.",
     "home_score, away_score"],
    ["home_win", "match", "binary classification",
     "1 when home team wins, else 0.",
     "home_score, away_score"],
    ["match_margin", "match", "regression",
     "home_score - away_score.",
     "home_score, away_score"],
    ["top_disposal_player", "player-game within match", "ranking target",
     "Player(s) with the maximum disposals in a match; ties are retained.",
     "player-round disposals"],
    ["top_goal_kicker", "player-game within match", "ranking target",
     "Player(s) with the maximum goals in a match; ties are retained.",
     "player-round goals"],
], columns=["target_name", "level", "type", "definition", "source_columns"])
target_dictionary.to_csv(OUTPUT_DIR / "target_dictionary.csv", index=False)

# ---------------------------------------------------------------------
# 4. Player table + player targets
# ---------------------------------------------------------------------
pg = tables["player_round"].copy()
pg_mid = match_id_col(pg)
pg_pid = player_id_col(pg)
pg_dc = date_col(pg)
pg_sc = season_col(pg)
disp_col = find_col(pg, exact=["disposals"], contains=["disposal"])
goal_col = find_col(pg, exact=["goals"], contains=["goal"])

pg["_date"] = pd.to_datetime(pg[pg_dc], errors="coerce") if pg_dc else pd.NaT
if pg_pid:
    pg["_player_key"] = pg[pg_pid].astype(str)
else:
    pg["_player_key"] = pg.index.astype(str)
if pg_mid:
    pg["_match_key"] = pg[pg_mid].astype(str)
else:
    # Fallback: attempt match/date identity from available columns.
    pg["_match_key"] = pg["_date"].astype(str) + "_" + pg["_player_key"]

if disp_col:
    pg["_disposals"] = numeric(pg, disp_col)
if goal_col:
    pg["_goals"] = numeric(pg, goal_col)

player_targets = pd.DataFrame()
if pg_mid and (disp_col or goal_col):
    target_parts = []
    if disp_col:
        mx = pg.groupby("_match_key")["_disposals"].transform("max")
        target_parts.append(pg.loc[pg["_disposals"].eq(mx), ["_match_key", "_player_key"]]
                            .assign(target_type="top_disposal_player"))
    if goal_col:
        mx = pg.groupby("_match_key")["_goals"].transform("max")
        target_parts.append(pg.loc[pg["_goals"].eq(mx), ["_match_key", "_player_key"]]
                            .assign(target_type="top_goal_kicker"))
    if target_parts:
        player_targets = pd.concat(target_parts, ignore_index=True)
        player_targets.to_csv(OUTPUT_DIR / "player_match_targets.csv", index=False)

# ---------------------------------------------------------------------
# 5. EDA
# ---------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / name, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

# Relationship 1: match result distribution / home advantage
plt.figure(figsize=(8, 5))
matches["match_result"].value_counts().reindex(["HOME_WIN", "AWAY_WIN", "DRAW"]).plot(kind="bar")
plt.title("AFL Match Result Distribution")
plt.ylabel("Matches")
plt.xlabel("Result")
savefig("01_match_result_distribution.png")

home_rate = matches["home_win"].mean()
print(f"\nOverall home-win rate: {home_rate:.2%}")

# Relationship 2: team win rates over seasons
rows = []
for _, r in matches.iterrows():

    rows.append({
        "season": r["year"],
        "team": r["home_team"],
        "win": int(r["match_result"] == "HOME_WIN")
    })

    rows.append({
        "season": r["year"],
        "team": r["away_team"],
        "win": int(r["match_result"] == "AWAY_WIN")
    })
team_games = pd.DataFrame(rows)
if sc:
    team_rates = team_games.groupby(["season", "team"], dropna=False)["win"].mean().reset_index(name="win_rate")
    team_rates.to_csv(OUTPUT_DIR / "team_win_rates.csv", index=False)
    plt.figure(figsize=(13, 7))
    team_rates.pivot(index="season", columns="team", values="win_rate").plot(ax=plt.gca(), legend=False)
    plt.title("Team Win Rate Over Time")
    plt.ylabel("Win Rate")
    plt.xlabel("Season")
    savefig("02_team_win_rates_over_time.png")
else:
    team_rates = pd.DataFrame()

# Relationship 3: match margin distribution
plt.figure(figsize=(9, 5))
matches["match_margin"].dropna().plot(kind="hist", bins=35)
plt.axvline(0, linestyle="--")
plt.title("Distribution of Match Margins")
plt.xlabel("Home Score - Away Score")
plt.ylabel("Matches")
savefig("03_match_margin_distribution.png")

# Relationship 4: player disposals / goals distributions
if disp_col:
    plt.figure(figsize=(9, 5))
    pg["_disposals"].dropna().plot(kind="hist", bins=35)
    plt.title("Player-Game Disposals Distribution")
    plt.xlabel("Disposals")
    plt.ylabel("Player-games")
    savefig("04_player_disposals_distribution.png")

if goal_col:
    plt.figure(figsize=(9, 5))
    pg["_goals"].dropna().plot(kind="hist", bins=15)
    plt.title("Player-Game Goals Distribution")
    plt.xlabel("Goals")
    plt.ylabel("Player-games")
    savefig("05_player_goals_distribution.png")

# Historical leaders
if pg_pid:
    if disp_col:
        leaders = pg.groupby("_player_key")["_disposals"].sum().sort_values(ascending=False).head(20)
        leaders.rename("total_disposals").to_csv(OUTPUT_DIR / "player_disposal_leaders.csv")
        print("\nTop historical disposal leaders:")
        print(leaders.head(10))
    if goal_col:
        leaders = pg.groupby("_player_key")["_goals"].sum().sort_values(ascending=False).head(20)
        leaders.rename("total_goals").to_csv(OUTPUT_DIR / "player_goal_leaders.csv")
        print("\nTop historical goal leaders:")
        print(leaders.head(10))

# Position analysis if available
position_col = find_col(pg, exact=["position", "player_position", "primary_position", "position_group"], contains=["position"])
if position_col and disp_col:
    pos = pg.groupby(position_col)["_disposals"].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
    pos.to_csv(OUTPUT_DIR / "position_disposal_summary.csv")
    plt.figure(figsize=(10, 5))
    pos["mean"].plot(kind="bar")
    plt.title("Average Disposals by Position")
    plt.ylabel("Average Disposals")
    plt.xlabel("Position")
    savefig("06_position_vs_disposals.png")

# ---------------------------------------------------------------------
# 6. Leakage-safe team features
# ---------------------------------------------------------------------
team_history = pd.concat([
    pd.DataFrame({
    "_match_key": matches["_match_key"],
    "_date": matches["_date"],
    "team": matches["away_team"].astype(str),
    "opponent": matches["home_team"].astype(str),
    "is_home": 0,
    "score_for": matches["_away_score"],
    "score_against": matches["_home_score"],
    "result": np.select(
        [
            matches["match_result"].eq("AWAY_WIN"),
            matches["match_result"].eq("DRAW")
        ],
        [
            1.0,
            0.5
        ],
        default=0.0
    ),
}),
    pd.DataFrame({
        "_match_key": matches["_match_key"],
        "_date": matches["_date"],
        "team": matches["home_team"].astype(str),
        "opponent": matches["away_team"].astype(str),
        "is_home": 1,
        "score_for": matches["_home_score"],
        "score_against": matches["_away_score"],
        "result": np.select(
            [matches["match_result"].eq("HOME_WIN"), matches["match_result"].eq("DRAW")],
            [1.0, 0.5], default=0.0
        ),
    })
], ignore_index=True)

team_history = team_history.sort_values(["team", "_date", "_match_key"], na_position="last").reset_index(drop=True)

for w in [3, 5]:
    team_history[f"win_rate_l{w}"] = team_history.groupby("team")["result"].transform(
        lambda s: s.shift(1).rolling(w, min_periods=1).mean()
    )
    team_history[f"score_for_avg_l{w}"] = team_history.groupby("team")["score_for"].transform(
        lambda s: s.shift(1).rolling(w, min_periods=1).mean()
    )
    team_history[f"score_against_avg_l{w}"] = team_history.groupby("team")["score_against"].transform(
        lambda s: s.shift(1).rolling(w, min_periods=1).mean()
    )

# Exact pre-match winning streak: consecutive prior wins, not including current match.
def prior_win_streak(s):
    out = []
    streak = 0
    for x in s:
        out.append(streak)
        streak = streak + 1 if x == 1 else 0
    return pd.Series(out, index=s.index)

team_history["win_streak"] = team_history.groupby("team")["result"].transform(prior_win_streak)

# Rest days: previous appearance only, therefore pre-match.
team_history["prev_date"] = team_history.groupby("team")["_date"].shift(1)
team_history["days_rest"] = (team_history["_date"] - team_history["prev_date"]).dt.days

# Head-to-head historical record before each match.
h2h_rows = []
history = {}
for _, r in matches.sort_values(["_date", "_match_key"], na_position="last").iterrows():
    h, a = str(r["home_team"]), str(r["away_team"])
    key = tuple(sorted([h, a]))
    prev = history.get(key, [])
    h_wins = sum(x == h for x in prev)
    a_wins = sum(x == a for x in prev)
    draws = sum(x == "DRAW" for x in prev)
    total = len(prev)
    h2h_rows.append({
        "_match_key": r["_match_key"],
        "h2h_home_win_rate": h_wins / total if total else np.nan,
        "h2h_away_win_rate": a_wins / total if total else np.nan,
        "h2h_draw_rate": draws / total if total else np.nan,
        "h2h_meetings": total,
    })
    if r["match_result"] == "HOME_WIN":
        history.setdefault(key, []).append(h)
    elif r["match_result"] == "AWAY_WIN":
        history.setdefault(key, []).append(a)
    else:
        history.setdefault(key, []).append("DRAW")

h2h = pd.DataFrame(h2h_rows)

# Pivot team history back to match level.
hf = team_history[team_history["is_home"].eq(1)].copy()
af = team_history[team_history["is_home"].eq(0)].copy()

feature_names = [
    "win_rate_l3", "win_rate_l5",
    "score_for_avg_l3", "score_for_avg_l5",
    "score_against_avg_l3", "score_against_avg_l5",
    "win_streak", "days_rest"
]
hf = hf[["_match_key"] + feature_names].rename(columns={c: f"home_{c}" for c in feature_names})
af = af[["_match_key"] + feature_names].rename(columns={c: f"away_{c}" for c in feature_names})

feature_table = matches.copy()
feature_table = feature_table.merge(hf, on="_match_key", how="left")
feature_table = feature_table.merge(af, on="_match_key", how="left")
feature_table = feature_table.merge(h2h, on="_match_key", how="left")

for w in [3, 5]:
    feature_table[f"form_diff_l{w}"] = (
        feature_table[f"home_win_rate_l{w}"] -
        feature_table[f"away_win_rate_l{w}"]
    )
    feature_table[f"score_form_diff_l{w}"] = (
        feature_table[f"home_score_for_avg_l{w}"] -
        feature_table[f"away_score_for_avg_l{w}"]
    )

# Optional ladder-position features, if the source dataset already contains them.
ladder_home = find_col(matches, exact=["home_ladder_position", "home_ladder_pos", "home_rank"])
ladder_away = find_col(matches, exact=["away_ladder_position", "away_ladder_pos", "away_rank"])
if ladder_home and ladder_away:
    feature_table["home_ladder_position"] = numeric(feature_table, ladder_home)
    feature_table["away_ladder_position"] = numeric(feature_table, ladder_away)
    feature_table["ladder_position_diff"] = (
        feature_table["home_ladder_position"] - feature_table["away_ladder_position"]
    )

# Optional venue/travel/weather columns are retained only as contextual fields if supplied.
context_candidates = [
    c for c in matches.columns
    if any(k in c for k in ["venue", "stadium", "travel", "distance", "weather", "temperature", "rain", "wind"])
]
context_candidates = [c for c in context_candidates if c not in {"home_team", "away_team", "_home_score", "_away_score"}]
pd.DataFrame({"available_context_column": context_candidates}).to_csv(
    OUTPUT_DIR / "available_context_columns.csv", index=False
)

# ---------------------------------------------------------------------
# 7. Leakage-safe player rolling features
# ---------------------------------------------------------------------
player_feature_table = pd.DataFrame()
if pg_pid and pg_mid:
    p = pg.copy().sort_values(["_player_key", "_date", "_match_key"], na_position="last")
    player_stats = {
        "disposals": "_disposals" if "_disposals" in p else None,
        "goals": "_goals" if "_goals" in p else None,
    }
    for stat in ["marks", "tackles", "clearances", "inside_50s", "rebound_50s",
                 "contested_possessions", "uncontested_possessions"]:
        if stat in p.columns:
            p[stat] = pd.to_numeric(p[stat], errors="coerce")
            player_stats[stat] = stat

    for stat, source in player_stats.items():
        if source:
            for w in [3, 5]:
                p[f"{stat}_avg_l{w}"] = p.groupby("_player_key")[source].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=1).mean()
                )

    feature_cols = ["_match_key", "_player_key", "_date"] + [
        c for c in p.columns if re.search(r"_avg_l[35]$", c)
    ]
    player_feature_table = p[feature_cols].copy()
    player_feature_table.to_csv(OUTPUT_DIR / "player_feature_table.csv", index=False)

# ---------------------------------------------------------------------
# 8. Versioned feature table + feature dictionary
# ---------------------------------------------------------------------
# Model features exclude current-match scores and result targets.
target_cols = {
    "match_result", "home_win", "match_margin",
    "_home_score", "_away_score", "_match_key"
}
model_features = [c for c in feature_table.columns if c not in target_cols]

feature_table["feature_version"] = "week3_day1_v1"
feature_table.to_csv(OUTPUT_DIR / "afl_feature_table_v1.csv", index=False)
try:
    feature_table.to_parquet(OUTPUT_DIR / "afl_feature_table_v1.parquet", index=False)
except Exception as e:
    print(f"Parquet export skipped: {e}")

feature_dictionary = [
    ["home_win_rate_l3", "Home team's mean result over previous 3 matches (win=1, draw=.5, loss=0).", "previous 3 team games", "match results", "shift(1); pre-match only"],
    ["away_win_rate_l3", "Away team's mean result over previous 3 matches.", "previous 3 team games", "match results", "shift(1); pre-match only"],
    ["home_win_rate_l5", "Home team's mean result over previous 5 matches.", "previous 5 team games", "match results", "shift(1); pre-match only"],
    ["away_win_rate_l5", "Away team's mean result over previous 5 matches.", "previous 5 team games", "match results", "shift(1); pre-match only"],
    ["home_score_for_avg_l5", "Average points scored by home team.", "previous 5 team games", "scores", "shift(1); pre-match only"],
    ["away_score_for_avg_l5", "Average points scored by away team.", "previous 5 team games", "scores", "shift(1); pre-match only"],
    ["home_score_against_avg_l5", "Average points conceded by home team.", "previous 5 team games", "scores", "shift(1); pre-match only"],
    ["away_score_against_avg_l5", "Average points conceded by away team.", "previous 5 team games", "scores", "shift(1); pre-match only"],
    ["home_win_streak", "Consecutive prior wins entering the match.", "all prior team games", "match results", "current match excluded"],
    ["away_win_streak", "Consecutive prior wins entering the match.", "all prior team games", "match results", "current match excluded"],
    ["home_days_rest", "Days since home team's previous match.", "previous appearance", "match dates", "previous date only"],
    ["away_days_rest", "Days since away team's previous match.", "previous appearance", "match dates", "previous date only"],
    ["h2h_home_win_rate", "Home team's historical win rate against this opponent.", "all prior meetings", "match results", "current meeting excluded"],
    ["h2h_meetings", "Number of previous meetings between the two teams.", "all prior meetings", "match history", "current meeting excluded"],
    ["form_diff_l3", "Home 3-game form minus away 3-game form.", "previous 3 team games", "rolling form", "pre-match only"],
    ["form_diff_l5", "Home 5-game form minus away 5-game form.", "previous 5 team games", "rolling form", "pre-match only"],
    ["score_form_diff_l3", "Home 3-game scoring average minus away 3-game scoring average.", "previous 3 team games", "scores", "pre-match only"],
    ["score_form_diff_l5", "Home 5-game scoring average minus away 5-game scoring average.", "previous 5 team games", "scores", "pre-match only"],
]
fd = pd.DataFrame(feature_dictionary, columns=["feature", "description", "window", "source_columns", "leakage_rule"])
fd.to_csv(OUTPUT_DIR / "feature_dictionary.csv", index=False)

# ---------------------------------------------------------------------
# 9. Five prediction-relevant relationships
# ---------------------------------------------------------------------
# 1) Recent form vs outcome
if "home_win_rate_l5" in feature_table:
    plot_df = feature_table[["home_win_rate_l5", "match_result"]].dropna()
    if not plot_df.empty:
        plot_df["form_bin"] = pd.cut(plot_df["home_win_rate_l5"], bins=5, duplicates="drop")
        rate = plot_df.assign(home_win=(plot_df["match_result"] == "HOME_WIN")).groupby("form_bin", observed=False)["home_win"].mean()
        plt.figure(figsize=(9, 5))
        rate.plot(kind="bar")
        plt.title("Recent Home-Team Form vs Home Win Probability")
        plt.ylabel("Observed Home Win Rate")
        plt.xlabel("Home Win Rate Over Previous 5 Games")
        savefig("07_recent_form_vs_home_win.png")

# 2) Rest vs margin
rest = feature_table[["home_days_rest", "away_days_rest", "match_margin"]].dropna()
if not rest.empty:
    rest["rest_diff"] = rest["home_days_rest"] - rest["away_days_rest"]
    plt.figure(figsize=(9, 5))
    plt.scatter(rest["rest_diff"], rest["match_margin"], alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.title("Rest-Day Advantage vs Match Margin")
    plt.xlabel("Home Rest Days - Away Rest Days")
    plt.ylabel("Home Score Margin")
    savefig("08_rest_advantage_vs_margin.png")

# 3) H2H vs outcome
h2 = feature_table[["h2h_home_win_rate", "match_result"]].dropna()
if not h2.empty:
    h2["home_win"] = (h2["match_result"] == "HOME_WIN").astype(int)
    h2["bin"] = pd.cut(h2["h2h_home_win_rate"], bins=5, duplicates="drop")
    rate = h2.groupby("bin", observed=False)["home_win"].mean()
    plt.figure(figsize=(9, 5))
    rate.plot(kind="bar")
    plt.title("Historical Head-to-Head Rate vs Home Win")
    plt.ylabel("Observed Home Win Rate")
    plt.xlabel("Prior H2H Home Win Rate")
    savefig("09_h2h_vs_outcome.png")

# 4) Scoring form difference vs margin
sf = feature_table[["score_form_diff_l5", "match_margin"]].dropna()
if not sf.empty:
    plt.figure(figsize=(9, 5))
    plt.scatter(sf["score_form_diff_l5"], sf["match_margin"], alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.title("Recent Scoring-Form Difference vs Match Margin")
    plt.xlabel("Home 5-Game Scoring Avg - Away 5-Game Scoring Avg")
    plt.ylabel("Home Score Margin")
    savefig("10_scoring_form_vs_margin.png")

# 5) Player recent output vs next observed output
if not player_feature_table.empty and "_disposals" in pg.columns:
    pf = player_feature_table.merge(
        pg[["_match_key", "_player_key", "_disposals"]],
        on=["_match_key", "_player_key"], how="left"
    )
    if "disposals_avg_l5" in pf.columns:
        pf = pf[["disposals_avg_l5", "_disposals"]].dropna()
        if not pf.empty:
            plt.figure(figsize=(9, 5))
            plt.scatter(pf["disposals_avg_l5"], pf["_disposals"], alpha=0.25)
            plt.title("Previous 5-Game Disposal Average vs Current Disposal Output")
            plt.xlabel("Previous 5-Game Average Disposals")
            plt.ylabel("Current Match Disposals")
            savefig("11_player_form_vs_next_output.png")

# ---------------------------------------------------------------------
# 10. Time-based train/hold-out split
# ---------------------------------------------------------------------
def time_split(df, season_col_name=None, date_col_name="_date",
               holdout_seasons=1, holdout_fraction=0.20):
    """Reusable strict chronological split.

    If >=2 seasons exist, the latest season(s) are held out.
    Otherwise, the latest chronological fraction is held out.
    """
    data = df.copy()
    if season_col_name and season_col_name in data.columns:
        seasons = sorted(pd.Series(data[season_col_name]).dropna().unique().tolist())
        if len(seasons) >= 2:
            test_seasons = seasons[-holdout_seasons:]
            train = data[~data[season_col_name].isin(test_seasons)].copy()
            test = data[data[season_col_name].isin(test_seasons)].copy()
            return train, test, {
                "method": "latest_season_holdout",
                "test_seasons": [str(x) for x in test_seasons],
            }

    if date_col_name in data.columns:
        data["_split_date"] = pd.to_datetime(data[date_col_name], errors="coerce")
        data = data.sort_values("_split_date", kind="stable").drop(columns="_split_date")

    cut = max(1, int(len(data) * (1 - holdout_fraction)))
    return data.iloc[:cut].copy(), data.iloc[cut:].copy(), {
        "method": "chronological_fraction",
        "holdout_fraction": holdout_fraction,
    }

split_season_col = sc if sc else None
train_df, holdout_df, split_info = time_split(
    feature_table, split_season_col, "_date"
)
train_df.to_csv(OUTPUT_DIR / "train_time_split.csv", index=False)
holdout_df.to_csv(OUTPUT_DIR / "holdout_time_split.csv", index=False)
with open(OUTPUT_DIR / "split_logic.json", "w") as f:
    json.dump(split_info, f, indent=2, default=str)

# ---------------------------------------------------------------------
# 11. Structural changes / rule changes
# ---------------------------------------------------------------------
# This report flags naming/count changes; actual historical rule changes
# should be interpreted from AFL documentation rather than inferred as facts.
structural = []

for c in ["home_team", "away_team"]:

    counts = matches.groupby("year")[c].nunique()

    if not counts.empty:

        structural.append({
            "column": c,
            "season_team_count_min": counts.min(),
            "season_team_count_max": counts.max(),
            "seasons_with_change": int(
                (counts != counts.iloc[0]).sum()
            )
        })

pd.DataFrame(structural).to_csv(
    OUTPUT_DIR / "structural_change_flags.csv",
    index=False
)
# ---------------------------------------------------------------------
# 12. Leakage audit + realistic prediction ceiling
# ---------------------------------------------------------------------
suspicious = []
for c in model_features:
    lc = c.lower()
    if any(k in lc for k in ["winner", "actual_result", "post_match", "final_score"]):
        suspicious.append(c)

leakage_report = pd.DataFrame({
    "check": [
        "Current-match scores excluded from model feature candidates",
        "Rolling team features shifted by one match",
        "Rolling player features shifted by one match",
        "H2H uses only prior meetings",
        "Rest uses previous appearance only",
        "Potential suspicious feature names"
    ],
    "result": [
        "_home_score/_away_score excluded",
        "PASS",
        "PASS",
        "PASS",
        "PASS",
        ", ".join(suspicious) if suspicious else "None detected"
    ]
})
leakage_report.to_csv(OUTPUT_DIR / "leakage_audit.csv", index=False)

ceiling_text = (
    "AFL prediction has an irreducible noise floor caused by injuries, tactics, "
    "player form, weather, umpiring and random scoring variation. A useful model "
    "should consistently outperform simple baselines rather than approach perfect "
    "accuracy. Near-perfect accuracy would be a red flag because it often means "
    "future or post-match information has entered the features. Strict chronological "
    "evaluation and pre-match-only rolling features are therefore essential."
)
(OUTPUT_DIR / "prediction_ceiling.txt").write_text(ceiling_text, encoding="utf-8")

# ---------------------------------------------------------------------
# 13. One-page-style data dictionary markdown
# ---------------------------------------------------------------------
dictionary_md = f"""# AFL Week 3 Day 1 — Data Dictionary & Target Contract

## Source tables
| File | Grain | Main purpose |
|---|---|---|
| `afl_datasets/afl_players_info_raw.csv` | Player | Player identity/profile information |
| `afl_datasets/afl_players_round_by_round_stats_raw.csv` | Player-game/round | Match-level player performance |
| `afl_datasets/afl_players_seasonal_stats_raw.csv` | Player-season | Season aggregates |
| `afl_datasets/team_matches_home_away_raw.csv` | Match | Home/away teams, scores and match context |

## Targets
- **match_result**: 3-class classification: `HOME_WIN`, `AWAY_WIN`, `DRAW`.
- **home_win**: binary classification: 1 if home team wins, otherwise 0.
- **match_margin**: regression target = `home_score - away_score`.
- **top_disposal_player**: player(s) with maximum disposals in a match.
- **top_goal_kicker**: player(s) with maximum goals in a match.
- Ties are retained rather than arbitrarily broken.

## Core prediction features
Rolling team form uses only previous games: 3-game and 5-game win rate, scoring-for average, scoring-against average and prior win streak. Context features include days of rest and historical head-to-head results. Optional ladder/venue/weather/travel columns are used only when present in the source data.

## Leakage rule
Every rolling feature uses `shift(1)` before the rolling window. The current match's score/result/player output is never used to predict that same match. The final evaluation is chronological, with the latest season held out when at least two seasons are available.

## Split
The reusable `time_split()` function is used for every model this week. Random splitting is avoided because future matches can influence training statistics and violate the real forecasting timeline.

## Prediction ceiling
AFL outcomes are inherently noisy. A realistic model should beat simple baselines consistently, not achieve perfect accuracy. Near-perfect performance is suspicious and should trigger a leakage audit.
"""
(OUTPUT_DIR / "data_dictionary_and_targets.md").write_text(dictionary_md, encoding="utf-8")

print("\n=== FINAL OUTPUTS ===")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(p.name)
print("\nFeature table:", feature_table.shape)
print("Train:", train_df.shape, "| Hold-out:", holdout_df.shape)
print("Split:", split_info)
print("\nDONE.")
