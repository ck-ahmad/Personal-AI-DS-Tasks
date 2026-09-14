# AFL Week 3 Day 1 — Data Dictionary & Target Contract

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
