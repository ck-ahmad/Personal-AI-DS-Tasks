
"""Compare the match model with a simple ladder-position baseline.

This is a dataset-local benchmark, not a claim of official AFL benchmark
status. It uses the hold-out rows and predicts the team with the better
pre-match ladder-position feature when available.

If the Day-1 feature table contains a column such as
`ladder_position_diff`, the script reports accuracy. Otherwise it exits with
an explicit message rather than fabricating a benchmark.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

FEATURE_FILE = Path("afl_outputs/afl_feature_table_v1.csv")
MODEL_FILE = Path("afl_models/match_winner_pipeline.joblib")

def main():
    if not FEATURE_FILE.exists():
        print("Missing afl_outputs/afl_feature_table_v1.csv. Run Day 1 first.")
        return
    df = pd.read_csv(FEATURE_FILE, low_memory=False)
    if "match_result" not in df:
        print("match_result is missing; cannot evaluate benchmark.")
        return

    ladder_cols = [c for c in df.columns if "ladder_position_diff" in c]
    if not ladder_cols:
        print("No ladder_position_diff feature found. No ladder baseline reported.")
        return

    col = ladder_cols[0]
    d = pd.to_datetime(df["_date"], errors="coerce") if "_date" in df else pd.Series(pd.NaT, index=df.index)
    if d.notna().any():
        work = df.assign(_sort_date=d).sort_values("_sort_date")
    else:
        work = df.copy()

    cut = max(1, int(len(work) * 0.8))
    test = work.iloc[cut:].copy()

    diff = pd.to_numeric(test[col], errors="coerce")
    pred = np.where(diff > 0, "HOME_WIN",
           np.where(diff < 0, "AWAY_WIN", "DRAW"))
    pred = pd.Series(pred, index=test.index)

    valid = diff.notna()
    y = test.loc[valid, "match_result"]
    p = pred.loc[valid]

    print("Ladder-position baseline")
    print(f"feature: {col}")
    print(f"hold-out rows scored: {len(y):,}")
    print(f"accuracy: {accuracy_score(y, p):.4f}")
    print(f"macro_f1: {f1_score(y, p, average='macro', zero_division=0):.4f}")
    print("\nCompare this output with afl_outputs/day2_match_metrics.csv.")
    print("Do not treat the benchmark as a public/official AFL standard.")

if __name__ == "__main__":
    main()
