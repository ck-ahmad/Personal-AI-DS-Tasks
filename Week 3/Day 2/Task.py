
"""
AFL Week 3 Day 2 — Prediction Models
Match Winner + Top Player

Run:
    python afl_week3_day2_models.py

This script uses the leakage-safe Day 1 artifacts:
    afl_outputs/afl_feature_table_v1.csv
    afl_outputs/player_feature_table.csv

It creates:
    afl_models/match_winner_pipeline.joblib
    afl_models/top_player_disposals_pipeline.joblib
    afl_models/top_player_goals_pipeline.joblib  (if goals exist)
    afl_models/model_metadata.joblib
    afl_outputs/day2_match_metrics.csv
    afl_outputs/day2_player_metrics.csv
    afl_outputs/day2_feature_importance.csv
    afl_outputs/day2_sniff_test.csv
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, top_k_accuracy_score,
    ndcg_score
)

warnings.filterwarnings("ignore")

BASE = Path(".")
DATA_DIR = BASE / "afl_datasets"
OUTPUT_DIR = BASE / "afl_outputs"
MODEL_DIR = BASE / "afl_models"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_FILE = OUTPUT_DIR / "afl_feature_table_v1.csv"
PLAYER_FEATURE_FILE = OUTPUT_DIR / "player_feature_table.csv"
PLAYER_TARGET_FILE = OUTPUT_DIR / "player_match_targets.csv"


def load_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURE_FILE} not found. Run Week 3 Day 1 first."
        )
    return pd.read_csv(FEATURE_FILE, low_memory=False)


def choose_match_features(df):
    """Only use Day-1 engineered, pre-match features.

    Raw team names/IDs, dates, scores and targets are excluded to avoid
    memorisation/leakage. Engineered form/H2H/rest/ladder features are used.
    """
    allowed = [
        c for c in df.columns
        if any(k in c for k in [
            "win_rate", "score_for_avg", "score_against_avg",
            "win_streak", "days_rest", "h2h_", "form_diff",
            "score_form_diff", "ladder_position_diff"
        ])
    ]
    # Explicitly exclude targets and post-match fields.
    bad = {
        "match_result", "home_win", "match_margin",
        "_home_score", "_away_score", "_match_key"
    }
    return [c for c in allowed if c not in bad]


def time_split(df):
    d = pd.to_datetime(df["_date"], errors="coerce") if "_date" in df else pd.Series(pd.NaT, index=df.index)
    if d.notna().sum() == 0:
        # Day-1 normally contains _date. Fallback to last 20%.
        cut = max(1, int(len(df) * 0.8))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # Hold out the latest season if season exists; otherwise latest 20% chronologically.
    work = df.copy()
    work["_split_date"] = d
    work = work.sort_values("_split_date")
    if "season" in work.columns and work["season"].nunique(dropna=True) >= 2:
        seasons = sorted(work["season"].dropna().unique())
        test_season = seasons[-1]
        test = work[work["season"] == test_season].copy()
        train = work[work["season"] < test_season].copy()
        if len(train) >= 10 and len(test) >= 5:
            return train, test

    cut = max(1, int(len(work) * 0.8))
    return work.iloc[:cut].copy(), work.iloc[cut:].copy()


def make_preprocessor(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat))
    return ColumnTransformer(transformers=transformers)


def evaluate_classifier(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "brier_home_win": np.nan,
        "roc_auc_ovr_macro": np.nan,
    }

    if "HOME_WIN" in classes:
        home_idx = classes.index("HOME_WIN")
        y_bin = (pd.Series(y_test).values == "HOME_WIN").astype(int)
        metrics["brier_home_win"] = brier_score_loss(y_bin, proba[:, home_idx])

    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_test, proba, multi_class="ovr", average="macro",
            labels=classes
        )
    except Exception:
        pass
    return metrics


def train_match_models():
    df = load_data()
    features = choose_match_features(df)
    if not features:
        raise ValueError("No Day-1 engineered match features were found.")

    # Day 1 target is 3-class: HOME_WIN / AWAY_WIN / DRAW.
    if "match_result" not in df:
        raise ValueError("match_result target missing from Day-1 feature table.")

    train, test = time_split(df)
    X_train, y_train = train[features], train["match_result"]
    X_test, y_test = test[features], test["match_result"]

    pre = make_preprocessor(X_train)

    models = {
        "LogisticRegression": Pipeline([
            ("prep", pre),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "GradientBoosting": Pipeline([
            ("prep", pre),
            ("model", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.04,
                max_depth=2, random_state=42
            ))
        ]),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        m = evaluate_classifier(model, X_test, y_test)
        m["model"] = name
        rows.append(m)
        fitted[name] = model

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "day2_match_metrics.csv", index=False)

    # Pick by macro F1 first, then Brier (lower is better).
    best_name = (
        metrics.sort_values(
            ["f1_macro", "brier_home_win"],
            ascending=[False, True]
        ).iloc[0]["model"]
    )
    best = fitted[best_name]
    joblib.dump(best, MODEL_DIR / "match_winner_pipeline.joblib")

    # Feature importance / coefficients.
    prep = best.named_steps["prep"]
    estimator = best.named_steps["model"]
    names = prep.get_feature_names_out()

    if hasattr(estimator, "coef_"):
        # Mean absolute coefficient across classes.
        values = np.mean(np.abs(estimator.coef_), axis=0)
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = np.zeros(len(names))

    fi = pd.DataFrame({
        "feature": names,
        "importance": values
    }).sort_values("importance", ascending=False)
    fi["model"] = best_name
    fi.to_csv(OUTPUT_DIR / "day2_feature_importance.csv", index=False)

    print("\n=== MATCH WINNER MODELS ===")
    print(metrics.round(4).to_string(index=False))
    print(f"\nFinal model: {best_name}")
    print(f"Train rows: {len(train):,} | Hold-out rows: {len(test):,}")
    print("\nTop match features:")
    print(fi.head(15).to_string(index=False))

    return df, train, test, features, best_name


def load_player_data():
    if not PLAYER_FEATURE_FILE.exists():
        raise FileNotFoundError(f"{PLAYER_FEATURE_FILE} not found. Run Day 1.")
    if not PLAYER_TARGET_FILE.exists():
        raise FileNotFoundError(f"{PLAYER_TARGET_FILE} not found. Run Day 1.")
    pf = pd.read_csv(PLAYER_FEATURE_FILE, low_memory=False)
    targets = pd.read_csv(PLAYER_TARGET_FILE, low_memory=False)
    return pf, targets


def player_holdout(pf):
    pf["_date"] = pd.to_datetime(pf["_date"], errors="coerce")
    pf = pf.sort_values(["_date", "_match_key"])
    # Use latest 20% by chronological match rows.
    dates = pf["_date"].dropna().sort_values()
    if len(dates) == 0:
        cut = int(len(pf) * .8)
        return pf.iloc[:cut], pf.iloc[cut:]
    unique_dates = dates.drop_duplicates().tolist()
    cut_date = unique_dates[max(0, int(len(unique_dates) * .8) - 1)]
    train = pf[pf["_date"] <= cut_date].copy()
    test = pf[pf["_date"] > cut_date].copy()
    return train, test

def train_top_player(stat="disposals"):
    print(f"\n=== TOP PLAYER MODEL: {stat.upper()} ===")

    raw_path = DATA_DIR / "afl_players_round_by_round_stats_raw.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing: {raw_path}")

    df = pd.read_csv(raw_path, low_memory=False)

    # Your actual AFL dataset columns
    required = ["player_id", "match_date", stat, "team", "opponent"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Clean
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df[stat] = pd.to_numeric(df[stat], errors="coerce")

    df = df.dropna(subset=["player_id", "match_date"])

    # Match key based on actual dataset
    df["_match_key"] = (
        df["match_date"].dt.strftime("%Y-%m-%d")
        + "__"
        + df["team"].astype(str)
        + "__"
        + df["opponent"].astype(str)
    )

    # Sort chronologically
    df = df.sort_values(
        ["player_id", "match_date", "id"]
    ).reset_index(drop=True)

    # Leakage-safe previous performance features
    feature_stats = [
        "disposals",
        "goals",
        "kicks",
        "marks",
        "handballs",
        "tackles",
        "fantasy_points",
    ]

    feature_cols = []

    for s in feature_stats:
        if s not in df.columns:
            continue

        df[s] = pd.to_numeric(df[s], errors="coerce")

        for window in [3, 5]:
            col = f"{s}_avg_l{window}"

            df[col] = (
                df.groupby("player_id")[s]
                .transform(
                    lambda x: x.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
            )

            feature_cols.append(col)

    # Only keep rows where target exists
    df = df.dropna(subset=[stat]).copy()

    # We need at least one historical feature
    feature_cols = [
        c for c in feature_cols
        if c in df.columns
    ]

    if not feature_cols:
        raise ValueError("No historical player features available.")

    # Chronological holdout
    dates = df["match_date"].sort_values().unique()

    if len(dates) < 2:
        raise ValueError("Not enough dates for train/holdout split.")

    cutoff = dates[int(len(dates) * 0.80)]

    train = df[df["match_date"] < cutoff].copy()
    test = df[df["match_date"] >= cutoff].copy()

    # Remove rows where all historical features are missing
    train = train.dropna(subset=feature_cols, how="all")
    test = test.dropna(subset=feature_cols, how="all")

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty player train/test split.")

    X_train = train[feature_cols].copy()
    y_train = train[stat].copy()

    X_test = test[feature_cols].copy()
    y_test = test[stat].copy()

    # Fill missing historical values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        max_depth=12,
        min_samples_leaf=3,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # ---------------------------------------------------------
    # TOP-5 HIT RATE
    # ---------------------------------------------------------

    test_eval = test[
        [
            "_match_key",
            "player_id",
            "team",
            "opponent",
            stat,
        ]
    ].copy()

    test_eval["prediction"] = predictions

    hits = []
    baseline_hits = []

    for match_key, group in test_eval.groupby("_match_key"):

        if len(group) == 0:
            continue

        # Actual top 5
        actual_top5 = set(
            group.nlargest(5, stat)["player_id"]
        )

        # Predicted top 5
        predicted_top5 = set(
            group.nlargest(5, "prediction")["player_id"]
        )

        hits.append(
            len(actual_top5.intersection(predicted_top5))
            / min(5, len(actual_top5))
        )

        # Baseline = previous L5 average
        baseline_col = f"{stat}_avg_l5"

        if baseline_col in group.columns:
            baseline_top5 = set(
                group.nlargest(
                    5,
                    baseline_col
                )["player_id"]
            )

            baseline_hits.append(
                len(actual_top5.intersection(baseline_top5))
                / min(5, len(actual_top5))
            )

    top5_hit_rate = (
        np.mean(hits)
        if hits
        else np.nan
    )

    baseline_top5_hit_rate = (
        np.mean(baseline_hits)
        if baseline_hits
        else np.nan
    )

    print(f"Train rows: {len(train):,}")
    print(f"Hold-out rows: {len(test):,}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Top-5 hit rate: {top5_hit_rate:.4f}")
    print(
        f"Baseline Top-5 hit rate: "
        f"{baseline_top5_hit_rate:.4f}"
    )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False
    )

    importance["model"] = "RandomForestRegressor"
    importance["stat"] = stat

    importance.to_csv(
        OUTPUT_DIR / f"day2_player_{stat}_feature_importance.csv",
        index=False
    )

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"top_player_{stat}_pipeline.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "stat": stat,
        },
        model_path,
    )

    # ---------------------------------------------------------
    # SAVE METRICS
    # ---------------------------------------------------------

    metrics = pd.DataFrame([
        {
            "stat": stat,
            "model": "RandomForestRegressor",
            "train_rows": len(train),
            "holdout_rows": len(test),
            "MAE": mae,
            "RMSE": rmse,
            "top5_hit_rate": top5_hit_rate,
            "baseline_top5_hit_rate": baseline_top5_hit_rate,
        }
    ])

    metrics.to_csv(
        OUTPUT_DIR / f"day2_player_{stat}_metrics.csv",
        index=False
    )

    print(f"\nSaved model:")
    print(model_path)

    return model, metrics


def sniff_test(df, match_test, model, features):
    """Show three held-out matches with actual result and model probabilities."""
    if match_test.empty:
        return
    sample = match_test.sort_values("_date").tail(3).copy()
    rows = []
    for _, r in sample.iterrows():
        X = pd.DataFrame([{c: r[c] for c in features}])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        cls = list(model.classes_)
        rows.append({
            "match_id": r.get("_match_key", ""),
            "home_team": r.get("home_team", r.get("home_team_name", "")),
            "away_team": r.get("away_team", r.get("away_team_name", "")),
            "actual": r["match_result"],
            "prediction": pred,
            "home_probability": proba[cls.index("HOME_WIN")] if "HOME_WIN" in cls else np.nan,
            "away_probability": proba[cls.index("AWAY_WIN")] if "AWAY_WIN" in cls else np.nan,
            "draw_probability": proba[cls.index("DRAW")] if "DRAW" in cls else np.nan,
            "manual_reason": "Check recent form, rest, H2H and ladder context manually.",
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "day2_sniff_test.csv", index=False)
    print("\n=== SNIFF TEST: 3 HELD-OUT MATCHES ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    df, train, test, features, best_name = train_match_models()
    best = joblib.load(MODEL_DIR / "match_winner_pipeline.joblib")
    sniff_test(df, test, best, features)

    train_top_player("disposals")

    # Train goals model too when the Day-1 source contains goal data.
    try:
        train_top_player("goals")
    except Exception as e:
        print(f"\nGoal model skipped: {e}")

    meta = {
        "match_features": features,
        "final_match_model": best_name,
        "target": "match_result",
        "classes": list(best.classes_),
        "notes": "All Day-1 rolling features are pre-match and shifted.",
    }
    joblib.dump(meta, MODEL_DIR / "model_metadata.joblib")

    print("\nDONE. Model artifacts saved in:", MODEL_DIR.resolve())


if __name__ == "__main__":
    main()
