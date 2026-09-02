# Adult Income Prediction — Day 3
# Feature Engineering + Cross-Validation + Statistical Testing

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, wilcoxon

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import make_scorer, precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
PRIMARY_METRIC = "precision"

OUTPUT_DIR = Path("adult_income_day3_outputs")
PLOT_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)

print("=" * 78)
print("ADULT INCOME — DAY 3")
print("FEATURE ENGINEERING + CROSS-VALIDATION")
print("=" * 78)

# ------------------------------------------------------------
# 1. Load and clean data
# ------------------------------------------------------------

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.copy()

for col in df.select_dtypes(include=["object", "category", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()

df = df.replace({"?": np.nan, "": np.nan})

target_col = "class" if "class" in df.columns else "income"

df[target_col] = (
    df[target_col]
    .astype("string")
    .str.replace(".", "", regex=False)
    .str.strip()
)

df["target"] = df[target_col].map({"<=50K": 0, ">50K": 1})
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

original_feature_columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

X = df[original_feature_columns].copy()
y = df["target"].copy()

# Same deterministic Day 1 final hold-out split.
# Day 3 CV uses only the development pool.
X_devpool, X_test, y_devpool, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("Development pool:", len(X_devpool))
print("Untouched final hold-out:", len(X_test))
print("Positive rate in CV pool:", f"{y_devpool.mean():.2%}")

# ------------------------------------------------------------
# 2. Leakage-safe feature engineering
# ------------------------------------------------------------

class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates row-level engineered features only.
    No target statistics are used, so the transformation is leakage-safe.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # 1. Age bucket — captures nonlinear career/life-stage effects.
        X["age_bucket"] = pd.cut(
            X["age"],
            bins=[0, 24, 34, 44, 54, 64, np.inf],
            labels=["17-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            include_lowest=True,
        ).astype("string")

        # 2. Hours bucket — captures part-time / standard / overtime patterns.
        X["hours_bucket"] = pd.cut(
            X["hours-per-week"],
            bins=[0, 24, 39, 40, 49, 59, np.inf],
            labels=[
                "part_time",
                "below_standard",
                "standard_40",
                "moderate_overtime",
                "high_overtime",
                "very_high_hours",
            ],
            include_lowest=True,
        ).astype("string")

        # 3. Capital gain flag — any capital gain may be a strong wealth signal.
        X["has_capital_gain"] = (X["capital-gain"].fillna(0) > 0).astype(int)

        # 4. Log capital gain — reduces extreme skew.
        X["log_capital_gain"] = np.log1p(
            X["capital-gain"].clip(lower=0).fillna(0)
        )

        # 5. Capital loss flag — indicates whether capital loss exists.
        X["has_capital_loss"] = (X["capital-loss"].fillna(0) > 0).astype(int)

        # 6. Log capital loss — reduces skew.
        X["log_capital_loss"] = np.log1p(
            X["capital-loss"].clip(lower=0).fillna(0)
        )

        # 7. Higher education flag — interpretable bachelor-level-or-higher signal.
        X["higher_education"] = (X["education-num"].fillna(0) >= 13).astype(int)

        # 8. Education × hours interaction.
        X["education_hours_interaction"] = (
            X["education-num"].fillna(0) * X["hours-per-week"].fillna(0)
        )

        # 9. Age × hours interaction.
        X["age_hours_interaction"] = (
            X["age"].fillna(0) * X["hours-per-week"].fillna(0)
        )

        # 10. Net capital signal.
        X["capital_net"] = (
            X["capital-gain"].fillna(0) - X["capital-loss"].fillna(0)
        )

        # --------------------------------------------------------
        # IMPORTANT: sklearn's SimpleImputer expects NumPy-style
        # missing values. Pandas nullable dtypes may contain pd.NA,
        # which can trigger: "boolean value of NA is ambiguous".
        # Normalize every numeric column to float64 + np.nan and
        # every categorical column to object + np.nan.
        # --------------------------------------------------------
        for col in [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "has_capital_gain",
            "log_capital_gain",
            "has_capital_loss",
            "log_capital_loss",
            "higher_education",
            "education_hours_interaction",
            "age_hours_interaction",
            "capital_net",
        ]:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

        for col in [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "age_bucket",
            "hours_bucket",
        ]:
            # Convert pandas StringDtype / pd.NA to ordinary object + np.nan.
            X[col] = X[col].astype(object)
            X[col] = X[col].where(pd.notna(X[col]), np.nan)

        return X

feature_engineer = AdultFeatureEngineer()

# ------------------------------------------------------------
# 3. Feature dictionary + univariate signal
# ------------------------------------------------------------

feature_dictionary = pd.DataFrame([
    ["age_bucket", "categorical", "bucket age into career/life stages",
     "Income relationships with age are likely nonlinear."],
    ["hours_bucket", "categorical", "bucket hours-per-week into workload ranges",
     "Part-time, standard, and overtime work may differ in income."],
    ["has_capital_gain", "binary", "capital-gain > 0",
     "Any capital gain may indicate stronger financial resources."],
    ["log_capital_gain", "numeric", "log1p(capital-gain)",
     "Compresses a highly skewed financial feature."],
    ["has_capital_loss", "binary", "capital-loss > 0",
     "Presence of capital loss may carry useful financial information."],
    ["log_capital_loss", "numeric", "log1p(capital-loss)",
     "Compresses capital-loss skew while keeping magnitude."],
    ["higher_education", "binary", "education-num >= 13",
     "Creates an interpretable higher-education indicator."],
    ["education_hours_interaction", "numeric interaction",
     "education-num * hours-per-week",
     "Combines education level with work intensity."],
    ["age_hours_interaction", "numeric interaction",
     "age * hours-per-week",
     "Combines career stage with workload."],
    ["capital_net", "numeric", "capital-gain - capital-loss",
     "Captures net capital-related financial signal."],
], columns=["name", "type", "creation_rule", "justification"])

engineered_dev = feature_engineer.transform(X_devpool)

numeric_engineered = [
    "has_capital_gain",
    "log_capital_gain",
    "has_capital_loss",
    "log_capital_loss",
    "higher_education",
    "education_hours_interaction",
    "age_hours_interaction",
    "capital_net",
]

categorical_engineered = ["age_bucket", "hours_bucket"]

signals = []

for col in numeric_engineered:
    values = pd.to_numeric(engineered_dev[col], errors="coerce")
    values = values.fillna(values.median())

    is_discrete = col in {
        "has_capital_gain",
        "has_capital_loss",
        "higher_education",
    }

    score = mutual_info_classif(
        values.to_frame(),
        y_devpool,
        discrete_features=is_discrete,
        random_state=RANDOM_STATE,
    )[0]

    signals.append([col, "mutual_information", float(score)])

for col in categorical_engineered:
    temp = pd.DataFrame({
        col: engineered_dev[col].astype("string").fillna("Missing").values,
        "target": y_devpool.values,
    })

    group_rate = temp.groupby(col)["target"].mean()
    spread = float(group_rate.max() - group_rate.min())

    signals.append([col, "target_rate_spread", spread])

signal_df = pd.DataFrame(
    signals,
    columns=["name", "predictive_signal_method", "predictive_signal"],
)

feature_dictionary = feature_dictionary.merge(signal_df, on="name", how="left")

print("\nENGINEERED FEATURE DICTIONARY")
print(feature_dictionary.to_string(index=False))

feature_dictionary.to_csv(
    OUTPUT_DIR / "engineered_feature_dictionary.csv",
    index=False,
)

# ------------------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------------------

base_numeric = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

engineered_numeric = numeric_engineered
all_numeric = base_numeric + engineered_numeric

base_categorical = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

engineered_categorical = categorical_engineered
all_categorical = base_categorical + engineered_categorical

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, all_numeric),
    ("cat", categorical_pipeline, all_categorical),
])

# ------------------------------------------------------------
# 5. Three models with identical preprocessing
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        random_state=RANDOM_STATE,
    ),
}

pipelines = {
    name: Pipeline([
        ("feature_engineering", AdultFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    for name, model in models.items()
}

# ------------------------------------------------------------
# 6. Stratified 5-fold cross-validation
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "precision": make_scorer(precision_score, zero_division=0),
    "roc_auc": "roc_auc",
    "f1": "f1",
}

summary_rows = []
fold_rows = []

for model_name, pipeline in pipelines.items():
    print(f"\nCross-validating: {model_name}")

    start = time.perf_counter()

    result = cross_validate(
        pipeline,
        X_devpool,
        y_devpool,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    elapsed = time.perf_counter() - start

    summary_rows.append({
        "Model": model_name,
        "Precision Mean": result["test_precision"].mean(),
        "Precision Std": result["test_precision"].std(ddof=1),
        "ROC AUC Mean": result["test_roc_auc"].mean(),
        "ROC AUC Std": result["test_roc_auc"].std(ddof=1),
        "F1 Mean": result["test_f1"].mean(),
        "F1 Std": result["test_f1"].std(ddof=1),
        "CV Time Seconds": elapsed,
    })

    for fold in range(N_SPLITS):
        fold_rows.append({
            "Model": model_name,
            "Fold": fold + 1,
            "Precision": result["test_precision"][fold],
            "ROC AUC": result["test_roc_auc"][fold],
            "F1": result["test_f1"][fold],
        })

cv_summary = pd.DataFrame(summary_rows).sort_values(
    "Precision Mean",
    ascending=False,
)

fold_scores = pd.DataFrame(fold_rows)

print("\nCV SUMMARY")
print(cv_summary.round(4).to_string(index=False))

cv_summary.to_csv(OUTPUT_DIR / "cv_model_comparison.csv", index=False)
fold_scores.to_csv(OUTPUT_DIR / "cv_fold_scores.csv", index=False)

# ------------------------------------------------------------
# 7. Boxplots
# ------------------------------------------------------------

for metric in ["Precision", "ROC AUC", "F1"]:
    names = fold_scores["Model"].unique().tolist()

    data = [
        fold_scores.loc[fold_scores["Model"] == name, metric].values
        for name in names
    ]

    plt.figure(figsize=(9, 6))
    plt.boxplot(data, tick_labels=names, showmeans=True)
    plt.ylabel(metric)
    plt.title(f"5-Fold Cross-Validation — {metric}")
    plt.xticks(rotation=15)
    plt.tight_layout()

    filename = metric.lower().replace(" ", "_")
    plt.savefig(PLOT_DIR / f"cv_boxplot_{filename}.png", dpi=180)
    plt.show()
    plt.close()

# ------------------------------------------------------------
# 8. Paired statistical test — top 2 precision models
# ------------------------------------------------------------

top_two = cv_summary.head(2)["Model"].tolist()
model_a, model_b = top_two

a_scores = (
    fold_scores[fold_scores["Model"] == model_a]
    .sort_values("Fold")["Precision"]
    .to_numpy()
)

b_scores = (
    fold_scores[fold_scores["Model"] == model_b]
    .sort_values("Fold")["Precision"]
    .to_numpy()
)

t_stat, t_p = ttest_rel(a_scores, b_scores)

try:
    w_stat, w_p = wilcoxon(a_scores, b_scores)
except ValueError:
    w_stat, w_p = np.nan, np.nan

mean_difference = a_scores.mean() - b_scores.mean()

stat_df = pd.DataFrame([{
    "Top Model": model_a,
    "Second Model": model_b,
    "Metric": "Precision",
    "Top Mean": a_scores.mean(),
    "Second Mean": b_scores.mean(),
    "Mean Difference": mean_difference,
    "Paired t statistic": t_stat,
    "Paired t p-value": t_p,
    "Wilcoxon statistic": w_stat,
    "Wilcoxon p-value": w_p,
}])

print("\nSTATISTICAL COMPARISON")
print(stat_df.round(6).to_string(index=False))

if t_p < 0.05:
    interpretation = (
        f"The paired t-test suggests a statistically detectable precision "
        f"difference between {model_a} and {model_b} at alpha=0.05. "
        f"The mean difference is {mean_difference:.4f}; practical value "
        f"still depends on whether this improvement matters to the use case."
    )
else:
    interpretation = (
        f"The paired t-test does not provide strong evidence of a precision "
        f"difference between {model_a} and {model_b} at alpha=0.05. "
        f"The observed mean difference is {mean_difference:.4f}; with only "
        f"five folds, small differences should not be over-interpreted."
    )

print("\nInterpretation:")
print(interpretation)

stat_df["Interpretation"] = interpretation
stat_df.to_csv(OUTPUT_DIR / "paired_statistical_test.csv", index=False)

# ------------------------------------------------------------
# 9. Fit models on CV development pool for interpretability
# ------------------------------------------------------------

fitted = {}

for name, pipeline in pipelines.items():
    pipeline.fit(X_devpool, y_devpool)
    fitted[name] = pipeline

engineered_fragments = [
    "age_bucket",
    "hours_bucket",
    "has_capital_gain",
    "log_capital_gain",
    "has_capital_loss",
    "log_capital_loss",
    "higher_education",
    "education_hours_interaction",
    "age_hours_interaction",
    "capital_net",
]

# Logistic Regression coefficients
log_pipe = fitted["Logistic Regression"]
log_names = log_pipe.named_steps["preprocessor"].get_feature_names_out()
log_coef = log_pipe.named_steps["model"].coef_[0]

log_coef_df = pd.DataFrame({
    "Feature": log_names,
    "Coefficient": log_coef,
})

eng_log_coef = log_coef_df[
    log_coef_df["Feature"].apply(
        lambda x: any(fragment in x for fragment in engineered_fragments)
    )
].copy()

eng_log_coef["AbsoluteCoefficient"] = eng_log_coef["Coefficient"].abs()
eng_log_coef = eng_log_coef.sort_values(
    "AbsoluteCoefficient",
    ascending=False,
)

print("\nENGINEERED LOGISTIC COEFFICIENTS")
print(eng_log_coef.head(30).round(4).to_string(index=False))

eng_log_coef.to_csv(
    OUTPUT_DIR / "engineered_logistic_coefficients.csv",
    index=False,
)

# Random Forest importances
rf_pipe = fitted["Random Forest"]
rf_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
rf_importance = rf_pipe.named_steps["model"].feature_importances_

rf_df = pd.DataFrame({
    "Feature": rf_names,
    "Importance": rf_importance,
})

eng_rf = rf_df[
    rf_df["Feature"].apply(
        lambda x: any(fragment in x for fragment in engineered_fragments)
    )
].copy()

eng_rf = eng_rf.sort_values("Importance", ascending=False)

print("\nENGINEERED RANDOM FOREST IMPORTANCE")
print(eng_rf.head(30).round(5).to_string(index=False))

eng_rf.to_csv(
    OUTPUT_DIR / "engineered_random_forest_importance.csv",
    index=False,
)

# ------------------------------------------------------------
# 10. Feature selection with SelectKBest(mutual information)
# ------------------------------------------------------------

best_model_name = cv_summary.iloc[0]["Model"]
best_estimator = models[best_model_name]

transform_only = Pipeline([
    ("feature_engineering", AdultFeatureEngineer()),
    ("preprocessor", preprocessor),
])

processed_matrix = transform_only.fit_transform(X_devpool, y_devpool)
processed_feature_count = processed_matrix.shape[1]

K_BEST = min(
    60,
    max(10, int(processed_feature_count * 0.70)),
)

selector = SelectKBest(
    score_func=mutual_info_classif,
    k=K_BEST,
)

selected_pipeline = Pipeline([
    ("feature_engineering", AdultFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("feature_selection", selector),
    ("model", best_estimator),
])

full_pipeline = pipelines[best_model_name]

start = time.perf_counter()
full_result = cross_validate(
    full_pipeline,
    X_devpool,
    y_devpool,
    cv=cv,
    scoring=scoring,
    n_jobs=1,
)
full_time = time.perf_counter() - start

start = time.perf_counter()
selected_result = cross_validate(
    selected_pipeline,
    X_devpool,
    y_devpool,
    cv=cv,
    scoring=scoring,
    n_jobs=1,
)
selected_time = time.perf_counter() - start

selection_df = pd.DataFrame([
    {
        "Version": "All features",
        "Model": best_model_name,
        "Feature Count": processed_feature_count,
        "Precision Mean": full_result["test_precision"].mean(),
        "Precision Std": full_result["test_precision"].std(ddof=1),
        "ROC AUC Mean": full_result["test_roc_auc"].mean(),
        "F1 Mean": full_result["test_f1"].mean(),
        "CV Time Seconds": full_time,
    },
    {
        "Version": f"SelectKBest MI (k={K_BEST})",
        "Model": best_model_name,
        "Feature Count": K_BEST,
        "Precision Mean": selected_result["test_precision"].mean(),
        "Precision Std": selected_result["test_precision"].std(ddof=1),
        "ROC AUC Mean": selected_result["test_roc_auc"].mean(),
        "F1 Mean": selected_result["test_f1"].mean(),
        "CV Time Seconds": selected_time,
    },
])

print("\nFEATURE SELECTION COMPARISON")
print(selection_df.round(4).to_string(index=False))

selection_df.to_csv(
    OUTPUT_DIR / "feature_selection_comparison.csv",
    index=False,
)

# Export selected feature names
selected_pipeline.fit(X_devpool, y_devpool)

support = selected_pipeline.named_steps["feature_selection"].get_support()
processed_names = selected_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

selected_names = processed_names[support]

pd.DataFrame({
    "SelectedFeature": selected_names
}).to_csv(
    OUTPUT_DIR / "selected_features_for_tuning.csv",
    index=False,
)

precision_delta = (
    selection_df.iloc[1]["Precision Mean"]
    - selection_df.iloc[0]["Precision Mean"]
)

time_delta = (
    selection_df.iloc[1]["CV Time Seconds"]
    - selection_df.iloc[0]["CV Time Seconds"]
)

if precision_delta >= -0.002:
    feature_selection_decision = (
        f"SelectKBest is worth keeping as a Day 4 candidate because "
        f"precision changed by {precision_delta:+.4f} while reducing "
        f"features from {processed_feature_count} to {K_BEST}."
    )
else:
    feature_selection_decision = (
        f"Keep the full engineered feature set because SelectKBest reduced "
        f"precision by {abs(precision_delta):.4f}, which is not desirable "
        f"for this precision-first problem."
    )

# ------------------------------------------------------------
# 11. Final recommendation
# ------------------------------------------------------------

recommendation = f"""
DAY 3 RECOMMENDATION
====================

Primary metric:
Precision

Best model by mean 5-fold CV precision:
{best_model_name}

Top two models:
1. {model_a}
2. {model_b}

Paired t-test p-value:
{t_p:.6f}

Mean precision difference:
{mean_difference:+.4f}

Statistical interpretation:
{interpretation}

Feature-selection decision:
{feature_selection_decision}

Training-time change with SelectKBest:
{time_delta:+.2f} seconds across the 5-fold comparison.

Recommended Day 4 direction:
- Tune {best_model_name} first.
- Keep {model_b if best_model_name == model_a else model_a} as a secondary candidate.
- Keep engineered features that show both univariate signal and model-level importance.
- Continue using CV for tuning.
- Keep the final Day 1 hold-out completely untouched until tuning is finished.
"""

print("\n" + recommendation)

(OUTPUT_DIR / "day3_recommendation.txt").write_text(
    recommendation,
    encoding="utf-8",
)

print("\n" + "=" * 78)
print("DAY 3 COMPLETE")
print("=" * 78)
print("Outputs saved to:", OUTPUT_DIR)
print("Final hold-out used for tuning? NO")
