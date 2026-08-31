# Generated from: adult_income_baselines_error_analysis.ipynb
# Converted at: 2026-08-31T08:02:16.934Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # UCI Adult Income — Baselines, Hold-out Evaluation & Initial Error Analysis
# 
# **Goal:** Predict whether annual income is **> $50K**. The notebook keeps the final 20% hold-out untouched until the end, uses a fixed random seed, evaluates two simple baselines, and identifies data/feature issues for the next modeling iteration.
# 
# **Primary metric:** **Precision**. In the stakeholder scenario used here, a positive prediction represents a person selected for higher-value outreach. False positives waste outreach effort, so we prioritize making contacted users likely to be genuinely >$50K. Recall remains a secondary metric so we can monitor how many positive cases we miss.
# 


# Reproducibility / imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEV_SIZE_WITHIN_REMAINDER = 0.125  # 10% of total, leaving ~70% train

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)


# ## Task 1 — Problem definition & success metric
# 
# **Target:** `income > 50K` → positive class `1`; `income <= 50K` → `0`.
# 
# **Business objective:** If the model is used to identify people for targeted, higher-value outreach, we want the people we contact to be genuinely likely to earn >$50K. Therefore, **precision** is the primary metric: higher precision means fewer wasted contacts. The trade-off is lower recall may miss some high-income people, but that is acceptable for this use case while we are optimizing outreach efficiency.
# 


# Load the UCI Adult / Census Income dataset from OpenML.
# fetch_openml caches the dataset locally after the first download.
adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame.copy()

print("Raw shape:", df.shape)
display(df.head())
print("\nRaw dtypes:")
display(df.dtypes.to_frame("dtype"))


# Clean whitespace and missing-value markers.
# OpenML usually supplies NaN already, but this also handles raw-UCI-style "?" / " ?".
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()

df = df.replace({"?": np.nan, "": np.nan})

# Normalize target and create binary target.
target_col = "class" if "class" in df.columns else "income"
df[target_col] = df[target_col].astype("string").str.replace(".", "", regex=False).str.strip()

df["target"] = df[target_col].map({">50K": 1, "<=50K": 0})

if df["target"].isna().any():
    # OpenML variants may use slightly different target column naming/formatting.
    unique_target = sorted(df[target_col].dropna().unique().tolist())
    print("Unrecognized target values:", unique_target)

# Drop rows with missing target only; feature missingness is handled in EDA for now.
df = df.dropna(subset=["target"]).copy()
df["target"] = df["target"].astype(int)

print("Cleaned shape:", df.shape)
print("Missing values by column:")
display(df.isna().sum().sort_values(ascending=False).to_frame("missing_count"))


# ### Class base rate
# 
# The positive-class base rate is the percentage of adults labeled `>50K`. A majority-class classifier will predict the negative class for everyone, so its accuracy will look high even though it finds **none** of the positive cases. This is why accuracy is not the primary decision metric here.
# 


class_counts = df["target"].value_counts().sort_index()
base_rate = df["target"].mean()

summary_class = pd.DataFrame({
    "class": [0, 1],
    "label": ["<=50K", ">50K"],
    "count": [class_counts.get(0, 0), class_counts.get(1, 0)],
})
summary_class["percent"] = summary_class["count"] / len(df) * 100

display(summary_class)
print(f"Positive-class base rate (>50K): {base_rate:.2%}")


# ## Task 2 — Quick EDA
# 
# We inspect shape, data types, numerical summaries, categorical value counts, missingness, and several feature distributions. The plots are saved so the notebook remains easy to review later.
# 


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

print("Shape:", df.shape)
print("\nNumeric columns:", numeric_cols)
print("\nCategorical columns:", categorical_cols)

display(df[numeric_cols].describe().T)

for col in categorical_cols:
    print(f"\n=== {col} ===")
    display(df[col].value_counts(dropna=False).head(15).to_frame("count"))


# Numeric histograms
plot_dir = Path("adult_income_plots")
plot_dir.mkdir(exist_ok=True)

plot_numeric = [c for c in ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"] if c in df.columns]

for col in plot_numeric:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col].dropna(), bins=30, kde=True)
    plt.title(f"Distribution: {col}")
    plt.tight_layout()
    plt.savefig(plot_dir / f"hist_{col}.png", dpi=150)
    plt.show()

# Categorical bar plots for a few high-value features
plot_cats = [c for c in ["education", "marital-status", "workclass", "occupation"] if c in df.columns]

for col in plot_cats:
    counts = df[col].fillna("Missing").value_counts().head(12)
    plt.figure(figsize=(8, 5))
    counts.sort_values().plot(kind="barh")
    plt.title(f"Top categories: {col}")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / f"bar_{col}.png", dpi=150)
    plt.show()


# Summary table with class counts and at least 3 notable feature distributions.
feature_summary = []

for col in ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]:
    if col in df.columns:
        s = df[col].dropna()
        feature_summary.append({
            "feature": col,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "p25": s.quantile(.25),
            "p75": s.quantile(.75),
            "missing": df[col].isna().sum()
        })

feature_summary = pd.DataFrame(feature_summary)

summary_dir = Path("adult_income_outputs")
summary_dir.mkdir(exist_ok=True)

summary_class.to_csv(summary_dir / "class_summary.csv", index=False)
feature_summary.to_csv(summary_dir / "feature_distribution_summary.csv", index=False)

display(summary_class)
display(feature_summary)


# ## Task 3 — Reproducible train / development / hold-out split
# 
# The final hold-out set is created first and is **not used for rule selection or tuning**. The remaining data are split into training and development sets for experimentation.
# 
# If the hold-out set is repeatedly inspected while choosing features or hyperparameters, information about the final test distribution leaks into the modeling process. The reported test score then becomes optimistic and no longer represents a clean estimate of performance on unseen data.
# 


# Separate features and target.
X = df.drop(columns=["target"])
y = df["target"]

# Final hold-out: ~20%, stratified and fixed seed.
X_devpool, X_test, y_devpool, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Optional development set: ~10% of total; train is ~70% total.
X_train, X_dev, y_train, y_dev = train_test_split(
    X_devpool, y_devpool,
    test_size=DEV_SIZE_WITHIN_REMAINDER,
    stratify=y_devpool,
    random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape[0]:,} ({len(X_train)/len(df):.1%})")
print(f"Dev:   {X_dev.shape[0]:,} ({len(X_dev)/len(df):.1%})")
print(f"Test:  {X_test.shape[0]:,} ({len(X_test)/len(df):.1%})")

print("\nClass rates:")
print("Train:", y_train.mean())
print("Dev:  ", y_dev.mean())
print("Test: ", y_test.mean())


# ## Task 4 — Simple baselines
# 
# Two intentionally simple baselines are evaluated:
# 
# 1. **Majority class:** always predicts `0` (`<=50K`), because it is the most common class.
# 2. **Education rule:** predicts `1` when `education-num >= 13`.
# 
# The education rule is a transparent business-style heuristic: `education-num >= 13` corresponds roughly to a bachelor's degree or higher and is plausibly associated with higher income. It is deliberately simple so that later models have to demonstrate real improvement over a meaningful reference point.
# 


# Baseline predictions on the untouched hold-out.
majority_class = int(y_train.mode().iloc[0])
pred_majority = np.full(len(y_test), majority_class, dtype=int)
score_majority = pred_majority.astype(float)

education_threshold = 13
pred_education = (
    pd.to_numeric(X_test["education-num"], errors="coerce")
    .fillna(0)
    .ge(education_threshold)
    .astype(int)
    .to_numpy()
)
score_education = pred_education.astype(float)

def evaluate_baseline(name, y_true, y_pred, y_score):
    return {
        "baseline": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }

results = pd.DataFrame([
    evaluate_baseline("Majority class", y_test, pred_majority, score_majority),
    evaluate_baseline("Education-num >= 13", y_test, pred_education, score_education),
])

display(results.style.format({
    "accuracy":"{:.3f}", "precision":"{:.3f}", "recall":"{:.3f}",
    "f1":"{:.3f}", "roc_auc":"{:.3f}", "pr_auc":"{:.3f}"
}))

results.to_csv(summary_dir / "baseline_metrics.csv", index=False)


# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, name, pred in zip(
    axes,
    ["Majority class", "Education-num >= 13"],
    [pred_majority, pred_education]
):
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["<=50K", ">50K"], cmap="Blues", ax=ax
    )
    ax.set_title(name)

plt.tight_layout()
plt.savefig(plot_dir / "baseline_confusion_matrices.png", dpi=160)
plt.show()


# ### Baseline interpretation
# 
# The majority baseline is useful mainly as a sanity check: it can achieve high accuracy because the negative class dominates, but its precision, recall, F1 and PR AUC for the positive class are effectively zero. The education rule should be substantially more useful because it identifies a subset enriched for the positive class, trading coverage for better precision. A real model should therefore beat the education heuristic on the **primary precision objective** while also providing materially better recall/F1 and PR AUC; merely matching majority-class accuracy would not be considered useful.
# 


# ## Task 5 — Initial error analysis
# 
# We inspect at least 10 false positives and 10 false negatives from the education rule. Because this is a rule-based baseline, these examples help reveal where the one-feature heuristic is too coarse.
# 


# Build an error-analysis table from the education-rule baseline.
error_df = X_test.copy()
error_df["actual"] = y_test.values
error_df["predicted"] = pred_education
error_df["error_type"] = np.select(
    [
        (error_df["actual"] == 0) & (error_df["predicted"] == 1),
        (error_df["actual"] == 1) & (error_df["predicted"] == 0),
    ],
    ["false_positive", "false_negative"],
    default="correct"
)

fp = error_df[error_df["error_type"] == "false_positive"].copy()
fn = error_df[error_df["error_type"] == "false_negative"].copy()

print("False positives:", len(fp))
display(fp.head(10))

print("False negatives:", len(fn))
display(fn.head(10))

fp.head(10).to_csv(summary_dir / "false_positive_sample.csv", index=False)
fn.head(10).to_csv(summary_dir / "false_negative_sample.csv", index=False)


# Compact quantitative comparison of FP/FN patterns.
def error_profile(frame, name):
    out = {"error_type": name, "n": len(frame)}
    for col in ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]:
        if col in frame.columns:
            out[f"{col}_median"] = pd.to_numeric(frame[col], errors="coerce").median()
    for col in ["workclass", "education", "marital-status", "occupation", "relationship", "sex"]:
        if col in frame.columns:
            mode = frame[col].mode(dropna=True)
            out[f"{col}_top"] = mode.iloc[0] if len(mode) else "Missing"
    return out

error_profile_table = pd.DataFrame([
    error_profile(fp, "False positives"),
    error_profile(fn, "False negatives")
])
display(error_profile_table)
error_profile_table.to_csv(summary_dir / "error_profile.csv", index=False)


# ### Concrete issues to address tomorrow
# 
# 1. **Categorical missing values:** impute or explicitly encode missing workclass/occupation/native-country rather than dropping rows.
# 2. **Categorical encoding:** use one-hot encoding for nominal variables and keep train/test columns aligned.
# 3. **Skewed capital features:** `capital-gain` and `capital-loss` are highly zero-inflated/skewed; consider binary indicators and/or log transforms.
# 4. **Nonlinear interactions:** income depends on combinations such as education × occupation × age × hours/week, which a single threshold cannot capture.
# 5. **Class imbalance:** evaluate precision/recall and PR AUC rather than relying on accuracy; consider class weighting if appropriate.
# 6. **Potentially sensitive attributes:** race and sex can affect predictions and fairness; treat them deliberately and report subgroup performance before any real deployment.
# 


# ## Primary metric for the rest of the week
# 
# **Primary metric: Precision.** The majority baseline demonstrates why accuracy is misleading: it can look strong while producing no positive predictions at all. The education threshold provides a more meaningful positive-class baseline, but it still leaves substantial room to improve precision while recovering more true >$50K cases. For the remaining modeling work, precision should be the optimization target, with **recall, F1, ROC AUC and especially PR AUC** retained as supporting diagnostics. The final hold-out should remain untouched until the chosen model and preprocessing pipeline are frozen.
# 


# Optional: generate a 1-page PDF summary after running the notebook.
# This cell uses the ACTUAL computed baseline metrics, class rate, and error counts.
from matplotlib.backends.backend_pdf import PdfPages

pdf_out = summary_dir / "adult_income_1page_summary.pdf"

m = results.set_index("baseline")
maj = m.loc["Majority class"]
edu = m.loc["Education-num >= 13"]

fig = plt.figure(figsize=(8.27, 11.69))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")

text = f'''
UCI ADULT INCOME — BASELINES & INITIAL ERROR ANALYSIS

Problem framing
Predict whether annual income exceeds $50K. For targeted higher-value outreach,
a positive prediction means a person is selected for contact. The business objective
is to keep contacted users likely to be >$50K, so precision is the primary metric.

Dataset / split
Rows: {len(df):,}    Positive rate: {base_rate:.2%}
Train: {len(X_train):,}    Dev: {len(X_dev):,}    Final hold-out: {len(X_test):,}
Split: stratified, random_state={RANDOM_STATE}; hold-out is used only for final evaluation.

Baseline results (hold-out)
                     Accuracy  Precision  Recall   F1     ROC AUC  PR AUC
Majority class        {maj.accuracy:.3f}     {maj.precision:.3f}    {maj.recall:.3f}  {maj.f1:.3f}  {maj.roc_auc:.3f}    {maj.pr_auc:.3f}
Education >= 13       {edu.accuracy:.3f}     {edu.precision:.3f}    {edu.recall:.3f}  {edu.f1:.3f}  {edu.roc_auc:.3f}    {edu.pr_auc:.3f}

Error analysis
Education-rule false positives: {len(fp):,}
Education-rule false negatives: {len(fn):,}
The rule is intentionally coarse: education alone cannot represent the combined
effects of age, occupation, marital status, hours worked, capital gains/losses,
and other variables.

Next steps
• Impute/encode missing categorical values consistently.
• One-hot encode nominal features.
• Address zero-inflated/skewed capital-gain and capital-loss.
• Add interaction/nonlinear features or a tree-based model.
• Monitor class imbalance with precision/recall and PR AUC.
• Evaluate subgroup/fairness behavior before any real-world deployment.

Success criterion
Optimize PRECISION on the development set while monitoring recall, F1 and PR AUC.
A useful model must materially outperform the education heuristic on the primary
business objective—not merely beat the majority baseline on accuracy.
'''

ax.text(0.055, 0.96, text, va="top", ha="left", fontsize=9.2, family="DejaVu Sans",
        linespacing=1.35)

with PdfPages(pdf_out) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
plt.close(fig)

print("Generated:", pdf_out)
