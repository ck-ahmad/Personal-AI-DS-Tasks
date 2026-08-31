# ============================================================
# UCI ADULT INCOME
# Baselines, Hold-out Evaluation, ML Model & Error Analysis
# ============================================================

# Goal:
# Predict whether annual income is > $50K.
#
# Primary business metric:
# PRECISION
#
# A positive prediction means that the person is selected for
# higher-value outreach. Higher precision means fewer wasted
# outreach contacts.
# ============================================================


# ============================================================
# 1. IMPORTS & REPRODUCIBILITY
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from IPython.display import display

from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from matplotlib.backends.backend_pdf import PdfPages


# Fixed random seed
RANDOM_STATE = 42

TEST_SIZE = 0.20

# 12.5% of the remaining 80% = 10% of total dataset
DEV_SIZE_WITHIN_REMAINDER = 0.125


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)


# Output folders
plot_dir = Path("adult_income_plots")
summary_dir = Path("adult_income_outputs")

plot_dir.mkdir(exist_ok=True)
summary_dir.mkdir(exist_ok=True)


print("=" * 70)
print("UCI ADULT INCOME — BASELINES + ML MODEL")
print("=" * 70)


# ============================================================
# 2. TASK 1 — PROBLEM DEFINITION
# ============================================================

print("\n" + "=" * 70)
print("TASK 1 — PROBLEM DEFINITION & SUCCESS METRIC")
print("=" * 70)

print("""
Target:
    >50K  -> Positive class (1)
    <=50K -> Negative class (0)

Business objective:
    Identify people who are likely to earn more than $50K
    for targeted higher-value outreach.

Primary metric:
    PRECISION

Why precision?
    A false positive means contacting someone who does not belong
    to the >50K group. This wastes outreach resources.

    Recall is still monitored because missing a high-income person
    is also important, but precision is the primary business goal.
""")


# ============================================================
# 3. TASK 2 — LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("TASK 2 — DATA LOADING")
print("=" * 70)

print("\nLoading UCI Adult dataset from OpenML...")

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy()

print("\nRaw shape:", df.shape)

display(df.head())


# ============================================================
# 4. CLEAN DATA
# ============================================================

print("\nCleaning data...")

# Strip whitespace from object/string columns
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()


# Convert ? / empty strings to NaN
df = df.replace({
    "?": np.nan,
    "": np.nan
})


# Target column
target_col = "class" if "class" in df.columns else "income"


# Clean target labels
df[target_col] = (
    df[target_col]
    .astype("string")
    .str.replace(".", "", regex=False)
    .str.strip()
)


# Convert target to binary
df["target"] = df[target_col].map({
    ">50K": 1,
    "<=50K": 0
})


# Check unknown target values
if df["target"].isna().any():

    print("\nWARNING: Unknown target values found:")

    print(
        df.loc[
            df["target"].isna(),
            target_col
        ].value_counts(dropna=False)
    )


# Remove rows where target is missing
df = df.dropna(
    subset=["target"]
).copy()


# Convert target to integer
df["target"] = df["target"].astype(int)


print("\nCleaned shape:", df.shape)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\nMissing values by column:")

missing_summary = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .to_frame("missing_count")
)

display(missing_summary)


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


class_counts = (
    df["target"]
    .value_counts()
    .sort_index()
)

base_rate = df["target"].mean()


summary_class = pd.DataFrame({
    "class": [0, 1],
    "label": ["<=50K", ">50K"],
    "count": [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
})


summary_class["percent"] = (
    summary_class["count"] /
    len(df) *
    100
)


display(summary_class)

print(
    f"\nPositive-class base rate (>50K): "
    f"{base_rate:.2%}"
)


summary_class.to_csv(
    summary_dir / "class_summary.csv",
    index=False
)


# ============================================================
# 7. TASK 2 — DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)


display(
    df.dtypes.to_frame("dtype")
)


# IMPORTANT:
# Do not include target or original class label in feature analysis.

feature_df = df.drop(
    columns=["target", target_col]
)


numeric_cols = (
    feature_df
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)


categorical_cols = [
    c for c in feature_df.columns
    if c not in numeric_cols
]


print("\nNumeric columns:")
print(numeric_cols)


print("\nCategorical columns:")
print(categorical_cols)


# ============================================================
# 8. NUMERIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC SUMMARY")
print("=" * 70)


numeric_summary = (
    feature_df[numeric_cols]
    .describe()
    .T
)


display(numeric_summary)


# ============================================================
# 9. CATEGORICAL VALUE COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL VALUE COUNTS")
print("=" * 70)


for col in categorical_cols:

    print(f"\n=== {col} ===")

    counts = (
        feature_df[col]
        .astype("string")
        .fillna("Missing")
        .value_counts()
        .head(15)
    )

    display(
        counts.to_frame("count")
    )


# ============================================================
# 10. HISTOGRAMS
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC VISUALIZATIONS")
print("=" * 70)


plot_numeric = [
    "age",
    "education-num",
    "hours-per-week",
    "capital-gain",
    "capital-loss"
]


for col in plot_numeric:

    if col not in feature_df.columns:
        continue

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        feature_df[col].dropna(),
        bins=30,
        kde=True
    )

    plt.title(
        f"Distribution of {col}"
    )

    plt.xlabel(col)
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        plot_dir / f"hist_{col}.png",
        dpi=150
    )

    plt.show()

    plt.close()


# ============================================================
# 11. CATEGORICAL BAR PLOTS
# ============================================================

plot_cats = [
    "education",
    "marital-status",
    "workclass",
    "occupation"
]


for col in plot_cats:

    if col not in feature_df.columns:
        continue

    plot_data = (
        feature_df[col]
        .astype("string")
        .fillna("Missing")
    )

    counts = (
        plot_data
        .value_counts()
        .head(12)
    )

    plt.figure(
        figsize=(9, 5)
    )

    counts.sort_values().plot(
        kind="barh"
    )

    plt.title(
        f"Top categories: {col}"
    )

    plt.xlabel("Count")

    plt.tight_layout()

    plt.savefig(
        plot_dir / f"bar_{col}.png",
        dpi=150
    )

    plt.show()

    plt.close()


# ============================================================
# 12. FEATURE DISTRIBUTION SUMMARY
# ============================================================

feature_summary = []


for col in plot_numeric:

    if col not in feature_df.columns:
        continue

    s = feature_df[col].dropna()

    feature_summary.append({

        "feature": col,

        "mean": s.mean(),

        "median": s.median(),

        "std": s.std(),

        "p25": s.quantile(0.25),

        "p75": s.quantile(0.75),

        "missing": feature_df[col].isna().sum()

    })


feature_summary = pd.DataFrame(
    feature_summary
)


display(feature_summary)


feature_summary.to_csv(
    summary_dir /
    "feature_distribution_summary.csv",
    index=False
)


# ============================================================
# 13. TASK 3 — REPRODUCIBLE SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TASK 3 — TRAIN / DEV / TEST SPLIT")
print("=" * 70)


# IMPORTANT:
# Remove BOTH target and original class label.
#
# This prevents target leakage.
#
# If "class" remained inside X, the model would receive the
# actual answer as an input feature.

X = df.drop(
    columns=[
        "target",
        target_col
    ]
)

y = df["target"]


print("\nFinal feature columns:")
print(X.columns.tolist())


# ------------------------------------------------------------
# Final 20% hold-out test set
# ------------------------------------------------------------

X_devpool, X_test, y_devpool, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    stratify=y,

    random_state=RANDOM_STATE

)


# ------------------------------------------------------------
# Development set
# ------------------------------------------------------------

X_train, X_dev, y_train, y_dev = train_test_split(

    X_devpool,
    y_devpool,

    test_size=DEV_SIZE_WITHIN_REMAINDER,

    stratify=y_devpool,

    random_state=RANDOM_STATE

)


print(
    f"\nTrain: {len(X_train):,} "
    f"({len(X_train) / len(df):.1%})"
)

print(
    f"Dev:   {len(X_dev):,} "
    f"({len(X_dev) / len(df):.1%})"
)

print(
    f"Test:  {len(X_test):,} "
    f"({len(X_test) / len(df):.1%})"
)


print("\nClass rates:")

print(
    f"Train: {y_train.mean():.4f}"
)

print(
    f"Dev:   {y_dev.mean():.4f}"
)

print(
    f"Test:  {y_test.mean():.4f}"
)


# ============================================================
# 14. TASK 4 — MAJORITY BASELINE
# ============================================================

print("\n" + "=" * 70)
print("TASK 4 — MAJORITY CLASS BASELINE")
print("=" * 70)


majority_class = int(
    y_train.mode().iloc[0]
)


pred_majority = np.full(
    len(y_test),
    majority_class,
    dtype=int
)


# Majority probability score
score_majority = pred_majority.astype(float)


# ============================================================
# 15. EDUCATION RULE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("EDUCATION RULE BASELINE")
print("=" * 70)


education_threshold = 13


pred_education = (

    pd.to_numeric(
        X_test["education-num"],
        errors="coerce"
    )

    .fillna(0)

    .ge(education_threshold)

    .astype(int)

    .to_numpy()

)


score_education = (
    pred_education.astype(float)
)


print(
    "Rule: education-num >= 13 -> >50K"
)


# ============================================================
# 16. BASELINE EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    y_pred,
    y_score
):

    return {

        "model": name,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_score
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_score
            )

    }


# ============================================================
# 17. BASELINE RESULTS
# ============================================================

baseline_results = pd.DataFrame([

    evaluate_model(
        "Majority class",
        y_test,
        pred_majority,
        score_majority
    ),

    evaluate_model(
        "Education-num >= 13",
        y_test,
        pred_education,
        score_education
    )

])


print("\nBaseline results:")


display(
    baseline_results.style.format({

        "accuracy": "{:.3f}",

        "precision": "{:.3f}",

        "recall": "{:.3f}",

        "f1": "{:.3f}",

        "roc_auc": "{:.3f}",

        "pr_auc": "{:.3f}"

    })
)


baseline_results.to_csv(
    summary_dir /
    "baseline_metrics.csv",
    index=False
)


# ============================================================
# 18. BASELINE CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4)
)


for ax, name, pred in zip(

    axes,

    [
        "Majority class",
        "Education-num >= 13"
    ],

    [
        pred_majority,
        pred_education
    ]

):

    ConfusionMatrixDisplay.from_predictions(

        y_test,

        pred,

        display_labels=[
            "<=50K",
            ">50K"
        ],

        cmap="Blues",

        ax=ax

    )

    ax.set_title(name)


plt.tight_layout()


plt.savefig(
    plot_dir /
    "baseline_confusion_matrices.png",
    dpi=160
)


plt.show()

plt.close()


# ============================================================
# 19. TASK 4B — REAL MACHINE LEARNING MODEL
# ============================================================

print("\n" + "=" * 70)
print("TASK 4B — HISTOGRAM GRADIENT BOOSTING MODEL")
print("=" * 70)


# Identify numerical features
numeric_features = (
    X_train
    .select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    )
    .columns
    .tolist()
)


# Identify categorical features
categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    )
    .columns
    .tolist()
)


print("\nNumeric features:")
print(numeric_features)


print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 20. NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )

])


# ============================================================
# 21. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )

])


# ============================================================
# 22. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([

    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )

])


# ============================================================
# 23. GRADIENT BOOSTING MODEL
# ============================================================

model = HistGradientBoostingClassifier(

    max_iter=300,

    learning_rate=0.08,

    max_leaf_nodes=31,

    min_samples_leaf=20,

    l2_regularization=1.0,

    random_state=RANDOM_STATE

)


# ============================================================
# 24. COMPLETE ML PIPELINE
# ============================================================

ml_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        model
    )

])


# ============================================================
# 25. TRAIN MODEL
# ============================================================

print("\nTraining model...")

ml_pipeline.fit(
    X_train,
    y_train
)


print("Training complete!")


# ============================================================
# 26. DEVELOPMENT SET EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("DEVELOPMENT SET RESULTS")
print("=" * 70)


dev_pred = ml_pipeline.predict(
    X_dev
)


dev_prob = ml_pipeline.predict_proba(
    X_dev
)[:, 1]


dev_results = evaluate_model(

    "HistGradientBoosting",

    y_dev,

    dev_pred,

    dev_prob

)


display(
    pd.DataFrame([dev_results]).style.format({

        "accuracy": "{:.3f}",

        "precision": "{:.3f}",

        "recall": "{:.3f}",

        "f1": "{:.3f}",

        "roc_auc": "{:.3f}",

        "pr_auc": "{:.3f}"

    })
)


print("\nDevelopment classification report:")

print(
    classification_report(
        y_dev,
        dev_pred,
        target_names=[
            "<=50K",
            ">50K"
        ]
    )
)


# ============================================================
# 27. CHECK IF ACCURACY EXCEEDS 85%
# ============================================================

dev_accuracy = dev_results["accuracy"]


print(
    f"\nDevelopment Accuracy: "
    f"{dev_accuracy:.2%}"
)


if dev_accuracy >= 0.85:

    print(
        "SUCCESS: Development accuracy is above 85%."
    )

else:

    print(
        "Accuracy is below 85%. "
        "Further tuning may be required."
    )


# ============================================================
# 28. FINAL HOLD-OUT TEST
# ============================================================
#
# IMPORTANT:
# The test set was not used for model training or tuning.
# This is the FIRST time we evaluate the trained model on it.
# ============================================================

print("\n" + "=" * 70)
print("FINAL HOLD-OUT TEST EVALUATION")
print("=" * 70)


test_pred = ml_pipeline.predict(
    X_test
)


test_prob = ml_pipeline.predict_proba(
    X_test
)[:, 1]


final_ml_results = evaluate_model(

    "HistGradientBoosting",

    y_test,

    test_pred,

    test_prob

)


final_ml_results_df = pd.DataFrame([
    final_ml_results
])


display(
    final_ml_results_df.style.format({

        "accuracy": "{:.3f}",

        "precision": "{:.3f}",

        "recall": "{:.3f}",

        "f1": "{:.3f}",

        "roc_auc": "{:.3f}",

        "pr_auc": "{:.3f}"

    })
)


final_ml_results_df.to_csv(

    summary_dir /
    "final_ml_results.csv",

    index=False

)


# ============================================================
# 29. FINAL ACCURACY CHECK
# ============================================================

final_accuracy = (
    final_ml_results["accuracy"]
)


print(
    f"\nFINAL TEST ACCURACY: "
    f"{final_accuracy:.2%}"
)


if final_accuracy >= 0.85:

    print(
        "✓ Target achieved: accuracy is above 85%."
    )

else:

    print(
        "⚠ Accuracy is below 85%. "
        "Additional feature engineering/tuning is needed."
    )


# ============================================================
# 30. FINAL CLASSIFICATION REPORT
# ============================================================

print("\nFinal classification report:")


print(
    classification_report(

        y_test,

        test_pred,

        target_names=[
            "<=50K",
            ">50K"
        ]

    )
)


# ============================================================
# 31. FINAL CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)


ConfusionMatrixDisplay.from_predictions(

    y_test,

    test_pred,

    display_labels=[
        "<=50K",
        ">50K"
    ],

    cmap="Blues"

)


plt.title(
    "HistGradientBoosting — Final Test Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    plot_dir /
    "final_ml_confusion_matrix.png",

    dpi=160

)


plt.show()

plt.close()


# ============================================================
# 32. COMPARE ALL MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


all_results = pd.concat([

    baseline_results,

    final_ml_results_df

], ignore_index=True)


display(
    all_results.style.format({

        "accuracy": "{:.3f}",

        "precision": "{:.3f}",

        "recall": "{:.3f}",

        "f1": "{:.3f}",

        "roc_auc": "{:.3f}",

        "pr_auc": "{:.3f}"

    })
)


all_results.to_csv(

    summary_dir /
    "all_model_results.csv",

    index=False

)


# ============================================================
# 33. TASK 5 — ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TASK 5 — INITIAL ERROR ANALYSIS")
print("=" * 70)


# Create error analysis dataframe
error_df = X_test.copy()


error_df["actual"] = y_test.to_numpy()


error_df["predicted"] = test_pred


error_df["predicted_probability"] = test_prob


error_df["error_type"] = np.select(

    [

        (
            (error_df["actual"] == 0) &
            (error_df["predicted"] == 1)
        ),

        (
            (error_df["actual"] == 1) &
            (error_df["predicted"] == 0)
        )

    ],

    [

        "false_positive",

        "false_negative"

    ],

    default="correct"

)


# False positives
fp = error_df[
    error_df["error_type"] ==
    "false_positive"
].copy()


# False negatives
fn = error_df[
    error_df["error_type"] ==
    "false_negative"
].copy()


print(
    "\nFalse positives:",
    len(fp)
)


print(
    "False negatives:",
    len(fn)
)


# ------------------------------------------------------------
# Sample at least 10 FP and 10 FN
# ------------------------------------------------------------

print("\n========== FALSE POSITIVE SAMPLE ==========")


display(
    fp.head(10)
)


print("\n========== FALSE NEGATIVE SAMPLE ==========")


display(
    fn.head(10)
)


# Save samples
fp.head(10).to_csv(

    summary_dir /
    "false_positive_sample.csv",

    index=False

)


fn.head(10).to_csv(

    summary_dir /
    "false_negative_sample.csv",

    index=False

)


# ============================================================
# 34. ERROR PROFILE
# ============================================================

def error_profile(
    frame,
    name
):

    out = {

        "error_type": name,

        "n": len(frame)

    }


    for col in [

        "age",

        "education-num",

        "hours-per-week",

        "capital-gain",

        "capital-loss"

    ]:

        if col in frame.columns:

            out[
                f"{col}_median"
            ] = pd.to_numeric(

                frame[col],

                errors="coerce"

            ).median()


    for col in [

        "workclass",

        "education",

        "marital-status",

        "occupation",

        "relationship",

        "sex"

    ]:

        if col in frame.columns:

            mode = frame[col].mode(
                dropna=True
            )

            out[
                f"{col}_top"
            ] = (
                mode.iloc[0]
                if len(mode)
                else "Missing"
            )


    return out


error_profile_table = pd.DataFrame([

    error_profile(
        fp,
        "False positives"
    ),

    error_profile(
        fn,
        "False negatives"
    )

])


print("\n========== ERROR PROFILE ==========")


display(
    error_profile_table
)


error_profile_table.to_csv(

    summary_dir /
    "error_profile.csv",

    index=False

)


# ============================================================
# 35. CONCRETE ISSUES FOR NEXT ITERATION
# ============================================================

print("\n" + "=" * 70)
print("DATA / FEATURE ISSUES FOR NEXT ITERATION")
print("=" * 70)


issues = [

    "1. Missing categorical values need consistent imputation.",

    "2. Categorical variables require robust encoding.",

    "3. Capital-gain and capital-loss are highly skewed and zero-inflated.",

    "4. Income depends on nonlinear interactions between education, age, occupation and hours worked.",

    "5. Class imbalance means accuracy alone is not sufficient.",

    "6. Race and sex should be evaluated carefully for fairness and subgroup performance."

]


for issue in issues:

    print(issue)


# ============================================================
# 36. PRIMARY METRIC
# ============================================================

print("\n" + "=" * 70)
print("PRIMARY METRIC")
print("=" * 70)


print("""
PRIMARY METRIC: PRECISION

Precision remains the primary business metric because the use case
assumes that positive predictions trigger higher-value outreach.

The majority baseline demonstrates that accuracy can be misleading:
predicting <=50K for everyone gives approximately 76% accuracy but
finds zero positive cases.

The education rule provides a stronger positive-class reference,
while the ML model should improve precision while maintaining useful
recall, F1 and PR AUC.

For future modeling, the development set should be used for tuning,
while the final 20% test set should remain untouched until the model
and preprocessing pipeline are finalized.
""")


# ============================================================
# 37. CREATE 1-PAGE PDF SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GENERATING 1-PAGE PDF SUMMARY")
print("=" * 70)


pdf_out = (
    summary_dir /
    "adult_income_1page_summary.pdf"
)


# Extract metrics
majority_row = baseline_results[
    baseline_results["model"] ==
    "Majority class"
].iloc[0]


education_row = baseline_results[
    baseline_results["model"] ==
    "Education-num >= 13"
].iloc[0]


ml_row = final_ml_results


# Create PDF page
fig = plt.figure(
    figsize=(8.27, 11.69)
)


ax = fig.add_axes(
    [0, 0, 1, 1]
)


ax.axis("off")


text = f"""
UCI ADULT INCOME
BASELINES, ML MODEL & INITIAL ERROR ANALYSIS

PROBLEM FRAMING
Predict whether an adult earns more than $50K per year.
The business scenario is targeted higher-value outreach.
Precision is the primary metric because false positives
represent wasted outreach resources.

DATASET
Total rows: {len(df):,}
Positive class (>50K): {base_rate:.2%}

TRAIN / DEV / TEST
Train: {len(X_train):,} ({len(X_train)/len(df):.1%})
Dev:   {len(X_dev):,} ({len(X_dev)/len(df):.1%})
Test:  {len(X_test):,} ({len(X_test)/len(df):.1%})

Split is stratified with random_state={RANDOM_STATE}.
The final test set was kept separate from model development.

BASELINE RESULTS

Model                  Accuracy   Precision   Recall    F1      ROC AUC   PR AUC

Majority class         {majority_row["accuracy"]:.3f}      {majority_row["precision"]:.3f}       {majority_row["recall"]:.3f}     {majority_row["f1"]:.3f}    {majority_row["roc_auc"]:.3f}     {majority_row["pr_auc"]:.3f}

Education >= 13        {education_row["accuracy"]:.3f}      {education_row["precision"]:.3f}       {education_row["recall"]:.3f}     {education_row["f1"]:.3f}    {education_row["roc_auc"]:.3f}     {education_row["pr_auc"]:.3f}

FINAL ML MODEL

HistGradientBoosting

Accuracy:  {ml_row["accuracy"]:.3f}
Precision: {ml_row["precision"]:.3f}
Recall:    {ml_row["recall"]:.3f}
F1:        {ml_row["f1"]:.3f}
ROC AUC:   {ml_row["roc_auc"]:.3f}
PR AUC:    {ml_row["pr_auc"]:.3f}

FINAL TEST ACCURACY: {ml_row["accuracy"]:.2%}

ERROR ANALYSIS

False positives: {len(fp):,}
False negatives: {len(fn):,}

The ML model can make mistakes because income depends on multiple
factors rather than a single feature. Important patterns include
education, age, occupation, work hours, marital status and capital
gains/losses.

NEXT STEPS

• Improve categorical missing-value handling.
• Experiment with encoding strategies.
• Handle skewed capital-gain and capital-loss.
• Explore nonlinear feature interactions.
• Tune the model using the development set.
• Monitor precision, recall, F1 and PR AUC.
• Evaluate subgroup/fairness performance.

PRIMARY SUCCESS METRIC

PRECISION

The final model should improve positive-class precision while
maintaining useful recall and PR AUC. Accuracy above 85% is a useful
technical target, but it should not replace the primary business
objective of reducing wasted positive predictions.
"""


ax.text(

    0.055,

    0.97,

    text,

    va="top",

    ha="left",

    fontsize=8.7,

    family="DejaVu Sans",

    linespacing=1.35

)


with PdfPages(pdf_out) as pdf:

    pdf.savefig(
        fig,
        bbox_inches="tight"
    )


plt.close(fig)


print(
    "\nGenerated PDF:",
    pdf_out
)


# ============================================================
# 38. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)


print(
    f"""
Dataset:
    {len(df):,} rows

Positive rate:
    {base_rate:.2%}

Train:
    {len(X_train):,}

Development:
    {len(X_dev):,}

Final Test:
    {len(X_test):,}

Baseline Accuracy:
    Majority       = {majority_row["accuracy"]:.2%}
    Education Rule = {education_row["accuracy"]:.2%}

ML Model:
    HistGradientBoosting

Final ML Accuracy:
    {ml_row["accuracy"]:.2%}

Final ML Precision:
    {ml_row["precision"]:.2%}

Final ML Recall:
    {ml_row["recall"]:.2%}

Final ML F1:
    {ml_row["f1"]:.2%}

Final ML ROC AUC:
    {ml_row["roc_auc"]:.3f}

Final ML PR AUC:
    {ml_row["pr_auc"]:.3f}

PDF:
    {pdf_out}
"""
)


if ml_row["accuracy"] >= 0.85:

    print(
        "✓ FINAL RESULT: Accuracy target >85% achieved."
    )

else:

    print(
        "⚠ FINAL RESULT: Accuracy is below 85%; "
        "continue tuning on the DEV set."
    )
