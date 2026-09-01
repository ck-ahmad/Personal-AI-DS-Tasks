# ============================================================
# Adult Income Prediction — Day 2
# Supervised Models + Leakage-Safe Preprocessing
# ============================================================
#
# Scenario:
# Using the same Adult dataset and train/dev/test methodology
# from Day 1, build the first real supervised ML models.
#
# Focus:
#   1. Repeatable preprocessing pipelines
#   2. No data leakage
#   3. Mixed numerical + categorical preprocessing
#   4. Logistic Regression
#   5. Decision Tree
#   6. Multiple evaluation metrics
#   7. ROC / PR curves
#   8. Confusion matrices
#   9. Model interpretability
#  10. Model selection for Day 3
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Day 1 split
TEST_SIZE = 0.20
DEV_SIZE = 0.125
# 0.125 of the remaining 80% = 10% of total data
# Final proportions:
# Train ≈ 70%
# Dev   ≈ 10%
# Test  ≈ 20%

OUTPUT_DIR = "adult_income_day2_outputs"
PLOT_DIR = "adult_income_day2_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD ADULT DATASET
# ============================================================

print("=" * 70)
print("LOADING ADULT DATASET")
print("=" * 70)

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy()

print(f"Raw dataset shape: {df.shape}")


# ============================================================
# 4. BASIC DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# ------------------------------------------------------------
# Remove whitespace from string columns
# ------------------------------------------------------------

for column in df.select_dtypes(include=["object", "category"]).columns:
    df[column] = df[column].astype(str).str.strip()


# ------------------------------------------------------------
# Replace '?' with NaN
# ------------------------------------------------------------
#
# The Adult dataset may contain missing categorical values
# represented as '?'.
#
# Converting '?' to NaN gives sklearn's imputers a standard
# missing-value representation.
# ------------------------------------------------------------

df = df.replace("?", np.nan)


# ------------------------------------------------------------
# Find target column
# ------------------------------------------------------------

if "class" in df.columns:
    TARGET_COLUMN = "class"
elif "income" in df.columns:
    TARGET_COLUMN = "income"
elif "target" in df.columns:
    TARGET_COLUMN = "target"
else:
    # OpenML Adult v2 normally uses "class"
    TARGET_COLUMN = df.columns[-1]

print(f"Target column: {TARGET_COLUMN}")


# ------------------------------------------------------------
# Normalize target labels
# ------------------------------------------------------------

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.replace(".", "", regex=False)
)


# ------------------------------------------------------------
# Convert target into binary format
# ------------------------------------------------------------

target_mapping = {
    "<=50K": 0,
    ">50K": 1
}

df[TARGET_COLUMN] = df[TARGET_COLUMN].map(target_mapping)

# Remove rows where target could not be converted
df = df.dropna(subset=[TARGET_COLUMN])

df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)


print("\nCleaned dataset shape:", df.shape)

print("\nTarget distribution:")
print(df[TARGET_COLUMN].value_counts())

print("\nTarget percentage:")
print(
    df[TARGET_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DEFINITIONS")
print("=" * 70)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# ------------------------------------------------------------
# Explicit numerical features
# ------------------------------------------------------------

numeric_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]


# ------------------------------------------------------------
# Explicit categorical features
# ------------------------------------------------------------

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]


# ------------------------------------------------------------
# Verify columns exist
# ------------------------------------------------------------

missing_numeric = [
    column for column in numeric_features
    if column not in X.columns
]

missing_categorical = [
    column for column in categorical_features
    if column not in X.columns
]

if missing_numeric:
    raise ValueError(
        f"Missing numerical columns: {missing_numeric}"
    )

if missing_categorical:
    raise ValueError(
        f"Missing categorical columns: {missing_categorical}"
    )


print("\nNumerical features:")
for feature in numeric_features:
    print(f"  - {feature}")

print("\nCategorical features:")
for feature in categorical_features:
    print(f"  - {feature}")


# ============================================================
# 6. TRAIN / DEV / HOLD-OUT TEST SPLIT
# ============================================================
#
# IMPORTANT:
#
# The final test set is separated before model training.
#
# The test set is NEVER used to:
#   - fit imputers
#   - fit scalers
#   - fit encoders
#   - select hyperparameters
#   - select features
#   - train models
#
# The preprocessing objects inside the sklearn Pipeline are
# fitted ONLY on the training data.
#
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / DEV / TEST SPLIT")
print("=" * 70)

# First split:
# 80% temporary data
# 20% final hold-out test

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# Second split:
# approximately 70% total training
# approximately 10% total development

X_train, X_dev, y_train, y_dev = train_test_split(
    X_temp,
    y_temp,
    test_size=DEV_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


print(f"Training samples:   {len(X_train):,}")
print(f"Development samples:{len(X_dev):,}")
print(f"Hold-out test:      {len(X_test):,}")

print("\nApproximate proportions:")
print(
    f"Train: {len(X_train) / len(X):.2%}"
)

print(
    f"Dev:   {len(X_dev) / len(X):.2%}"
)

print(
    f"Test:  {len(X_test) / len(X):.2%}"
)


# ============================================================
# 7. PREPROCESSING PIPELINES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PREPROCESSING PIPELINE")
print("=" * 70)


# ------------------------------------------------------------
# NUMERICAL PIPELINE
# ------------------------------------------------------------
#
# SimpleImputer(strategy="median")
#
# Why median?
#
# Numerical Adult features such as capital-gain and
# capital-loss can be highly skewed and contain extreme values.
#
# Median is less sensitive to outliers than the mean.
#
# IMPORTANT:
# The median is learned only from X_train because this
# transformer is inside the Pipeline.
#
#
# StandardScaler
#
# Standardization puts numerical features onto a comparable
# scale:
#
#       x_scaled = (x - mean) / standard_deviation
#
# This is especially useful for Logistic Regression because
# regularized linear models are sensitive to feature scale.
#
# Alternative considered:
#   - mean imputation
#
# Skipped because mean can be strongly influenced by outliers
# and skewed distributions.
#
# Alternative considered:
#   - KNNImputer
#
# Skipped because it is computationally more expensive and is
# unnecessary for this first baseline.
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# CATEGORICAL PIPELINE
# ------------------------------------------------------------
#
# SimpleImputer(strategy="most_frequent")
#
# Missing categorical values are replaced with the most
# frequent category learned from the training data.
#
# Alternative:
#   SimpleImputer(strategy="constant", fill_value="Missing")
#
# This is also reasonable because missingness can itself
# contain information.
#
# For this baseline, most_frequent is kept simple.
#
#
# OneHotEncoder
#
# Categorical variables cannot be directly used by most
# sklearn classifiers.
#
# One-hot encoding creates a binary feature for each category.
#
# handle_unknown="ignore" is VERY important.
#
# If a category appears in validation/test data that was not
# observed during training, the pipeline will not crash.
#
# Instead, the unseen category is encoded as all zeros for
# that feature group.
#
# Alternative considered:
#   OrdinalEncoder
#
# Skipped because assigning numerical ordering to categories
# such as occupation or marital status could incorrectly imply
# that one category is greater/less than another.
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ============================================================
# 8. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
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
    ],
    remainder="drop"
)


print("Preprocessing pipeline created successfully.")


# ============================================================
# 9. LOGISTIC REGRESSION PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LOGISTIC REGRESSION PIPELINE")
print("=" * 70)

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=RANDOM_STATE,
                solver="liblinear",
                penalty="l2",
                max_iter=1000
            )
        )
    ]
)


# ============================================================
# 10. DECISION TREE PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("BUILDING DECISION TREE PIPELINE")
print("=" * 70)

decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# 11. TRAIN MODELS
# ============================================================
#
# IMPORTANT:
#
# Only X_train and y_train are passed to .fit().
#
# The hold-out test set is NOT touched during training.
#
# Because preprocessing is inside each Pipeline:
#
#       Imputer
#       ↓
#       Scaler / Encoder
#       ↓
#       Model
#
# every preprocessing step is fitted only on X_train.
#
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

print("\nTraining Logistic Regression...")

logistic_pipeline.fit(
    X_train,
    y_train
)

print("Logistic Regression training complete.")


print("\nTraining Decision Tree...")

decision_tree_pipeline.fit(
    X_train,
    y_train
)

print("Decision Tree training complete.")


# ============================================================
# 12. DAY 1 BASELINE
# ============================================================
#
# Recreate the Day 1 baselines using the same hold-out test.
#
# Majority baseline:
#     Always predict the majority class.
#
# Education baseline:
#     education-num >= 13 -> >50K
#     otherwise -> <=50K
#
# These baselines are NOT trained ML models.
# They provide reference points for evaluating whether the
# supervised models actually learn useful patterns.
# ============================================================

print("\n" + "=" * 70)
print("DAY 1 BASELINES")
print("=" * 70)


# ------------------------------------------------------------
# Majority-class baseline
# ------------------------------------------------------------

majority_class = y_train.mode()[0]

majority_predictions = np.full(
    shape=len(y_test),
    fill_value=majority_class
)


# ------------------------------------------------------------
# Education rule baseline
# ------------------------------------------------------------

education_predictions = (
    X_test["education-num"]
    >= 13
).astype(int)


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate_predictions(
    model_name,
    y_true,
    y_pred,
    y_probability
):
    """
    Calculate the required classification metrics.

    Metrics:
        Accuracy
        Precision
        Recall
        F1
        ROC AUC
        PR AUC
    """

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "ROC AUC": roc_auc_score(
            y_true,
            y_probability
        ),

        "PR AUC": average_precision_score(
            y_true,
            y_probability
        )
    }


# ============================================================
# 14. EVALUATE DAY 1 BASELINES
# ============================================================

# Majority baseline does not have a meaningful probability
# ranking. We use its constant prediction as a probability
# score for reference.

majority_probability = np.full(
    len(y_test),
    majority_class,
    dtype=float
)


majority_metrics = evaluate_predictions(
    "Day 1 — Majority Class",
    y_test,
    majority_predictions,
    majority_probability
)


# Education rule probability score:
# 0 or 1 according to the rule.

education_probability = education_predictions.astype(float)

education_metrics = evaluate_predictions(
    "Day 1 — Education >= 13",
    y_test,
    education_predictions,
    education_probability
)


# ============================================================
# 15. LOGISTIC REGRESSION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION — HOLD-OUT TEST EVALUATION")
print("=" * 70)

logistic_predictions = logistic_pipeline.predict(
    X_test
)

logistic_probabilities = logistic_pipeline.predict_proba(
    X_test
)[:, 1]


logistic_metrics = evaluate_predictions(
    "Day 2 — Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
)


# ============================================================
# 16. DECISION TREE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE — HOLD-OUT TEST EVALUATION")
print("=" * 70)

tree_predictions = decision_tree_pipeline.predict(
    X_test
)

tree_probabilities = decision_tree_pipeline.predict_proba(
    X_test
)[:, 1]


tree_metrics = evaluate_predictions(
    "Day 2 — Decision Tree",
    y_test,
    tree_predictions,
    tree_probabilities
)


# ============================================================
# 17. COMPARISON TABLE
# ============================================================

results = pd.DataFrame(
    [
        majority_metrics,
        education_metrics,
        logistic_metrics,
        tree_metrics
    ]
)

results = results.set_index("Model")


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.round(4).to_string()
)


# Save metrics
results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    )
)


# ============================================================
# 18. IDENTIFY BEST MODEL BY PRECISION
# ============================================================

best_precision_model = results[
    "Precision"
].idxmax()

best_f1_model = results[
    "F1"
].idxmax()

best_pr_auc_model = results[
    "PR AUC"
].idxmax()


print("\nBest model by Precision:")
print(best_precision_model)

print("\nBest model by F1:")
print(best_f1_model)

print("\nBest model by PR AUC:")
print(best_pr_auc_model)


# ============================================================
# 19. ROC CURVES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING ROC CURVES")
print("=" * 70)

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_probabilities
)

fpr_tree, tpr_tree, _ = roc_curve(
    y_test,
    tree_probabilities
)


plt.figure(figsize=(9, 7))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=(
        f"Logistic Regression "
        f"(AUC={logistic_metrics['ROC AUC']:.3f})"
    )
)

plt.plot(
    fpr_tree,
    tpr_tree,
    label=(
        f"Decision Tree "
        f"(AUC={tree_metrics['ROC AUC']:.3f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves — Adult Income Prediction"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "roc_curves.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 20. PRECISION-RECALL CURVES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PRECISION-RECALL CURVES")
print("=" * 70)

precision_logistic, recall_logistic, _ = (
    precision_recall_curve(
        y_test,
        logistic_probabilities
    )
)

precision_tree, recall_tree, _ = (
    precision_recall_curve(
        y_test,
        tree_probabilities
    )
)


plt.figure(figsize=(9, 7))

plt.plot(
    recall_logistic,
    precision_logistic,
    label=(
        f"Logistic Regression "
        f"(PR AUC={logistic_metrics['PR AUC']:.3f})"
    )
)

plt.plot(
    recall_tree,
    precision_tree,
    label=(
        f"Decision Tree "
        f"(PR AUC={tree_metrics['PR AUC']:.3f})"
    )
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curves — Adult Income Prediction"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "precision_recall_curves.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 21. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression Confusion Matrix:")
print(logistic_cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=logistic_cm,
    display_labels=["<=50K", ">50K"]
)

disp.plot()

plt.title(
    "Logistic Regression — Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "logistic_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------

tree_cm = confusion_matrix(
    y_test,
    tree_predictions
)

print("\nDecision Tree Confusion Matrix:")
print(tree_cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=tree_cm,
    display_labels=["<=50K", ">50K"]
)

disp.plot()

plt.title(
    "Decision Tree — Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "decision_tree_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 22. ERROR TYPE ANALYSIS
# ============================================================

def error_analysis(
    model_name,
    cm,
    precision_value
):
    """
    Analyze false positives and false negatives.
    """

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"True Negatives : {tn:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"True Positives : {tp:,}")

    print(f"\nFalse Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")

    if fp > fn:
        print(
            "\nMore common error: FALSE POSITIVES"
        )

        print(
            "This is important because precision is the "
            "primary business metric."
        )

        print(
            "A high number of false positives means that "
            "more people predicted as >50K actually belong "
            "to the <=50K class."
        )

    elif fn > fp:
        print(
            "\nMore common error: FALSE NEGATIVES"
        )

        print(
            "The model is missing more actual >50K cases."
        )

        print(
            "This reduces recall and means the model may "
            "be too conservative when identifying positive cases."
        )

    else:
        print(
            "\nFalse positives and false negatives are equal."
        )

    print(
        f"\nPrecision: {precision_value:.4f}"
    )


error_analysis(
    "Logistic Regression",
    logistic_cm,
    logistic_metrics["Precision"]
)

error_analysis(
    "Decision Tree",
    tree_cm,
    tree_metrics["Precision"]
)


# ============================================================
# 23. LOGISTIC REGRESSION INTERPRETABILITY
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION INTERPRETABILITY")
print("=" * 70)


# ------------------------------------------------------------
# Extract fitted preprocessing pipeline
# ------------------------------------------------------------

logistic_preprocessor = (
    logistic_pipeline
    .named_steps["preprocessor"]
)


# ------------------------------------------------------------
# Get feature names after preprocessing
# ------------------------------------------------------------
#
# This includes:
#
# Numerical features
#
# plus one-hot encoded categorical features.
#
# Example:
#
# education_Bachelors
# education_Masters
# occupation_Exec-managerial
#
# etc.
# ------------------------------------------------------------

feature_names = (
    logistic_preprocessor
    .get_feature_names_out()
)


# ------------------------------------------------------------
# Extract logistic coefficients
# ------------------------------------------------------------

logistic_model = (
    logistic_pipeline
    .named_steps["classifier"]
)

coefficients = logistic_model.coef_[0]


coefficient_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients
    }
)


# ------------------------------------------------------------
# Sort coefficients
# ------------------------------------------------------------

coefficient_df["Absolute_Coefficient"] = (
    coefficient_df["Coefficient"]
    .abs()
)


# ============================================================
# TOP 10 POSITIVE COEFFICIENTS
# ============================================================

top_positive = (
    coefficient_df
    .sort_values(
        by="Coefficient",
        ascending=False
    )
    .head(10)
    .copy()
)


print("\nTop 10 Positive Coefficients:")
print(
    top_positive[
        ["Feature", "Coefficient"]
    ].round(4).to_string(index=False)
)


# ============================================================
# TOP 10 NEGATIVE COEFFICIENTS
# ============================================================

top_negative = (
    coefficient_df
    .sort_values(
        by="Coefficient",
        ascending=True
    )
    .head(10)
    .copy()
)


print("\nTop 10 Negative Coefficients:")
print(
    top_negative[
        ["Feature", "Coefficient"]
    ].round(4).to_string(index=False)
)


# Save coefficients
coefficient_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "logistic_coefficients.csv"
    ),
    index=False
)

top_positive[
    ["Feature", "Coefficient"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_10_positive_coefficients.csv"
    ),
    index=False
)

top_negative[
    ["Feature", "Coefficient"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_10_negative_coefficients.csv"
    ),
    index=False
)


# ============================================================
# 24. AUTOMATIC COEFFICIENT INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("COEFFICIENT INTERPRETATION")
print("=" * 70)


print("\nPositive coefficients:")
print(
    "A positive coefficient increases the model's log-odds "
    "of predicting >50K, holding other features constant."
)

for _, row in top_positive.iterrows():

    print(
        f"  {row['Feature']}: "
        f"{row['Coefficient']:.4f}"
    )


print("\nNegative coefficients:")
print(
    "A negative coefficient decreases the model's log-odds "
    "of predicting >50K, holding other features constant."
)

for _, row in top_negative.iterrows():

    print(
        f"  {row['Feature']}: "
        f"{row['Coefficient']:.4f}"
    )


# ============================================================
# 25. DECISION TREE INTERPRETABILITY
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE INTERPRETABILITY")
print("=" * 70)


tree_model = (
    decision_tree_pipeline
    .named_steps["classifier"]
)


# ------------------------------------------------------------
# Tree depth
# ------------------------------------------------------------

tree_depth = tree_model.get_depth()

tree_leaves = tree_model.get_n_leaves()

print(f"Tree depth: {tree_depth}")
print(f"Number of leaves: {tree_leaves}")


# ============================================================
# 26. TRAIN VS TEST SCORE
# ============================================================

tree_train_score = (
    decision_tree_pipeline.score(
        X_train,
        y_train
    )
)

tree_test_score = (
    decision_tree_pipeline.score(
        X_test,
        y_test
    )
)


print(
    f"\nDecision Tree training accuracy: "
    f"{tree_train_score:.4f}"
)

print(
    f"Decision Tree hold-out test accuracy: "
    f"{tree_test_score:.4f}"
)

print(
    f"Train - Test gap: "
    f"{tree_train_score - tree_test_score:.4f}"
)


# ------------------------------------------------------------
# Overfitting interpretation
# ------------------------------------------------------------

if (
    tree_train_score - tree_test_score
    > 0.10
):
    tree_overfit_comment = (
        "The tree shows a substantial train-test gap, "
        "suggesting overfitting."
    )

elif (
    tree_train_score - tree_test_score
    > 0.05
):
    tree_overfit_comment = (
        "The tree shows some evidence of overfitting "
        "because training performance is noticeably higher "
        "than hold-out performance."
    )

else:
    tree_overfit_comment = (
        "The train-test gap is relatively small, suggesting "
        "limited evidence of severe overfitting."
    )


print("\nOverfitting assessment:")
print(tree_overfit_comment)


# ============================================================
# 27. TOP TREE SPLITS
# ============================================================

print("\n" + "=" * 70)
print("TOP DECISION TREE SPLITS")
print("=" * 70)


tree_feature_names = (
    decision_tree_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

tree_structure = export_text(
    tree_model,
    feature_names=list(tree_feature_names),
    max_depth=3
)


print("\nTree structure up to depth 3:")
print(tree_structure)


# ------------------------------------------------------------
# Extract first 3 actual split nodes
# ------------------------------------------------------------

children_left = tree_model.tree_.children_left
children_right = tree_model.tree_.children_right
tree_features = tree_model.tree_.feature
tree_thresholds = tree_model.tree_.threshold


split_rows = []

for node_id in range(
    len(tree_features)
):

    # Leaf nodes have feature == -2
    if tree_features[node_id] != -2:

        feature_index = tree_features[node_id]

        split_rows.append(
            {
                "Node": node_id,
                "Feature": tree_feature_names[
                    feature_index
                ],
                "Threshold": tree_thresholds[
                    node_id
                ]
            }
        )

    if len(split_rows) == 3:
        break


top_3_splits_df = pd.DataFrame(
    split_rows
)


print("\nTop 3 splits:")

print(
    top_3_splits_df.to_string(
        index=False
    )
)


top_3_splits_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree_top_3_splits.csv"
    ),
    index=False
)


# ============================================================
# 28. DECISION TREE LOGIC COMMENT
# ============================================================

print("\nDecision Tree logic assessment:")

print(
    """
The decision tree should be examined for whether its highest-
level splits are based on meaningful income-related variables.

Features such as education, education-num, age, capital-gain,
occupation, marital status, and hours-per-week can reasonably
appear as important predictors because they may contain useful
information about income.

However, very deep trees can create highly specific rules that
fit training observations instead of learning general patterns.

Therefore, the tree depth and train-test performance gap should
be considered before selecting the tree for further development.
"""
)


# ============================================================
# 29. FEATURE IMPORTANCE
# ============================================================

feature_importance_df = pd.DataFrame(
    {
        "Feature": tree_feature_names,
        "Importance": tree_model.feature_importances_
    }
)

feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop Decision Tree Features:")

print(
    feature_importance_df
    .head(20)
    .round(4)
    .to_string(index=False)
)


feature_importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 30. MODEL COMPARISON — PRIMARY METRIC
# ============================================================

print("\n" + "=" * 70)
print("PRIMARY METRIC COMPARISON")
print("=" * 70)


print(
    f"\nLogistic Regression Precision: "
    f"{logistic_metrics['Precision']:.4f}"
)

print(
    f"Decision Tree Precision: "
    f"{tree_metrics['Precision']:.4f}"
)


precision_difference = (
    logistic_metrics["Precision"]
    - tree_metrics["Precision"]
)


print(
    f"\nLogistic - Tree precision difference: "
    f"{precision_difference:.4f}"
)


# ============================================================
# 31. DAY 3 MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("DAY 3 MODEL SELECTION")
print("=" * 70)


logistic_precision = logistic_metrics["Precision"]
tree_precision = tree_metrics["Precision"]

logistic_f1 = logistic_metrics["F1"]
tree_f1 = tree_metrics["F1"]

logistic_pr_auc = logistic_metrics["PR AUC"]
tree_pr_auc = tree_metrics["PR AUC"]


# Determine candidate models
candidates = []

if logistic_precision >= tree_precision:
    candidates.append(
        "Logistic Regression"
    )

if tree_precision >= logistic_precision:
    candidates.append(
        "Decision Tree"
    )


print("\nCandidate models based on precision:")
for candidate in candidates:
    print(f"  - {candidate}")


# ============================================================
# 32. AUTOMATIC WRITE-UP
# ============================================================

if logistic_precision > tree_precision:

    primary_model = "Logistic Regression"

    writeup = f"""
DAY 2 MODEL SELECTION WRITE-UP
==============================

The Day 2 experiments introduced the first supervised learning
models using a leakage-safe preprocessing architecture. Numerical
features were processed using median imputation followed by
StandardScaler, while categorical features were processed using
most-frequent imputation followed by OneHotEncoder with
handle_unknown='ignore'. The complete preprocessing process was
placed inside sklearn Pipelines and ColumnTransformer so that
imputation, scaling, and encoding were fitted only on the training
data. This prevents information from the hold-out test set from
leaking into model training.

Logistic Regression achieved a precision of
{logistic_precision:.4f}, recall of {logistic_metrics['Recall']:.4f},
F1 score of {logistic_f1:.4f}, ROC AUC of
{logistic_metrics['ROC AUC']:.4f}, and PR AUC of
{logistic_pr_auc:.4f}. Decision Tree achieved a precision of
{tree_precision:.4f}, recall of {tree_metrics['Recall']:.4f},
F1 score of {tree_f1:.4f}, ROC AUC of
{tree_metrics['ROC AUC']:.4f}, and PR AUC of
{tree_pr_auc:.4f}. Based primarily on the precision objective,
Logistic Regression is the leading candidate for further
development. Its coefficients also provide a useful level of
interpretability for understanding which encoded features are
associated with higher or lower predicted income.

The Decision Tree remains a useful candidate for investigating
nonlinear relationships and feature interactions. However, its
training accuracy was {tree_train_score:.4f} compared with a
hold-out test accuracy of {tree_test_score:.4f}, giving a train-test
gap of {tree_train_score - tree_test_score:.4f}. This gap should be
considered when deciding whether additional tree regularization is
necessary. For Day 3, the primary direction will be to improve the
strongest candidate while also testing controlled tree
regularization and threshold selection.

Potential preprocessing experiments for Day 3 include comparing
most-frequent categorical imputation against an explicit
'Missing' category, engineering indicators for capital-gain and
capital-loss, applying log transformations to highly skewed
features, and investigating whether removing or separately
handling potentially sensitive attributes changes model behavior.
"""


else:

    primary_model = "Decision Tree"

    writeup = f"""
DAY 2 MODEL SELECTION WRITE-UP
==============================

The Day 2 experiments introduced the first supervised learning
models using a leakage-safe preprocessing architecture. Numerical
features were processed using median imputation followed by
StandardScaler, while categorical features were processed using
most-frequent imputation followed by OneHotEncoder with
handle_unknown='ignore'. The complete preprocessing process was
placed inside sklearn Pipelines and ColumnTransformer so that
imputation, scaling, and encoding were fitted only on the training
data. This prevents information from the hold-out test set from
leaking into model training.

Logistic Regression achieved a precision of
{logistic_precision:.4f}, recall of {logistic_metrics['Recall']:.4f},
F1 score of {logistic_f1:.4f}, ROC AUC of
{logistic_metrics['ROC AUC']:.4f}, and PR AUC of
{logistic_pr_auc:.4f}. Decision Tree achieved a precision of
{tree_precision:.4f}, recall of {tree_metrics['Recall']:.4f},
F1 score of {tree_f1:.4f}, ROC AUC of
{tree_metrics['ROC AUC']:.4f}, and PR AUC of
{tree_pr_auc:.4f}. Based primarily on the precision objective,
Decision Tree is the leading candidate for further development.
Its ability to learn nonlinear relationships and interactions
makes it useful for this dataset.

Logistic Regression will remain an important comparison model
because its coefficients provide a more interpretable view of the
relationship between encoded features and the predicted income
class. The Decision Tree's training accuracy was
{tree_train_score:.4f} compared with a hold-out test accuracy of
{tree_test_score:.4f}, giving a train-test gap of
{tree_train_score - tree_test_score:.4f}. For Day 3, the main focus
will be improving the selected candidate while testing controlled
tree regularization and threshold selection.

Potential preprocessing experiments for Day 3 include comparing
most-frequent categorical imputation against an explicit
'Missing' category, engineering indicators for capital-gain and
capital-loss, applying log transformations to highly skewed
features, and investigating whether removing or separately
handling potentially sensitive attributes changes model behavior.
"""


print(writeup)


# ============================================================
# 33. SAVE WRITE-UP
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "day2_model_selection_writeup.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(writeup)


# ============================================================
# 34. SAVE PREPROCESSING INFORMATION
# ============================================================

preprocessing_summary = pd.DataFrame(
    {
        "Pipeline": [
            "Numerical",
            "Categorical"
        ],

        "Steps": [
            "Median Imputation -> StandardScaler",
            "Most Frequent Imputation -> OneHotEncoder"
        ],

        "Reason": [
            "Median is robust to skew/outliers; scaling helps regularized linear models.",
            "One-hot encoding avoids artificial ordinal relationships; unknown categories are safely ignored."
        ]
    }
)


preprocessing_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "preprocessing_summary.csv"
    ),
    index=False
)


# ============================================================
# 35. SAVE TRAIN / DEV / TEST INFORMATION
# ============================================================

split_summary = pd.DataFrame(
    {
        "Dataset": [
            "Training",
            "Development",
            "Hold-out Test"
        ],

        "Rows": [
            len(X_train),
            len(X_dev),
            len(X_test)
        ],

        "Percentage": [
            len(X_train) / len(X),
            len(X_dev) / len(X),
            len(X_test) / len(X)
        ]
    }
)


split_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_split_summary.csv"
    ),
    index=False
)


# ============================================================
# 36. SAVE ERROR SUMMARY
# ============================================================

logistic_tn, logistic_fp, logistic_fn, logistic_tp = (
    logistic_cm.ravel()
)

tree_tn, tree_fp, tree_fn, tree_tp = (
    tree_cm.ravel()
)


error_summary = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree"
        ],

        "True_Negative": [
            logistic_tn,
            tree_tn
        ],

        "False_Positive": [
            logistic_fp,
            tree_fp
        ],

        "False_Negative": [
            logistic_fn,
            tree_fn
        ],

        "True_Positive": [
            logistic_tp,
            tree_tp
        ]
    }
)


error_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "error_summary.csv"
    ),
    index=False
)


# ============================================================
# 37. EXPORT PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["Actual"] = y_test.values

prediction_output["Logistic_Prediction"] = (
    logistic_predictions
)

prediction_output["Logistic_Probability"] = (
    logistic_probabilities
)

prediction_output["Tree_Prediction"] = (
    tree_predictions
)

prediction_output["Tree_Probability"] = (
    tree_probabilities
)


prediction_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "holdout_predictions.csv"
    ),
    index=False
)


# ============================================================
# 38. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DAY 2 COMPLETE")
print("=" * 70)

print("\nModels trained:")
print("  1. Logistic Regression")
print("  2. Decision Tree")

print("\nPrimary metric:")
print("  Precision")

print("\nLogistic Regression:")
print(
    f"  Accuracy : {logistic_metrics['Accuracy']:.4f}"
)
print(
    f"  Precision: {logistic_metrics['Precision']:.4f}"
)
print(
    f"  Recall   : {logistic_metrics['Recall']:.4f}"
)
print(
    f"  F1       : {logistic_metrics['F1']:.4f}"
)
print(
    f"  ROC AUC  : {logistic_metrics['ROC AUC']:.4f}"
)
print(
    f"  PR AUC   : {logistic_metrics['PR AUC']:.4f}"
)


print("\nDecision Tree:")
print(
    f"  Accuracy : {tree_metrics['Accuracy']:.4f}"
)
print(
    f"  Precision: {tree_metrics['Precision']:.4f}"
)
print(
    f"  Recall   : {tree_metrics['Recall']:.4f}"
)
print(
    f"  F1       : {tree_metrics['F1']:.4f}"
)
print(
    f"  ROC AUC  : {tree_metrics['ROC AUC']:.4f}"
)
print(
    f"  PR AUC   : {tree_metrics['PR AUC']:.4f}"
)


print("\nDecision Tree:")
print(
    f"  Depth    : {tree_depth}"
)
print(
    f"  Leaves   : {tree_leaves}"
)
print(
    f"  Train Acc: {tree_train_score:.4f}"
)
print(
    f"  Test Acc : {tree_test_score:.4f}"
)


print("\nDay 3 candidate(s):")

for candidate in candidates:
    print(f"  → {candidate}")


print("\nOutputs saved to:")
print(
    f"  {OUTPUT_DIR}/"
)

print("\nPlots saved to:")
print(
    f"  {PLOT_DIR}/"
)

print("\n" + "=" * 70)
print("NO-LEAKAGE PIPELINE COMPLETE")
print("=" * 70)

print("\nDay 2 pipeline completed successfully.")
