# Adult Income Prediction — Day 5 Final Production Project
# Final validation + error analysis + interpretation + inference + reporting

from __future__ import annotations

import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
PRIMARY_METRIC = "precision"
DEFAULT_THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "final_model.joblib"
OUTPUT_DIR = BASE_DIR / "adult_income_day5_outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Same raw Adult features used on Days 1–3.
ORIGINAL_FEATURE_COLUMNS = [
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

BASE_NUMERIC = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
]
BASE_CATEGORICAL = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country"
]
ENGINEERED_NUMERIC = [
    "has_capital_gain", "log_capital_gain", "has_capital_loss", "log_capital_loss",
    "higher_education", "education_hours_interaction", "age_hours_interaction", "capital_net"
]
ENGINEERED_CATEGORICAL = ["age_bucket", "hours_bucket"]
ALL_NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC
ALL_CATEGORICAL = BASE_CATEGORICAL + ENGINEERED_CATEGORICAL


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    """Day 3 row-level feature engineering; no target-derived values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.replace({pd.NA: np.nan})
        X = X.replace({"?": np.nan, "": np.nan})

        for col in BASE_NUMERIC:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X["age_bucket"] = pd.cut(
            X["age"], bins=[0, 24, 34, 44, 54, 64, np.inf],
            labels=["17-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            include_lowest=True,
        ).astype(object)

        X["hours_bucket"] = pd.cut(
            X["hours-per-week"], bins=[0, 24, 39, 40, 49, 59, np.inf],
            labels=["part_time", "below_standard", "standard_40", "moderate_overtime",
                    "high_overtime", "very_high_hours"],
            include_lowest=True,
        ).astype(object)

        X["has_capital_gain"] = (X["capital-gain"].fillna(0) > 0).astype(float)
        X["log_capital_gain"] = np.log1p(X["capital-gain"].clip(lower=0).fillna(0))
        X["has_capital_loss"] = (X["capital-loss"].fillna(0) > 0).astype(float)
        X["log_capital_loss"] = np.log1p(X["capital-loss"].clip(lower=0).fillna(0))
        X["higher_education"] = (X["education-num"].fillna(0) >= 13).astype(float)
        X["education_hours_interaction"] = (
            X["education-num"].fillna(0) * X["hours-per-week"].fillna(0)
        )
        X["age_hours_interaction"] = (
            X["age"].fillna(0) * X["hours-per-week"].fillna(0)
        )
        X["capital_net"] = (
            X["capital-gain"].fillna(0) - X["capital-loss"].fillna(0)
        )

        for col in ALL_NUMERIC:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")

        for col in ALL_CATEGORICAL:
            X[col] = X[col].astype(object)
            X[col] = X[col].where(pd.notna(X[col]), np.nan)

        return X


def load_adult() -> tuple[pd.DataFrame, pd.Series, str]:
    """Load Adult and return clean features + binary target."""
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

    X = df[ORIGINAL_FEATURE_COLUMNS].copy()
    y = df["target"].copy()
    return X, y, target_col


def load_model_artifact(path: Path) -> tuple[Any, float, dict[str, Any]]:
    """Load a pipeline or a bundle/dict containing model + threshold + metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Place the Day 4 saved artifact in the same folder as this script."
        )

    obj = joblib.load(path)
    metadata: dict[str, Any] = {}

    if isinstance(obj, dict):
        model = obj.get("model") or obj.get("pipeline") or obj.get("estimator")
        threshold = obj.get("threshold", DEFAULT_THRESHOLD)
        metadata = obj.get("metadata", {}) or {}
        if model is None:
            raise ValueError("Artifact dictionary must contain 'model', 'pipeline', or 'estimator'.")
    else:
        model = obj
        threshold = DEFAULT_THRESHOLD

        # Optional threshold exposed as an attribute.
        for attr in ("threshold", "classification_threshold", "decision_threshold"):
            if hasattr(model, attr):
                threshold = float(getattr(model, attr))
                break

    threshold = float(threshold)
    if not 0 < threshold < 1:
        raise ValueError(f"Classification threshold must be between 0 and 1; got {threshold}.")

    if not hasattr(model, "predict_proba"):
        raise TypeError("Final model must expose predict_proba() for ROC-AUC, PR-AUC, Brier score, and thresholding.")

    return model, threshold, metadata


def get_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("Expected predict_proba() with two class probabilities.")
    return probs[:, 1]


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Brier": brier_score_loss(y_true, probabilities),
    }


def save_final_metrics(metrics: dict[str, float]) -> pd.DataFrame:
    row = pd.DataFrame([metrics])
    row.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)
    return row


def load_shortlisted_models() -> pd.DataFrame:
    """Read optional Day 4 comparison results without fabricating missing values."""
    candidates = [
        BASE_DIR / "adult_income_day4_outputs" / "day4_model_comparison.csv",
        BASE_DIR / "adult_income_day4_outputs" / "tuned_model_comparison.csv",
        BASE_DIR / "day4_model_comparison.csv",
        BASE_DIR / "tuned_model_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["Source"] = str(path)
            return df
    return pd.DataFrame()


def append_selected_model(shortlisted: pd.DataFrame, metrics: dict[str, float], metadata: dict[str, Any]) -> pd.DataFrame:
    selected_name = metadata.get("model_name", "Selected Final Model")
    row = pd.DataFrame([{
        "Model": selected_name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "ROC-AUC": metrics["ROC-AUC"],
        "PR-AUC": metrics["PR-AUC"],
        "Brier": metrics["Brier"],
        "Status": "SELECTED FINAL MODEL",
    }])

    if shortlisted.empty:
        table = row
    else:
        table = shortlisted.copy()
        if "Model" not in table.columns:
            table.insert(0, "Model", "Day 4 shortlisted model")
        table["Status"] = table.get("Status", "Day 4 shortlisted")
        table = pd.concat([table, row], ignore_index=True, sort=False)

    table.to_csv(OUTPUT_DIR / "final_metrics_comparison.csv", index=False)
    return table


def error_analysis(model: Any, X_test: pd.DataFrame, y_test: pd.Series, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    predictions = (probabilities >= threshold).astype(int)
    result = X_test.copy()
    result["actual"] = y_test.to_numpy()
    result["probability"] = probabilities
    result["prediction"] = predictions

    result["error_type"] = np.select(
        [
            (result["actual"] == 0) & (result["prediction"] == 1),
            (result["actual"] == 1) & (result["prediction"] == 0),
        ],
        ["False Positive", "False Negative"],
        default="Correct",
    )

    result.to_csv(OUTPUT_DIR / "test_predictions_with_error_labels.csv", index=False)

    mistakes = result[result["error_type"] != "Correct"].copy()
    mistakes.to_csv(OUTPUT_DIR / "misclassified_test_examples.csv", index=False)

    # Group-level error rates for important categorical features.
    subgroup_rows = []
    for col in ["education", "occupation", "workclass", "marital-status", "sex", "age_bucket", "hours_bucket"]:
        if col in result.columns:
            grouped = result.groupby(col, dropna=False)
            for value, g in grouped:
                subgroup_rows.append({
                    "Feature": col,
                    "Group": "Missing" if pd.isna(value) else str(value),
                    "Count": len(g),
                    "Accuracy": (g["actual"] == g["prediction"]).mean(),
                    "Precision": precision_score(g["actual"], g["prediction"], zero_division=0),
                    "Recall": recall_score(g["actual"], g["prediction"], zero_division=0),
                    "FalsePositiveRate": ((g["actual"] == 0) & (g["prediction"] == 1)).sum() / max((g["actual"] == 0).sum(), 1),
                    "FalseNegativeRate": ((g["actual"] == 1) & (g["prediction"] == 0)).sum() / max((g["actual"] == 1).sum(), 1),
                })
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(OUTPUT_DIR / "subgroup_error_analysis.csv", index=False)

    return result


def confusion_matrix_plot(y_true: pd.Series, predictions: np.ndarray) -> None:
    cm = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    ax.set_title("Final Model — Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["<=50K", ">50K"])
    ax.set_yticks([0, 1], labels=["<=50K", ">50K"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "final_confusion_matrix.png", dpi=180)
    plt.close(fig)

    pd.DataFrame(
        cm,
        index=["Actual <=50K", "Actual >50K"],
        columns=["Predicted <=50K", "Predicted >50K"],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")


def roc_pr_plot(y_true: pd.Series, probabilities: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Final Model — ROC Curve")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "final_roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Final Model — Precision–Recall Curve")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "final_precision_recall_curve.png", dpi=180)
    plt.close(fig)


def calibration_plot(y_true: pd.Series, probabilities: np.ndarray) -> float:
    frac_pos, mean_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=f"Final model (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "calibration_curve.png", dpi=180)
    plt.close(fig)
    return brier


def learning_curve_plot(model: Any, X_devpool: pd.DataFrame, y_devpool: pd.Series) -> None:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    sizes = np.linspace(0.2, 1.0, 5)
    train_sizes, train_scores, val_scores = learning_curve(
        clone(model), X_devpool, y_devpool,
        cv=cv,
        scoring="precision",
        train_sizes=sizes,
        n_jobs=1,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training precision")
    ax.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="Validation precision")
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Precision")
    ax.set_title("Learning Curve — Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "learning_curve_precision.png", dpi=180)
    plt.close(fig)

    pd.DataFrame({
        "train_size": train_sizes,
        "train_precision_mean": train_scores.mean(axis=1),
        "train_precision_std": train_scores.std(axis=1),
        "validation_precision_mean": val_scores.mean(axis=1),
        "validation_precision_std": val_scores.std(axis=1),
    }).to_csv(OUTPUT_DIR / "learning_curve_scores.csv", index=False)


def permutation_feature_importance(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Model-agnostic importance on original input columns; safe for arbitrary final pipeline."""
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    df = pd.DataFrame({
        "Feature": X_test.columns,
        "ImportanceMean": perm.importances_mean,
        "ImportanceStd": perm.importances_std,
    }).sort_values("ImportanceMean", ascending=False)
    df.to_csv(OUTPUT_DIR / "permutation_feature_importance.csv", index=False)

    top = df.head(12).sort_values("ImportanceMean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["Feature"], top["ImportanceMean"], xerr=top["ImportanceStd"])
    ax.set_xlabel("Mean decrease in precision after permutation")
    ax.set_title("Final Model — Top Feature Importance")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)
    return df


def extract_native_importance(model: Any) -> pd.DataFrame | None:
    """Try to expose native coefficients/feature importances when the artifact is a sklearn Pipeline."""
    estimator = model
    preprocessor = None
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        estimator = steps.get("model", estimator)
        preprocessor = steps.get("preprocessor")

    if preprocessor is None:
        return None

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    values = None
    kind = None
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).ravel()
        kind = "Coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).ravel()
        kind = "Importance"

    if values is None or len(values) != len(names):
        return None

    out = pd.DataFrame({"Feature": names, kind: values})
    out["AbsoluteValue"] = out[kind].abs()
    out = out.sort_values("AbsoluteValue", ascending=False)
    out.to_csv(OUTPUT_DIR / "native_model_feature_importance.csv", index=False)
    return out


def write_inference_examples(model: Any, threshold: float, X_test: pd.DataFrame) -> pd.DataFrame:
    examples = X_test.sample(n=min(8, len(X_test)), random_state=RANDOM_STATE).reset_index(drop=True)
    probabilities = get_probability(model, examples)
    predictions = (probabilities >= threshold).astype(int)
    out = examples.copy()
    out["probability"] = probabilities
    out["prediction"] = predictions
    out["prediction_label"] = np.where(predictions == 1, ">50K", "<=50K")
    out.to_csv(OUTPUT_DIR / "inference_smoke_test.csv", index=False)
    return out


def write_environment_file() -> None:
    import sklearn
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": __import__("matplotlib").__version__,
    }
    with open(OUTPUT_DIR / "environment_versions.json", "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)


def write_report(
    metrics: dict[str, float],
    threshold: float,
    metadata: dict[str, Any],
    cm: np.ndarray,
    importance: pd.DataFrame,
    shortlisted: pd.DataFrame,
) -> None:
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    tn = int(cm[0, 0])
    model_name = metadata.get("model_name", "Selected Final Model")
    best_params = metadata.get("best_params", "Available in Day 4 artifact metadata")

    lines = [
        "# Adult Income Prediction — Final Day 5 Report",
        "",
        "## 1. Problem Definition",
        "The project predicts whether an Adult dataset record belongs to the >50K income class. Day 5 converts the tuned Day 4 solution into a reproducible, test-validated and inference-ready machine learning project.",
        "",
        "## 2. Data Preparation",
        f"The Adult dataset was loaded from OpenML. The deterministic split used throughout the project was stratified with random_state={RANDOM_STATE}: 80% development pool and 20% final hold-out test. The final hold-out was not used for model fitting, feature selection, threshold selection, or hyperparameter tuning.",
        "",
        "Feature engineering remains row-level only: age and hours buckets, capital-gain/loss indicators and log transforms, higher-education flag, education×hours and age×hours interactions, and net capital. The saved Day 4 artifact contains the actual preprocessing used in final inference.",
        "",
        "## 3. Model Development",
        f"Selected final model: **{model_name}**. The final pipeline is loaded directly from `final_model.joblib`, so preprocessing is not manually repeated during inference.",
        "",
        "## 4. Model Tuning",
        f"Best parameters: `{best_params}`. The final classification threshold used for validation is **{threshold:.4f}**.",
        "",
        "## 5. Diagnostics",
        "The project generates a precision learning curve and a calibration curve. The learning curve is used to inspect whether training and validation precision converge as data increases. Calibration is assessed with a calibration plot and Brier score.",
        "",
        "## 6. Final Results",
        "",
        "| Metric | Final Test Result |",
        "|---|---:|",
        f"| Accuracy | {metrics['Accuracy']:.4f} |",
        f"| Precision | {metrics['Precision']:.4f} |",
        f"| Recall | {metrics['Recall']:.4f} |",
        f"| F1 | {metrics['F1']:.4f} |",
        f"| ROC-AUC | {metrics['ROC-AUC']:.4f} |",
        f"| PR-AUC | {metrics['PR-AUC']:.4f} |",
        f"| Brier score | {metrics['Brier']:.4f} |",
        "",
        f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}.",
        f"False positives are records predicted >50K when the true class is <=50K. False negatives are records predicted <=50K when the true class is >50K. Because precision is the primary Day 1 objective, false positives are particularly important to monitor.",
        "",
        "## 7. Model Interpretation",
        "The final project exports model-agnostic permutation importance on the original input features and, where supported by the saved pipeline, native coefficients or feature importances after preprocessing. The most influential features should be reviewed for sensible relationships with income and for possible bias or proxy effects.",
        "",
        "Top original-input features from permutation importance:",
    ]

    for _, row in importance.head(5).iterrows():
        lines.append(f"- **{row['Feature']}**: mean precision decrease {row['ImportanceMean']:.5f}.")

    lines += [
        "",
        "## 8. Production Readiness",
        "The trained pipeline is stored as `final_model.joblib`. The inference workflow loads this single artifact, accepts raw Adult-style records, applies the saved preprocessing, returns probability for >50K, and converts that probability into the final class using the saved threshold.",
        "",
        "The inference smoke test evaluates eight examples that were not used for training. The results are saved in `inference_smoke_test.csv`.",
        "",
        "## 9. Limitations & Future Improvements",
        "The model is trained on the historical Adult dataset and should not be treated as a real-world income or eligibility decision system without additional validation. Important limitations include dataset age, possible demographic and socioeconomic bias, dependence on the chosen threshold, and subgroup performance differences. Future work should include external validation, fairness evaluation, threshold analysis by operational cost, more representative data, and monitoring for distribution shift.",
        "",
        "## Reproducibility",
        "Run `python Task_5_Adult_Income_Final.py` with `final_model.joblib` in the same directory. The script writes all metrics, plots, subgroup analysis, feature-importance outputs, environment versions, and the final 2–3 page report source into `adult_income_day5_outputs/`.",
        "",
        "### Key files",
        "- `final_model.joblib` — saved Day 4 production artifact",
        "- `inference.py` — lightweight inference entry point",
        "- `requirements.txt` — environment specification",
        "- `adult_income_day5_outputs/final_test_metrics.csv` — final metrics",
        "- `adult_income_day5_outputs/final_metrics_comparison.csv` — Day 4 shortlist + final model when Day 4 comparison CSV is available",
        "- `adult_income_day5_outputs/plots/` — confusion, ROC, PR, calibration, learning-curve, and feature-importance plots",
    ]

    if shortlisted.empty:
        lines += [
            "",
            "> Note: no Day 4 comparison CSV was present in the execution directory, so this report does not invent shortlisted-model metrics. Add the Day 4 comparison CSV and rerun to populate that table.",
        ]

    (OUTPUT_DIR / "final_project_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 80)
    print("ADULT INCOME — DAY 5 FINAL PROJECT")
    print("FINAL VALIDATION + ERROR ANALYSIS + PRODUCTION INFERENCE")
    print("=" * 80)

    write_environment_file()

    # 1. Load model artifact.
    model, threshold, metadata = load_model_artifact(ARTIFACT_PATH)
    print(f"Loaded artifact: {ARTIFACT_PATH}")
    print(f"Classification threshold: {threshold:.4f}")
    print(f"Model: {metadata.get('model_name', type(model).__name__)}")

    # 2. Load exactly the same raw dataset and recreate the untouched hold-out.
    X, y, target_col = load_adult()
    X_devpool, X_test, y_devpool, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Explicit leakage checks.
    train_ids = set(X_devpool.index)
    test_ids = set(X_test.index)
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, "Leakage detected: development and final-test row indices overlap."
    print(f"Development pool: {len(X_devpool)}")
    print(f"Final untouched hold-out: {len(X_test)}")
    print(f"Index overlap: {len(overlap)}")

    # 3. Final validation.
    probabilities = get_probability(model, X_test)
    predictions = (probabilities >= threshold).astype(int)
    metrics = evaluate_predictions(y_test, probabilities, threshold)
    metrics_df = save_final_metrics(metrics)

    print("\nFINAL TEST METRICS")
    print(metrics_df.round(6).to_string(index=False))

    # 4. Shortlisted comparison.
    shortlisted = load_shortlisted_models()
    comparison = append_selected_model(shortlisted, metrics, metadata)
    print("\nFINAL METRICS COMPARISON")
    print(comparison.round(6).to_string(index=False))

    # 5. Error analysis + subgroup analysis.
    prediction_table = error_analysis(model, X_test, y_test, probabilities, threshold)
    fp_count = int(((prediction_table["actual"] == 0) & (prediction_table["prediction"] == 1)).sum())
    fn_count = int(((prediction_table["actual"] == 1) & (prediction_table["prediction"] == 0)).sum())
    print(f"\nFalse positives: {fp_count}")
    print(f"False negatives: {fn_count}")

    # 6. Visual diagnostics.
    confusion_matrix_plot(y_test, predictions)
    roc_pr_plot(y_test, probabilities)
    calibration_plot(y_test, probabilities)
    learning_curve_plot(model, X_devpool, y_devpool)

    # 7. Feature interpretation.
    importance = permutation_feature_importance(model, X_test, y_test)
    native = extract_native_importance(model)
    print("\nTOP PERMUTATION FEATURES")
    print(importance.head(10).round(6).to_string(index=False))
    if native is not None:
        print("\nTOP NATIVE MODEL FEATURES")
        print(native.head(10).round(6).to_string(index=False))

    # 8. Inference smoke test.
    smoke = write_inference_examples(model, threshold, X_test)
    print("\nINFERENCE SMOKE TEST")
    print(smoke[["probability", "prediction_label"]].round(4).to_string(index=False))

    # 9. Final report.
    cm = confusion_matrix(y_test, predictions)
    write_report(metrics, threshold, metadata, cm, importance, shortlisted)

    print("\nOutputs written to:", OUTPUT_DIR.resolve())
    print("Completed successfully.")


if __name__ == "__main__":
    main()
