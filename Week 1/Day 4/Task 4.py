# ============================================================
# UCI ADULT INCOME — MODEL TUNING, REGULARIZATION &
# REPRODUCIBLE PIPELINES
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn import __version__ as sklearn_version

from sklearn.datasets import fetch_openml

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    learning_curve
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import (
    CalibrationDisplay,
    CalibratedClassifierCV
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42

print("Python:", sys.version)
print("Scikit-learn:", sklearn_version)
print("Random State:", RANDOM_STATE)


# ============================================================
# 3. FETCH UCI ADULT INCOME DATASET
# ============================================================

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

X = adult.data.copy()
y = adult.target.copy()

print("Dataset shape:", X.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.value_counts())


# ============================================================
# 4. CLEAN TARGET
# ============================================================

# Remove spaces from target labels
y = y.astype(str).str.strip()

print("\nTarget classes:")
print(y.unique())


# Convert target to binary:
# <=50K  -> 0
# >50K   -> 1

y = y.map({
    "<=50K": 0,
    ">50K": 1
})

if y.isnull().any():
    raise ValueError("Unexpected target values found.")

y = y.astype(int)

print("\nBinary target distribution:")
print(y.value_counts())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(X.isnull().sum())


# ============================================================
# 6. TRAIN / TEST SPLIT
#
# TEST SET WILL REMAIN COMPLETELY UNTOUCHED
# ============================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTraining set:", X_train_full.shape)
print("Test set:", X_test.shape)


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
#
# Validation is used for threshold selection.
# Test remains untouched until final evaluation.
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

print("\nModel training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Final test set:", X_test.shape)


# ============================================================
# 8. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 9. PREPROCESSING PIPELINES
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


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
# 10. STRATIFIED K-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 11. LOGISTIC REGRESSION PIPELINE
# ============================================================

logistic_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# 12. LOGISTIC REGRESSION SEARCH
#
# L1 / L2 regularization
# C = inverse regularization strength
# ============================================================

logistic_params = {
    "model__penalty": [
        "l1",
        "l2"
    ],

    "model__C": np.logspace(
        -3,
        2,
        30
    ),

    "model__solver": [
        "liblinear"
    ]
}


logistic_search = RandomizedSearchCV(
    estimator=logistic_pipeline,
    param_distributions=logistic_params,

    n_iter=50,

    scoring="f1",

    cv=cv,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1,

    refit=True
)


print("\nTraining Logistic Regression...")

logistic_search.fit(
    X_train,
    y_train
)


print("\nBest Logistic Regression parameters:")
print(logistic_search.best_params_)

print(
    "Best CV F1:",
    round(logistic_search.best_score_, 4)
)


# ============================================================
# 13. GRADIENT BOOSTING PIPELINE
# ============================================================

gradient_pipeline = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "model",
        GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# 14. GRADIENT BOOSTING SEARCH
# ============================================================

gradient_params = {

    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1,
        0.2
    ],

    "model__n_estimators": [
        50,
        100,
        150,
        200,
        300
    ],

    "model__max_depth": [
        1,
        2,
        3,
        4,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2,
        5,
        10,
        20
    ],

    "model__max_features": [
        None,
        "sqrt",
        "log2"
    ],

    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ]
}


gradient_search = RandomizedSearchCV(
    estimator=gradient_pipeline,

    param_distributions=gradient_params,

    n_iter=75,

    scoring="f1",

    cv=cv,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1,

    refit=True
)


print("\nTraining Gradient Boosting...")

gradient_search.fit(
    X_train,
    y_train
)


print("\nBest Gradient Boosting parameters:")
print(gradient_search.best_params_)

print(
    "Best CV F1:",
    round(gradient_search.best_score_, 4)
)


# ============================================================
# 15. COMPARE THE TWO MODELS
# ============================================================

models = {

    "Logistic Regression":
        logistic_search.best_estimator_,

    "Gradient Boosting":
        gradient_search.best_estimator_
}


def evaluate_model(
    model,
    X_data,
    y_data,
    threshold=0.5
):

    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "Accuracy":
            accuracy_score(
                y_data,
                predictions
            ),

        "Precision":
            precision_score(
                y_data,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_data,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_data,
                predictions,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_data,
                probabilities
            ),

        "Brier Score":
            brier_score_loss(
                y_data,
                probabilities
            ),

        "Log Loss":
            log_loss(
                y_data,
                probabilities
            )
    }


comparison = []


for name, model in models.items():

    metrics = evaluate_model(
        model,
        X_val,
        y_val
    )

    metrics["Model"] = name

    comparison.append(metrics)


comparison_df = pd.DataFrame(
    comparison
)


comparison_df = comparison_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "Brier Score",
        "Log Loss"
    ]
]


print("\nMODEL COMPARISON")
display(comparison_df)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

best_model_name = comparison_df.loc[
    comparison_df["F1"].idxmax(),
    "Model"
]

best_model = models[
    best_model_name
]


print(
    "\nSelected Best Model:",
    best_model_name
)


# ============================================================
# 17. TASK 3 — LEARNING CURVE
# ============================================================

train_sizes, train_scores, validation_scores = learning_curve(

    best_model,

    X_train,

    y_train,

    cv=cv,

    scoring="f1",

    train_sizes=np.linspace(
        0.1,
        1.0,
        8
    ),

    n_jobs=-1
)


train_mean = train_scores.mean(
    axis=1
)

train_std = train_scores.std(
    axis=1
)

validation_mean = validation_scores.mean(
    axis=1
)

validation_std = validation_scores.std(
    axis=1
)


plt.figure(figsize=(9, 6))


plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training F1"
)


plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15
)


plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Validation F1"
)


plt.fill_between(
    train_sizes,
    validation_mean - validation_std,
    validation_mean + validation_std,
    alpha=0.15
)


plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")

plt.title(
    "Learning Curve — Best Model"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 18. OVERFITTING / UNDERFITTING DIAGNOSIS
# ============================================================

train_final = train_mean[-1]

validation_final = validation_mean[-1]

gap = train_final - validation_final


print(
    "Training F1:",
    round(train_final, 4)
)

print(
    "Validation F1:",
    round(validation_final, 4)
)

print(
    "Train-Validation Gap:",
    round(gap, 4)
)


if gap > 0.10:

    print(
        "\nDiagnosis: POSSIBLE OVERFITTING"
    )

    print(
        "Recommended fixes:"
    )

    print(
        "- Stronger regularization"
    )

    print(
        "- Lower tree depth"
    )

    print(
        "- Increase min_samples_leaf"
    )

    print(
        "- Collect more training data"
    )


elif (
    train_final < 0.70
    and validation_final < 0.70
):

    print(
        "\nDiagnosis: POSSIBLE UNDERFITTING"
    )

    print(
        "Recommended fixes:"
    )

    print(
        "- Reduce regularization"
    )

    print(
        "- Increase model complexity"
    )

    print(
        "- Add better features"
    )


else:

    print(
        "\nDiagnosis: Reasonably balanced model"
    )


# ============================================================
# 19. EFFECT OF REGULARIZATION
#    Logistic Regression: C vs F1
# ============================================================

C_values = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
]


train_f1 = []
validation_f1 = []


for C in C_values:

    model = Pipeline([

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                C=C,
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )
    ])


    scores = []


    for train_idx, val_idx in cv.split(
        X_train,
        y_train
    ):

        X_tr = X_train.iloc[
            train_idx
        ]

        X_cv = X_train.iloc[
            val_idx
        ]

        y_tr = y_train.iloc[
            train_idx
        ]

        y_cv = y_train.iloc[
            val_idx
        ]


        model.fit(
            X_tr,
            y_tr
        )


        train_pred = model.predict(
            X_tr
        )

        val_pred = model.predict(
            X_cv
        )


        scores.append([

            f1_score(
                y_tr,
                train_pred
            ),

            f1_score(
                y_cv,
                val_pred
            )
        ])


    scores = np.array(scores)


    train_f1.append(
        scores[:, 0].mean()
    )

    validation_f1.append(
        scores[:, 1].mean()
    )


plt.figure(figsize=(9, 6))


plt.semilogx(
    C_values,
    train_f1,
    marker="o",
    label="Training F1"
)


plt.semilogx(
    C_values,
    validation_f1,
    marker="o",
    label="Validation F1"
)


plt.xlabel(
    "C — Inverse Regularization Strength"
)

plt.ylabel("F1 Score")

plt.title(
    "Effect of Regularization on Logistic Regression"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 20. TASK 4 — PROBABILITY CALIBRATION
#
# Use validation set to evaluate calibration.
# ============================================================

calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    method="sigmoid",
    cv=5
)


print(
    "\nTraining calibrated model..."
)


calibrated_model.fit(
    X_train,
    y_train
)


# ============================================================
# 21. CALIBRATION PLOT
# ============================================================

plt.figure(figsize=(9, 6))


CalibrationDisplay.from_estimator(

    best_model,

    X_val,

    y_val,

    n_bins=10,

    name="Before Calibration"
)


CalibrationDisplay.from_estimator(

    calibrated_model,

    X_val,

    y_val,

    n_bins=10,

    name="After Calibration"
)


plt.title(
    "Probability Calibration"
)

plt.grid(True)

plt.show()


# ============================================================
# 22. BRiER SCORE
# ============================================================

uncalibrated_prob = (
    best_model
    .predict_proba(X_val)[:, 1]
)


calibrated_prob = (
    calibrated_model
    .predict_proba(X_val)[:, 1]
)


uncalibrated_brier = brier_score_loss(
    y_val,
    uncalibrated_prob
)


calibrated_brier = brier_score_loss(
    y_val,
    calibrated_prob
)


print(
    "Brier Score Before Calibration:",
    round(uncalibrated_brier, 5)
)


print(
    "Brier Score After Calibration:",
    round(calibrated_brier, 5)
)


# ============================================================
# 23. TASK 4 — THRESHOLD SELECTION
#
# IMPORTANT:
# Threshold is optimized on VALIDATION SET,
# NOT on the final TEST SET.
# ============================================================

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        calibrated_prob >= threshold
    ).astype(int)


    threshold_results.append({

        "Threshold":
            threshold,

        "Accuracy":
            accuracy_score(
                y_val,
                predictions
            ),

        "Precision":
            precision_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_val,
                predictions,
                zero_division=0
            )
    })


threshold_df = pd.DataFrame(
    threshold_results
)


best_threshold_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]


best_threshold = float(
    best_threshold_row["Threshold"]
)


print(
    "\nBest Classification Threshold:",
    round(best_threshold, 3)
)


print(
    "\nValidation Metrics at Best Threshold:"
)


display(
    best_threshold_row.to_frame().T
)


# ============================================================
# 24. THRESHOLD CURVES
# ============================================================

plt.figure(figsize=(9, 6))


plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)


plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)


plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1"
)


plt.axvline(
    best_threshold,
    linestyle="--",
    label=(
        f"Best Threshold = "
        f"{best_threshold:.2f}"
    )
)


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel("Score")

plt.title(
    "Threshold Selection"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 25. CONFUSION MATRIX — DEFAULT THRESHOLD
# ============================================================

default_predictions = (
    calibrated_prob >= 0.50
).astype(int)


cm_default = confusion_matrix(
    y_val,
    default_predictions
)


print(
    "Confusion Matrix — Threshold 0.50"
)

print(cm_default)


# ============================================================
# 26. CONFUSION MATRIX — OPTIMAL THRESHOLD
# ============================================================

optimal_predictions = (
    calibrated_prob >= best_threshold
).astype(int)


cm_optimal = confusion_matrix(
    y_val,
    optimal_predictions
)


print(
    "\nConfusion Matrix — Optimal Threshold"
)

print(cm_optimal)


# ============================================================
# 27. ROC CURVE — VALIDATION
# ============================================================

fpr, tpr, _ = roc_curve(
    y_val,
    calibrated_prob
)


auc = roc_auc_score(
    y_val,
    calibrated_prob
)


plt.figure(figsize=(8, 6))


plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.3f}"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 28. PRECISION-RECALL CURVE
# ============================================================

precision, recall, _ = precision_recall_curve(
    y_val,
    calibrated_prob
)


plt.figure(figsize=(8, 6))


plt.plot(
    recall,
    precision
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve"
)

plt.grid(True)

plt.show()


# ============================================================
# 29. FINAL MODEL
#
# Now we can use TRAIN + VALIDATION data.
#
# Test remains untouched.
# ============================================================

X_final_train = pd.concat(
    [X_train, X_val],
    axis=0
)

y_final_train = pd.concat(
    [y_train, y_val],
    axis=0
)


print(
    "\nFinal training data:",
    X_final_train.shape
)


# ============================================================
# 30. RETRAIN BEST MODEL WITH CALIBRATION
# ============================================================

final_calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    method="sigmoid",
    cv=5
)


print(
    "\nTraining final calibrated model..."
)


final_calibrated_model.fit(
    X_final_train,
    y_final_train
)


# ============================================================
# 31. FINAL TEST PREDICTIONS
#
# The test set is used ONLY here.
# ============================================================

test_probabilities = (
    final_calibrated_model
    .predict_proba(X_test)[:, 1]
)


test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# 32. FINAL TEST METRICS
# ============================================================

final_metrics = {

    "Accuracy":
        accuracy_score(
            y_test,
            test_predictions
        ),

    "Precision":
        precision_score(
            y_test,
            test_predictions,
            zero_division=0
        ),

    "Recall":
        recall_score(
            y_test,
            test_predictions,
            zero_division=0
        ),

    "F1":
        f1_score(
            y_test,
            test_predictions,
            zero_division=0
        ),

    "ROC-AUC":
        roc_auc_score(
            y_test,
            test_probabilities
        ),

    "Brier Score":
        brier_score_loss(
            y_test,
            test_probabilities
        ),

    "Log Loss":
        log_loss(
            y_test,
            test_probabilities
        )
}


final_metrics_df = pd.DataFrame(
    [final_metrics]
)


print(
    "\n======================================"
)

print(
    "FINAL TEST PERFORMANCE"
)

print(
    "======================================"
)


display(
    final_metrics_df
)


# ============================================================
# 33. FINAL CLASSIFICATION REPORT
# ============================================================

print(
    "\nFINAL CLASSIFICATION REPORT"
)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "<=50K",
            ">50K"
        ],
        zero_division=0
    )
)


# ============================================================
# 34. FINAL CONFUSION MATRIX
# ============================================================

final_cm = confusion_matrix(
    y_test,
    test_predictions
)


print(
    "\nFINAL TEST CONFUSION MATRIX"
)

print(final_cm)


plt.figure(figsize=(6, 5))


plt.imshow(final_cm)


plt.title(
    f"Final Test Confusion Matrix\n"
    f"Threshold = {best_threshold:.2f}"
)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)


plt.xticks(
    [0, 1],
    ["<=50K", ">50K"]
)

plt.yticks(
    [0, 1],
    ["<=50K", ">50K"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            final_cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.show()


# ============================================================
# 35. FINAL ROC CURVE
# ============================================================

test_fpr, test_tpr, _ = roc_curve(
    y_test,
    test_probabilities
)


test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


plt.figure(figsize=(8, 6))


plt.plot(
    test_fpr,
    test_tpr,
    label=f"ROC-AUC = {test_auc:.3f}"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Final Test ROC Curve"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 36. SAVE FINAL MODEL ARTIFACT
# ============================================================

artifact = {

    "model":
        final_calibrated_model,

    "threshold":
        best_threshold,

    "model_name":
        best_model_name,

    "best_parameters":
        (
            logistic_search.best_params_
            if best_model_name ==
            "Logistic Regression"
            else gradient_search.best_params_
        ),

    "random_state":
        RANDOM_STATE,

    "sklearn_version":
        sklearn_version,

    "features":
        list(X.columns),

    "target":
        "income",

    "target_mapping":
        {
            "<=50K": 0,
            ">50K": 1
        }
}


MODEL_PATH = "adult_income_final_model.joblib"


joblib.dump(
    artifact,
    MODEL_PATH
)


print(
    "\nModel saved successfully:"
)

print(
    MODEL_PATH
)


# ============================================================
# 37. HOW-TO-INFER FUNCTION
# ============================================================

def load_adult_income_model(
    model_path="adult_income_final_model.joblib"
):

    artifact = joblib.load(
        model_path
    )

    model = artifact["model"]

    threshold = artifact["threshold"]

    return model, threshold


def predict_income(
    new_data,
    model_path="adult_income_final_model.joblib"
):

    model, threshold = (
        load_adult_income_model(
            model_path
        )
    )


    probabilities = (
        model
        .predict_proba(new_data)[:, 1]
    )


    predictions = (
        probabilities >= threshold
    ).astype(int)


    result = new_data.copy()


    result["probability_>50K"] = (
        probabilities
    )


    result["prediction"] = (
        predictions
    )


    result["income_prediction"] = (
        np.where(
            predictions == 1,
            ">50K",
            "<=50K"
        )
    )


    return result


# ============================================================
# 38. EXAMPLE INFERENCE
# ============================================================

# Example:
#
# new_person = X_test.iloc[[0]]
#
# result = predict_income(
#     new_person
# )
#
# print(result)
#
# ============================================================


# ============================================================
# 39. FINAL SUMMARY FOR REPORT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print(
    "Dataset: UCI Adult Income"
)

print(
    "Best Model:",
    best_model_name
)

print(
    "Best Threshold:",
    round(best_threshold, 3)
)

print(
    "\nBest Hyperparameters:"
)


if best_model_name == "Logistic Regression":

    print(
        logistic_search.best_params_
    )

else:

    print(
        gradient_search.best_params_
    )


print(
    "\nFinal Test Metrics:"
)


for metric, value in final_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


print(
    "\nExpected Production Behavior:"
)

print(
    "- Preprocessing is included inside the pipeline."
)

print(
    "- Unknown categorical values are handled safely."
)

print(
    "- Probabilities are calibrated."
)

print(
    "- Classification threshold is optimized using validation data."
)

print(
    "- Final test set was kept untouched until final evaluation."
)

print(
    "- Random state is fixed for reproducibility."
)

print(
    "- Complete model artifact is saved using joblib."
)
