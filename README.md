<p align="center">
  <img src="https://img.shields.io/badge/AI-Artificial%20Intelligence-8A2BE2?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/ML-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/DS-Data%20Science-00A67E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

<p align="center">
  <b>Practical AI • Machine Learning • Data Science</b>
  <br>
  Learning through real datasets, experimentation, evaluation, and iterative improvement.
</p>

---

## 📖 About This Repository

**Personal-AI-DS-Tasks** is a practical collection of Artificial Intelligence, Machine Learning, and Data Science tasks designed to build skills through hands-on experimentation with real datasets.

The repository focuses on developing the complete ML workflow rather than only training models.

It covers:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 📊 Data Science
* 🐍 Python
* 📈 Data Analysis
* 🧹 Data Preprocessing
* 🧪 Model Evaluation
* 🔍 Error Analysis
* 📊 Model Comparison
* 🔬 Experimentation
* 🚀 Iterative Model Improvement

The overall philosophy of this repository is:

> **Understand → Clean → Analyze → Baseline → Preprocess → Build → Evaluate → Interpret → Improve**

Each task documents:

> **What I did → How I did it → What I found → What problems I faced → What I will improve next**

---

# 🛠️ Skills Covered

## 🤖 Artificial Intelligence

* AI problem definition
* Business-oriented problem solving
* Classification problems
* Data-driven decision making
* Model interpretation
* Error analysis
* Fairness considerations
* Model selection

## 🧠 Machine Learning

* Supervised Learning
* Binary Classification
* Baseline Modeling
* Logistic Regression
* Decision Trees
* Train / Development / Test Splitting
* Stratified Sampling
* Reproducible Experiments
* Pipeline-based preprocessing
* Feature transformation
* Numerical feature scaling
* Categorical feature encoding
* Precision
* Recall
* F1 Score
* ROC AUC
* PR AUC
* Confusion Matrix
* False Positive / False Negative Analysis
* Model Comparison
* Overfitting Analysis
* Model Interpretability

## 📊 Data Science

* Data Loading
* Data Cleaning
* Missing Value Analysis
* Exploratory Data Analysis
* Numerical Statistics
* Categorical Analysis
* Feature Distributions
* Data Visualization
* Class Distribution Analysis
* Feature Investigation
* Error Pattern Analysis
* Model Performance Analysis
* Feature Interpretation

---

# 💻 Tools & Technologies

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white" />
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square" />
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
<img src="https://img.shields.io/badge/OpenML-Data%20Source-1F77B4?style=flat-square" />

</p>

---

# 📌 Task 1 — Adult Income Prediction

## 💰 UCI Adult / Census Income

### 🎯 Objective

Task 1 focuses on the **UCI Adult / Census Income dataset**.

The goal is to predict whether a person earns more than **$50,000 per year**.

### Target Definition

| Income  | Class |
| ------- | :---: |
| `<=50K` |  `0`  |
| `>50K`  |  `1`  |

The project was designed as a starting point for a real-world binary classification problem where the objective is not only to train a model, but to understand the dataset, establish meaningful baselines, build reliable evaluation procedures, and analyze model errors.

---

# 💼 Business Problem

A possible business application is **targeted higher-value outreach**.

For example, a company may want to identify people who are more likely to belong to the higher-income group before contacting them.

In this scenario, incorrectly predicting someone as a high-income individual can result in:

* Wasted outreach effort
* Unnecessary marketing cost
* Lower-quality targeting

Therefore, **precision** was selected as the primary business metric.

---

# 🎯 Primary Metric — Precision

Precision answers:

> **When the model predicts that someone earns more than $50K, how often is that prediction correct?**

Precision was prioritized because the assumed business scenario places more importance on avoiding unnecessary positive predictions.

Recall is still monitored because a model that misses too many actual high-income individuals could also have limited business value.

The project therefore does not optimize for accuracy alone.

---

# 📂 Dataset

The dataset was loaded directly from **OpenML** using Scikit-learn.

```python
from sklearn.datasets import fetch_openml

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)

df = adult.frame.copy()
```

Using `fetch_openml()` makes the dataset acquisition reproducible and avoids manually preparing the raw UCI files.

---

# 🧹 Data Cleaning

Before analysis and modeling, the dataset was cleaned and standardized.

### What I did

* Loaded the dataset into a Pandas DataFrame
* Inspected the raw shape
* Inspected column names and data types
* Removed unnecessary whitespace
* Converted `?` and `" ?"` values into missing values
* Normalized target labels
* Removed trailing `.` from target labels when required
* Converted income into binary `0/1`
* Removed rows with missing target values
* Checked remaining missing values

### Missing Values

The Adult dataset can represent missing categorical information using:

```text
?
```

These values were converted to:

```python
np.nan
```

This allows missing values to be handled consistently by Pandas and the sklearn preprocessing pipeline.

### Target Conversion

The target was converted to:

```python
{
    ">50K": 1,
    "<=50K": 0
}
```

This transforms the problem into a standard binary classification task.

---

# 🔎 Exploratory Data Analysis

After cleaning the dataset, I performed **Exploratory Data Analysis (EDA)** to understand the data before modeling.

## 📋 Dataset Inspection

I examined:

* Dataset shape
* Column names
* Numerical columns
* Categorical columns
* Data types
* Numerical statistics
* Missing values
* Category frequencies
* Class distribution

### Numerical Analysis

For numerical features, I calculated:

* Mean
* Median
* Standard deviation
* 25th percentile
* 75th percentile
* Missing-value count

Important numerical features included:

* `age`
* `education-num`
* `hours-per-week`
* `capital-gain`
* `capital-loss`

---

# 📊 Visual Analysis

Visualizations were created to understand important feature distributions.

### Numerical Features

Histograms were created for:

* Age
* Education number
* Hours per week
* Capital gain
* Capital loss

### Categorical Features

Bar plots were created for:

* Education
* Marital status
* Workclass
* Occupation

These visualizations helped identify:

* Common categories
* Skewed numerical features
* Concentration around certain values
* Potential missing-value problems
* Differences in feature distributions

---

# 📈 Class Distribution

The number and percentage of:

* Negative examples: `<=50K`
* Positive examples: `>50K`

were calculated.

The positive class is smaller than the negative class.

This demonstrated why **accuracy alone should not be the primary metric**.

A model can achieve relatively high accuracy by predicting the majority class while performing poorly on the positive class.

---

# 🔀 Reproducible Train / Development / Test Split

A reliable evaluation strategy was created before modeling.

The experiment uses:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
```

The final hold-out test set was separated before experimentation.

### Dataset Split

* 🟦 **Training:** approximately 70%
* 🟨 **Development:** approximately 10%
* 🟥 **Final Hold-out Test:** approximately 20%

The split was stratified using the target variable so that the class proportions remain similar between datasets.

### 🔒 Hold-out Test Principle

The final test set represents unseen data.

It must not be used for:

* Feature selection
* Model selection
* Hyperparameter tuning
* Preprocessing decisions
* Repeated experimentation

Using the test set during development can cause **test-set leakage** and lead to overly optimistic results.

Therefore:

> **The final hold-out test is reserved for final evaluation.**

---

# 🧪 Day 1 — Baseline Modeling

Before training real supervised models, two simple baselines were established.

The purpose of these baselines was to create a minimum performance reference.

---

## 🧱 Baseline 1 — Majority Class

The majority-class baseline predicts the most common class for every observation.

For this dataset:

```text
Predict <=50K for everyone
```

This baseline demonstrates how far a model can get without learning meaningful relationships from the features.

It can achieve reasonable accuracy because the negative class is more common, but it provides essentially no useful positive-class identification.

---

## 🎓 Baseline 2 — Education Rule

The second baseline uses only:

```text
education-num
```

The rule was:

```text
education-num >= 13
        ↓
Predict >50K
```

Otherwise:

```text
Predict <=50K
```

Education was selected because it has a meaningful relationship with income.

The rule remains intentionally simple so that future machine-learning models have a clear and interpretable benchmark.

---

# 📏 Day 1 Evaluation Framework

The baselines were evaluated using:

| Metric        | Purpose                                          |
| ------------- | ------------------------------------------------ |
| **Accuracy**  | Overall percentage of correct predictions        |
| **Precision** | Correctness of positive predictions              |
| **Recall**    | Percentage of actual positives identified        |
| **F1 Score**  | Balance between precision and recall             |
| **ROC AUC**   | Overall discrimination/ranking ability           |
| **PR AUC**    | Positive-class performance under class imbalance |

---

# 🧮 Day 1 Confusion Matrix

The four prediction outcomes are:

|                     |   Actual <=50K |    Actual >50K |
| ------------------- | -------------: | -------------: |
| **Predicted <=50K** |  True Negative | False Negative |
| **Predicted >50K**  | False Positive |  True Positive |

### False Positive

The model predicts:

```text
>50K
```

but the actual class is:

```text
<=50K
```

This is particularly important because false positives directly reduce precision.

### False Negative

The model predicts:

```text
<=50K
```

but the actual class is:

```text
>50K
```

False negatives reduce recall.

---

# 🔍 Day 1 Error Analysis

For the education-based baseline, predictions were divided into:

* False positives
* False negatives

At least 10 examples from each category were inspected.

Features examined included:

* `age`
* `education`
* `education-num`
* `hours-per-week`
* `capital-gain`
* `capital-loss`
* `workclass`
* `occupation`
* `marital-status`
* `relationship`
* `sex`

This showed that income cannot be reliably explained using education alone.

---

# 💡 Day 1 Main Insight

The education rule provides a useful interpretable baseline, but it is too simple to represent the full income prediction problem.

Income can depend on combinations of:

> **Education + Age + Occupation + Work Hours + Workclass + Capital Features**

A real supervised model can learn these relationships simultaneously.

This created the motivation for the next modeling stage.

---

# 🚀 Task 2 — First Real Supervised Models

The second stage moves from simple hand-written baselines to **real supervised machine-learning models**.

The focus was:

* Leakage-safe preprocessing
* Repeatable sklearn pipelines
* Correct handling of mixed numerical and categorical data
* Logistic Regression
* Decision Tree
* Multi-metric evaluation
* Model interpretability
* Model comparison
* Candidate selection for further development

---

# 🧹 Task 2 — Preprocessing Plan

The Adult dataset contains both numerical and categorical features.

Instead of manually preprocessing the data before training, preprocessing was placed inside sklearn pipelines.

This ensures that transformations are learned only from the training data.

---

# 🔢 Numerical Features

The numerical feature set includes:

```text
age
fnlwgt
education-num
capital-gain
capital-loss
hours-per-week
```

These features represent continuous or numerical measurements.

---

# 🔤 Categorical Features

The categorical feature set includes:

```text
workclass
education
marital-status
occupation
relationship
race
sex
native-country
```

These features represent discrete categories.

---

# 🏗️ Preprocessing Architecture

A `ColumnTransformer` was used to apply different preprocessing operations to numerical and categorical features.

### Numerical Pipeline

```text
Numerical Features
        ↓
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

### Categorical Pipeline

```text
Categorical Features
        ↓
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

The complete structure is:

```text
                    ┌── Numerical ──→ Median Imputation ──→ StandardScaler ──┐
Raw Features ───────┤                                                         ├──→ Model
                    └── Categorical → Most-Frequent Imputation → One-Hot ────┘
```

---

# 🧠 Why Median Imputation?

Median imputation was selected for numerical features because it is:

* Simple
* Robust
* Easy to reproduce
* Less sensitive to extreme values than mean imputation

This is particularly useful because features such as `capital-gain` and `capital-loss` are highly skewed.

### Alternatives Considered

Other approaches could include:

* Mean imputation
* KNN imputation
* Iterative imputation
* Model-based imputation

These were not selected for this stage because they introduce additional complexity that is unnecessary for the first supervised-model baseline.

---

# 🔤 Why One-Hot Encoding?

One-hot encoding converts categorical values into numerical indicator features.

For example:

```text
workclass = Private
```

can become:

```text
workclass_Private = 1
```

while other workclass indicators become `0`.

`handle_unknown="ignore"` was intentionally used so that an unseen category during validation or testing does not cause the pipeline to fail.

### Alternatives Considered

Alternatives include:

* Ordinal encoding
* Target encoding
* Frequency encoding
* Hash encoding

Ordinal encoding was not preferred because categorical values such as occupation or workclass do not necessarily have a natural numerical order.

Target encoding was also skipped at this stage because it requires additional leakage-aware handling.

---

# 🔒 Why Pipelines Matter

The preprocessing operations were combined with the estimator inside a single sklearn `Pipeline`.

This means:

```text
Training Data
     ↓
Fit Imputer
     ↓
Fit Scaler
     ↓
Fit Encoder
     ↓
Fit Model
```

The transformations are learned from training data only.

When evaluating on unseen data:

```text
Hold-out Data
     ↓
Existing fitted preprocessing
     ↓
Existing fitted model
     ↓
Prediction
```

The hold-out data does not influence the fitting of the preprocessing parameters.

This provides a repeatable and leakage-resistant workflow.

---

# 🤖 Task 3 — Logistic Regression

The first supervised model was **Logistic Regression**.

The model was selected because it provides:

* A strong and simple classification baseline
* Fast training
* Regularization
* Probabilistic predictions
* Interpretable coefficients
* A useful comparison against nonlinear models

A solver supporting regularization was used.

The model was placed inside the complete preprocessing pipeline.

Conceptually:

```text
Raw Data
   ↓
Preprocessing
   ↓
One-Hot Encoded Features
   ↓
Logistic Regression
   ↓
Probability / Class Prediction
```

---

# 🌳 Task 4 — Decision Tree Classifier

The second supervised model was a **Decision Tree Classifier**.

Decision Trees were selected because they can:

* Learn nonlinear relationships
* Capture feature interactions
* Work naturally with transformed features
* Produce interpretable decision rules
* Provide feature importance
* Be compared directly against a linear classifier

The tree was also placed inside the same preprocessing framework.

Conceptually:

```text
Raw Data
   ↓
Preprocessing
   ↓
Encoded Features
   ↓
Decision Tree
   ↓
Probability / Class Prediction
```

---

# 🧪 Training Strategy

Both models were trained using:

```text
Training Set only
```

The hold-out test set was not used for model fitting.

The development set remains available for future:

* Hyperparameter tuning
* Threshold optimization
* Model selection
* Feature engineering experiments

This keeps the final hold-out evaluation meaningful.

---

# 📊 Task 5 — Model Evaluation

Both supervised models were evaluated on the final hold-out test set.

The following metrics were calculated:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* PR AUC

The Day 1 baselines were included in the same comparison framework.

---

# 📈 Model Comparison

The final comparison table is generated directly from the experiment.

| Model               |  Accuracy | Precision |    Recall |        F1 |   ROC AUC |    PR AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| Majority Class      | Generated | Generated | Generated | Generated | Generated | Generated |
| Education ≥ 13      | Generated | Generated | Generated | Generated | Generated | Generated |
| Logistic Regression | Generated | Generated | Generated | Generated | Generated | Generated |
| Decision Tree       | Generated | Generated | Generated | Generated | Generated | Generated |

Keeping all models in one table makes it easier to determine whether the supervised models provide meaningful improvement over the Day 1 baselines.

---

# 📉 ROC Curve

ROC curves were generated for both supervised models.

The ROC curve evaluates the relationship between:

```text
True Positive Rate
        vs.
False Positive Rate
```

ROC AUC provides a threshold-independent summary of ranking performance.

---

# 📊 Precision–Recall Curve

Precision–Recall curves were also generated.

This is especially useful because the positive class is smaller than the negative class.

The curve shows the trade-off between:

```text
Precision
     vs.
Recall
```

This is particularly relevant because **precision is the primary business metric** for this project.

---

# 🧮 Confusion Matrix Analysis

A confusion matrix was generated for each supervised model.

The main question was:

> **Which error type is more common — false positives or false negatives?**

If false positives are high:

```text
Precision ↓
```

This is especially problematic because precision is the selected primary metric.

If false negatives are high:

```text
Recall ↓
```

This means the model is missing many actual `>50K` cases.

Therefore, confusion matrices provide additional information that cannot be obtained from accuracy alone.

---

# 🎯 Decision-Making Based on Errors

The model selection process considers the business objective.

Because precision is prioritized:

> A candidate with slightly lower accuracy but substantially better precision may be more valuable than a model with higher accuracy but many false positives.

However, precision cannot be considered in isolation.

The final candidate should maintain a reasonable balance between:

* Precision
* Recall
* F1
* PR AUC
* Overall discrimination

---

# 🔬 Task 6 — Logistic Regression Interpretability

Logistic Regression provides a useful interpretability advantage.

After fitting the preprocessing pipeline, the transformed feature names can be recovered using:

```python
get_feature_names_out()
```

This allows the learned coefficients to be mapped back to the original numerical and one-hot encoded features.

---

# 📈 Top Positive Coefficients

The 10 largest positive coefficients are extracted.

A positive coefficient means that, holding other model features constant, increasing the corresponding transformed feature is associated with a higher model tendency toward:

```text
>50K
```

For categorical features, a positive coefficient corresponds to a particular category being associated with the positive class relative to the encoding reference structure.

---

# 📉 Top Negative Coefficients

The 10 most negative coefficients are also extracted.

A negative coefficient indicates that the corresponding transformed feature is associated with a lower model tendency toward:

```text
>50K
```

This provides a transparent way to inspect what the model learned.

The coefficients should be interpreted as **model associations**, not as proof of causation.

---

# 🌳 Task 7 — Decision Tree Interpretability

The Decision Tree was inspected to determine whether it was learning sensible patterns.

The following were examined:

* Tree depth
* Training score
* Hold-out score
* Top feature splits
* Important decision rules

---

# ⚠️ Overfitting Check

The difference between training and test performance was examined.

For example:

```text
Training Score >> Test Score
```

may indicate that the tree is overfitting.

A smaller gap suggests better generalization.

Decision Trees are particularly susceptible to overfitting when allowed to grow too deeply.

---

# 🔝 Top 3 Tree Splits

The first three important splits were extracted from the fitted tree.

These splits help answer:

* Which features are being used first?
* Are the decisions sensible?
* Is the model relying on meaningful attributes?
* Does the tree appear excessively complex?

The tree's logic was compared against domain expectations and the EDA findings from Day 1.

---

# 🧠 Model Interpretation

The two models provide complementary strengths.

### Logistic Regression

Strong for:

* Interpretability
* Coefficient analysis
* Stable baseline performance
* Understanding feature associations
* Regularized linear decision boundaries

### Decision Tree

Strong for:

* Nonlinear relationships
* Feature interactions
* Human-readable decision rules
* Capturing threshold-based behavior

This makes them useful complementary candidates for future experiments.

---

# ⚠️ Issues & Challenges

## 1. Mixed Data Types

The dataset contains both numerical and categorical features.

### Solution

Use `ColumnTransformer` to apply feature-specific preprocessing.

---

## 2. Missing Values

Categorical and numerical columns can contain missing values.

### Solution

Use:

```text
Median imputation → numerical features
Most-frequent imputation → categorical features
```

inside the pipeline.

---

## 3. Categorical Encoding

Machine-learning models cannot directly consume raw categorical strings.

### Solution

Use:

```text
OneHotEncoder(handle_unknown="ignore")
```

---

## 4. Data Leakage

Performing preprocessing before splitting the dataset can allow information from validation or test data to influence preprocessing.

### Solution

Place preprocessing inside sklearn pipelines and fit models only on training data.

---

## 5. High-Dimensional One-Hot Features

One-hot encoding can significantly increase the number of input features.

### Solution

Use sparse representations where supported and use regularized models such as Logistic Regression.

---

## 6. Skewed Numerical Features

`capital-gain` and `capital-loss` remain strongly skewed.

### Future Improvement

Test:

* Log transformations
* Binary indicators
* Robust transformations
* Alternative feature engineering

---

## 7. Model Complexity

Decision Trees can easily become too complex.

### Future Improvement

Experiment with:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`
* Cost-complexity pruning

---

## 8. Class Imbalance

The positive class is smaller than the negative class.

### Future Improvement

Continue monitoring:

* Precision
* Recall
* F1
* PR AUC

and investigate class weighting or threshold optimization.

---

## 9. Sensitive Attributes

The dataset contains features such as:

* Race
* Sex

These variables introduce potential fairness concerns.

Before any real-world deployment, subgroup performance and fairness should be investigated.

---

# 🖼️ Task 1–2 Pipeline

<p align="center">
  <img src="./assets/adult_income_pipeline.png" alt="Adult Income Prediction Machine Learning Pipeline" width="100%">
</p>

<p align="center">
  <i>End-to-end Adult Income classification workflow</i>
</p>

---

# 📁 Generated Outputs

The project produces reusable outputs including:

```text
adult_income_outputs/
│
├── class_summary.csv
├── feature_distribution_summary.csv
├── baseline_metrics.csv
├── false_positive_sample.csv
├── false_negative_sample.csv
├── error_profile.csv
├── model_comparison.csv
├── logistic_top_positive_coefficients.csv
├── logistic_top_negative_coefficients.csv
└── decision_tree_top_splits.csv
```

Visualizations are stored under:

```text
adult_income_plots/
```

including:

* Numerical histograms
* Categorical bar plots
* Baseline confusion matrices
* Logistic Regression confusion matrix
* Decision Tree confusion matrix
* ROC curves
* Precision–Recall curves

---

# 🗂️ Repository Structure

```text
Personal-AI-DS-Tasks/
│
├── 📄 Adult Income Detail.py
│
├── 📄 adult_income_supervised_models.py
│
├── 📓 adult_income_baselines_error_analysis.ipynb
│
├── 📂 assets/
│   └── 🖼️ adult_income_pipeline.png
│
├── 📂 adult_income_plots/
│   ├── hist_age.png
│   ├── hist_education-num.png
│   ├── hist_hours-per-week.png
│   ├── hist_capital-gain.png
│   ├── hist_capital-loss.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── logistic_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   └── ...
│
├── 📂 adult_income_outputs/
│   ├── class_summary.csv
│   ├── feature_distribution_summary.csv
│   ├── baseline_metrics.csv
│   ├── false_positive_sample.csv
│   ├── false_negative_sample.csv
│   ├── error_profile.csv
│   ├── model_comparison.csv
│   ├── logistic_top_positive_coefficients.csv
│   ├── logistic_top_negative_coefficients.csv
│   └── decision_tree_top_splits.csv
│
└── 📖 README.md
```

---

# 🧰 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ck-ahmad/Personal-AI-DS-Tasks.git
cd Personal-AI-DS-Tasks
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the Day 1 notebook

```bash
jupyter notebook
```

Open:

```text
adult_income_baselines_error_analysis.ipynb
```

and run the cells from top to bottom.

---

### 4. Run the supervised-model stage

The supervised model implementation can be executed with:

```bash
python "adult_income_supervised_models.py"
```

The script:

1. Loads the Adult dataset
2. Cleans the data
3. Creates the train/dev/test split
4. Builds preprocessing pipelines
5. Builds Logistic Regression and Decision Tree pipelines
6. Fits both models on training data
7. Evaluates them on the hold-out test set
8. Generates comparison metrics
9. Creates ROC and Precision–Recall curves
10. Generates confusion matrices
11. Extracts Logistic Regression coefficients
12. Inspects Decision Tree depth and splits
13. Saves reusable outputs

The dataset is fetched automatically through OpenML.

---

# 🔐 Reproducibility & Leakage Prevention

This project emphasizes reproducibility.

The experiment uses:

```python
RANDOM_STATE = 42
```

Preprocessing is performed inside sklearn pipelines.

The final hold-out test is not used during model fitting or preprocessing.

The overall structure is:

```text
                    ┌─────────────────────────┐
                    │       Raw Dataset       │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │      Data Cleaning      │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Stratified Train/Dev/   │
                    │     Hold-out Split      │
                    └────────────┬────────────┘
                                 ↓
              ┌──────────────────┴──────────────────┐
              ↓                                     ↓
       Training Data                         Hold-out Test
              ↓                                     │
       ColumnTransformer                            │
              ↓                                     │
      ┌───────┴────────┐                            │
      ↓                ↓                            │
 Numerical        Categorical                       │
      ↓                ↓                            │
 Imputer          Imputer                           │
      ↓                ↓                            │
 Scaler         One-Hot Encoder                     │
      └───────┬────────┘                            │
              ↓                                     │
       ┌──────┴───────┐                             │
       ↓              ↓                             │
 Logistic Regression  Decision Tree                 │
       ↓              ↓                             │
       └──────┬───────┘                             │
              ↓                                     ↓
        Model Selection ←────── Final Evaluation ───┘
```

---

# 📊 Task 2 Results

The exact model metrics are generated directly by the reproducible implementation.

| Model               |  Accuracy | Precision |    Recall |        F1 |   ROC AUC |    PR AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| Majority Class      | Generated | Generated | Generated | Generated | Generated | Generated |
| Education ≥ 13      | Generated | Generated | Generated | Generated | Generated | Generated |
| Logistic Regression | Generated | Generated | Generated | Generated | Generated | Generated |
| Decision Tree       | Generated | Generated | Generated | Generated | Generated | Generated |

This avoids hard-coding results that could become inconsistent with the actual experiment.

---

# 🧭 Model Selection for the Next Stage

The goal of this stage is not to declare a permanent final model.

Instead, the objective is to identify promising candidates for further experimentation.

### Logistic Regression Candidate

Logistic Regression is a strong candidate when:

* Precision is competitive
* Performance is stable
* Coefficients provide useful insight
* The linear decision boundary provides sufficient predictive power

Its major advantage is interpretability.

### Decision Tree Candidate

Decision Tree is a strong candidate when:

* It captures nonlinear relationships
* It improves recall or PR AUC
* Its precision remains acceptable
* The training/test gap is manageable
* Its top splits appear meaningful

Its major advantage is its ability to capture nonlinear interactions.

---

# 📝 Day 2 / Day 3 Write-Up

The first supervised modeling stage extends the Day 1 baseline analysis by introducing a complete leakage-resistant preprocessing and modeling workflow. Numerical and categorical features were handled separately through a `ColumnTransformer`, with median imputation and scaling applied to numerical variables and most-frequent imputation followed by one-hot encoding applied to categorical variables. Both Logistic Regression and Decision Tree models were implemented inside sklearn pipelines, ensuring that preprocessing is learned from the training data rather than manually applied using information from the hold-out set.

The two supervised models were then compared against the Day 1 majority-class and education-rule baselines using accuracy, precision, recall, F1, ROC AUC, and PR AUC. Confusion matrices, ROC curves, and Precision–Recall curves were used to understand model behavior beyond a single score. Logistic Regression provides an interpretable linear baseline through its coefficients, while the Decision Tree provides a way to investigate nonlinear relationships and feature interactions. The model(s) with the strongest precision-oriented performance and acceptable generalization will be carried forward into the next stage.

---

# 🔮 Preprocessing Changes to Test Next

The following preprocessing and modeling changes are candidates for the next stage:

* [ ] Test `most_frequent` vs explicit `"Missing"` category
* [ ] Test log transformation for `capital-gain`
* [ ] Test log transformation for `capital-loss`
* [ ] Add binary indicators for capital features
* [ ] Test RobustScaler
* [ ] Investigate class weighting
* [ ] Test Logistic Regression regularization strength
* [ ] Tune Decision Tree depth
* [ ] Tune minimum samples per split
* [ ] Tune minimum samples per leaf
* [ ] Investigate probability threshold optimization
* [ ] Compare precision-recall trade-offs
* [ ] Perform deeper error analysis
* [ ] Evaluate subgroup performance
* [ ] Consider additional models

---

# 🚀 Future Modeling Roadmap

The project will progressively move from simple models toward stronger and more carefully evaluated solutions.

### Completed

* [x] Problem Definition
* [x] Business Objective
* [x] Dataset Loading
* [x] Data Cleaning
* [x] Exploratory Data Analysis
* [x] Class Distribution Analysis
* [x] Train / Development / Test Split
* [x] Majority Baseline
* [x] Education Rule Baseline
* [x] Multi-Metric Baseline Evaluation
* [x] Error Analysis
* [x] Leakage-Safe Preprocessing
* [x] Numerical Pipeline
* [x] Categorical Pipeline
* [x] Logistic Regression
* [x] Decision Tree
* [x] ROC Analysis
* [x] Precision–Recall Analysis
* [x] Confusion Matrix Analysis
* [x] Logistic Regression Interpretability
* [x] Decision Tree Interpretability

### 🔜 Next

* [ ] Feature Engineering
* [ ] Hyperparameter Tuning
* [ ] Threshold Optimization
* [ ] Random Forest
* [ ] Gradient Boosting
* [ ] XGBoost
* [ ] Model Comparison
* [ ] Deeper Error Analysis
* [ ] Fairness / Subgroup Evaluation
* [ ] Final Model Selection
* [ ] Final Hold-out Evaluation
* [ ] Model Export
* [ ] Reusable Prediction Pipeline

---

# 📌 Task Completion

| Component                     | Status |
| ----------------------------- | :----: |
| Problem Definition            |    ✅   |
| Business Objective            |    ✅   |
| Primary Metric Selection      |    ✅   |
| Dataset Loading               |    ✅   |
| Missing Value Handling        |    ✅   |
| Target Conversion             |    ✅   |
| Class Base Rate               |    ✅   |
| Numerical EDA                 |    ✅   |
| Categorical EDA               |    ✅   |
| Visualizations                |    ✅   |
| Summary Tables                |    ✅   |
| Stratified Hold-out Split     |    ✅   |
| Development Split             |    ✅   |
| Majority Baseline             |    ✅   |
| Education Rule Baseline       |    ✅   |
| Accuracy                      |    ✅   |
| Precision                     |    ✅   |
| Recall                        |    ✅   |
| F1 Score                      |    ✅   |
| ROC AUC                       |    ✅   |
| PR AUC                        |    ✅   |
| Confusion Matrices            |    ✅   |
| False Positive Analysis       |    ✅   |
| False Negative Analysis       |    ✅   |
| Feature/Error Analysis        |    ✅   |
| ColumnTransformer             |    ✅   |
| Numerical Imputation          |    ✅   |
| Numerical Scaling             |    ✅   |
| Categorical Imputation        |    ✅   |
| One-Hot Encoding              |    ✅   |
| Leakage-Safe Pipeline         |    ✅   |
| Logistic Regression           |    ✅   |
| Decision Tree                 |    ✅   |
| ROC Curves                    |    ✅   |
| Precision–Recall Curves       |    ✅   |
| Logistic Coefficient Analysis |    ✅   |
| Decision Tree Depth Analysis  |    ✅   |
| Top Tree Splits               |    ✅   |
| Model Comparison              |    ✅   |
| Model Selection               |    ✅   |
| Advanced Models               |   🔜   |
| Hyperparameter Tuning         |   🔜   |
| Threshold Optimization        |   🔜   |
| Final Model                   |   🔜   |

---

# 🎯 Final Takeaway

This project is progressing from simple data analysis toward a complete, reproducible machine-learning workflow.

**Day 1** established the problem, business objective, dataset understanding, exploratory analysis, reproducible split, simple baselines, multi-metric evaluation, and initial error analysis.

**The supervised-modeling stage** moved beyond hand-written rules by introducing proper preprocessing and real machine-learning models. Numerical and categorical features are handled through a `ColumnTransformer`, preprocessing is integrated directly into sklearn pipelines, and Logistic Regression and Decision Tree models are trained only on the training data.

The models are evaluated against the original baselines using multiple complementary metrics rather than relying only on accuracy. ROC and Precision–Recall curves provide threshold-independent performance analysis, while confusion matrices reveal the actual types of classification errors. Logistic Regression coefficients and Decision Tree splits provide additional interpretability.

The most important lesson from this stage is:

> **A good machine-learning workflow is not just about training a model — it is about preventing leakage, preprocessing correctly, evaluating honestly, understanding errors, and using evidence to decide what to improve next.**

---

<p align="center">

### 🧠 Learn → Analyze → Baseline → Preprocess → Build → Evaluate → Interpret → Improve

<br>

**Personal AI • Machine Learning • Data Science**

</p>
