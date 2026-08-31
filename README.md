# 🧠 Personal-AI-DS-Tasks

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

**Personal-AI-DS-Tasks** is a collection of practical tasks focused on building and improving skills in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 📊 Data Science
* 🐍 Python
* 📈 Data Analysis
* 🧪 Model Evaluation
* 🔍 Error Analysis
* ⚙️ Model Optimization
* 🛡️ Data Leakage Prevention

The purpose of this repository is to practice the **complete data science and machine-learning workflow**, starting from understanding a problem and dataset, moving through preprocessing and analysis, establishing baselines, improving model performance, and finally evaluating results and identifying remaining weaknesses.

Each task documents:

> **What I did → How I did it → What I found → What problems I faced → How I improved it → What I will improve next**

---

# 🛠️ Skills Covered

## 🤖 Artificial Intelligence

* AI problem definition
* Business-oriented problem solving
* Binary classification
* Data-driven decision making
* Model evaluation
* Error analysis
* Performance optimization
* Responsible AI considerations

## 🧠 Machine Learning

* Supervised Learning
* Binary Classification
* Baseline Modeling
* Feature Engineering
* Data Preprocessing
* Missing Value Handling
* Categorical Encoding
* Train / Development / Test Splitting
* Stratified Sampling
* Reproducible Experiments
* Gradient Boosting
* Model Optimization
* Precision
* Recall
* F1 Score
* ROC AUC
* PR AUC
* Confusion Matrix
* False Positive / False Negative Analysis
* Model Comparison
* Data Leakage Prevention

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
* Experiment Tracking

## ⚙️ Model Development

* Baseline → ML model progression
* Preprocessing pipelines
* Median imputation
* Ordinal encoding
* Gradient boosting
* Validation-based experimentation
* Hold-out test evaluation
* Performance comparison
* Error-driven improvement

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

The task started as a simple binary classification problem and was progressively developed into a complete machine-learning workflow.

The objective was not simply to obtain a high accuracy score. Instead, I focused on:

* Building a reliable data pipeline
* Preventing data leakage
* Establishing meaningful baselines
* Understanding the data through EDA
* Improving predictive performance using a real ML model
* Evaluating multiple metrics
* Performing error analysis
* Maintaining a clean train/dev/test methodology

---

# 📊 Dataset Overview

The Adult dataset contains demographic, employment, education, and financial-related attributes used to predict income class.

The dataset includes features such as:

### Numerical Features

* `age`
* `fnlwgt`
* `education-num`
* `capital-gain`
* `capital-loss`
* `hours-per-week`

### Categorical Features

* `workclass`
* `education`
* `marital-status`
* `occupation`
* `relationship`
* `race`
* `sex`
* `native-country`

### Target

```text
income
```

with two classes:

```text
<=50K
>50K
```

---

# 📂 Dataset Loading

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

Using `fetch_openml()` makes the dataset-loading process reproducible and avoids manually downloading and preparing the raw dataset.

---

# 🧹 Data Cleaning

Before modeling, I cleaned the dataset to make the data consistent and suitable for machine learning.

### What I Did

* Loaded the dataset into Pandas
* Inspected shape and data types
* Checked duplicate rows
* Removed unnecessary whitespace
* Detected `?` and `" ?"` missing-value representations
* Converted missing-value markers to `NaN`
* Normalized target labels
* Removed trailing `.` from target labels when present
* Converted income into binary `0/1`
* Removed rows with missing target values
* Checked missing-value counts
* Separated features from the target

### Dataset Size

The cleaned dataset contained approximately:

```text
48,842 rows
```

with a positive-class rate of approximately:

```text
23.93%
```

This means that the `>50K` class is the minority class.

---

# 🔎 Missing Value Analysis

Missing values were identified particularly in categorical features.

Important missing-value counts included approximately:

| Feature          | Missing Values |
| ---------------- | -------------: |
| `occupation`     |          2,809 |
| `workclass`      |          2,799 |
| `native-country` |            857 |

Instead of allowing these values to break the ML pipeline, missing values were handled during preprocessing.

This was especially important because categorical variables cannot be directly passed into many machine-learning algorithms.

---

# 🎯 Target Conversion

The original income labels were normalized and converted into binary values:

```python
{
    ">50K": 1,
    "<=50K": 0
}
```

This converted the problem into a standard binary classification task.

---

# 🔎 Exploratory Data Analysis

After cleaning, I performed **Exploratory Data Analysis (EDA)** to understand the dataset before training the model.

## 📋 Dataset Inspection

I examined:

* Dataset shape
* Column names
* Data types
* Numerical features
* Categorical features
* Missing values
* Unique categories
* Numerical statistics
* Target distribution

---

# 📊 Numerical Analysis

For numerical features, I calculated:

* Mean
* Median
* Standard deviation
* Minimum
* Maximum
* 25th percentile
* 75th percentile
* Missing-value counts

Important numerical features included:

* `age`
* `fnlwgt`
* `education-num`
* `capital-gain`
* `capital-loss`
* `hours-per-week`

---

# 📈 Categorical Analysis

I investigated the distribution of categorical variables including:

* Workclass
* Education
* Marital status
* Occupation
* Relationship
* Race
* Sex
* Native country

This helped identify:

* High-frequency categories
* Rare categories
* Missing values
* Feature diversity
* Potential predictive relationships

---

# 📊 Visual Analysis

I generated visualizations for important features.

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

* Skewed distributions
* Dominant categories
* Concentrated values
* Missing-value patterns
* Potential relationships with the target

---

# 📈 Class Distribution

The positive class represented approximately:

```text
23.93%
```

of the dataset.

Therefore, the dataset is not perfectly balanced.

This is important because a model can obtain relatively high accuracy by favoring the majority class without providing useful positive predictions.

For this reason, accuracy was not treated as the only performance measure.

---

# 🔀 Reproducible Train / Dev / Test Split

I created a reproducible three-way split.

```python
RANDOM_STATE = 42
```

The final split was approximately:

| Dataset     |   Rows | Percentage |
| ----------- | -----: | :--------: |
| Training    | 34,188 |     70%    |
| Development |  4,885 |     10%    |
| Final Test  |  9,769 |     20%    |

The split was **stratified on the target variable**.

This preserves a similar class distribution across training, development, and test data.

---

# 🔒 Data Leakage Prevention

One of the important improvements during development was identifying and preventing possible **target leakage**.

The target column was explicitly separated from the feature matrix.

Conceptually:

```python
X = df.drop(columns=["target", target_col])
y = df[target_col]
```

The target was therefore prevented from accidentally becoming an input feature.

This is important because including the target or information derived directly from the target can produce artificially high accuracy while making the model invalid for real-world prediction.

---

# 🧪 Baseline Modeling

Before improving the model, I established simple baselines.

These provided reference points for determining whether the ML model actually learned useful patterns.

---

# 🧱 Baseline 1 — Majority Class

The first baseline predicts the most common class for every sample.

Because `<=50K` is the majority class, the baseline essentially predicts:

```text
<=50K
```

for everyone.

### Purpose

The majority baseline answers:

> How well can we perform without using any meaningful feature information?

It is therefore a minimum reference point.

---

# 🎓 Baseline 2 — Education Rule

The second baseline used:

```text
education-num
```

with the rule:

```text
education-num >= 13
        ↓
Predict >50K
```

Otherwise:

```text
Predict <=50K
```

This provided a more meaningful feature-based baseline.

---

# ❗ Why the Baselines Were Not Enough

The two baselines were intentionally simple.

However, income depends on multiple variables simultaneously.

For example:

```text
Education
    +
Age
    +
Occupation
    +
Workclass
    +
Hours Worked
    +
Capital Gain/Loss
    ↓
Income Prediction
```

A single education threshold cannot capture these interactions.

This motivated the transition from rule-based baselines to a proper machine-learning model.

---

# 🚀 Accuracy Improvement — What I Actually Did

The biggest improvement in Task 1 was moving from simple rules to a **real supervised ML model with preprocessing**.

Instead of relying only on:

```text
education-num >= 13
```

I allowed the model to learn relationships from the full feature set.

---

# 🧠 Model Used — HistGradientBoosting

I introduced a **HistGradientBoosting** classifier as the main ML model.

Gradient boosting works by building a sequence of decision-tree-based learners where later learners focus on correcting errors made by earlier learners.

This makes it substantially more capable than a single manually defined threshold.

### Why HistGradientBoosting?

It was selected because it can model:

* Nonlinear relationships
* Feature interactions
* Complex decision boundaries
* Different relationships between numerical features and income

This is particularly useful for the Adult dataset because income is influenced by combinations of demographic, education, employment, and financial features.

---

# ⚙️ Preprocessing Improvements

The ML stage introduced a proper preprocessing workflow.

## 1. Median Imputation

Missing numerical values were handled using **median imputation**.

Instead of dropping potentially useful rows, missing numerical values can be replaced using the median calculated from the training data.

This provides a robust solution that is less affected by extreme values than mean imputation.

---

## 2. Categorical Encoding

Categorical variables cannot directly be interpreted as numerical values by the model.

Therefore, categorical data was transformed using an **ordinal encoding approach**.

This allowed categorical variables such as:

```text
workclass
occupation
marital-status
relationship
education
native-country
```

to be represented numerically for model training.

Unknown categories encountered outside the fitting data were handled safely rather than causing the pipeline to fail.

---

# 🔄 Full ML Pipeline

The improved workflow became:

```text
Raw Adult Dataset
       ↓
Data Cleaning
       ↓
Target Separation
       ↓
Train / Dev / Test Split
       ↓
Numerical Imputation
       ↓
Categorical Encoding
       ↓
HistGradientBoosting
       ↓
Predictions
       ↓
Evaluation
       ↓
Error Analysis
```

This was a major improvement over the original rule-based approach.

---

# 🎯 Why This Improved Accuracy

The improvement came from allowing the model to learn from **all relevant features simultaneously** rather than relying on one manually selected feature.

The model can learn patterns such as:

```text
Education + Age
Education + Occupation
Occupation + Hours Worked
Age + Capital Gain
Workclass + Education
```

and many other nonlinear relationships.

This gives the classifier much more information than the education-only baseline.

---

# 📈 Accuracy Improvement Strategy

The improvement process followed an iterative approach:

### Step 1 — Establish Baselines

I first measured:

* Majority-class performance
* Education-rule performance

### Step 2 — Identify Weaknesses

Error analysis showed that simple rules produced:

* False positives
* False negatives
* Poor coverage of complex income patterns

### Step 3 — Use More Features

Instead of using only `education-num`, the ML model used the complete cleaned feature set.

### Step 4 — Handle Missing Values

Missing values were incorporated into a structured preprocessing stage.

### Step 5 — Encode Categorical Features

Categorical information was transformed into a representation usable by the ML model.

### Step 6 — Train a Nonlinear Model

HistGradientBoosting was used to learn nonlinear relationships and feature interactions.

### Step 7 — Validate on Development Data

The development set was used for experimentation and model assessment without touching the final hold-out test set.

### Step 8 — Final Hold-out Evaluation

After the development process, the final test set remained reserved for final evaluation.

---

# 🧪 Model Evaluation

The model was evaluated using multiple metrics rather than accuracy alone.

| Metric        | Why It Matters                             |
| ------------- | ------------------------------------------ |
| **Accuracy**  | Overall correctness                        |
| **Precision** | Correctness of positive predictions        |
| **Recall**    | Ability to find actual `>50K` cases        |
| **F1 Score**  | Balance between precision and recall       |
| **ROC AUC**   | Overall class discrimination               |
| **PR AUC**    | Positive-class performance under imbalance |

---

# 🎯 Primary Business Metric — Precision

The main business-oriented metric remained:

# **Precision**

Precision answers:

> When the model predicts `>50K`, how often is that prediction correct?

This was important because the selected business scenario assumes that positive predictions may lead to targeted outreach.

A false positive can therefore represent:

* Wasted outreach
* Unnecessary marketing cost
* Poor targeting

However, recall and the other metrics were also monitored to ensure that improving precision did not result in a model that simply predicts the negative class.

---

# 🧮 Confusion Matrix

The model was evaluated using a confusion matrix.

|                   | Actual `<=50K` |  Actual `>50K` |
| ----------------- | -------------: | -------------: |
| Predicted `<=50K` |  True Negative | False Negative |
| Predicted `>50K`  | False Positive |  True Positive |

### False Positive

```text
Predicted >50K
Actual <=50K
```

### False Negative

```text
Predicted <=50K
Actual >50K
```

These errors were analyzed to understand where the model still struggles.

---

# 🔍 Error Analysis

After training the ML model, I continued the error-analysis process.

I inspected:

* False positives
* False negatives
* Numerical feature profiles
* Categorical feature patterns
* Common characteristics of incorrectly classified samples

This allowed me to move beyond simply asking:

> "What is the accuracy?"

and instead ask:

> "Why is the model making these mistakes?"

---

# 🧠 Key Error Analysis Insight

The main lesson from the baseline stage was that income prediction cannot be reliably represented by one simple rule.

The ML model can use combinations of:

```text
Age
Education
Occupation
Workclass
Hours-per-week
Capital Gain
Capital Loss
Marital Status
Relationship
```

to build a more flexible decision function.

This is the main reason the ML stage is expected to outperform the simple education heuristic.

---

# ⚠️ Issues & Challenges

## 1. Missing Categorical Values

Several categorical features contained missing values.

### Solution

Missing values were incorporated into the preprocessing pipeline rather than allowing them to cause model failures.

---

## 2. Categorical Features

The dataset contains many categorical variables.

### Solution

Categorical features were encoded before being passed to the model.

---

## 3. Highly Skewed Capital Features

`capital-gain` and `capital-loss` contain many zero values and relatively few large values.

### Impact

These features have highly non-uniform distributions.

### Future Improvement

Potential improvements include:

* Log transformations
* Binary indicators
* Additional feature engineering

---

## 4. Class Imbalance

The positive class represents approximately 23.93% of the dataset.

### Impact

Accuracy can become misleading when the majority class dominates.

### Solution

I monitored:

* Precision
* Recall
* F1
* ROC AUC
* PR AUC

rather than relying on accuracy alone.

---

## 5. Data Leakage Risk

During development, it was important to ensure that the target variable was not accidentally included among the features.

### Solution

The target was explicitly separated before model training.

The final test set was also kept untouched during experimentation.

---

## 6. Feature Interactions

A manually selected rule cannot capture complex interactions.

### Solution

HistGradientBoosting was introduced because it can learn nonlinear patterns and interactions from multiple features.

---

## 7. Potentially Sensitive Features

The dataset contains attributes such as:

* Race
* Sex

These features can raise fairness concerns.

Before real-world deployment, subgroup performance and fairness should be evaluated carefully.

---

# 📊 Task 1 Results

The project compares simple baselines against the improved ML model.

| Model                    |                  Accuracy |                 Precision |                    Recall |                        F1 |                   ROC AUC |                    PR AUC |
| ------------------------ | ------------------------: | ------------------------: | ------------------------: | ------------------------: | ------------------------: | ------------------------: |
| Majority Class           |     Generated in notebook |     Generated in notebook |     Generated in notebook |     Generated in notebook |     Generated in notebook |     Generated in notebook |
| Education ≥ 13           |     Generated in notebook |     Generated in notebook |     Generated in notebook |     Generated in notebook |     Generated in notebook |     Generated in notebook |
| **HistGradientBoosting** | **Generated in notebook** | **Generated in notebook** | **Generated in notebook** | **Generated in notebook** | **Generated in notebook** | **Generated in notebook** |

The exact metrics are generated directly from the reproducible notebook.

This ensures that the README does not manually hard-code experimental values that could become inconsistent with the actual implementation.

---

# 📈 Baseline → ML Improvement

The development process can be summarized as:

```text
Majority Class
      ↓
Simple Reference
      ↓
Education ≥ 13 Rule
      ↓
Identify False Positives / False Negatives
      ↓
Use Full Feature Set
      ↓
Handle Missing Values
      ↓
Encode Categorical Variables
      ↓
HistGradientBoosting
      ↓
Learn Nonlinear Relationships
      ↓
Improved Predictive Performance
```

The key improvement was not simply changing one parameter.

It was changing the problem from:

> **A manually defined single-feature rule**

into:

> **A complete supervised machine-learning pipeline capable of learning relationships across the dataset.**

---

# 📁 Generated Outputs

The analysis and modeling workflow produces reusable outputs including:

```text
adult_income_outputs/
│
├── class_summary.csv
├── feature_distribution_summary.csv
├── baseline_metrics.csv
├── false_positive_sample.csv
├── false_negative_sample.csv
└── error_profile.csv
```

Visualizations are stored under:

```text
adult_income_plots/
```

including:

* Numerical histograms
* Categorical bar plots
* Confusion matrices
* Model evaluation visualizations

---

# 🖼️ Task 1 Pipeline

<p align="center">
  <img src="./assets/adult_income_pipeline.png" alt="Adult Income Prediction Machine Learning Pipeline" width="100%">
</p>

<p align="center">
  <i>Task 1 — End-to-end Adult Income classification and improvement pipeline</i>
</p>

---

# 🧭 Complete Task Workflow

The complete Task 1 workflow is:

```text
Problem Definition
        ↓
Business Objective
        ↓
Dataset Loading
        ↓
Data Cleaning
        ↓
Missing Value Analysis
        ↓
Exploratory Data Analysis
        ↓
Class Distribution
        ↓
Stratified Train / Dev / Test Split
        ↓
Data Leakage Prevention
        ↓
Majority Baseline
        ↓
Education Rule Baseline
        ↓
Error Analysis
        ↓
Identify Model Limitations
        ↓
Preprocessing Pipeline
        ↓
Median Imputation
        ↓
Categorical Encoding
        ↓
HistGradientBoosting
        ↓
Development Evaluation
        ↓
Error Analysis
        ↓
Final Hold-out Evaluation
        ↓
Improvement Planning
```

---

# 🚀 Next Modeling Stage

The current ML stage establishes a strong foundation for further experimentation.

### Planned improvements

* [ ] Compare Logistic Regression
* [ ] Compare Decision Tree
* [ ] Compare Random Forest
* [ ] Compare Gradient Boosting alternatives
* [ ] Evaluate XGBoost
* [ ] Perform systematic hyperparameter tuning
* [ ] Optimize classification threshold
* [ ] Optimize precision while monitoring recall
* [ ] Investigate feature importance
* [ ] Perform deeper error analysis
* [ ] Engineer additional features
* [ ] Evaluate subgroup performance
* [ ] Investigate fairness metrics
* [ ] Compare model stability across random seeds
* [ ] Freeze the final preprocessing + model pipeline
* [ ] Perform final evaluation on the untouched test set

---

# 📌 What I Improved

The most important improvements made during Task 1 were:

### Before

```text
Raw Dataset
   ↓
Basic Cleaning
   ↓
EDA
   ↓
Majority Baseline
   ↓
Education Rule
```

### After

```text
Raw Dataset
   ↓
Cleaning
   ↓
Target Separation
   ↓
Leakage Prevention
   ↓
Stratified Train / Dev / Test
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Full Feature Set
   ↓
HistGradientBoosting
   ↓
Development Evaluation
   ↓
Error Analysis
   ↓
Final Hold-out Evaluation
```

### Main Improvements

| Area             | Improvement                                      |
| ---------------- | ------------------------------------------------ |
| Data             | Better cleaning and missing-value handling       |
| Features         | Used the full feature set instead of one feature |
| Categorical Data | Added categorical encoding                       |
| Numerical Data   | Added median imputation                          |
| Model            | Replaced simple rules with HistGradientBoosting  |
| Generalization   | Used train/dev/test methodology                  |
| Leakage          | Explicitly separated target from features        |
| Evaluation       | Added six evaluation metrics                     |
| Errors           | Added FP/FN analysis                             |
| Reproducibility  | Fixed random seed and structured pipeline        |

---

# 🗂️ Repository Structure

```text
Personal-AI-DS-Tasks/
│
├── 📄 Adult Income Detail.py
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
│   └── ...
│
├── 📂 adult_income_outputs/
│   ├── class_summary.csv
│   ├── feature_distribution_summary.csv
│   ├── baseline_metrics.csv
│   ├── false_positive_sample.csv
│   ├── false_negative_sample.csv
│   └── error_profile.csv
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

### 3. Run the notebook

```bash
jupyter notebook
```

Open:

```text
adult_income_baselines_error_analysis.ipynb
```

and run the cells from top to bottom.

The dataset is fetched automatically through OpenML.

---

# 📌 Task 1 Completion

| Component                 | Status |
| ------------------------- | :----: |
| Problem Definition        |    ✅   |
| Business Objective        |    ✅   |
| Primary Metric Selection  |    ✅   |
| Dataset Loading           |    ✅   |
| Data Cleaning             |    ✅   |
| Missing Value Analysis    |    ✅   |
| Target Conversion         |    ✅   |
| Class Base Rate           |    ✅   |
| Numerical EDA             |    ✅   |
| Categorical EDA           |    ✅   |
| Visualizations            |    ✅   |
| Summary Tables            |    ✅   |
| Stratified Train Split    |    ✅   |
| Development Split         |    ✅   |
| Hold-out Test Split       |    ✅   |
| Data Leakage Prevention   |    ✅   |
| Majority Baseline         |    ✅   |
| Education Rule Baseline   |    ✅   |
| Full Feature Set          |    ✅   |
| Median Imputation         |    ✅   |
| Categorical Encoding      |    ✅   |
| HistGradientBoosting      |    ✅   |
| Accuracy Evaluation       |    ✅   |
| Precision Evaluation      |    ✅   |
| Recall Evaluation         |    ✅   |
| F1 Score Evaluation       |    ✅   |
| ROC AUC Evaluation        |    ✅   |
| PR AUC Evaluation         |    ✅   |
| Confusion Matrices        |    ✅   |
| False Positive Analysis   |    ✅   |
| False Negative Analysis   |    ✅   |
| Error Profiling           |    ✅   |
| Accuracy Improvement      |    ✅   |
| Model Development         |    ✅   |
| Issues Identified         |    ✅   |
| Improvement Plan          |    ✅   |
| Advanced Model Comparison |   🔜   |
| Hyperparameter Tuning     |   🔜   |
| Threshold Optimization    |   🔜   |
| Final Model               |   🔜   |

---

# 🎯 Final Takeaway

Task 1 was not only about predicting whether a person earns more than `$50K`.

The main objective was to build a **reliable and progressively improving machine-learning workflow**.

I started by defining the business problem and selecting **precision** as the primary metric. I then loaded and cleaned the Adult dataset, handled missing-value representations, converted the target into a binary classification problem, and performed detailed exploratory data analysis.

I created a reproducible **70% training / 10% development / 20% hold-out test split** using stratified sampling and explicitly separated the target variable from the feature matrix to reduce the risk of data leakage.

I first established two simple baselines:

1. **Majority-class prediction**
2. **Education ≥ 13 rule**

These baselines helped demonstrate the limitations of simple prediction strategies.

The major improvement came from moving to a real ML pipeline using **median imputation, categorical encoding, the complete feature set, and HistGradientBoosting**.

Instead of relying on one manually selected feature, the model can learn nonlinear relationships and interactions across:

```text
Age
Education
Occupation
Workclass
Hours-per-week
Capital Gain
Capital Loss
Marital Status
Relationship
```

This made the modeling process substantially more representative of a real-world machine-learning workflow.

I then evaluated the model using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* PR AUC
* Confusion Matrix

and continued to analyze false positives and false negatives to understand remaining model weaknesses.

The biggest lesson from Task 1 was:

> **Improving model performance is not just about choosing a more powerful algorithm. It requires better data preparation, correct feature handling, leakage prevention, appropriate evaluation, and iterative error analysis.**

The current model provides the next foundation for further improvements through model comparison, hyperparameter tuning, threshold optimization, feature engineering, and deeper fairness/error analysis.

---

<p align="center">

### 🧠 Learn → Analyze → Build → Evaluate → Improve

<br>

**Personal AI • Machine Learning • Data Science**

</p>
