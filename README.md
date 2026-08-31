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

The purpose of this repository is to practice the **complete data science and machine-learning workflow**, starting from understanding a problem and dataset, moving through preprocessing and analysis, and finally evaluating results and identifying improvements.

Each task documents:

> **What I did → How I did it → What I found → What problems I faced → What I will improve next**

---

# 🛠️ Skills Covered

## 🤖 Artificial Intelligence

* AI problem definition
* Business-oriented problem solving
* Classification problems
* Model evaluation
* Error analysis
* Data-driven decision making

## 🧠 Machine Learning

* Supervised Learning
* Binary Classification
* Baseline Modeling
* Train / Development / Test Splitting
* Stratified Sampling
* Reproducible Experiments
* Precision
* Recall
* F1 Score
* ROC AUC
* PR AUC
* Confusion Matrix
* False Positive / False Negative Analysis
* Model Comparison

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

## 💻 Tools & Technologies

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

The task was designed as a starting point for a real-world binary classification problem where the goal is not only to build a model, but also to understand the data, establish reliable baselines, evaluate performance correctly, and analyze model errors.

---

# 💼 Business Problem

A possible business application is **targeted higher-value outreach**.

For example, a company may want to identify people who are more likely to belong to the higher-income group before contacting them.

In this scenario, incorrectly predicting someone as a high-income individual can result in:

* Wasted outreach effort
* Unnecessary marketing cost
* Lower-quality targeting

Therefore, I selected:

# 🎯 Precision as the Primary Metric

**Precision answers:**

> When the model predicts that someone earns more than $50K, how often is that prediction correct?

I prioritized precision over recall because the business scenario places more importance on avoiding unnecessary positive predictions.

Recall is still monitored because a model that misses too many actual high-income individuals would also have limited business value.

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

Using `fetch_openml()` makes the dataset loading process reproducible and avoids manually preparing the raw UCI files.

---

# 🧹 Data Cleaning

Before performing analysis, I cleaned the dataset to make the data consistent and suitable for evaluation.

### What I did

* Loaded the dataset into a Pandas DataFrame
* Inspected the raw shape and data types
* Removed unnecessary whitespace
* Converted `?` values into missing values
* Handled raw-UCI-style `" ?"` values
* Normalized the target labels
* Removed the trailing `.` when necessary
* Converted income into binary `0/1`
* Removed rows where the target was missing
* Checked remaining missing values

### Missing Values

The Adult dataset can represent missing categorical information using:

```text
?
```

I converted these values to:

```python
np.nan
```

This allows Pandas and later preprocessing pipelines to handle missing values correctly.

### Target Conversion

The income target was converted into:

```python
{
    ">50K": 1,
    "<=50K": 0
}
```

This transforms the original problem into a standard binary classification task.

---

# 🔎 Exploratory Data Analysis

After cleaning the dataset, I performed **Exploratory Data Analysis (EDA)** to understand the structure and behavior of the data before modeling.

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

I created visualizations to understand important feature distributions.

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

The notebook also saves these plots for later review.

---

# 📈 Class Distribution

I calculated the number and percentage of:

* Negative examples: `<=50K`
* Positive examples: `>50K`

The positive-class percentage is important because the dataset is not perfectly balanced.

This immediately showed why **accuracy alone should not be the main metric**.

A model can obtain relatively high accuracy by predicting the majority class while failing to identify the positive class.

---

# 🔀 Reproducible Train / Dev / Test Split

A major part of this task was creating a reliable evaluation setup.

I used:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
```

The final test set was created first and kept completely separate from experimentation.

### Dataset Split

* 🟦 **Training:** approximately 70%
* 🟨 **Development:** approximately 10%
* 🟥 **Final Hold-out Test:** approximately 20%

The split was **stratified on the target** so that the proportion of positive and negative examples remains similar across the datasets.

### Why the Hold-out Test Matters

The final test set represents unseen data.

If I repeatedly inspect the test results while selecting features, rules, or hyperparameters, I indirectly start optimizing for the test set.

That causes **test-set leakage** and can make the final reported performance look better than the model's true performance.

Therefore:

> 🔒 **The hold-out test set is only used for final evaluation.**

---

# 🧪 Baseline Modeling

Before building advanced machine-learning models, I created two simple baselines.

The purpose of baselines is to establish a **minimum performance reference**.

A future ML model should provide meaningful improvement over these simple approaches.

---

## 🧱 Baseline 1 — Majority Class

The first baseline predicts the most common class for every person.

In this dataset, the negative class is the majority.

Therefore, the baseline essentially says:

> **"Predict <=50K for everyone."**

### Why use it?

Although this is a very simple model, it is important as a sanity check.

It shows what performance can be achieved without learning anything meaningful from the features.

The majority baseline can have reasonable accuracy because the negative class dominates, but it does not identify the positive class effectively.

---

# 🎓 Baseline 2 — Education Rule

The second baseline uses a single feature:

```text
education-num
```

The rule is:

```text
education-num >= 13
        ↓
Predict >50K
```

Otherwise:

```text
Predict <=50K
```

### Why did I choose this rule?

Education is reasonably associated with income.

A threshold of `13` provides a simple and interpretable heuristic that roughly represents **bachelor-level education or above**.

The important point is that this is intentionally simple.

It gives future ML models a meaningful baseline to beat.

---

# 📏 Evaluation Framework

Both baselines were evaluated on the **untouched hold-out test set**.

I calculated six metrics:

| Metric        | Purpose                                                      |
| ------------- | ------------------------------------------------------------ |
| **Accuracy**  | Overall percentage of correct predictions                    |
| **Precision** | Correctness of positive predictions                          |
| **Recall**    | Percentage of actual positive cases found                    |
| **F1 Score**  | Balance between precision and recall                         |
| **ROC AUC**   | Overall ranking/discrimination ability                       |
| **PR AUC**    | Positive-class performance, especially useful with imbalance |

---

# 🎯 Why Precision?

For this particular business scenario:

```text
Predicted >50K
       ↓
Person is contacted
```

A false positive means:

```text
Predicted >50K
Actual <=50K
```

This represents an unnecessary contact.

Therefore, precision is the main metric used to judge whether the model is producing useful positive predictions.

However, I continue to monitor:

* Recall
* F1
* ROC AUC
* PR AUC

so that improving precision does not completely destroy coverage of actual positive cases.

---

# 🧮 Confusion Matrix

I also generated confusion matrices for both baseline approaches.

The four possible outcomes are:

|                     |   Actual <=50K |    Actual >50K |
| ------------------- | -------------: | -------------: |
| **Predicted <=50K** |  True Negative | False Negative |
| **Predicted >50K**  | False Positive |  True Positive |

### Important Errors

**False Positive**

> Model predicts `>50K`, but actual income is `<=50K`.

**False Negative**

> Model predicts `<=50K`, but actual income is `>50K`.

These errors are especially useful for understanding why the simple baseline is limited.

---

# 🔍 Initial Error Analysis

After evaluating the education-based rule, I performed an initial error analysis.

I separated predictions into:

### ❌ False Positives

People who were predicted as:

```text
>50K
```

but actually belonged to:

```text
<=50K
```

### ❌ False Negatives

People who were predicted as:

```text
<=50K
```

but actually belonged to:

```text
>50K
```

I inspected at least:

* 10 false positives
* 10 false negatives

---

# 🧠 What I Looked For

For the false-positive and false-negative examples, I examined features including:

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

I also generated a compact error profile using numerical medians and common categorical values.

---

# 💡 Main Error Analysis Insight

The education rule is useful as a baseline, but it is too simple to represent the actual income problem.

A person's income is affected by multiple factors at the same time.

For example:

**Education + Age + Occupation + Work Hours + Workclass + Capital Features**

can provide much more information than education alone.

This explains why a single threshold can produce both false positives and false negatives.

---

# ⚠️ Issues & Challenges I Identified

## 1. Missing Categorical Values

Features such as workclass, occupation, and native country can contain missing values.

### Improvement

Use consistent imputation or explicitly encode missing categories instead of ignoring them.

---

## 2. Categorical Encoding

Many important features are categorical.

Examples:

* Workclass
* Occupation
* Marital status
* Relationship
* Native country

Most ML algorithms require these values to be converted into numerical representations.

### Improvement

Use techniques such as **One-Hot Encoding** while ensuring train and test columns remain aligned.

---

## 3. Highly Skewed Capital Features

`capital-gain` and `capital-loss` contain many zero values and a relatively small number of large values.

This creates strongly skewed distributions.

### Improvement

Consider:

* Binary indicators such as `capital-gain > 0`
* Log transformations
* Other feature transformations

---

## 4. Feature Interactions

The current education rule only looks at one feature.

However, income may depend on combinations such as:

> Education × Occupation × Age × Hours Worked

### Improvement

Use feature engineering or models capable of learning nonlinear relationships.

---

## 5. Class Imbalance

The positive class is smaller than the negative class.

This means a high accuracy score does not automatically mean the model is useful.

### Improvement

Continue monitoring:

* Precision
* Recall
* F1
* PR AUC

and consider class weighting where appropriate.

---

## 6. Potentially Sensitive Features

The dataset contains attributes such as:

* Race
* Sex

These features can affect predictions and raise important fairness considerations.

### Improvement

Before any real-world deployment, subgroup performance and fairness should be evaluated carefully.

---

# 🖼️ Task 1 Pipeline

<p align="center">
  <img src="./assets/adult_income_pipeline.png" alt="Adult Income Prediction Machine Learning Pipeline" width="100%">
</p>

<p align="center">
  <i>Task 1 — End-to-end Adult Income classification workflow</i>
</p>

---

# 📊 Task 1 Results

The notebook evaluates both baselines on the final hold-out set using the same evaluation framework.

| Baseline       |              Accuracy |             Precision |                Recall |                    F1 |               ROC AUC |                PR AUC |
| -------------- | --------------------: | --------------------: | --------------------: | --------------------: | --------------------: | --------------------: |
| Majority Class | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook |
| Education ≥ 13 | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook | Generated in notebook |

The exact values are generated directly from the reproducible notebook so that the reported results always correspond to the actual dataset and fixed split.

### Baseline Interpretation

The majority-class baseline is useful as a minimum reference point but does not meaningfully identify positive cases.

The education-based rule is more informative because it uses an actual feature to identify a subset of people more likely to belong to the `>50K` class.

A future machine-learning model should therefore provide a **material improvement over the education heuristic**, especially on the primary precision objective.

Simply achieving higher accuracy than the majority baseline would not be enough.

---

# 📁 Generated Outputs

The analysis produces reusable outputs including:

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

Visualizations are saved under:

```text
adult_income_plots/
```

including:

* Numerical histograms
* Categorical bar plots
* Baseline confusion matrices

---

# 🧭 Complete Task Workflow

<p align="center">
  <img src="./assets/adult_income_pipeline.png" alt="Complete Task 1 Pipeline" width="95%">
</p>

The project follows a reproducible progression:

**Problem Definition → Data Loading → Data Cleaning → EDA → Class Analysis → Reproducible Split → Baseline Modeling → Evaluation → Error Analysis → Improvement Planning**

---

# 🚀 Next Modeling Stage

The baseline stage establishes the foundation for more advanced experiments.

### Planned improvements

* [ ] Handle missing categorical values
* [ ] Build a preprocessing pipeline
* [ ] One-hot encode categorical variables
* [ ] Engineer useful numerical features
* [ ] Transform skewed capital features
* [ ] Train Logistic Regression
* [ ] Train Decision Tree
* [ ] Train Random Forest
* [ ] Train Gradient Boosting / XGBoost
* [ ] Compare models
* [ ] Tune hyperparameters using the development set
* [ ] Optimize precision
* [ ] Monitor recall, F1 and PR AUC
* [ ] Perform deeper error analysis
* [ ] Evaluate subgroup performance
* [ ] Freeze the final pipeline
* [ ] Perform one final evaluation on the untouched hold-out set

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

Open the Adult Income notebook and run the cells from top to bottom.

The dataset is fetched automatically through OpenML.

---

# 📌 Task 1 Completion

| Component                 | Status |
| ------------------------- | :----: |
| Problem Definition        |    ✅   |
| Business Objective        |    ✅   |
| Primary Metric Selection  |    ✅   |
| Dataset Loading           |    ✅   |
| Missing Value Handling    |    ✅   |
| Target Conversion         |    ✅   |
| Class Base Rate           |    ✅   |
| Numerical EDA             |    ✅   |
| Categorical EDA           |    ✅   |
| Visualizations            |    ✅   |
| Summary Tables            |    ✅   |
| Stratified Hold-out Split |    ✅   |
| Development Split         |    ✅   |
| Majority Baseline         |    ✅   |
| Education Rule Baseline   |    ✅   |
| Accuracy                  |    ✅   |
| Precision                 |    ✅   |
| Recall                    |    ✅   |
| F1 Score                  |    ✅   |
| ROC AUC                   |    ✅   |
| PR AUC                    |    ✅   |
| Confusion Matrices        |    ✅   |
| False Positive Analysis   |    ✅   |
| False Negative Analysis   |    ✅   |
| Feature/Error Analysis    |    ✅   |
| Issues Identified         |    ✅   |
| Improvement Plan          |    ✅   |
| Advanced Models           |   🔜   |
| Hyperparameter Tuning     |   🔜   |
| Final Model               |   🔜   |

---

# 🎯 Final Takeaway

Task 1 was not only about predicting income.

The main objective was to establish a **reliable machine-learning workflow** before moving to complex models.

I started by defining the business problem and choosing **precision** as the primary metric. I then loaded and cleaned the Adult dataset, explored its numerical and categorical features, analyzed the class distribution, and created a reproducible stratified train/dev/test split.

After that, I built two transparent baselines — a **majority-class predictor** and an **education-based rule** — and evaluated them using accuracy, precision, recall, F1, ROC AUC, PR AUC, and confusion matrices.

Finally, I performed initial false-positive and false-negative analysis to understand where the simple rule fails. This analysis identified missing categorical values, categorical encoding, skewed capital features, feature interactions, class imbalance, and fairness considerations as the main areas to address in the next modeling stage.

> **The baseline is the starting point — the goal of the next stage is to build a model that learns the relationships the simple rules cannot capture.**

---

<p align="center">

### 🧠 Learn → Analyze → Build → Evaluate → Improve

<br>

**Personal AI • Machine Learning • Data Science**

</p>
