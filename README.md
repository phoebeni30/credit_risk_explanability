# 🏦 Credit Risk Explainability
> **Trustworthy Machine Learning for Home Credit Default Prediction**

## 📑 Outline
1. [Problem Statement](#1-problem-statement)
2. [Project Description](#2-project-description)
3. [Analytical Pipeline](#3-analytical-pipeline)
4. [Dataset Characteristics](#4-dataset-characteristics)
5. [Project Architecture](#5-project-architecture)
6. [Results & Model Performance](#6-results--model-performance)
7. [Explainability Methods](#7-explainability-methods)
8. [Conclusion & Future Work](#8-conclusion--future-work)

---

## 1. Problem Statement

In consumer lending, a model that only predicts **who will default** is insufficient. Regulators and customers alike demand to know **why** a decision was made — and **what can be done to change it**.

This project addresses two core challenges:

| Challenge | Description |
| :--- | :--- |
| **Prediction** | Accurately identify customers at risk of default on the Home Credit dataset |
| **Explainability** | Provide transparent, actionable explanations at both global and local levels |

**The stakes:**
- A false negative (missed default) causes direct financial loss
- A false positive (wrongly rejected customer) causes reputational and regulatory risk
- A "black box" model — even if accurate — cannot satisfy modern credit regulation (GDPR, SR 11-7)

---

## 2. Project Description

### 🌟 Overview
This project builds a complete **Trustworthy ML pipeline** for credit risk, combining:
- Hyperparameter-optimized **Random Forest** trained with Optuna
- A suite of **global and local explainability** methods (SHAP, LIME, PDP, ICE)
- **Counterfactual generation** via MAPOCAM and DiCE — telling customers exactly what to change

### 🎯 Objectives
- **Modeling:** Train and optimize a credit risk classifier on the Home Credit dataset using Optuna-powered hyperparameter search
- **Global Explainability:** Understand which features drive default risk across the entire population
- **Local Explainability:** Explain individual predictions for specific customers
- **Counterfactuals:** Generate actionable recommendations for customers predicted to default

---

## 3. Analytical Pipeline

1. **Data Cleaning:** Load and preprocess Home Credit data via `DataCleaner`
2. **Feature Engineering:** EBE encoding for categorical features, normalization, missing value imputation via `create_pipeline`
3. **Model Training:**
   - Baseline: `RandomForestClassifier` with default parameters
   - Optimized: Optuna hyperparameter search (`n_trials=5`, `class_weight="balanced"`)
   - Threshold selection via **KS statistic** to handle 8% class imbalance
4. **Global Explainability:**
   - SHAP Feature Importances → Parallel Coordinates Plot
   - Partial Dependence Plots (PDP)
   - Individual Conditional Expectation (ICE)
5. **Local Explainability:**
   - SHAP local explanation per customer
   - LIME local explanation per customer
6. **Counterfactual Explanations:**
   - **MAPOCAM** — minimum-change counterfactuals
   - **DiCE** — diverse counterfactual scenarios

---

## 4. Dataset Characteristics

### 📂 Source & Scope
- **Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
- **Domain:** Consumer lending (non-bank financial institutions)
- **Target:** `DEFAULT` — binary (0 = repaid, 1 = defaulted)

### 📊 Key Statistics
| Property | Value |
| :--- | :--- |
| **Total rows** | ~307,511 |
| **Features** | 118 (after cleaning) |
| **Default rate** | ~8% |
| **Train / Val / Test split** | 60% / 20% / 20% |
| **Class imbalance** | Severe — requires `class_weight="balanced"` |

### ⚠️ Key Challenges
- **Severe class imbalance (8%):** Standard models predict all class 0 → useless Recall
- **High dimensionality (118 features):** Requires feature selection and efficient explainability methods
- **Mixed feature types:** Numeric, categorical, and ordinal features requiring different encoding strategies

---

## 5. Project Architecture

```text
credit_risk_explanability/
├── src/
│   ├── data_loader.py          # DataCleaner — load & preprocess Home Credit
│   ├── models.py               # get_models() — model registry
│   ├── training.py             # create_pipeline, ks_threshold
│   ├── optuna_optimize.py      # optimize_model_fast with Optuna
│   ├── evaluate.py             # get_metrics — AUC, F1, Recall, Precision
│   └── explanability.py        # SHAP, LIME, PDP, ICE, MAPOCAM, DiCE
├── outputs/
│   └── models/
│       └── randomforest_params_optimized.joblib
├── run_training.ipynb          # Model training & evaluation
├── run_explainability.ipynb    # All explainability methods
├── requirements.txt
└── README.md
```

---

## 6. Results & Model Performance

### 📊 Baseline vs Optimized

| Metric | Baseline | Optimized |
| :--- | :--- | :--- |
| **AUC** | 0.708 | **0.744** |
| **Balanced Accuracy** | 0.501 | **0.682** |
| **Recall** | 0.003 | **0.715** |
| **Precision** | 0.765 | 0.151 |
| **F1** | 0.005 | **0.249** |
| **Accuracy** | 0.920 | 0.654 |

### 💡 Key Findings
- **Baseline is degenerate** — Accuracy of 92% achieved by predicting all class 0, Recall ≈ 0
- **Optimized model** recovers Recall to 71.5% — critical for credit risk where missing a default is costly
- **Trade-off is intentional:** `class_weight="balanced"` boosts Recall at the cost of Precision — acceptable in lending where false negatives are more expensive than false positives
- **Best hyperparameters found by Optuna:** `n_estimators=97`, `max_depth=12`, `min_samples_leaf=40`, `class_weight="balanced"`

---

## 7. Explainability Methods

### 🌍 Global — SHAP Feature Importances
Top features driving default risk across all customers:

| Rank | Feature | Interpretation |
| :--- | :--- | :--- |
| 1 | `EXT_SOURCE_3` | External credit score — most predictive signal |
| 2 | `EXT_SOURCE_2` | External credit score |
| 3 | `DAYS_BIRTH` | Age — older customers default less |
| 4 | `DAYS_EMPLOYED` | Job tenure — longer = more stable |
| 5 | `AMT_GOODS_PRICE` | Loan goods value — non-linear relationship |

### 📈 Global — PDP & ICE
- **PDP (`AMT_GOODS_PRICE`):** Reveals non-linear relationship — lowest risk at 700k–800k price range; high risk for very cheap goods (low-income proxy)
- **ICE (`EXT_SOURCE_2`):** Heterogeneous response — some customers are highly sensitive to this score, others are not

### 🔍 Local — SHAP & LIME
For individual predictions, both methods consistently identify `EXT_SOURCE_1/2/3` and `DAYS_EMPLOYED` as the dominant factors. SHAP provides theoretically grounded Shapley values; LIME offers faster approximate explanations.

### 🔄 Counterfactual Explanations

| Method | Approach | Best For |
| :--- | :--- | :--- |
| **MAPOCAM** | Minimum-change path search | Actionable advice to real customers |
| **DiCE** | Diverse optimization | Multiple scenario exploration |

**Example MAPOCAM output for a predicted default:**
> *"If your job tenure increases from 4.6 years to 13 years, your default prediction changes to 0."*

---

## 8. Conclusion & Future Work

### 🔑 Key Takeaways
- A high-accuracy model is **not enough** in credit risk — explainability and fairness are equally critical
- `class_weight="balanced"` is **non-negotiable** for imbalanced datasets like Home Credit (8% default rate)
- SHAP and MAPOCAM together provide a complete picture: *why* someone is high-risk and *what* they can do about it
- MAPOCAM outperforms DiCE for this dataset due to realistic, incremental counterfactuals

### 🚀 Future Improvements
- **More models:** Extend to `LGBMClassifier`, `XGBClassifier` — faster and often more accurate on tabular data
- **Fairness analysis:** Audit model for demographic bias across gender, age groups
- **Feature selection:** Reduce from 118 to top-30 SHAP features to speed up Optuna tuning
- **Calibration:** Apply Platt scaling or isotonic regression to fix Brier Score degradation
- **Increase `n_trials`:** From 5 → 50+ trials with `n_jobs=-1` for better hyperparameter optimization
