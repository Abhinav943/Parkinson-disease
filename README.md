# 🧠 Parkinson's Disease Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)

---

## Overview

Parkinson's Disease (PD) is a progressive neurological disorder affecting movement. Early detection is critical for improving patient outcomes. This project leverages voice measurement data to build and compare multiple ML classifiers for automated PD detection, achieving up to **96.6% accuracy** on the test set.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | UCI Machine Learning Repository |
| **Name** | Parkinsons Disease Dataset |
| **Instances** | 195 |
| **Features** | 23 (22 input + 1 target) |
| **Target** | `status` — `1` = Parkinson's, `0` = Healthy |
| **Class Distribution** | 147 Parkinson's, 48 Healthy (imbalanced) |

The dataset contains biomedical voice measurements including fundamental frequency, jitter, shimmer, noise-to-harmonic ratios, and nonlinear dynamic features.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/parkinsons-detection.git
cd parkinsons-detection

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib joblib
```

---

## Methodology

### 1. Data Preprocessing
- Dropped the non-informative `name` column
- Verified no missing values or duplicate rows
- Cast `status` column to `uint8` to save memory

### 2. Exploratory Data Analysis
- **Count plot** — revealed class imbalance (147 vs 48 samples)
- **Correlation heatmap** — showed high multicollinearity among jitter and shimmer features
- **Box plots** — patients with lower HNR, MDVP:Fo(Hz), MDVP:Fhi(Hz), and MDVP:Flo(Hz) tend to have Parkinson's
- **Pair plots** — confirmed strong correlation within jitter and shimmer feature groups

### 3. Class Balancing
Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance classes:
- Before: 195 samples (147 PD, 48 healthy)
- After: 294 samples (147 each)

### 4. Feature Scaling
Applied **MinMaxScaler** with range `(-1, 1)` to normalize all features.

### 5. Train/Test Split
80/20 split → 235 training samples, 59 test samples.

### 6. Hyperparameter Tuning
All models optimized using **GridSearchCV** with 5-fold cross-validation (3-fold for XGBoost).

---

## Models & Results

| Metric | Decision Tree | Random Forest | Logistic Regression | SVM | Naive Bayes | KNN | XGBoost |
|---|---|---|---|---|---|---|---|
| **Accuracy** | 0.932 | **0.966** | 0.831 | **0.966** | 0.763 | **0.966** | 0.915 |
| **F1-Score** | 0.920 | **0.962** | 0.783 | 0.960 | 0.650 | 0.960 | 0.909 |
| **Recall** | 0.885 | **0.962** | 0.692 | 0.923 | 0.500 | 0.923 | **0.962** |
| **Precision** | 0.958 | **0.962** | 0.900 | **1.000** | 0.929 | **1.000** | 0.862 |
| **R2-Score** | 0.725 | **0.862** | 0.312 | **0.862** | 0.037 | **0.862** | 0.656 |

### Best Hyperparameters

| Model | Best Parameters |
|---|---|
| Decision Tree | `criterion=entropy`, `max_depth=6`, `random_state=120` |
| Random Forest | `criterion=entropy`, `max_depth=7`, `n_estimators=125`, `random_state=200` |
| SVM | `C=100`, `gamma=1`, `kernel=rbf` |
| XGBoost | `eta=0.1`, `max_depth=7`, `reg_lambda=1`, `random_state=300` |

---

## Key Findings

- **Random Forest, SVM, and KNN** tied for the best overall accuracy at **96.6%**
- **SVM and KNN** achieved perfect precision of **1.0** (zero false positives)
- **Naive Bayes** performed the worst (76.3% accuracy), struggling with the highly correlated feature space
- **XGBoost** achieved perfect scores on the training set but showed signs of overfitting on test data
- Lower values of `HNR`, `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, and `MDVP:Flo(Hz)` are strong indicators of Parkinson's Disease
- Jitter and shimmer feature groups are heavily inter-correlated, suggesting dimensionality reduction (e.g., PCA) could be explored

---

## Technologies Used

- **Data:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **ML Models:** `scikit-learn`, `xgboost`
- **Class Balancing:** `imbalanced-learn` (SMOTE)
- **Model Persistence:** `joblib`

---

## ⚠️ Disclaimer

This project is for academic and research purposes only. It is not intended for clinical diagnosis or medical decision-making. Always consult a qualified medical professional for health-related decisions.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Abhinav943/Parkinson-disease/blob/main/LICENSE) file for details.
