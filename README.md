# Parkinson's Disease Detection: A Statistical & Mathematical Modeling Approach

## Overview
This repository contains the implementation of a machine learning pipeline designed to detect Parkinson's disease from biomedical voice measurements. Moving beyond basic model implementation, this project heavily emphasizes the mathematical foundations of classification algorithms, focusing on statistical feature scaling, optimization of decision boundaries, and rigorous probability analysis for medical diagnostics.

## Mathematical Foundations & Optimization
This project benchmarks multiple classification algorithms, evaluating their underlying mathematical trade-offs when applied to high-dimensional biological data.

### 1. Statistical Feature Scaling
Biomedical datasets frequently exhibit extreme variance across features (e.g., vocal fundamental frequencies vs. acoustic jitter measurements). To ensure stable convergence during algorithmic optimization, strict data standardization was applied. Features were centered and scaled to unit variance:
$$z = \frac{x - \mu}{\sigma}$$
This transformation ensures that features with larger magnitudes do not mathematically dominate the objective functions of distance-based models like SVMs.

### 2. Support Vector Machines (SVM) & Hyperplane Optimization
The primary model utilized is a Support Vector Machine. The objective is to find an optimal decision boundary (hyperplane) that maximizes the margin between healthy and Parkinson's-positive data points. 

Mathematically, this is framed as a constrained optimization problem, where we minimize:
$$\min_{w,b} \frac{1}{2} ||w||^2$$
subject to the condition $y_i(w \cdot x_i + b) \ge 1$. To handle non-linearly separable biological data, the model utilizes the kernel trick, implicitly mapping the input vectors into a higher-dimensional feature space to establish a robust classification boundary using calculus and linear algebra concepts.

### 3. Decision Trees & Information Theory
To provide an interpretable, non-linear alternative, Decision Tree classifiers were also implemented. The model constructs a directed acyclic graph by recursively splitting the dataset to minimize impurity. The splits were optimized using information-theoretic metrics such as Gini Impurity:
$$G = 1 - \sum_{i=1}^{C} p_i^2$$
where $p_i$ is the probability of an element belonging to a specific class at a given node.

## Dataset
The model was trained on a dataset of biomedical voice measurements from individuals with and without Parkinson's disease. Key features include:
* **MDVP:Fo (Hz):** Average vocal fundamental frequency.
* **MDVP:Jitter (%) & MDVP:Shimmer:** Measures of variation in fundamental frequency and amplitude.
* **PPE (Pitch Period Entropy):** A nonlinear measure of fundamental frequency variation.

## Evaluation & Precision-Recall Trade-offs
In medical diagnostics, the cost of a False Negative (failing to detect the disease) is significantly higher than a False Positive. Therefore, the evaluation phase heavily weighed **Recall (Sensitivity)** alongside overall accuracy.

### Key Results:
* **SVM Accuracy:** 81%
* **Decision Tree Accuracy:** 92%
* **Optimized Recall Score:** 94%

By analyzing the confusion matrix and adjusting the decision thresholds via probability analysis, the model's precision-recall trade-off was tuned to minimize type II errors, ensuring high reliability for preliminary medical screening.

## Setup & Execution
1. Clone the repository: `git clone https://github.com/Abhinav943/Minor-Project-Parkinson-disease-.git`
2. Install required dependencies. 
3. Execute the Jupyter Notebook to view the mathematical transformations and model training process: `jupyter notebook Parkinson_Disease_Detection.ipynb`
