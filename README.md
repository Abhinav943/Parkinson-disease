# Parkinson's Disease Detection Using Machine Learning

## Project Overview
This project applies Machine Learning algorithms to acoustic features extracted from voice recordings to detect Parkinson's disease. [cite_start]By analyzing various vocal metrics, the models predict whether a patient is healthy or has Parkinson's disease[cite: 2, 236]. 

## Dataset Description
[cite_start]The project utilizes a dataset containing 195 instances and 24 features[cite: 116, 117]. 
* [cite_start]**Target Variable:** `status` (1 = Parkinson's, 0 = Healthy)[cite: 275].
* [cite_start]**Key Features:** Various acoustic measurements including Average Pitch (`MDVP:Fo(Hz)`), Maximum Pitch (`MDVP:Fhi(Hz)`), Minimum Pitch (`MDVP:Flo(Hz)`), Jitter, Shimmer, Harmonics-to-Noise Ratio (`HNR`), and Pitch Period Entropy (`PPE`)[cite: 110, 111, 112, 113].
* [cite_start]**Data Cleansing:** The non-predictive `name` column was removed prior to model training[cite: 118, 119]. 

## Technologies & Libraries Used
The following Python libraries were utilized for data manipulation, visualization, and machine learning:
* [cite_start]`pandas` & `numpy` [cite: 3, 4]
* [cite_start]`seaborn` & `matplotlib.pyplot` [cite: 5, 6]
* [cite_start]`scikit-learn` (`train_test_split`, `StandardScaler`, `LogisticRegression`, `DecisionTreeClassifier`, `SVC`, `LinearRegression`, metrics) [cite: 7, 8, 9, 10, 11, 12, 516]
* [cite_start]`imblearn` (`SMOTE`) [cite: 13]

## Methodology
1. [cite_start]**Exploratory Data Analysis (EDA):** * Generated a Feature Correlation Heatmap to identify relationships and multicollinearity among variables (e.g., strong correlations between Jitter and Shimmer metrics)[cite: 128, 131, 193].
   * [cite_start]Created a scatter plot analyzing the relationship between Average Frequency and Maximum Frequency, colored by patient status[cite: 196, 197].
2. [cite_start]**Data Preprocessing:** * Features and target labels were separated into `X` and `y` variables[cite: 227, 228].
   * [cite_start]The data was split into an 80% training set and a 20% testing set[cite: 230].
   * [cite_start]Applied `StandardScaler` to normalize features for distance-based algorithms[cite: 231, 232].
3. **Model Implementation:** Four distinct algorithms were trained and evaluated:
   * [cite_start]**Logistic Regression** [cite: 237]
   * [cite_start]**Decision Tree Classifier** (Max Depth = 4) [cite: 333]
   * [cite_start]**Support Vector Machine (SVM)** (RBF Kernel) [cite: 456]
   * [cite_start]**Linear Regression** (Output thresholded at $\geq 0.5$ for binary classification) [cite: 516, 530]

## Results & Evaluation
[cite_start]Model performance was evaluated using Accuracy, Precision, Recall, F1-scores, and Confusion Matrices[cite: 240, 241, 273]. 

| Model | Accuracy | Class 1 (Parkinson's) F1-Score |
| :--- | :--- | :--- |
| **Decision Tree** | [cite_start]92.31% [cite: 340] | [cite_start]0.95 [cite: 353] |
| **Logistic Regression** | [cite_start]89.74% [cite: 245] | [cite_start]0.94 [cite: 258] |
| **SVM (RBF Kernel)** | [cite_start]89.74% [cite: 460] | [cite_start]0.94 [cite: 472] |
| **Linear Regression** | [cite_start]87.18% [cite: 545] | [cite_start]0.92 [cite: 557] |

* [cite_start]**Feature Importance:** The Decision Tree identified `PPE` (Pitch Period Entropy) as the most critical root node feature for splitting the data[cite: 404, 453]. [cite_start]Logistic Regression identified `PPE`, `D2`, and `spread1` as heavily weighted coefficients[cite: 312, 313, 314]. 
* [cite_start]**Note on False Negatives:** The Decision Tree effectively minimized False Negatives (0 cases missed), which is a critical metric in medical diagnostics[cite: 382].

## How to Run
1. Ensure Python and JupyterLab/Jupyter Notebook are installed.
2. Install the required dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn`
3. [cite_start]Place `parkinsons.csv` in the same root directory as the notebook[cite: 14].
4. Run all cells in `Parkinson_disease_using_ML.ipynb` to reproduce the standard scaling, model training, and visualizations.