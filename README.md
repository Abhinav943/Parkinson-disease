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

## 📌 Overview
This project applies machine learning techniques to detect Parkinson’s disease using voice-based biomarkers. By analyzing acoustic features such as frequency variations, jitter, shimmer, and nonlinear dynamics, the models classify individuals as healthy or affected.

---

## 📂 Dataset
- **File:** `parkinsons.csv`  
- **Samples:** 195  
- **Features:** 24  
- **Target Variable:** `status` (0 = healthy, 1 = Parkinson’s)

### 🔧 Preprocessing
- Removed non-informative `name` column  
- Applied feature scaling using `StandardScaler`  
- Split dataset into 80% training and 20% testing  

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Environment:** Jupyter Notebook  

### 📚 Libraries
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- imbalanced-learn  

---

## 📊 Exploratory Data Analysis
- Correlation heatmap for feature relationships  
- Scatter plots for class separability  
- Feature distribution analysis  

---

## 🤖 Models Implemented

| Model                | Accuracy | Key Insight |
|---------------------|----------|------------|
| Decision Tree        | **92.31%** | Best performer; PPE most important feature |
| Logistic Regression  | 89.74%   | High precision; weaker on minority class |
| SVM (RBF Kernel)     | 89.74%   | Comparable to Logistic Regression |
| Linear Regression    | 87.18%   | Adapted using classification threshold |

---

## 📈 Key Insights
- Pitch Period Entropy (PPE) is the most influential feature  
- Tree-based models performed best  
- Class imbalance affects minority class prediction  
- Nonlinear patterns are important in diagnosis  

---

## 📉 Visualizations
- Correlation Heatmap  
- Scatter Plots  
- Confusion Matrices  
- Decision Tree Visualization  
- Feature Importance Charts  
- Model Comparison Graph  

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/Abhinav943/Parkinson-disease

# Navigate into project
cd parkinsons-ml

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook
```

---

## 📌 Future Improvements
- Handle imbalance using SMOTE  
- Hyperparameter tuning (GridSearchCV)  
- Try ensemble models (Random Forest, XGBoost)  
- Deploy with Streamlit or Flask  

---

## 🎯 Applications
- Early detection of Parkinson’s disease  
- Healthcare ML model interpretability  
- Voice-based diagnostic tools  

---

## ⭐ Support
If you found this useful, consider giving it a star ⭐

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.