import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("parkinsons.csv")

# Drop non-informative column
if "name" in df.columns:
    df = df.drop("name", axis=1)

# Split features and target
X = df.drop("status", axis=1)
y = df["status"]

# Standardize features (same as training pipeline idea)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Decision Tree model (your best model)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_scaled, y)

st.title("🧠 Parkinson's Disease Detection App")
st.write("Using Decision Tree Classifier")

st.subheader("Enter Voice Feature Values")

# Collect user inputs
input_data = []

for col in X.columns:
    val = st.number_input(col, value=float(X[col].mean()))
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Parkinson's Disease Detected")
    else:
        st.success("✅ No Parkinson's Disease Detected")

    st.write("### Prediction Confidence")
    st.write(f"Healthy: {probability[0][0]:.2f}")
    st.write(f"Parkinson's: {probability[0][1]:.2f}")