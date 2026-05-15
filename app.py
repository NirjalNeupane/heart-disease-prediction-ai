"""
Heart Disease Prediction Web App
--------------------------------
Author: Nirjal Neupane
"""

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1. LOAD DATA
# ===============================

data = pd.read_csv("heart_disease_cleveland.csv")

X = data.drop("target", axis=1)
y = data["target"]

# Fill missing values
X = X.fillna(X.median())

# ===============================
# 2. TRAIN MODEL (Random Forest)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ===============================
# 3. STREAMLIT UI
# ===============================

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details below:")

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [1, 0])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("Oldpeak (0–5)", 0.0, 5.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3])

# ===============================
# 4. PREDICTION
# ===============================

if st.button("Predict"):

    # Create input as dataframe
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]], columns=X.columns)

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\nProbability: {probability:.2f}")
