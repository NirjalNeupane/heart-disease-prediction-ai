"""
Heart Disease Prediction System
---------------------------------------
Author: Nirjal Neupane

Description:
This program predicts heart disease using machine learning.
It compares three models:
    1. Logistic Regression
    2. Random Forest
    3. Support Vector Machine (SVM)

Dataset:
UCI Heart Disease Dataset (Cleveland subset)
File: heart_disease_cleveland.csv
"""

# ====================================================
# 1. IMPORT REQUIRED LIBRARIES
# ====================================================

import pandas as pd
import numpy as np

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ====================================================
# 2. LOAD THE DATASET
# ====================================================

# Load Cleveland dataset
data = pd.read_csv("heart_disease_cleveland.csv")

# Display basic dataset information
print("=====================================")
print("Dataset Information")
print("=====================================")
print("Shape:", data.shape)
print(data.head())

# ====================================================
# 3. DATA PREPROCESSING
# ====================================================

# Separate features (X) and target variable (y)
X = data.drop("target", axis=1)
y = data["target"]

# Handle missing values (if present)
# Using median is suitable for medical data
X = X.fillna(X.median())

# ====================================================
# 4. TRAIN-TEST SPLIT
# ====================================================

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps class balance
)

# ====================================================
# 5. FEATURE SCALING
# ====================================================

# Standard scaling improves model performance (especially SVM & LR)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================================================
# 6. INITIALIZE MODELS
# ====================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        probability=True
    )
}

# Store results for comparison
results = []

# ====================================================
# 7. TRAIN & EVALUATE MODELS
# ====================================================

print("\n=====================================")
print("MODEL TRAINING AND EVALUATION")
print("=====================================\n")

for model_name, model in models.items():

    print(f"🔹 Training {model_name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probability for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:\n", cm)

    # Save results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

# ====================================================
# 8. FINAL MODEL COMPARISON
# ====================================================

results_df = pd.DataFrame(results)

print("\n=====================================")
print("FINAL MODEL COMPARISON")
print("=====================================")

# Sort by Accuracy (best first)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print(results_df)

# ====================================================
# 9. CONCLUSION OUTPUT
# ====================================================

best_model = results_df.iloc[0]["Model"]
print(f"\n✅ Best performing model: {best_model}")

# ====================================================
# 10. VISUALIZATION (GRAPHS WINDOW)
# ====================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# --- Confusion Matrix Plot (last model only for demo) ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# --- ROC Curve for all models ---
plt.figure(figsize=(8, 6))

for model_name, model in models.items():
    # Predict probabilities again
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=model_name)

# Diagonal line (random guess)
plt.plot([0, 1], [0, 1], linestyle='--')

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# --- Model Accuracy Comparison Bar Chart ---
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.show()
