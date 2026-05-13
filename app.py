import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

st.title("Credit Card Fraud Detection System")

# Load dataset
df = pd.read_csv("creditcard.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

# Features & Target
X = df.drop('Class', axis=1)
y = df['Class']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE (handle imbalance)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
mlp_pred = mlp.predict(X_test)

# Accuracy
lr_acc = accuracy_score(y_test, lr_pred)
dt_acc = accuracy_score(y_test, dt_pred)
mlp_acc = accuracy_score(y_test, mlp_pred)

st.subheader("Model Accuracy")

accuracy_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Neural Network"],
    "Accuracy": [lr_acc, dt_acc, mlp_acc]
})

st.write(accuracy_df)

# Accuracy graph
fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x="Model", y="Accuracy", data=accuracy_df, ax=ax)
plt.ylim(0,1)

st.pyplot(fig)

# Confusion matrix
st.subheader("Neural Network Confusion Matrix")

cm = confusion_matrix(y_test, mlp_pred)

fig2, ax2 = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)

st.pyplot(fig2)

st.success("Fraud Detection System Running Successfully 🚀")