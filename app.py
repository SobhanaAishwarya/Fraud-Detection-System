import os
import pandas as pd

os.system("kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip")

df = pd.read_csv("data/creditcard.csv").sample(5000)
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
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.write("Machine Learning based Fraud Detection using Credit Card Transactions")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("creditcard.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

# ---------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------

st.subheader("Fraud vs Normal Transactions")

class_counts = df['Class'].value_counts()

fig0, ax0 = plt.subplots(figsize=(5, 5))

ax0.pie(
    class_counts,
    labels=['Normal', 'Fraud'],
    autopct='%1.1f%%',
    startangle=90
)

st.pyplot(fig0)

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = df.drop('Class', axis=1)
y = df['Class']

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# SMOTE FOR IMBALANCED DATA
# ---------------------------------------------------

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

st.subheader("SMOTE Applied Successfully")

smote_df = pd.DataFrame({
    "Before SMOTE": y.value_counts(),
    "After SMOTE": pd.Series(y_resampled).value_counts()
})

st.write(smote_df)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# MACHINE LEARNING MODELS
# ---------------------------------------------------

lr = LogisticRegression(max_iter=1000)

dt = DecisionTreeClassifier(random_state=42)

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------

lr.fit(X_train, y_train)

dt.fit(X_train, y_train)

mlp.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

lr_pred = lr.predict(X_test)

dt_pred = dt.predict(X_test)

mlp_pred = mlp.predict(X_test)

# ---------------------------------------------------
# ACCURACY
# ---------------------------------------------------

lr_acc = accuracy_score(y_test, lr_pred)

dt_acc = accuracy_score(y_test, dt_pred)

mlp_acc = accuracy_score(y_test, mlp_pred)

# ---------------------------------------------------
# ACCURACY TABLE
# ---------------------------------------------------

st.subheader("Model Accuracy")

accuracy_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Neural Network"
    ],
    "Accuracy": [
        lr_acc,
        dt_acc,
        mlp_acc
    ]
})

st.write(accuracy_df)

# ---------------------------------------------------
# ACCURACY GRAPH
# ---------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(7, 5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=accuracy_df,
    ax=ax1
)

plt.ylim(0, 1)

st.pyplot(fig1)

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

st.subheader("Neural Network Confusion Matrix")

cm = confusion_matrix(y_test, mlp_pred)

fig2, ax2 = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax2
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

st.pyplot(fig2)

# ---------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------

st.subheader("Isolation Forest Fraud Detection")

iso_forest = IsolationForest(
    contamination=0.01,
    random_state=42
)

y_pred_iforest = iso_forest.fit_predict(X_scaled)

y_pred_iforest = [1 if x == -1 else 0 for x in y_pred_iforest]

fraud_detected = sum(y_pred_iforest)

st.write("Fraud Transactions Detected:", fraud_detected)

# ---------------------------------------------------
# REAL TIME FRAUD PREDICTION
# ---------------------------------------------------

st.subheader("💳 Predict Transaction")

input_data = []

# Time
time_value = st.number_input("Time", value=0.0)

input_data.append(time_value)

# V1 to V28
for i in range(1, 29):

    value = st.number_input(f"V{i}", value=0.0)

    input_data.append(value)

# Amount
amount_value = st.number_input("Amount", value=0.0)

input_data.append(amount_value)

# Convert to array
input_array = np.array(input_data).reshape(1, -1)

# Prediction button
if st.button("Predict Fraud"):

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict using Neural Network
    prediction = mlp.predict(input_scaled)

    if prediction[0] == 1:

        st.error("⚠ Fraudulent Transaction Detected")

    else:

        st.success("✅ Legitimate Transaction")

# ---------------------------------------------------
# SUCCESS MESSAGE
# ---------------------------------------------------

st.success("🚀 Fraud Detection System Running Successfully")