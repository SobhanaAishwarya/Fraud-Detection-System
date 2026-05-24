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
st.title("💳 Fraud Detection System (ML Project)")
st.write("Machine Learning-based Fraud Detection App")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("fraud_detection_dataset_500.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

# ---------------------------------------------------
# AUTO DETECT TARGET COLUMN
# ---------------------------------------------------
possible_targets = ["Class", "class", "Fraud", "fraud", "label", "target"]

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    st.error("❌ No target column found! Please name it as Class/Fraud/label/target")
    st.stop()

st.write(f"🎯 Target Column Detected: **{target_col}**")

# ---------------------------------------------------
# SPLIT FEATURES & TARGET
# ---------------------------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

# ---------------------------------------------------
# HANDLE NON-NUMERIC DATA (IMPORTANT FIX)
# ---------------------------------------------------
X = pd.get_dummies(X)

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# SMOTE (Only if imbalance exists)
# ---------------------------------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

st.subheader("Class Distribution After SMOTE")
st.write(pd.Series(y_resampled).value_counts())

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
# MODELS
# ---------------------------------------------------
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
mlp_pred = mlp.predict(X_test)

# ---------------------------------------------------
# ACCURACY
# ---------------------------------------------------
st.subheader("Model Accuracy Comparison")

accuracy_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Neural Network"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, mlp_pred)
    ]
})

st.write(accuracy_df)

# ---------------------------------------------------
# BAR PLOT
# ---------------------------------------------------
fig1, ax1 = plt.subplots()
sns.barplot(x="Model", y="Accuracy", data=accuracy_df, ax=ax1)
plt.ylim(0, 1)
st.pyplot(fig1)

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
st.subheader("Confusion Matrix (Neural Network)")

cm = confusion_matrix(y_test, mlp_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
st.pyplot(fig2)

# ---------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------
st.subheader("Anomaly Detection (Isolation Forest)")

iso = IsolationForest(contamination=0.01, random_state=42)
y_pred_iso = iso.fit_predict(X_scaled)

y_pred_iso = [1 if i == -1 else 0 for i in y_pred_iso]

st.write("Anomalies detected:", sum(y_pred_iso))

# ---------------------------------------------------
# USER INPUT PREDICTION
# ---------------------------------------------------
st.subheader("Predict New Transaction")

input_data = []

for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

if st.button("Predict"):
    input_scaled = scaler.transform(input_array)
    prediction = mlp.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")