import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
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
st.title("💳 Fraud Detection System (ML Demo App)")
st.write("Machine Learning-based classification system")

# ---------------------------------------------------
# LOAD DATASET (FIXED - NO CSV)
# ---------------------------------------------------
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Class"] = data.target

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

# ---------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------
st.subheader("Class Distribution")

class_counts = df["Class"].value_counts()

fig0, ax0 = plt.subplots()
ax0.pie(class_counts, labels=["Class 0", "Class 1"], autopct="%1.1f%%")
st.pyplot(fig0)

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# SMOTE
# ---------------------------------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

st.subheader("SMOTE Applied")

smote_df = pd.DataFrame({
    "Before": y.value_counts(),
    "After": pd.Series(y_resampled).value_counts()
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
st.subheader("Model Accuracy")

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
# ACCURACY GRAPH
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
# INPUT PREDICTION
# ---------------------------------------------------
st.subheader("Predict New Data")

input_data = []

for i in range(X.shape[1]):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

if st.button("Predict"):
    input_scaled = scaler.transform(input_array)
    prediction = mlp.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent / Class 1")
    else:
        st.success("✅ Normal / Class 0")

# ---------------------------------------------------
# SUCCESS
# ---------------------------------------------------
st.success("🚀 App Running Successfully on Streamlit")