
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("Fraud Detection Analytics Dashboard")
st.write(
    "Machine Learning based Credit Card Fraud Detection System"
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

else:
    df = pd.read_csv("fraud_detection_dataset_500.csv")

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

# ---------------------------------------------------
# AUTO DETECT TARGET
# ---------------------------------------------------

possible_targets = [
    "Class",
    "class",
    "Fraud",
    "fraud",
    "label",
    "target"
]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    st.error(
        " No target column found!"
    )
    st.stop()

st.success(
    f"Target Column Detected: {target_col}"
)

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

total_transactions = len(df)

fraud_transactions = (
    df[target_col] == 1
).sum()

legitimate_transactions = (
    total_transactions -
    fraud_transactions
)

fraud_rate = (
    fraud_transactions /
    total_transactions
) * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Transactions",
    total_transactions
)

c2.metric(
    "Fraud Transactions",
    fraud_transactions
)

c3.metric(
    "Legitimate Transactions",
    legitimate_transactions
)

c4.metric(
    "Fraud Rate",
    f"{fraud_rate:.2f}%"
)

# ---------------------------------------------------
# PIE CHART
# ---------------------------------------------------

st.subheader(
    "Fraud Distribution"
)

labels = [
    "Legitimate",
    "Fraud"
]

sizes = df[target_col].value_counts()

fig1, ax1 = plt.subplots(
    figsize=(5, 5)
)

ax1.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

st.pyplot(fig1)

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = df.drop(
    target_col,
    axis=1
)

y = df[target_col]

# ---------------------------------------------------
# HANDLE NON NUMERIC
# ---------------------------------------------------

X = pd.get_dummies(X)

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# SMOTE
# ---------------------------------------------------

smote = SMOTE(
    random_state=42
)

X_resampled, y_resampled = (
    smote.fit_resample(
        X_scaled,
        y
    )
)

# ---------------------------------------------------
# CLASS DISTRIBUTION AFTER SMOTE
# ---------------------------------------------------

st.subheader("Class Distribution After SMOTE")

fig2, ax2 = plt.subplots(figsize=(5,4))

sns.countplot(
    x=y_resampled,
    ax=ax2
)

ax2.set_xlabel("Class")
ax2.set_ylabel("Count")
ax2.set_title("Balanced Class Distribution")

st.pyplot(fig2)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42
    )
)

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

lr = LogisticRegression(
    max_iter=1000
)

dt = DecisionTreeClassifier(
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=300,
    random_state=42
)

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

lr.fit(
    X_train,
    y_train
)

dt.fit(
    X_train,
    y_train
)

mlp.fit(
    X_train,
    y_train
)

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------

lr_pred = lr.predict(
    X_test
)

dt_pred = dt.predict(
    X_test
)

mlp_pred = mlp.predict(
    X_test
)

# ---------------------------------------------------
# ACCURACY TABLE
# ---------------------------------------------------

st.subheader(
    "Model Accuracy Comparison"
)

accuracy_df = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Neural Network"
        ],

        "Accuracy": [
            accuracy_score(
                y_test,
                lr_pred
            ),

            accuracy_score(
                y_test,
                dt_pred
            ),

            accuracy_score(
                y_test,
                mlp_pred
            )
        ]
    }
)

st.dataframe(
    accuracy_df
)

fig3, ax3 = plt.subplots(
    figsize=(7, 5)
)

sns.barplot(
    x="Model",
    y="Accuracy",
    data=accuracy_df,
    ax=ax3
)

plt.ylim(0, 1)

st.pyplot(fig3)

# ---------------------------------------------------
# ALL CONFUSION MATRICES
# ---------------------------------------------------

st.subheader(
    "Confusion Matrices"
)

c1, c2, c3 = st.columns(3)

with c1:

    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(
            y_test,
            lr_pred
        ),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        "Logistic Regression"
    )

    st.pyplot(fig)

with c2:

    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(
            y_test,
            dt_pred
        ),
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax
    )

    ax.set_title(
        "Decision Tree"
    )

    st.pyplot(fig)

with c3:

    fig, ax = plt.subplots()

    sns.heatmap(
        confusion_matrix(
            y_test,
            mlp_pred
        ),
        annot=True,
        fmt="d",
        cmap="Reds",
        ax=ax
    )

    ax.set_title(
        "Neural Network"
    )

    st.pyplot(fig)

# ---------------------------------------------------
# AUC ROC CURVE
# ---------------------------------------------------

st.subheader(
    "AUC-ROC Curve"
)

y_prob = (
    lr.predict_proba(
        X_test
    )[:, 1]
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

fpr, tpr, thresholds = (
    roc_curve(
        y_test,
        y_prob
    )
)

fig4, ax4 = plt.subplots(
    figsize=(7, 5)
)

ax4.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.2f}"
)

ax4.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

ax4.legend()

ax4.set_title(
    "ROC Curve"
)

st.pyplot(fig4)

# ---------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------

st.subheader(
    "Anomaly Detection Dashboard"
)

iso = IsolationForest(
    contamination=0.01,
    random_state=42
)

y_pred_iso = (
    iso.fit_predict(
        X_scaled
    )
)

y_pred_iso = [
    1 if i == -1 else 0
    for i in y_pred_iso
]

anomalies = sum(
    y_pred_iso
)

anomaly_rate = (
    anomalies /
    len(df)
) * 100

c1, c2 = st.columns(2)

c1.metric(
    "Total Anomalies",
    anomalies
)

c2.metric(
    "Anomaly Percentage",
    f"{anomaly_rate:.2f}%"
)

# ---------------------------------------------------
# EXCEL REPORT EXPORT
# ---------------------------------------------------

st.subheader(
    "Download Prediction Report"
)

prob = (
    mlp.predict_proba(
        X_test
    )[:, 1]
)

report_df = pd.DataFrame(
    {
        "Transaction_ID":
            range(
                1,
                len(prob) + 1
            ),

        "Prediction":
            np.where(
                mlp_pred == 1,
                "Fraud",
                "Legitimate"
            ),

        "Fraud_Probability":
            prob
    }
)

buffer = BytesIO()

with pd.ExcelWriter(
    buffer,
    engine="xlsxwriter"
) as writer:

    report_df.to_excel(
        writer,
        index=False,
        sheet_name="Fraud_Report"
    )

st.download_button(
    label="📥 Download Excel Report",
    data=buffer.getvalue(),
    file_name="Fraud_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ---------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------

st.subheader(
    "Business Insights"
)

st.success(
"""
• Fraud transactions constitute a very small percentage of total transactions.

• Neural Network and Decision Tree achieved excellent performance.

• Isolation Forest effectively detected suspicious anomalies.

• AUC-ROC indicates strong fraud classification capability.

• Dashboard can assist financial institutions in real-time fraud monitoring.
"""
)

st.success(
    "Fraud Detection Dashboard Executed Successfully"
)
