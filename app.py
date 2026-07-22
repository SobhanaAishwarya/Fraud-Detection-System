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
    layout="wide",
)

# ---------------------------------------------------
# THEME
# ---------------------------------------------------

NAVY = "#0F1B2D"
SLATE = "#334155"
SLATE_LIGHT = "#64748B"
ACCENT = "#2563EB"
GOLD = "#C8973C"
LINE = "#E4E7EC"
PALETTE = [ACCENT, GOLD, NAVY, SLATE_LIGHT]

st.markdown(
    """
    <style>
    .stApp { background: #F5F7FA; font-family: 'Inter', 'Segoe UI', sans-serif; }
    .block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1220px; }
    h1, h2, h3 { color: #0F1B2D; font-weight: 700; }

    .app-header { margin-bottom: 6px; }
    .app-header .eyebrow {
        font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #2563EB; margin-bottom: 4px;
    }
    .app-header h1 { font-size: 26px; margin: 0; }
    .app-header p { font-size: 14px; color: #64748B; margin: 4px 0 0; }

    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 2px 10px rgba(15,27,45,0.06);
    }
    [data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #0F1B2D; font-weight: 800; }

    .section-title {
        font-size: 12.5px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; color: #64748B; margin: 30px 0 14px;
    }

    .chart-card {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 14px;
        padding: 18px 20px 6px; box-shadow: 0 2px 10px rgba(15,27,45,0.06);
        margin-bottom: 18px;
    }
    .chart-card h4 {
        margin: 0 0 8px; font-size: 13px; font-weight: 700; color: #0F1B2D;
    }

    .insight-card {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-left: 4px solid #2563EB;
        border-radius: 12px; padding: 18px 22px; box-shadow: 0 2px 10px rgba(15,27,45,0.06);
    }
    .insight-card ul { margin: 0; padding-left: 18px; }
    .insight-card li { color: #334155; font-size: 14px; margin-bottom: 8px; line-height: 1.5; }

    [data-testid="stDataFrame"] { border: 1px solid #E4E7EC; border-radius: 12px; }
    [data-testid="stTable"] { border: 1px solid #E4E7EC; border-radius: 12px; }
    [data-testid="stFileUploader"] {
        background: #FFFFFF; border: 1.5px dashed #2563EB; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": LINE,
    "axes.labelcolor": SLATE,
    "text.color": NAVY,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.grid": True,
    "grid.color": "#EEF1F5",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})
sns.set_style("white")
sns.set_palette(PALETTE)


def chart_card_open(title: str) -> None:
    st.markdown(f'<div class="chart-card"><h4>{title}</h4>', unsafe_allow_html=True)


def chart_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="app-header">
      <div class="eyebrow">Risk Analytics</div>
      <h1>Fraud Detection Analytics Dashboard</h1>
      <p>Machine learning based credit card fraud detection system.</p>
    </div>
    """,
    unsafe_allow_html=True,
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

st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)
st.caption(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

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
        "No target column found!"
    )
    st.stop()

st.success(
    f"Target column detected: {target_col}"
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

st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

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

st.markdown('<div class="section-title">Class Distribution — Before SMOTE</div>', unsafe_allow_html=True)

labels = [
    "Legitimate",
    "Fraud"
]

sizes = df[target_col].value_counts().reindex([0, 1]).fillna(0)

before_counts = pd.DataFrame({
    "Class": labels,
    "Count": sizes.values.astype(int),
})

pie_col, bar_col = st.columns(2)

with pie_col:

    chart_card_open("Legitimate vs Fraud")

    fig1, ax1 = plt.subplots(figsize=(5.0, 5.0), dpi=140)

    wedges, _, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=[NAVY, GOLD],
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
        textprops={"fontsize": 11.5, "color": SLATE},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    fig1.tight_layout()

    st.pyplot(fig1, use_container_width=False)

    chart_card_close()

with bar_col:

    categorical_candidates = [
        "Transaction_Type", "Device_Type", "Location_Type",
        "Merchant_Category", "Payment_Method",
    ]
    compare_col = next((c for c in categorical_candidates if c in df.columns), None)

    if compare_col is not None:

        chart_card_open(f"{compare_col.replace('_', ' ')} vs {labels[1]}")

        fig1b, ax1b = plt.subplots(figsize=(5.6, 5.0), dpi=140)

        plot_df = df.copy()
        plot_df[target_col] = plot_df[target_col].map({0: "Legitimate", 1: "Fraud"})

        sns.countplot(
            x=compare_col,
            hue=target_col,
            data=plot_df,
            ax=ax1b,
            palette=[NAVY, GOLD],
        )

        ax1b.set_xlabel("")
        ax1b.set_ylabel("Transactions")
        ax1b.tick_params(axis="x", rotation=15)
        ax1b.legend(title="", frameon=False)
        fig1b.tight_layout()

        st.pyplot(fig1b, use_container_width=True)

    elif "Transaction_Amount" in df.columns:

        chart_card_open(f"Transaction Amount vs {labels[1]}")

        fig1b, ax1b = plt.subplots(figsize=(5.6, 5.0), dpi=140)

        plot_df = df.copy()
        plot_df[target_col] = plot_df[target_col].map({0: "Legitimate", 1: "Fraud"})

        sns.boxplot(
            x=target_col,
            y="Transaction_Amount",
            data=plot_df,
            ax=ax1b,
            palette=[NAVY, GOLD],
        )

        ax1b.set_xlabel("")
        fig1b.tight_layout()

        st.pyplot(fig1b, use_container_width=True)

    else:

        chart_card_open("Feature Comparison")
        st.caption("No comparable feature found in this dataset.")

    chart_card_close()

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
# CLASS DISTRIBUTION: BEFORE VS AFTER SMOTE
# ---------------------------------------------------

st.markdown('<div class="section-title">Class Distribution — Before vs After SMOTE</div>', unsafe_allow_html=True)

after_counts = pd.DataFrame({
    "Class": ["Legitimate", "Fraud"],
    "Count": [
        int((y_resampled == 0).sum()),
        int((y_resampled == 1).sum())
    ]
})

before_col, after_col = st.columns(2)

with before_col:
    st.dataframe(before_counts, use_container_width=True, hide_index=True)

    chart_card_open("Before SMOTE")

    fig_before, ax_before = plt.subplots(figsize=(5.6, 4.2), dpi=140)

    sns.barplot(
        x="Class",
        y="Count",
        data=before_counts,
        ax=ax_before,
        palette=[NAVY, GOLD],
    )

    ax_before.set_xlabel("")
    ax_before.set_ylabel("Transactions")
    for container in ax_before.containers:
        ax_before.bar_label(container, fmt="%d", padding=3, fontsize=10, color=SLATE)
    fig_before.tight_layout()

    st.pyplot(fig_before, use_container_width=True)

    chart_card_close()

with after_col:
    st.dataframe(after_counts, use_container_width=True, hide_index=True)

    chart_card_open("After SMOTE")

    fig_after, ax_after = plt.subplots(figsize=(5.6, 4.2), dpi=140)

    sns.barplot(
        x="Class",
        y="Count",
        data=after_counts,
        ax=ax_after,
        palette=[NAVY, GOLD],
    )

    ax_after.set_xlabel("")
    ax_after.set_ylabel("Transactions")
    for container in ax_after.containers:
        ax_after.bar_label(container, fmt="%d", padding=3, fontsize=10, color=SLATE)
    fig_after.tight_layout()

    st.pyplot(fig_after, use_container_width=True)

    chart_card_close()

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

st.markdown('<div class="section-title">Model Accuracy Comparison</div>', unsafe_allow_html=True)

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
    accuracy_df,
    use_container_width=True,
    hide_index=True,
)

chart_card_open("Accuracy by Model")

fig3, ax3 = plt.subplots(figsize=(11, 4.2), dpi=140)

sns.barplot(
    x="Model",
    y="Accuracy",
    data=accuracy_df,
    ax=ax3,
    palette=PALETTE[:3],
)

ax3.set_ylim(0, 1)
ax3.set_xlabel("")
fig3.tight_layout()

st.pyplot(fig3, use_container_width=True)

chart_card_close()

# ---------------------------------------------------
# ALL CONFUSION MATRICES
# ---------------------------------------------------

st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:

    chart_card_open("Logistic Regression")

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=140)

    sns.heatmap(
        confusion_matrix(
            y_test,
            lr_pred
        ),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
    )
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    chart_card_close()

with c2:

    chart_card_open("Decision Tree")

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=140)

    sns.heatmap(
        confusion_matrix(
            y_test,
            dt_pred
        ),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
    )
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    chart_card_close()

with c3:

    chart_card_open("Neural Network")

    fig, ax = plt.subplots(figsize=(4, 3.6), dpi=140)

    sns.heatmap(
        confusion_matrix(
            y_test,
            mlp_pred
        ),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
    )
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    chart_card_close()

# ---------------------------------------------------
# AUC ROC CURVE
# ---------------------------------------------------

st.markdown('<div class="section-title">AUC-ROC Curve</div>', unsafe_allow_html=True)

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

roc_col, _ = st.columns([1, 1])

with roc_col:

    chart_card_open(f"ROC Curve — Logistic Regression (AUC = {auc_score:.2f})")

    fig4, ax4 = plt.subplots(figsize=(5.2, 4.4), dpi=140)

    ax4.plot(
        fpr,
        tpr,
        color=ACCENT,
        linewidth=2.4,
        label=f"AUC = {auc_score:.2f}",
    )

    ax4.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=SLATE_LIGHT,
        linewidth=1.3,
    )

    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend(frameon=False)
    fig4.tight_layout()

    st.pyplot(fig4, use_container_width=False)

    chart_card_close()

# ---------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------

st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)

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

st.markdown('<div class="section-title">Download Prediction Report</div>', unsafe_allow_html=True)

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
    label="Download Excel Report",
    data=buffer.getvalue(),
    file_name="Fraud_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ---------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------

st.markdown('<div class="section-title">Business Insights</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="insight-card">
      <ul>
        <li>Fraud transactions constitute a very small percentage of total transactions.</li>
        <li>Neural Network and Decision Tree achieved excellent performance.</li>
        <li>Isolation Forest effectively detected suspicious anomalies.</li>
        <li>AUC-ROC indicates strong fraud classification capability.</li>
        <li>Dashboard can assist financial institutions in real-time fraud monitoring.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
