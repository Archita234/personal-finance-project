import random
from datetime import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Health & Anomaly Dashboard",
    layout="wide",
)

sns.set_theme(style="whitegrid")

CATEGORIES = [
    "Food & Dining", "Groceries", "Shopping", "Travel", "Fuel", "Utilities",
    "Mobile & Internet", "Entertainment", "Healthcare", "Education",
    "Personal Care", "Investments", "Miscellaneous",
]
PAYMENT_TYPES = ["Credit", "Debit"]
PAYMENT_MODE = {
    "Credit": ["Bank Transfer", "Cash Deposit", "Interest Credit"],
    "Debit": ["ATM Withdrawal", "UPI"],
}
MERCHANTS = {
    "Food & Dining": ["Domino's", "McDonald's", "KFC", "Starbucks"],
    "Groceries": ["D-Mart", "Reliance Fresh", "Blinkit", "Zepto"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio"],
    "Travel": ["Uber", "Ola", "IRCTC", "Rapido"],
    "Fuel": ["Indian Oil", "HP Petrol Pump", "Bharat Petroleum"],
    "Utilities": ["Electricity Board", "Water Department", "Gas Agency"],
    "Mobile & Internet": ["Jio", "Airtel", "Vi"],
    "Entertainment": ["Netflix", "BookMyShow", "Spotify"],
    "Healthcare": ["Apollo Pharmacy", "MedPlus", "Fortis Hospital"],
    "Education": ["Coursera", "Udemy", "College Fee"],
    "Personal Care": ["Nykaa", "Salon", "Barber Shop"],
    "Investments": ["Groww", "Zerodha", "Mutual Fund SIP"],
    "Salary": ["ABC Pvt Ltd", "XYZ Technologies"],
    "Miscellaneous": ["Local Shop", "Gift Store"],
}
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


# --------------------------------------------------------------------------------------
# Data pipeline (cached so it only reruns when the uploaded file / seed changes)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean_transactions(file_bytes: bytes) -> pd.DataFrame:
    import io
    data1 = pd.read_csv(io.BytesIO(file_bytes))

    data1 = data1.drop(columns=["CustomerID", "CustomerDOB", "CustGender"], errors="ignore")
    data1.rename(columns={"TransactionAmount (INR)": "TransactionAmount"}, inplace=True)

    data1["TransactionTime"] = (
        data1["TransactionTime"].astype(float).astype(int).astype(str).str.zfill(6)
    )
    data1["TransactionTime"] = pd.to_datetime(
        data1["TransactionTime"], format="%H%M%S"
    ).dt.time

    data1["DateTime"] = pd.to_datetime(
        data1["TransactionDate"].astype(str) + " " + data1["TransactionTime"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
    )

    data1["TransactionDay"] = data1["DateTime"].dt.day_name()
    data1["TransactionMonth"] = data1["DateTime"].dt.month_name()
    data1["EventDate"] = data1["DateTime"].dt.day
    data1["IsWeekend"] = data1["TransactionDay"].isin(["Saturday", "Sunday"]).astype(int)

    return data1.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_synthetic_layers(data1: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Recreates data2 (category/merchant/payment) and data3 (income/budget) and merges them in."""
    rnd = random.Random(seed)

    customer_ids = [f"C0{i}" for i in range(1, 500)]

    rows = []
    for i in range(len(data1)):
        category = rnd.choice(CATEGORIES)
        merchant = rnd.choice(MERCHANTS[category])
        ptype = rnd.choice(PAYMENT_TYPES)
        pmode = rnd.choice(PAYMENT_MODE[ptype])
        cust = rnd.choice(customer_ids)
        rows.append({
            "TransactionID": data1.loc[i, "TransactionID"] if "TransactionID" in data1.columns else f"T{i}",
            "CustomerID": cust,
            "Category": category,
            "Merchant": merchant,
            "PaymentType": ptype,
            "PaymentMode": pmode,
        })
    data2 = pd.DataFrame(rows)

    row = []
    for customer in customer_ids:
        base_income = rnd.randint(60000, 100000)
        for month in MONTHS:
            income = base_income + rnd.randint(-2000, 3000)
            budget = rnd.randint(int(0.6 * income), int(0.9 * income))
            row.append({
                "CustomerID": customer,
                "TransactionMonth": month,
                "Income": income,
                "Budget": budget,
            })
    data3 = pd.DataFrame(row)

    data = pd.merge(data1, data2, on="TransactionID")
    data = pd.merge(data, data3, on=["CustomerID", "TransactionMonth"], how="inner")
    return data


@st.cache_data(show_spinner=False)
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["Monthly_Spending"] = data.groupby("CustomerID")["TransactionAmount"].transform("sum")
    data["Savings"] = data["Budget"] - data["Monthly_Spending"]
    data["Investment"] = data["Income"] - data["Budget"]

    # --- Vectorized anomaly flag (equivalent to the original per-row loop, respecting
    # Python's "and" binding tighter than "or": last clause is (H and I), not (H) or (I)) ---
    A = data["TransactionAmount"] <= 0
    B = data["Monthly_Spending"] > data["Budget"]
    C = data["Monthly_Spending"] > 0.8 * data["Income"]
    D = data["Savings"] < 0.25 * data["Budget"]
    E = data["Savings"] > 5 * data["Income"]
    F = data["TransactionTime"].apply(lambda t: time(0, 0, 0) <= t <= time(4, 0, 0))
    G = data["Savings"] < data["Monthly_Spending"]
    H = data["TransactionAmount"] * 0.8 > data["Income"]
    I = data["EventDate"] <= 7

    is_anomaly = A | B | C | D | E | F | G | (H & I)
    data["Is_Anomaly"] = is_anomaly.astype(int)

    # --- Vectorized health score (same rule set as the original loop) ---
    spending_ratio = data["Monthly_Spending"] / data["Income"]
    score1 = np.select(
        [spending_ratio <= 0.60, spending_ratio <= 0.80, spending_ratio <= 1.00],
        [30, 25, 15], default=5,
    )

    savings_ratio = data["Savings"] / data["Income"]
    score2 = np.select(
        [savings_ratio >= 0.30, savings_ratio >= 0.20, savings_ratio >= 0.10, savings_ratio >= 0],
        [20, 15, 10, 5], default=0,
    )

    score3 = np.where(data["Monthly_Spending"] <= data["Budget"], 15, 5)

    investment_ratio = data["Investment"] / data["Income"]
    score4 = np.select(
        [investment_ratio >= 0.10, investment_ratio >= 0.05, investment_ratio > 0],
        [10, 7, 5], default=0,
    )

    balance = data["CustAccountBalance"]
    spending = data["Monthly_Spending"]
    score5 = np.select(
        [balance >= 3 * spending, balance >= spending, balance >= 0.5 * spending],
        [15, 10, 5], default=0,
    )

    behaviour = 10 - 5 * (data["Is_Anomaly"] == 1).astype(int) - 2 * (data["IsWeekend"] == 1).astype(int)
    behaviour = behaviour.clip(lower=0)

    data["HealthScore"] = score1 + score2 + score3 + score4 + score5 + behaviour

    def categorize(score):
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Average"
        elif score >= 40:
            return "Poor"
        else:
            return "Critical"

    data["HealthCategory"] = data["HealthScore"].apply(categorize)
    data = data.dropna().reset_index(drop=True)
    return data


# --------------------------------------------------------------------------------------
# Sidebar — data input
# --------------------------------------------------------------------------------------
st.sidebar.title("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload bank_transactions.csv", type=["csv"])
seed = st.sidebar.number_input("Random seed (synthetic category/income data)", value=42, step=1)
sample_size = st.sidebar.slider(
    "Rows to use (for speed on large files)", min_value=1000, max_value=50000, value=5000, step=1000
)

st.title("💰 Financial Health & Anomaly Detection Dashboard")

if uploaded is None:
    st.info(
        "Upload the `bank_transactions.csv` file in the sidebar to get started. "
        "This file must contain the standard columns: CustomerID, CustomerDOB, CustGender, "
        "TransactionID, TransactionDate, TransactionTime, CustAccountBalance, "
        "TransactionAmount (INR)."
    )
    st.stop()

with st.spinner("Reading and cleaning transactions..."):
    raw_bytes = uploaded.getvalue()
    data1_full = load_and_clean_transactions(raw_bytes)

if len(data1_full) > sample_size:
    data1 = data1_full.sample(sample_size, random_state=int(seed)).reset_index(drop=True)
    st.sidebar.caption(f"Using a random sample of {sample_size:,} of {len(data1_full):,} rows.")
else:
    data1 = data1_full

with st.spinner("Building synthetic category / income layers and merging..."):
    merged = build_synthetic_layers(data1, seed=int(seed))

with st.spinner("Engineering features (spending, savings, anomaly flag, health score)..."):
    data = engineer_features(merged)

st.success(f"Pipeline complete — {len(data):,} rows ready for analysis.")

tabs = st.tabs([
    "Overview",
    "Spending & Trends",
    "Customer & Payment Insights",
    "Health Score",
    "Anomaly Detection Model",
    "Health Score Model",
    "Clustering",
])

# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Processed data preview")
    st.dataframe(data.head(50), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(data):,}")
    c2.metric("Unique customers", f"{data['CustomerID'].nunique():,}")
    c3.metric("Anomaly rate", f"{data['Is_Anomaly'].mean()*100:.1f}%")
    c4.metric("Avg. Health Score", f"{data['HealthScore'].mean():.1f}")

    st.download_button(
        "Download processed dataset (CSV)",
        data.to_csv(index=False).encode("utf-8"),
        file_name="processed_transactions.csv",
        mime="text/csv",
    )

    st.subheader("Distribution of Transaction Amount")
    fig, ax = plt.subplots()
    ax.hist(data["TransactionAmount"], bins=30)
    ax.set_xlabel("Transaction Amount")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# --------------------------------------------------------------------------------------
# Spending & Trends
# --------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Monthly Spending Trend")
    monthly_spending = data.groupby("TransactionMonth")["TransactionAmount"].sum().reindex(MONTHS).dropna().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_spending["TransactionMonth"], monthly_spending["TransactionAmount"],
            color="red", marker="o", linestyle="--")
    ax.set_title("Monthly Spending Trend", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount Spent (₹)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("Total Amount Spent by Category")
    category_spending = data.groupby("Category")["TransactionAmount"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(category_spending["Category"], category_spending["TransactionAmount"], color="steelblue")
    ax.set_title("Total Amount Spent by Category")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Daily Spending")
    daily_spending = data.groupby("EventDate")["TransactionAmount"].sum().reset_index()
    fig, ax = plt.subplots()
    sns.scatterplot(x="EventDate", y="TransactionAmount", data=daily_spending, ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Merchants")
    merchant = data.groupby("Merchant")["TransactionAmount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    merchant.head(10).plot(kind="bar", color="green", ax=ax)
    ax.set_title("Top 10 Merchants")
    plt.tight_layout()
    st.pyplot(fig)

# --------------------------------------------------------------------------------------
# Customer & Payment Insights
# --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Top 10 Spending Customers")
    top_customers = (
        data.groupby("CustomerID")["TransactionAmount"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_customers, x="CustomerID", y="TransactionAmount", ax=ax)
    ax.set_title("Top 10 Spending Customers", fontsize=16, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Payment Type Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x="PaymentType", order=data["PaymentType"].value_counts().index, ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    with col2:
        st.subheader("Payment Mode Usage")
        fig, ax = plt.subplots()
        data["PaymentMode"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("Average Spending: Weekday vs Weekend")
    week = data.groupby("IsWeekend")["TransactionAmount"].mean()
    week.index = ["Weekday", "Weekend"]
    fig, ax = plt.subplots()
    week.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Spending")
    plt.setp(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig)

    st.subheader("Income vs Monthly Spending")
    fig, ax = plt.subplots()
    ax.scatter(data["Income"], data["Monthly_Spending"], alpha=0.5)
    ax.set_xlabel("Income")
    ax.set_ylabel("Monthly Spending")
    st.pyplot(fig)

    st.subheader("Anomaly Distribution")
    fig, ax = plt.subplots()
    data["Is_Anomaly"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# --------------------------------------------------------------------------------------
# Health Score
# --------------------------------------------------------------------------------------
with tabs[3]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Financial Health Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(data["HealthScore"], bins=20)
        st.pyplot(fig)

    with col2:
        st.subheader("Financial Health Categories")
        fig, ax = plt.subplots()
        data["HealthCategory"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.subheader("Savings Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data["Savings"], ax=ax)
    ax.set_xlabel("Savings")
    ax.set_ylabel("Number of Records")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# --------------------------------------------------------------------------------------
# Helper: shared feature matrix
# --------------------------------------------------------------------------------------
FEATURE_COLS = ["Income", "Budget", "TransactionAmount", "CustAccountBalance", "Monthly_Spending"]


def run_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    results["Logistic Regression"] = (model_lr, model_lr.predict(X_test))

    model_rg = RidgeClassifier(alpha=1.0)
    model_rg.fit(X_train, y_train)
    results["Ridge Classifier"] = (model_rg, model_rg.predict(X_test))

    model_dt = DecisionTreeRegressor()
    model_dt.fit(X_train, y_train)
    results["Decision Tree (regressor, rounded)"] = (model_dt, np.rint(model_dt.predict(X_test)).astype(int))

    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    results["Random Forest"] = (model_rf, model_rf.predict(X_test))

    return X_train, X_test, y_train, y_test, results


# --------------------------------------------------------------------------------------
# Anomaly Detection Model
# --------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Anomaly Detection — Model Comparison")
    st.caption(
        "Note: Decision Tree here is a regressor (as in the original script), so its output "
        "is rounded to the nearest class for scoring."
    )

    X = data[FEATURE_COLS]
    y = data["Is_Anomaly"]

    X_train, X_test, y_train, y_test, results = run_classification_models(X, y)

    acc_rows = []
    for name, (model, preds) in results.items():
        acc_rows.append({"Model": name, "Accuracy": accuracy_score(y_test, preds)})
    acc_df = pd.DataFrame(acc_rows).sort_values("Accuracy", ascending=False)
    st.dataframe(acc_df, use_container_width=True, hide_index=True)

    chosen = st.selectbox("View detailed classification report for:", list(results.keys()), key="anomaly_model_select")
    model, preds = results[chosen]
    st.code(classification_report(y_test, preds), language="text")

    rf_model = results["Random Forest"][0]
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance on Anomaly Detection")
    st.pyplot(fig)

# --------------------------------------------------------------------------------------
# Health Score Model
# --------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Health Score — Model Comparison")
    st.caption(
        "Note: HealthScore is a continuous-ish integer score. Classifiers (Logistic/Ridge/"
        "Random Forest) treat each score value as its own class, so accuracy here reflects "
        "exact-score matches, not closeness."
    )

    X = data[FEATURE_COLS]
    y = data["HealthScore"]

    X_train, X_test, y_train, y_test, results = run_classification_models(X, y)

    acc_rows = []
    for name, (model, preds) in results.items():
        acc_rows.append({"Model": name, "Accuracy (exact match)": accuracy_score(y_test, preds)})
    acc_df = pd.DataFrame(acc_rows).sort_values("Accuracy (exact match)", ascending=False)
    st.dataframe(acc_df, use_container_width=True, hide_index=True)

    chosen = st.selectbox("View detailed classification report for:", list(results.keys()), key="health_model_select")
    model, preds = results[chosen]
    st.code(classification_report(y_test, preds, zero_division=0), language="text")

    rf_model = results["Random Forest"][0]
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance on Health Score")
    st.pyplot(fig)

# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("K-Means Clustering")
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=3)

    X = data[FEATURE_COLS]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster = kmeans.fit_predict(X)

    plot_df = data.copy()
    plot_df["Cluster"] = cluster.astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x="Income", y="TransactionAmount", hue="Cluster", ax=ax, palette="tab10")
    ax.set_title("Customer Segments (Income vs Transaction Amount)")
    st.pyplot(fig)

    st.dataframe(
        plot_df.groupby("Cluster")[FEATURE_COLS].mean().round(2),
        use_container_width=True,
    )
