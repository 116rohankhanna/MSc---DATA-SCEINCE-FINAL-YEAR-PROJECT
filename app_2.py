# ============================================================
# AI-POWERED AUTOMATED STOCK REPORT GENERATION SYSTEM
# MSc (Data Science) Major Project
# ============================================================

# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Powered Automated Stock Report Generation System",
    layout="wide"
)

# ============================================================
# PROJECT TITLE
# ============================================================

st.title("AI-Powered Automated Stock Report Generation System")

st.markdown("""
This intelligent financial analytics platform uses:
- Machine Learning
- Technical Indicators
- Predictive Analytics
- Risk Analysis
- Automated Financial Reporting
- Interactive Data Visualization
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

# ============================================================
# ALLOWED DATASET FILES
# ============================================================

allowed_files = [
    "nifty_data.csv",
    "fstock_data.csv",
    "Upload_Data.csv",
    "Data.csv",
    "sample.csv"
]

# ============================================================
# FILE NAME VALIDATION
# ============================================================

if uploaded_file is not None:

    if uploaded_file.name not in allowed_files:

        st.error("Invalid File Uploaded")

        st.warning("Please upload only the approved project datasets.")

        st.info("Allowed Files:")

        for file in allowed_files:
            st.write(file)

        st.stop()

# ============================================================
# LOAD DATASET FUNCTION
# ============================================================

@st.cache_data

def load_data(file):

    if file.name.endswith("csv"):
        df = pd.read_csv(file)

    else:
        df = pd.read_excel(file)

    return df

# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

# ============================================================
# DATA PREPROCESSING FUNCTION
# ============================================================


def preprocess_data(df):

    df = df.copy()

    df.columns = [col.strip() for col in df.columns]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if len(missing_columns) > 0:

        st.error(f"Missing Columns: {missing_columns}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df = df.drop_duplicates()

    df = df.sort_values(["Ticker", "Date"])

    return df

# ============================================================
# TECHNICAL INDICATOR FUNCTIONS
# ============================================================


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def create_features(df):

    df = df.copy()

    df["Return"] = df.groupby("Ticker")["Close"].pct_change()

    df["MA20"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(20).mean()
    )

    df["MA50"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(50).mean()
    )

    df["EMA12"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.ewm(span=12).mean()
    )

    df["EMA26"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.ewm(span=26).mean()
    )

    df["MACD"] = df["EMA12"] - df["EMA26"]

    df["RSI"] = df.groupby("Ticker")["Close"].transform(
        lambda x: calculate_rsi(x)
    )

    df["Volatility"] = df.groupby("Ticker")["Return"].transform(
        lambda x: x.rolling(20).std()
    )

    df["Momentum"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.diff(10)
    )

    rolling_mean = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(20).mean()
    )

    rolling_std = df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(20).std()
    )

    df["Upper_Band"] = rolling_mean + (rolling_std * 2)
    df["Lower_Band"] = rolling_mean - (rolling_std * 2)

    df["Target"] = np.where(
        df.groupby("Ticker")["Close"].shift(-1) > df["Close"],
        1,
        0
    )

    df = df.dropna()

    return df

# ============================================================
# MACHINE LEARNING MODELS
# ============================================================


def train_models(X_train, y_train):

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Random Forest":
            RandomForestClassifier(n_estimators=100),

        "Decision Tree":
            DecisionTreeClassifier(),

        "Support Vector Machine":
            SVC(probability=True),

        "Gradient Boosting":
            GradientBoostingClassifier(),

        "XGBoost":
            XGBClassifier(eval_metric='logloss')
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models

# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data(uploaded_file)

    st.success("Dataset Loaded Successfully")

    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # ========================================================
    # PREPROCESSING
    # ========================================================

    df = preprocess_data(df)

    st.subheader("Preprocessed Dataset")
    st.dataframe(df.head())

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = create_features(df)

    st.subheader("Feature Engineered Dataset")
    st.dataframe(df.head())

    # ========================================================
    # STOCK FILTER
    # ========================================================

    stock_list = sorted(df["Ticker"].unique())

    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        stock_list
    )

    filtered_df = df[df["Ticker"] == selected_stock]

    # ========================================================
    # CANDLESTICK CHART
    # ========================================================

    st.subheader("Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(

        x=filtered_df["Date"],

        open=filtered_df["Open"],

        high=filtered_df["High"],

        low=filtered_df["Low"],

        close=filtered_df["Close"]
    )])

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # RSI GRAPH
    # ========================================================

    st.subheader("RSI Indicator")

    fig_rsi = px.line(
        filtered_df,
        x="Date",
        y="RSI"
    )

    st.plotly_chart(fig_rsi, use_container_width=True)

    # ========================================================
    # MACD GRAPH
    # ========================================================

    st.subheader("MACD Indicator")

    fig_macd = px.line(
        filtered_df,
        x="Date",
        y="MACD"
    )

    st.plotly_chart(fig_macd, use_container_width=True)

    # ========================================================
    # FEATURES FOR ML
    # ========================================================

    features = [
        "RSI",
        "MACD",
        "MA20",
        "MA50",
        "Volatility",
        "Momentum"
    ]

    X = df[features]
    y = df["Target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        shuffle=False
    )

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    trained_models = train_models(X_train, y_train)

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    st.subheader("Machine Learning Model Performance")

    results = []

    for name, model in trained_models.items():

        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4)
        })

    result_df = pd.DataFrame(results)

    st.dataframe(result_df)

    # ========================================================
    # ENSEMBLE PREDICTION
    # ========================================================

    prediction_votes = []

    latest_data = X_scaled[-1].reshape(1, -1)

    for name, model in trained_models.items():

        pred = model.predict(latest_data)[0]

        prediction_votes.append(pred)

    vote_total = sum(prediction_votes)

    if vote_total >= 4:
        final_signal = "BUY"

    elif vote_total <= 2:
        final_signal = "SELL"

    else:
        final_signal = "HOLD"

    confidence = (max(vote_total, 6 - vote_total) / 6) * 100

    # ========================================================
    # RISK ANALYSIS
    # ========================================================

    latest_volatility = filtered_df["Volatility"].iloc[-1]

    if latest_volatility < 0.01:
        risk_level = "Low"

    elif latest_volatility < 0.02:
        risk_level = "Medium"

    else:
        risk_level = "High"

    # ========================================================
    # PREDICTION OUTPUT
    # ========================================================

    st.subheader("AI Prediction Output")

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction Signal", final_signal)
    col2.metric("Confidence Score", f"{confidence:.2f}%")
    col3.metric("Risk Level", risk_level)

    # ========================================================
    # AI REPORT GENERATION
    # ========================================================

    latest_close = filtered_df["Close"].iloc[-1]

    target_price = latest_close * 1.05
    stop_loss = latest_close * 0.95

    report = f"""

    STOCK ANALYSIS REPORT

    Stock Name: {selected_stock}

    Current Closing Price:
    {latest_close:.2f}

    AI Prediction Signal:
    {final_signal}

    Prediction Confidence:
    {confidence:.2f}%

    Risk Category:
    {risk_level}

    RSI Interpretation:
    RSI measures momentum and trend strength.

    MACD Interpretation:
    MACD identifies bullish or bearish trend movement.

    Suggested Target Price:
    {target_price:.2f}

    Suggested Stop Loss:
    {stop_loss:.2f}

    Educational Insight:
    The generated prediction is based on technical indicators,
    historical stock behavior, and ensemble machine learning.

    """

    st.subheader("AI-Generated Stock Report")

    st.text(report)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader("Feature Importance Analysis")

    rf_model = trained_models["Random Forest"]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    })

    fig_importance = px.bar(
        importance_df,
        x="Feature",
        y="Importance"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    # ========================================================
    # BACKTESTING
    # ========================================================

    st.subheader("Backtesting Analysis")

    filtered_df["Signal"] = np.where(
        filtered_df["Return"] > 0,
        1,
        -1
    )

    filtered_df["Strategy_Return"] = (
        filtered_df["Return"] *
        filtered_df["Signal"].shift(1)
    )

    filtered_df["Cumulative_Market_Return"] = (
        1 + filtered_df["Return"]
    ).cumprod()

    filtered_df["Cumulative_Strategy_Return"] = (
        1 + filtered_df["Strategy_Return"]
    ).cumprod()

    fig_backtest = go.Figure()

    fig_backtest.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Cumulative_Market_Return"],
        name="Market Return"
    ))

    fig_backtest.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Cumulative_Strategy_Return"],
        name="AI Strategy Return"
    ))

    st.plotly_chart(fig_backtest, use_container_width=True)

    # ========================================================
    # DOWNLOAD DATASET
    # ========================================================

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Processed Dataset",
        data=csv,
        file_name="processed_stock_analysis.csv",
        mime="text/csv"
    )

# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.info("Please upload a stock market dataset to continue.")

    st.markdown("""
    Supported Files:

    - nifty50_5y_data.csv
    - filtered_stock_data.csv
    - Upload_Stock_Data.csv
    - Tatamotors_Data.csv
    - sample_stock_format.csv
    """)

# ============================================================
# END OF PROJECT
# ============================================================
