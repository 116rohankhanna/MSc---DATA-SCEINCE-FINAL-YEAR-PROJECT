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
from plotly.subplots import make_subplots
import warnings
import time
import io
import base64
from datetime import datetime, timedelta
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stock Report System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown("""
<style>

/* ─── Base Theme ─── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1520 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1723 0%, #1a2332 100%);
    border-right: 1px solid rgba(59,130,246,0.2);
}

/* ─── Title ─── */
.main-title {
    font-family: 'Georgia', serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #34d399, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1rem 0 0.25rem 0;
    letter-spacing: -0.5px;
}

.sub-title {
    font-family: 'Georgia', serif;
    font-size: 1rem;
    color: #64748b;
    text-align: center;
    margin-bottom: 2rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ─── KPI Cards ─── */
.kpi-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.9) 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(59,130,246,0.2);
}

.kpi-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'Georgia', serif;
    color: #f1f5f9;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* ─── Signal Badge ─── */
.signal-buy {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 1.5px solid #34d399;
    color: #6ee7b7;
    border-radius: 12px;
    padding: 0.8rem 1.6rem;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: 2px;
    box-shadow: 0 0 30px rgba(52,211,153,0.25);
}

.signal-sell {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1.5px solid #f87171;
    color: #fca5a5;
    border-radius: 12px;
    padding: 0.8rem 1.6rem;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: 2px;
    box-shadow: 0 0 30px rgba(248,113,113,0.25);
}

.signal-hold {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 1.5px solid #fbbf24;
    color: #fde68a;
    border-radius: 12px;
    padding: 0.8rem 1.6rem;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: 2px;
    box-shadow: 0 0 30px rgba(251,191,36,0.25);
}

/* ─── Section Headers ─── */
.section-header {
    font-family: 'Georgia', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #60a5fa;
    padding-left: 0.8rem;
    margin: 1.8rem 0 1rem 0;
}

/* ─── Report Box ─── */
.report-box {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.9));
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 16px;
    padding: 2rem;
    font-family: 'Courier New', monospace;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.85;
    white-space: pre-wrap;
    box-shadow: inset 0 0 60px rgba(0,0,0,0.3);
}

/* ─── Risk Badge ─── */
.risk-low  { color: #34d399; font-weight: 700; }
.risk-med  { color: #fbbf24; font-weight: 700; }
.risk-high { color: #f87171; font-weight: 700; }

/* ─── Info Banner ─── */
.info-banner {
    background: linear-gradient(90deg, rgba(59,130,246,0.12), rgba(168,85,247,0.08));
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* ─── Plotly Dark Fix ─── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ─── Streamlit overrides ─── */
[data-testid="metric-container"] {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }

.stDataFrame { border-radius: 12px; overflow: hidden; }
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }

h1, h2, h3 { color: #e2e8f0 !important; }
p, li { color: #94a3b8; }

.stSelectbox label, .stMultiselect label,
.stFileUploader label { color: #94a3b8 !important; }

</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_FILES = [
    "nifty50_5y_data.csv",
    "filtered_stock_data.csv",
    "Upload_Stock_Data.csv",
    "Tatamotors_Data.csv",
    "sample_stock_format.csv"
]

REQUIRED_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]

FEATURES = ["RSI", "MACD", "MA20", "MA50", "Volatility", "Momentum",
            "Upper_Band", "Lower_Band", "Volume_MA20", "Return_Lag1"]

CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,26,0.8)",
    font=dict(color="#94a3b8", family="Georgia, serif"),
    xaxis=dict(gridcolor="rgba(59,130,246,0.1)", linecolor="rgba(59,130,246,0.2)"),
    yaxis=dict(gridcolor="rgba(59,130,246,0.1)", linecolor="rgba(59,130,246,0.2)"),
    margin=dict(l=50, r=30, t=50, b=40)
)

# ============================================================
# PAGE HEADER
# ============================================================

st.markdown('<div class="main-title">📈 AI-Powered Stock Report Generation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">MSc Data Science · Machine Learning · Technical Analysis · Predictive Analytics</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset",
        type=["csv", "xlsx"],
        help="Upload one of the approved project datasets"
    )

    st.markdown("---")
    st.markdown("**⚙️ Analysis Settings**")

    ma_short = st.slider("Short MA Period", 5, 30, 20, 1)
    ma_long  = st.slider("Long MA Period", 30, 100, 50, 1)
    rsi_period = st.slider("RSI Period", 7, 21, 14, 1)
    vol_window = st.slider("Volatility Window", 10, 30, 20, 1)

    st.markdown("---")
    st.markdown("**🤖 Model Settings**")

    n_estimators = st.slider("RF / GB Trees", 50, 300, 100, 50)
    test_size    = st.slider("Test Split %", 10, 40, 20, 5) / 100

    st.markdown("---")
    st.markdown("**📌 Approved Datasets**")
    for f in ALLOWED_FILES:
        st.markdown(f"• `{f}`")

# ============================================================
# FILE VALIDATION
# ============================================================

if uploaded_file is not None and uploaded_file.name not in ALLOWED_FILES:
    st.error("❌ Invalid file. Please upload only approved project datasets listed in the sidebar.")
    st.stop()

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(file_bytes, file_name):
    if file_name.endswith(".csv"):
        # Handle multi-header nifty50 format
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            if df.columns[0] == "Ticker":
                # Wide-format: melt to long
                tickers_row = df.iloc[0]
                price_row   = df.iloc[1]
                data_rows   = df.iloc[3:].copy()
                data_rows.columns = df.columns
                data_rows = data_rows.rename(columns={data_rows.columns[0]: "Date"})

                records = []
                col_idx = 1
                seen_tickers = {}
                for c in df.columns[1:]:
                    ticker = tickers_row[c]
                    if pd.isna(ticker) or str(ticker).strip() == "":
                        col_idx += 1
                        continue
                    price_type = price_row[c]
                    seen_tickers.setdefault(ticker, {})
                    seen_tickers[ticker][price_type] = c

                long_records = []
                for ticker, cols in seen_tickers.items():
                    needed = {"Open","High","Low","Close","Volume"}
                    if not needed.issubset(set(cols.keys())):
                        continue
                    for _, row in data_rows.iterrows():
                        long_records.append({
                            "Date":   row["Date"],
                            "Ticker": ticker,
                            "Open":   row[cols["Open"]],
                            "High":   row[cols["High"]],
                            "Low":    row[cols["Low"]],
                            "Close":  row[cols["Close"]],
                            "Volume": row[cols["Volume"]],
                        })
                return pd.DataFrame(long_records)

        except Exception:
            pass

        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    return df

# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates()
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)
    return df

# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    delta   = series.diff()
    gain    = delta.clip(lower=0)
    loss    = -delta.clip(upper=0)
    avg_g   = gain.ewm(com=period-1, min_periods=period).mean()
    avg_l   = loss.ewm(com=period-1, min_periods=period).mean()
    rs      = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_stochastic(high, low, close, k=14, d=3):
    low_min  = low.rolling(k).min()
    high_max = high.rolling(k).max()
    stoch_k  = 100 * (close - low_min) / (high_max - low_min + 1e-9)
    stoch_d  = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calculate_atr(high, low, close, period=14):
    hl  = high - low
    hpc = (high - close.shift()).abs()
    lpc = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_obv(close, volume):
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def calculate_williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll + 1e-9)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df, ma_s=20, ma_l=50, rsi_p=14, vol_w=20):
    df = df.copy()

    g = df.groupby("Ticker")

    df["Return"]        = g["Close"].pct_change()
    df["Return_Lag1"]   = g["Return"].shift(1)
    df["Return_Lag2"]   = g["Return"].shift(2)
    df["Log_Return"]    = np.log(df["Close"] / g["Close"].shift(1))

    df["MA20"]  = g["Close"].transform(lambda x: x.rolling(ma_s).mean())
    df["MA50"]  = g["Close"].transform(lambda x: x.rolling(ma_l).mean())
    df["MA200"] = g["Close"].transform(lambda x: x.rolling(200).mean())

    df["EMA12"] = g["Close"].transform(lambda x: x.ewm(span=12).mean())
    df["EMA26"] = g["Close"].transform(lambda x: x.ewm(span=26).mean())
    df["MACD"]  = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"]   = df.groupby("Ticker")["MACD"].transform(lambda x: x.ewm(span=9).mean())
    df["MACD_Hist"]     = df["MACD"] - df["MACD_Signal"]

    df["RSI"] = g["Close"].transform(lambda x: calculate_rsi(x, rsi_p))

    roll_mean = g["Close"].transform(lambda x: x.rolling(ma_s).mean())
    roll_std  = g["Close"].transform(lambda x: x.rolling(ma_s).std())
    df["Upper_Band"]  = roll_mean + 2 * roll_std
    df["Lower_Band"]  = roll_mean - 2 * roll_std
    df["BB_Width"]    = (df["Upper_Band"] - df["Lower_Band"]) / (roll_mean + 1e-9)
    df["BB_Position"] = (df["Close"] - df["Lower_Band"]) / (df["Upper_Band"] - df["Lower_Band"] + 1e-9)

    df["Volatility"]    = g["Return"].transform(lambda x: x.rolling(vol_w).std())
    df["Momentum"]      = g["Close"].transform(lambda x: x.diff(10))
    df["ROC"]           = g["Close"].transform(lambda x: x.pct_change(10) * 100)

    df["ATR"]           = df.groupby("Ticker").apply(
        lambda x: calculate_atr(x["High"], x["Low"], x["Close"])
    ).reset_index(level=0, drop=True)

    df["OBV"]           = df.groupby("Ticker").apply(
        lambda x: calculate_obv(x["Close"], x["Volume"])
    ).reset_index(level=0, drop=True)

    stoch_data = df.groupby("Ticker").apply(
        lambda x: calculate_stochastic(x["High"], x["Low"], x["Close"])
    )
    df["Stoch_K"] = df.groupby("Ticker").apply(
        lambda x: calculate_stochastic(x["High"], x["Low"], x["Close"])[0]
    ).reset_index(level=0, drop=True)
    df["Stoch_D"] = df.groupby("Ticker").apply(
        lambda x: calculate_stochastic(x["High"], x["Low"], x["Close"])[1]
    ).reset_index(level=0, drop=True)

    df["Williams_R"] = df.groupby("Ticker").apply(
        lambda x: calculate_williams_r(x["High"], x["Low"], x["Close"])
    ).reset_index(level=0, drop=True)

    df["Volume_MA20"]   = g["Volume"].transform(lambda x: x.rolling(20).mean())
    df["Volume_Ratio"]  = df["Volume"] / (df["Volume_MA20"] + 1)
    df["Volume_Spike"]  = (df["Volume"] > 2 * df["Volume_MA20"]).astype(int)

    df["Price_Range"]   = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["Gap_Pct"]       = (df["Open"] - g["Close"].shift(1)) / (g["Close"].shift(1) + 1e-9)
    df["MA20_Distance"] = (df["Close"] - df["MA20"]) / (df["MA20"] + 1e-9)
    df["MA50_Distance"] = (df["Close"] - df["MA50"]) / (df["MA50"] + 1e-9)
    df["Above_MA20"]    = (df["Close"] > df["MA20"]).astype(int)
    df["Above_MA50"]    = (df["Close"] > df["MA50"]).astype(int)
    df["Golden_Cross"]  = ((df["MA20"] > df["MA50"]) & (g["MA20"].shift(1) <= g["MA50"].shift(1))).astype(int)

    df["Target"] = np.where(g["Close"].shift(-1) > df["Close"], 1, 0)

    df = df.dropna()
    return df

# ============================================================
# ML MODEL TRAINING
# ============================================================

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "MA20", "MA50", "Volatility", "Momentum", "ROC",
    "Upper_Band", "Lower_Band", "BB_Width", "BB_Position",
    "Volume_Ratio", "Volume_Spike",
    "Return_Lag1", "Return_Lag2",
    "Stoch_K", "Stoch_D",
    "Williams_R", "ATR",
    "MA20_Distance", "MA50_Distance",
    "Above_MA20", "Above_MA50"
]

def train_models(X_train, y_train, n_est=100):
    models = {
        "Logistic Regression":   LogisticRegression(max_iter=2000, C=0.5),
        "Random Forest":         RandomForestClassifier(n_estimators=n_est, max_depth=8, random_state=42),
        "Decision Tree":         DecisionTreeClassifier(max_depth=6, random_state=42),
        "Support Vector Machine":SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.05, random_state=42),
        "XGBoost":               XGBClassifier(n_estimators=n_est, max_depth=4, learning_rate=0.05,
                                               eval_metric="logloss", random_state=42, verbosity=0),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

# ============================================================
# RISK METRICS
# ============================================================

def compute_risk_metrics(series):
    ret = series.pct_change().dropna()
    if len(ret) < 20:
        return {}
    sharpe  = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
    neg     = ret[ret < 0]
    sortino = (ret.mean() / neg.std()) * np.sqrt(252) if len(neg) > 0 and neg.std() != 0 else 0
    cumret  = (1 + ret).cumprod()
    peak    = cumret.cummax()
    dd      = (cumret - peak) / peak
    max_dd  = dd.min()
    calmar  = (ret.mean() * 252) / abs(max_dd) if max_dd != 0 else 0
    var_95  = np.percentile(ret, 5)
    cvar_95 = ret[ret <= var_95].mean()
    beta    = 1.0  # Placeholder (requires market index data)
    return {
        "Annual Return %": round(ret.mean() * 252 * 100, 2),
        "Annual Volatility %": round(ret.std() * np.sqrt(252) * 100, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Calmar Ratio": round(calmar, 3),
        "Max Drawdown %": round(max_dd * 100, 2),
        "VaR 95% (Daily)": round(var_95 * 100, 2),
        "CVaR 95% (Daily)": round(cvar_95 * 100, 2),
        "Skewness": round(float(stats.skew(ret)), 3),
        "Kurtosis": round(float(stats.kurtosis(ret)), 3),
    }

# ============================================================
# GENERATE AI REPORT TEXT
# ============================================================

def generate_report(stock, df_stock, signal, confidence, risk_level, metrics,
                    model_results, selected_features, latest_close,
                    target_price, stop_loss, rsi_val, macd_val):

    now = datetime.now().strftime("%d %B %Y, %H:%M")
    best_model = model_results.sort_values("F1 Score", ascending=False).iloc[0]

    rsi_interp = ("Oversold — potential reversal signal" if rsi_val < 30
                  else "Overbought — caution on new longs" if rsi_val > 70
                  else "Neutral momentum zone")

    macd_trend = "Bullish crossover" if macd_val > 0 else "Bearish crossover"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║        AI-POWERED AUTOMATED STOCK ANALYSIS REPORT           ║
║              MSc Data Science · Major Project               ║
╚══════════════════════════════════════════════════════════════╝

  Report Generated  : {now}
  Stock Symbol      : {stock}
  Analysis Period   : {df_stock['Date'].min().strftime('%d %b %Y')} → {df_stock['Date'].max().strftime('%d %b %Y')}
  Total Trading Days: {len(df_stock):,}

══════════════════════════════════════════════════════════════
  1. PRICE SUMMARY
══════════════════════════════════════════════════════════════

  Current Close Price : ₹ {latest_close:,.2f}
  52-Week High        : ₹ {df_stock['High'].rolling(252).max().iloc[-1]:,.2f}
  52-Week Low         : ₹ {df_stock['Low'].rolling(252).min().iloc[-1]:,.2f}
  30-Day Avg Volume   : {int(df_stock['Volume'].tail(30).mean()):,}

══════════════════════════════════════════════════════════════
  2. AI PREDICTION OUTPUT
══════════════════════════════════════════════════════════════

  ► Ensemble Signal   : {signal}
  ► Confidence Score  : {confidence:.1f}%
  ► Risk Category     : {risk_level}
  ► Suggested Target  : ₹ {target_price:,.2f}  (+5%)
  ► Suggested Stop    : ₹ {stop_loss:,.2f}  (-5%)
  ► Risk:Reward Ratio : 1 : 1

══════════════════════════════════════════════════════════════
  3. TECHNICAL INDICATOR SUMMARY
══════════════════════════════════════════════════════════════

  RSI ({rsi_val:.1f})    : {rsi_interp}
  MACD ({macd_val:.2f}) : {macd_trend}
  MA20              : ₹ {df_stock['MA20'].iloc[-1]:,.2f}
  MA50              : ₹ {df_stock['MA50'].iloc[-1]:,.2f}
  Upper Bollinger   : ₹ {df_stock['Upper_Band'].iloc[-1]:,.2f}
  Lower Bollinger   : ₹ {df_stock['Lower_Band'].iloc[-1]:,.2f}
  ATR (Volatility)  : ₹ {df_stock['ATR'].iloc[-1]:,.2f}
  BB Position       : {df_stock['BB_Position'].iloc[-1]*100:.1f}% (0=Lower Band, 100=Upper Band)

══════════════════════════════════════════════════════════════
  4. RISK METRICS
══════════════════════════════════════════════════════════════
"""
    for k, v in metrics.items():
        report += f"\n  {k:<26}: {v}"

    report += f"""

══════════════════════════════════════════════════════════════
  5. BEST PERFORMING ML MODEL
══════════════════════════════════════════════════════════════

  Model             : {best_model['Model']}
  Accuracy          : {best_model['Accuracy']*100:.2f}%
  Precision         : {best_model['Precision']*100:.2f}%
  Recall            : {best_model['Recall']*100:.2f}%
  F1 Score          : {best_model['F1 Score']*100:.2f}%
  AUC-ROC           : {best_model['AUC-ROC']*100:.2f}%

══════════════════════════════════════════════════════════════
  6. EDUCATIONAL INSIGHT
══════════════════════════════════════════════════════════════

  This report is generated by an ensemble of {len(model_results)} ML models
  trained on {len(selected_features)} technical features derived from OHLCV data.
  The system applies RSI, MACD, Bollinger Bands, Stochastic,
  Williams %R, ATR, OBV, and momentum indicators.

  ⚠ DISCLAIMER: This is an academic project for educational
    purposes only. It does not constitute financial advice.
    Always conduct independent research before investing.

══════════════════════════════════════════════════════════════
  END OF REPORT
══════════════════════════════════════════════════════════════
"""
    return report

# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # ── Load & Preprocess ──────────────────────────────────
    with st.spinner("⏳ Loading and preprocessing data..."):
        file_bytes = uploaded_file.read()
        df_raw = load_data(file_bytes, uploaded_file.name)

    st.success(f"✅ Dataset **{uploaded_file.name}** loaded — {len(df_raw):,} rows")

    with st.expander("🔍 Raw Dataset Preview", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)

    df = preprocess_data(df_raw)

    # ── Feature Engineering ────────────────────────────────
    with st.spinner("⚙️ Engineering features & computing indicators..."):
        df = create_features(df, ma_s=ma_short, ma_l=ma_long,
                             rsi_p=rsi_period, vol_w=vol_window)

    # ── Stock Selection ────────────────────────────────────
    stock_list = sorted(df["Ticker"].unique())
    selected_stock = st.sidebar.selectbox("📊 Select Stock", stock_list)
    fd = df[df["Ticker"] == selected_stock].copy().reset_index(drop=True)

    # ── Tabs Layout ────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "📈 Charts & Indicators",
        "🤖 ML Models",
        "⚠️ Risk Analysis",
        "📉 Backtesting",
        "📋 Full Report"
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Dataset & Stock Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Stocks</div>
                <div class="kpi-value">{df['Ticker'].nunique()}</div>
                <div class="kpi-sub">unique tickers</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Date Range</div>
                <div class="kpi-value">{df['Date'].nunique()}</div>
                <div class="kpi-sub">trading days</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-value">{len(df):,}</div>
                <div class="kpi-sub">after preprocessing</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Features</div>
                <div class="kpi-value">{len(FEATURE_COLS)}</div>
                <div class="kpi-sub">engineered</div></div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">Selected: {selected_stock}</div>', unsafe_allow_html=True)

        latest = fd.iloc[-1]
        prev   = fd.iloc[-2]
        chg    = latest["Close"] - prev["Close"]
        chg_p  = (chg / prev["Close"]) * 100

        ka, kb, kc, kd, ke = st.columns(5)
        ka.metric("Close", f"₹{latest['Close']:,.2f}", f"{chg_p:+.2f}%")
        kb.metric("Open",  f"₹{latest['Open']:,.2f}")
        kc.metric("High",  f"₹{latest['High']:,.2f}")
        kd.metric("Low",   f"₹{latest['Low']:,.2f}")
        ke.metric("Volume", f"{int(latest['Volume']):,}")

        st.markdown("---")

        c_l, c_r = st.columns([2,1])
        with c_l:
            st.markdown("**📅 Recent Data (Last 10 Trading Days)**")
            cols_show = ["Date","Open","High","Low","Close","Volume","RSI","MACD","Return"]
            st.dataframe(
                fd[cols_show].tail(10).round(3).reset_index(drop=True),
                use_container_width=True
            )
        with c_r:
            st.markdown("**📊 Descriptive Statistics**")
            st.dataframe(
                fd[["Open","High","Low","Close","Volume"]].describe().round(2),
                use_container_width=True
            )

    # ════════════════════════════════════════════════════════
    # TAB 2 — CHARTS & INDICATORS
    # ════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="section-header">📈 Price & Technical Charts</div>', unsafe_allow_html=True)

        # Candlestick + Volume
        fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.75, 0.25], vertical_spacing=0.03)

        fig_candle.add_trace(go.Candlestick(
            x=fd["Date"], open=fd["Open"], high=fd["High"],
            low=fd["Low"], close=fd["Close"],
            increasing_line_color="#34d399", decreasing_line_color="#f87171",
            increasing_fillcolor="rgba(52,211,153,0.4)",
            decreasing_fillcolor="rgba(248,113,113,0.4)",
            name="OHLC"
        ), row=1, col=1)

        fig_candle.add_trace(go.Scatter(x=fd["Date"], y=fd["MA20"],
            line=dict(color="#60a5fa", width=1.5), name=f"MA{ma_short}"), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=fd["Date"], y=fd["MA50"],
            line=dict(color="#f59e0b", width=1.5), name=f"MA{ma_long}"), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=fd["Date"], y=fd["Upper_Band"],
            line=dict(color="rgba(167,139,250,0.5)", width=1, dash="dot"),
            name="Upper BB"), row=1, col=1)
        fig_candle.add_trace(go.Scatter(x=fd["Date"], y=fd["Lower_Band"],
            line=dict(color="rgba(167,139,250,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(167,139,250,0.04)",
            name="Lower BB"), row=1, col=1)

        colors = ["#34d399" if r >= 0 else "#f87171" for r in fd["Return"]]
        fig_candle.add_trace(go.Bar(x=fd["Date"], y=fd["Volume"],
            marker_color=colors, name="Volume", opacity=0.8), row=2, col=1)

        fig_candle.update_layout(**CHART_TEMPLATE, height=600,
            title=f"{selected_stock} — Price Action & Volume",
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h"))
        fig_candle.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)

        # RSI + Stochastic
        col_l, col_r = st.columns(2)
        with col_l:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=fd["Date"], y=fd["RSI"],
                line=dict(color="#a78bfa", width=2), name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f87171", opacity=0.7)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#34d399", opacity=0.7)
            fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", line_width=0)
            fig_rsi.update_layout(**CHART_TEMPLATE, height=280,
                title="RSI Indicator", yaxis=dict(**CHART_TEMPLATE["yaxis"], range=[0,100]))
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_r:
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=fd["Date"], y=fd["Stoch_K"],
                line=dict(color="#60a5fa", width=2), name="Stoch %K"))
            fig_stoch.add_trace(go.Scatter(x=fd["Date"], y=fd["Stoch_D"],
                line=dict(color="#f59e0b", width=1.5, dash="dot"), name="Stoch %D"))
            fig_stoch.add_hline(y=80, line_dash="dash", line_color="#f87171", opacity=0.7)
            fig_stoch.add_hline(y=20, line_dash="dash", line_color="#34d399", opacity=0.7)
            fig_stoch.update_layout(**CHART_TEMPLATE, height=280,
                title="Stochastic Oscillator", yaxis=dict(**CHART_TEMPLATE["yaxis"], range=[0,100]))
            st.plotly_chart(fig_stoch, use_container_width=True)

        # MACD
        fig_macd = make_subplots(rows=1, cols=1)
        fig_macd.add_trace(go.Scatter(x=fd["Date"], y=fd["MACD"],
            line=dict(color="#60a5fa", width=2), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=fd["Date"], y=fd["MACD_Signal"],
            line=dict(color="#f59e0b", width=1.5), name="Signal"))
        hist_colors = ["#34d399" if v >= 0 else "#f87171" for v in fd["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(x=fd["Date"], y=fd["MACD_Hist"],
            marker_color=hist_colors, name="Histogram", opacity=0.7))
        fig_macd.update_layout(**CHART_TEMPLATE, height=300, title="MACD Indicator")
        st.plotly_chart(fig_macd, use_container_width=True)

        # Williams %R + OBV
        col_wl, col_obv = st.columns(2)
        with col_wl:
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Scatter(x=fd["Date"], y=fd["Williams_R"],
                line=dict(color="#fb923c", width=2), name="Williams %R"))
            fig_wr.add_hline(y=-20, line_dash="dash", line_color="#f87171", opacity=0.7)
            fig_wr.add_hline(y=-80, line_dash="dash", line_color="#34d399", opacity=0.7)
            fig_wr.update_layout(**CHART_TEMPLATE, height=280, title="Williams %R",
                yaxis=dict(**CHART_TEMPLATE["yaxis"], range=[-100, 0]))
            st.plotly_chart(fig_wr, use_container_width=True)

        with col_obv:
            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(x=fd["Date"], y=fd["OBV"],
                fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
                line=dict(color="#60a5fa", width=2), name="OBV"))
            fig_obv.update_layout(**CHART_TEMPLATE, height=280, title="On-Balance Volume (OBV)")
            st.plotly_chart(fig_obv, use_container_width=True)

        # Correlation Heatmap
        st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
        corr_cols = ["RSI","MACD","Volatility","Momentum","BB_Width","Volume_Ratio",
                     "Stoch_K","Williams_R","ROC","ATR","Return"]
        corr_data = fd[corr_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr_data.values, x=corr_cols, y=corr_cols,
            colorscale="RdBu", zmid=0,
            text=corr_data.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="r", tickfont=dict(color="#94a3b8"))
        ))
        fig_corr.update_layout(**CHART_TEMPLATE, height=450, title="Indicator Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 3 — ML MODELS
    # ════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="section-header">🤖 Machine Learning Model Training & Evaluation</div>', unsafe_allow_html=True)

        available_features = [f for f in FEATURE_COLS if f in fd.columns]
        selected_features  = st.multiselect(
            "Select Feature Set", available_features,
            default=available_features[:12]
        )

        if len(selected_features) < 2:
            st.warning("Please select at least 2 features.")
            st.stop()

        X = df[selected_features]
        y = df["Target"]

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False
        )

        with st.spinner("🔄 Training 6 ML models..."):
            t0 = time.time()
            trained_models = train_models(X_train, y_train, n_estimators)
            train_time = time.time() - t0

        st.success(f"✅ All models trained in {train_time:.1f}s")

        results = []
        for name, model in trained_models.items():
            preds  = model.predict(X_test)
            proba  = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else preds
            results.append({
                "Model":     name,
                "Accuracy":  round(accuracy_score(y_test, preds), 4),
                "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
                "Recall":    round(recall_score(y_test, preds, zero_division=0), 4),
                "F1 Score":  round(f1_score(y_test, preds, zero_division=0), 4),
                "AUC-ROC":   round(roc_auc_score(y_test, proba), 4),
            })

        result_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)

        st.markdown("**📊 Model Performance Comparison**")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Radar chart
        fig_radar = go.Figure()
        metrics_radar = ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
        colors_r = ["#60a5fa","#34d399","#f59e0b","#f87171","#a78bfa","#fb923c"]
        for i, row in result_df.iterrows():
            vals = [row[m] for m in metrics_radar] + [row[metrics_radar[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=metrics_radar + [metrics_radar[0]],
                fill="toself", name=row["Model"],
                line=dict(color=colors_r[i % 6]),
                fillcolor=colors_r[i % 6].replace("#", "rgba(").rstrip(")") + ",0.08)"
            ))
        fig_radar.update_layout(**CHART_TEMPLATE, height=420,
            title="Model Performance Radar",
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(range=[0,1], color="#64748b",
                                       gridcolor="rgba(59,130,246,0.2)"),
                       angularaxis=dict(color="#94a3b8")))
        st.plotly_chart(fig_radar, use_container_width=True)

        # Feature Importance
        st.markdown('<div class="section-header">🔍 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        rf_model    = trained_models["Random Forest"]
        imp_df      = pd.DataFrame({
            "Feature":    selected_features,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h",
            marker=dict(
                color=imp_df["Importance"],
                colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#60a5fa"]],
                showscale=False
            )
        ))
        fig_imp.update_layout(**CHART_TEMPLATE, height=400, title="Random Forest Feature Importances")
        st.plotly_chart(fig_imp, use_container_width=True)

        # Confusion matrix for best model
        best_name  = result_df.iloc[0]["Model"]
        best_model = trained_models[best_name]
        best_preds = best_model.predict(X_test)
        cm         = confusion_matrix(y_test, best_preds)

        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted 0","Predicted 1"],
            y=["Actual 0","Actual 1"],
            colorscale=[[0,"#0a0e1a"],[1,"#3b82f6"]],
            text=cm, texttemplate="%{text}",
            colorbar=dict(title="Count", tickfont=dict(color="#94a3b8"))
        ))
        fig_cm.update_layout(**CHART_TEMPLATE, height=350,
            title=f"Confusion Matrix — {best_name}")
        st.plotly_chart(fig_cm, use_container_width=True)

        # ── Ensemble Prediction ────────────────────────────
        st.markdown('<div class="section-header">🎯 Ensemble AI Prediction</div>', unsafe_allow_html=True)

        if all(f in fd.columns for f in selected_features):
            latest_X  = scaler.transform(fd[selected_features].tail(1))
            votes      = []
            proba_sum  = 0
            for name, model in trained_models.items():
                p = model.predict(latest_X)[0]
                votes.append(p)
                if hasattr(model, "predict_proba"):
                    proba_sum += model.predict_proba(latest_X)[0][1]

            vote_total = sum(votes)
            avg_proba  = proba_sum / len(trained_models)
            confidence = avg_proba * 100

            if vote_total >= 4:
                final_signal = "BUY"
                signal_class = "signal-buy"
            elif vote_total <= 2:
                final_signal = "SELL"
                signal_class = "signal-sell"
            else:
                final_signal = "HOLD"
                signal_class = "signal-hold"

            latest_volatility = fd["Volatility"].iloc[-1]
            risk_level = "Low" if latest_volatility < 0.01 else "Medium" if latest_volatility < 0.025 else "High"

            latest_close = fd["Close"].iloc[-1]
            target_price = latest_close * 1.05
            stop_loss    = latest_close * 0.95
            rsi_val      = fd["RSI"].iloc[-1]
            macd_val     = fd["MACD"].iloc[-1]

            sa, sb, sc, sd = st.columns(4)
            with sa:
                st.markdown(f'<div class="{signal_class}">{final_signal}</div>', unsafe_allow_html=True)
            with sb:
                st.metric("Confidence", f"{confidence:.1f}%")
            with sc:
                st.metric("Risk Level", risk_level)
            with sd:
                st.metric("Models Agreeing (Buy)", f"{vote_total} / 6")

            st.markdown("")
            e1, e2, e3 = st.columns(3)
            e1.metric("Current Price", f"₹{latest_close:,.2f}")
            e2.metric("Target (+5%)", f"₹{target_price:,.2f}")
            e3.metric("Stop Loss (-5%)", f"₹{stop_loss:,.2f}")

    # ════════════════════════════════════════════════════════
    # TAB 4 — RISK ANALYSIS
    # ════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-header">⚠️ Risk Analysis & Portfolio Metrics</div>', unsafe_allow_html=True)

        metrics = compute_risk_metrics(fd["Close"])

        m_cols = st.columns(4)
        metric_items = list(metrics.items())
        for i, (k, v) in enumerate(metric_items):
            with m_cols[i % 4]:
                color = "#34d399" if (
                    ("Return" in k and v > 0) or
                    ("Sharpe" in k and v > 1) or
                    ("Sortino" in k and v > 1)
                ) else "#f87171" if (
                    ("Return" in k and v < 0) or
                    ("Drawdown" in k and v < -20) or
                    ("VaR" in k and v < -3)
                ) else "#f59e0b"
                st.markdown(f"""<div class="kpi-card">
                    <div class="kpi-label">{k}</div>
                    <div class="kpi-value" style="font-size:1.4rem;color:{color}">{v}</div>
                </div>""", unsafe_allow_html=True)

        # Rolling Volatility
        st.markdown('<div class="section-header">📉 Rolling Volatility & Drawdown</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fd["Roll_Vol_20"] = fd["Return"].rolling(20).std() * np.sqrt(252) * 100
            fd["Roll_Vol_60"] = fd["Return"].rolling(60).std() * np.sqrt(252) * 100
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=fd["Date"], y=fd["Roll_Vol_20"],
                line=dict(color="#60a5fa", width=2), name="20-Day Vol"))
            fig_vol.add_trace(go.Scatter(x=fd["Date"], y=fd["Roll_Vol_60"],
                line=dict(color="#f59e0b", width=1.5), name="60-Day Vol"))
            fig_vol.update_layout(**CHART_TEMPLATE, height=330, title="Rolling Annualised Volatility (%)")
            st.plotly_chart(fig_vol, use_container_width=True)

        with col_b:
            cum_ret  = (1 + fd["Return"]).cumprod()
            peak     = cum_ret.cummax()
            drawdown = (cum_ret - peak) / peak * 100
            fig_dd   = go.Figure()
            fig_dd.add_trace(go.Scatter(x=fd["Date"], y=drawdown,
                fill="tozeroy", fillcolor="rgba(248,113,113,0.15)",
                line=dict(color="#f87171", width=2), name="Drawdown"))
            fig_dd.update_layout(**CHART_TEMPLATE, height=330, title="Drawdown from Peak (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

        # Return Distribution
        fig_ret_dist = go.Figure()
        fig_ret_dist.add_trace(go.Histogram(
            x=fd["Return"] * 100,
            nbinsx=60,
            marker=dict(color="#3b82f6", opacity=0.7, line=dict(color="#60a5fa", width=0.5)),
            name="Daily Returns"
        ))
        var_95 = np.percentile(fd["Return"]*100, 5)
        fig_ret_dist.add_vline(x=var_95, line_dash="dash",
                                line_color="#f87171", annotation_text=f"VaR 95%: {var_95:.2f}%",
                                annotation_font_color="#f87171")
        fig_ret_dist.update_layout(**CHART_TEMPLATE, height=320,
            title="Daily Return Distribution",
            xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig_ret_dist, use_container_width=True)

        # Multi-stock risk comparison
        if df["Ticker"].nunique() > 1:
            st.markdown('<div class="section-header">📊 Cross-Stock Volatility Comparison</div>', unsafe_allow_html=True)
            risk_summary = []
            for t in df["Ticker"].unique():
                sub = df[df["Ticker"] == t]["Close"]
                r   = sub.pct_change().dropna()
                if len(r) < 20:
                    continue
                risk_summary.append({
                    "Ticker": t,
                    "Annual Return %": round(r.mean() * 252 * 100, 2),
                    "Annual Vol %":    round(r.std() * np.sqrt(252) * 100, 2),
                    "Sharpe":          round((r.mean() / r.std()) * np.sqrt(252), 2) if r.std() != 0 else 0,
                })
            risk_df = pd.DataFrame(risk_summary)
            fig_scatter = px.scatter(
                risk_df, x="Annual Vol %", y="Annual Return %",
                text="Ticker", color="Sharpe",
                color_continuous_scale="Bluered_r",
                title="Risk vs Return Scatter (All Stocks)"
            )
            fig_scatter.update_traces(textposition="top center", marker=dict(size=8))
            fig_scatter.update_layout(**CHART_TEMPLATE, height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 5 — BACKTESTING
    # ════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-header">📉 Strategy Backtesting</div>', unsafe_allow_html=True)

        bt = fd.copy()
        bt["Buy_Hold_Return"]   = (1 + bt["Return"]).cumprod()

        # MA Crossover Strategy
        bt["MA_Signal"]         = np.where(bt["MA20"] > bt["MA50"], 1, -1)
        bt["MA_Ret"]            = bt["Return"] * bt["MA_Signal"].shift(1)
        bt["MA_Cum"]            = (1 + bt["MA_Ret"].fillna(0)).cumprod()

        # RSI Strategy
        bt["RSI_Signal"]        = np.where(bt["RSI"] < 30, 1, np.where(bt["RSI"] > 70, -1, 0))
        bt["RSI_Ret"]           = bt["Return"] * bt["RSI_Signal"].shift(1)
        bt["RSI_Cum"]           = (1 + bt["RSI_Ret"].fillna(0)).cumprod()

        # MACD Strategy
        bt["MACD_Sig_Strat"]    = np.where(bt["MACD"] > bt["MACD_Signal"], 1, -1)
        bt["MACD_Ret"]          = bt["Return"] * bt["MACD_Sig_Strat"].shift(1)
        bt["MACD_Cum"]          = (1 + bt["MACD_Ret"].fillna(0)).cumprod()

        fig_bt = go.Figure()
        strategies = [
            ("Buy & Hold",          "Buy_Hold_Return", "#94a3b8"),
            ("MA Crossover",        "MA_Cum",          "#60a5fa"),
            ("RSI Mean-Reversion",  "RSI_Cum",         "#34d399"),
            ("MACD Trend-Follow",   "MACD_Cum",        "#f59e0b"),
        ]
        for label, col, color in strategies:
            fig_bt.add_trace(go.Scatter(x=bt["Date"], y=bt[col],
                line=dict(color=color, width=2), name=label))

        fig_bt.update_layout(**CHART_TEMPLATE, height=500,
            title=f"Backtesting — Cumulative Returns ({selected_stock})",
            yaxis_title="Cumulative Return (₹1 invested)")
        st.plotly_chart(fig_bt, use_container_width=True)

        # Strategy metrics table
        def backtest_metrics(cum_series, ret_series):
            total = cum_series.iloc[-1] - 1
            ann   = ret_series.mean() * 252
            vol   = ret_series.std() * np.sqrt(252)
            sh    = ann / vol if vol != 0 else 0
            peak  = cum_series.cummax()
            dd    = ((cum_series - peak) / peak).min()
            wins  = (ret_series > 0).sum()
            total_trades = (ret_series != 0).sum()
            wr    = wins / total_trades if total_trades > 0 else 0
            return {
                "Total Return %":   round(total * 100, 2),
                "Ann. Return %":    round(ann * 100, 2),
                "Ann. Volatility %":round(vol * 100, 2),
                "Sharpe Ratio":     round(sh, 3),
                "Max Drawdown %":   round(dd * 100, 2),
                "Win Rate %":       round(wr * 100, 2),
            }

        rows = []
        for label, col, _ in strategies:
            ret_col = col.replace("_Cum","_Ret").replace("Buy_Hold_Return","Return")
            r = backtest_metrics(bt[col], bt.get(ret_col, bt["Return"]))
            r["Strategy"] = label
            rows.append(r)

        bt_df = pd.DataFrame(rows)[["Strategy","Total Return %","Ann. Return %",
                                    "Ann. Volatility %","Sharpe Ratio","Max Drawdown %","Win Rate %"]]
        st.markdown("**📊 Strategy Performance Summary**")
        st.dataframe(bt_df, use_container_width=True, hide_index=True)

        # Monthly returns heatmap
        st.markdown('<div class="section-header">📅 Monthly Returns Heatmap</div>', unsafe_allow_html=True)
        bt["Year"]  = bt["Date"].dt.year
        bt["Month"] = bt["Date"].dt.month
        monthly     = bt.groupby(["Year","Month"])["Return"].sum().unstack() * 100
        fig_monthly = go.Figure(go.Heatmap(
            z=monthly.values,
            x=[f"M{m}" for m in monthly.columns],
            y=monthly.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            text=monthly.round(1).values,
            texttemplate="%{text}%",
            colorbar=dict(title="Return %", tickfont=dict(color="#94a3b8"))
        ))
        fig_monthly.update_layout(**CHART_TEMPLATE, height=350,
            title="Monthly Return Heatmap (%)")
        st.plotly_chart(fig_monthly, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # TAB 6 — FULL REPORT
    # ════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="section-header">📋 AI-Generated Full Stock Report</div>', unsafe_allow_html=True)

        try:
            risk_metrics = compute_risk_metrics(fd["Close"])

            full_report = generate_report(
                stock=selected_stock,
                df_stock=fd,
                signal=final_signal,
                confidence=confidence,
                risk_level=risk_level,
                metrics=risk_metrics,
                model_results=result_df,
                selected_features=selected_features,
                latest_close=latest_close,
                target_price=target_price,
                stop_loss=stop_loss,
                rsi_val=rsi_val,
                macd_val=macd_val,
            )
            st.markdown(f'<div class="report-box">{full_report}</div>', unsafe_allow_html=True)

            # Downloads
            st.markdown("---")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button(
                    "📄 Download Report (.txt)",
                    data=full_report,
                    file_name=f"{selected_stock}_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            with d2:
                csv_data = fd.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📊 Download Processed Data (.csv)",
                    data=csv_data,
                    file_name=f"{selected_stock}_processed.csv",
                    mime="text/csv"
                )
            with d3:
                metrics_csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "🤖 Download Model Metrics (.csv)",
                    data=metrics_csv,
                    file_name=f"{selected_stock}_model_metrics.csv",
                    mime="text/csv"
                )

        except NameError:
            st.info("⚠️ Please run the ML Models tab first to generate predictions before viewing the full report.")

# ============================================================
# NO FILE UPLOADED — LANDING STATE
# ============================================================

else:
    st.markdown('<div class="info-banner">📂 Upload one of the approved datasets using the sidebar to begin analysis.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features_list = [
        ("📊 Technical Analysis", ["RSI · MACD · Bollinger Bands", "Stochastic Oscillator", "Williams %R · ATR", "OBV · Momentum · ROC"]),
        ("🤖 ML Models", ["Logistic Regression", "Random Forest · XGBoost", "Gradient Boosting", "Support Vector Machine"]),
        ("📉 Analytics", ["Backtesting (4 strategies)", "Risk Metrics (Sharpe/VaR)", "Monthly Return Heatmap", "Feature Importance"]),
    ]
    for col, (title, items) in zip([c1,c2,c3], features_list):
        with col:
            bullet = "\n".join([f"  • {i}" for i in items])
            st.markdown(f"""<div class="kpi-card" style="text-align:left">
                <div class="kpi-label" style="font-size:0.85rem;color:#60a5fa">{title}</div>
                <div style="color:#94a3b8;font-size:0.88rem;line-height:2;margin-top:0.6rem">
                    {"<br>".join(["• "+i for i in items])}
                </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📌 Approved Dataset Files:**")
    for f in ALLOWED_FILES:
        st.markdown(f"&nbsp;&nbsp;• `{f}`")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#334155;font-size:0.78rem">'
    'AI-Powered Stock Report Generation System · MSc Data Science Major Project · '
    'For Educational Purposes Only · Not Financial Advice</p>',
    unsafe_allow_html=True
)

# ============================================================
# END OF APPLICATION
# ============================================================
