import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import os
import subprocess
import plotly.express as px
import plotly.graph_objects as go
import joblib
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === TRAINING SCRIPT ===
def train_and_save_model():
    ticker = "AAPL"
    df = yf.download(ticker, period="1y")

    if df.empty:
        print("No data downloaded.")
        return

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()
    X = df[["SMA_20", "SMA_50", "RSI", "MACD"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {accuracy:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/predictor_model.pkl")
    print("Model saved to models/predictor_model.pkl")

# Uncomment below to train model directly from Streamlit if needed
# if st.sidebar.button("Train ML Model"):
#     train_and_save_model()

# === APP CODE ===
# Load politician trade data
def load_politician_data():
    csv_path = "data/politician_trades.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
        df.columns = df.columns.str.strip().str.lower()
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].str.upper()
        return df
    return pd.DataFrame()

# Compute RSI
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta > 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load historical stock data
def get_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, group_by='ticker')
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        st.error(f"No data returned for {ticker}. Please check the ticker symbol.")
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.loc[:, ~df.columns.duplicated()]

    required_cols = {"Close", "Volume"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing required columns in downloaded data: {required_cols - set(df.columns)}")
        st.dataframe(df.head())
        return pd.DataFrame()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    return df

# Load ML model
def load_model():
    try:
        model = joblib.load("models/predictor_model.pkl")
        return model
    except Exception as e:
        st.warning("⚠️ ML model not found or failed to load.")
        return None

# Load sentiment pipeline (e.g., from transformers)
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except:
        return None

# ... rest of the Streamlit app logic remains unchanged ...
