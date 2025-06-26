import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import os
import joblib
import plotly.graph_objects as go
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

# === APP CODE ===

def load_politician_data():
    csv_path = "data/politician_trades.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
        df.columns = df.columns.str.strip().str.lower()
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].str.upper()
        return df
    return pd.DataFrame()

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

def load_model():
    try:
        model = joblib.load("models/predictor_model.pkl")
        return model
    except Exception as e:
        st.warning("⚠️ ML model not found or failed to load.")
        return None

def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except:
        return None

# === STREAMLIT UI ===
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Movement Predictor")
st.write("Use the sidebar to train model or load stock data.")

with st.sidebar:
    st.header("Options")
    ticker = st.text_input("Enter Ticker Symbol", "AAPL")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    show_train = st.button("Train Model")

if show_train:
    with st.spinner("Training model..."):
        train_and_save_model()
    st.success("Model trained and saved.")

if ticker:
    df = get_stock_data(ticker, period)
    if not df.empty:
        st.subheader(f"📊 Stock Data: {ticker}")
        st.plotly_chart(go.Figure(go.Scatter(x=df.index, y=df["Close"], name="Close")), use_container_width=True)

        st.subheader("📈 Moving Averages")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode='lines', name='SMA 20'))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode='lines', name='SMA 50'))
        st.plotly_chart(fig_ma, use_container_width=True)

        st.subheader("📉 RSI and MACD Indicators")
        fig_indicators = go.Figure()
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='RSI'))
        fig_indicators.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode='lines', name='MACD'))
        st.plotly_chart(fig_indicators, use_container_width=True)

        model = load_model()
        if model:
            features = df[["SMA_20", "SMA_50", "RSI", "MACD"]].dropna()
            if not features.empty:
                preds = model.predict(features)
                preds_df = pd.DataFrame({"Date": features.index, "Prediction": preds})
                st.subheader("🤖 Model Predictions")
                fig_preds = go.Figure(go.Scatter(x=preds_df["Date"], y=preds_df["Prediction"], mode='lines+markers', name='Prediction'))
                fig_preds.update_layout(yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Down', 'Up']))
                st.plotly_chart(fig_preds, use_container_width=True)

                st.info("Prediction shown for each trading day as Up (1) or Down (0).")

# === TOP POLITICIAN PICKS (placeholder for future step) ===
st.markdown("---")
st.subheader("🏛️ Top Politician Trade Picks (Coming Soon)")
st.info("This section will highlight the top 10 trade picks based on recent politician activity.")
