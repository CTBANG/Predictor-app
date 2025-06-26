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
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_default_news_api_key")

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
        model_path = "models/predictor_model.pkl"
        if not os.path.exists(model_path):
            st.warning("⚠️ ML model not found. Auto-training...")
            train_and_save_model()
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(f"⚠️ ML model not found or failed to load: {e}")
        return None

def fetch_news_sentiment(query="stock market"):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=20)
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for article in articles["articles"]:
        text = article["title"] + ". " + (article.get("description") or "")
        score = analyzer.polarity_scores(text)["compound"]
        sentiments.append({"source": article["source"]["name"], "title": article["title"], "score": score})
    return sentiments

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
        st.plotly_chart(go.Figure(go.Scatter(x=df.index, y=df["Close"], name="Close Price")), use_container_width=True)

        st.subheader("📈 Moving Averages")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode='lines', name='20-Day SMA'))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode='lines', name='50-Day SMA'))
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

        st.subheader("📰 News Sentiment Overlay")
        sentiments = fetch_news_sentiment(ticker)
        if sentiments:
            for entry in sentiments[:5]:
                st.write(f"[{entry['source']}] {entry['title']} → **Sentiment**: {entry['score']:.2f}")
        else:
            st.warning("No sentiment data retrieved.")

# === TOP POLITICIAN PICKS ===
st.markdown("---")
st.subheader("🏛️ Top Politician Trade Picks")

df_politicians = load_politician_data()
if not df_politicians.empty:
    recent_trades = df_politicians.sort_values("transaction_date", ascending=False)
    top_tickers = recent_trades["ticker"].value_counts().head(10).index.tolist()
    st.write("Top 10 Tickers Recently Traded by Politicians:", top_tickers)
    st.dataframe(recent_trades[recent_trades["ticker"].isin(top_tickers)][["politician", "ticker", "transaction_date", "transaction"]])
else:
    st.warning("No politician trade data found.")
