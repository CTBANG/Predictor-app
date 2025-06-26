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

# Load politician trade data
def load_politician_data():
    csv_path = "data/politician_trades.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
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

# Load ML model (placeholder, assumes model.pkl exists)
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

st.set_page_config(layout="wide")
st.title("📊 Real-Time Stock + Politician Trading Insights")

if st.button("🔄 Refresh All Data"):
    with st.spinner("Refreshing stock and politician data..."):
        try:
            subprocess.run(["python", "scraper_politicians.py"], check=True)
            st.success("Data refreshed successfully!")
        except Exception as e:
            st.error(f"Failed to refresh data: {e}")

if st.button("🔁 Manually Update Politician Trades"):
    with st.spinner("Running scraper..."):
        try:
            subprocess.run(["python", "scraper_politicians.py"], check=True)
            st.success("Politician trade data updated!")
        except Exception as e:
            st.error(f"Failed to run scraper: {e}")
            st.write(e)

pol_data = load_politician_data()
model = load_model()
sentiment_pipe = load_sentiment_pipeline()

# Top 10 ticker picks from most traded by politicians
def get_top_10_picks(pol_df):
    if pol_df.empty:
        return []
    top_tickers = pol_df['ticker'].value_counts().head(10).index.tolist()
    return top_tickers

st.subheader("🔥 Top 10 Politician-Traded Stocks")
top_10 = get_top_10_picks(pol_data)

for ticker in top_10:
    with st.expander(f"📌 {ticker} Analysis"):
        stock_data = get_stock_data(ticker)
        if not stock_data.empty:
            try:
                info = yf.Ticker(ticker).info
                company_name = info.get("longName", ticker)
            except:
                company_name = ticker

            st.markdown(f"#### 📈 Price & Indicators for {company_name} ({ticker})")

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], mode='lines', name='Close Price', line=dict(color='deepskyblue')))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_20"], mode='lines', name='20-Day SMA', line=dict(color='blue')))
            fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_50"], mode='lines', name='50-Day SMA', line=dict(color='pink')))
            fig1.update_layout(title=f"{ticker} Stock Price with 20 & 50-Day SMAs", xaxis_title="Date", yaxis_title="Stock Price (USD)", legend_title="Indicators")
            st.plotly_chart(fig1, use_container_width=True)

            st.plotly_chart(px.line(stock_data, x=stock_data.index, y="RSI", title=f"{ticker} Relative Strength Index (RSI)"), use_container_width=True)
            st.plotly_chart(px.line(stock_data, x=stock_data.index, y="MACD", title=f"{ticker} MACD (12-26 EMA)"), use_container_width=True)
            st.plotly_chart(px.bar(stock_data, x=stock_data.index, y="Volume", title=f"{ticker} Daily Trading Volume"), use_container_width=True)

            pol_data_ticker = pol_data[pol_data["ticker"] == ticker]
            if not pol_data_ticker.empty:
                st.markdown(f"#### 🧑‍⚖️ Trades by Top Politicians in {ticker}")
                st.dataframe(pol_data_ticker.sort_values(by="transaction_date", ascending=False))
            else:
                st.warning(f"No politician trade data found for {ticker}.")

            # Placeholder ML prediction
            if model:
                try:
                    features = stock_data[["SMA_20", "SMA_50", "RSI", "MACD"]].dropna().iloc[-1].values.reshape(1, -1)
                    prediction = model.predict(features)[0]
                    st.info(f"📈 Predicted 1-day move: {'↑ Up' if prediction == 1 else '↓ Down'}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# Individual ticker analysis
st.subheader("🔍 Analyze a Specific Ticker")
ticker = st.text_input("Enter stock ticker:", "AAPL").upper()
if ticker:
    stock_data = get_stock_data(ticker)
    if not stock_data.empty:
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
        except:
            company_name = ticker

        st.subheader(f"📈 Price & Indicators for {company_name} ({ticker})")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], mode='lines', name='Close Price', line=dict(color='deepskyblue')))
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_20"], mode='lines', name='20-Day SMA', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_50"], mode='lines', name='50-Day SMA', line=dict(color='pink')))
        fig1.update_layout(title=f"{ticker} Stock Price with 20 & 50-Day SMAs", xaxis_title="Date", yaxis_title="Stock Price (USD)", legend_title="Indicators")
        st.plotly_chart(fig1, use_container_width=True)

        st.plotly_chart(px.line(stock_data, x=stock_data.index, y="RSI", title=f"{ticker} Relative Strength Index (RSI)"), use_container_width=True)
        st.plotly_chart(px.line(stock_data, x=stock_data.index, y="MACD", title=f"{ticker} MACD (12-26 EMA)"), use_container_width=True)
        st.plotly_chart(px.bar(stock_data, x=stock_data.index, y="Volume", title=f"{ticker} Daily Trading Volume"), use_container_width=True)

    pol_data_ticker = pol_data[pol_data["ticker"] == ticker]
    if not pol_data_ticker.empty:
        st.subheader(f"🧑‍⚖️ Trades by Top Politicians in {ticker}")
        st.dataframe(pol_data_ticker.sort_values(by="transaction_date", ascending=False))
    else:
        st.warning(f"No politician trade data found for {ticker}.")
