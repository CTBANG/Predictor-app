import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import os
import subprocess

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
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load historical stock data
def get_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, group_by="ticker", auto_adjust=True)
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()

    if df.empty or ticker not in df.columns.levels[-1]:
        st.error(f"No data returned or missing expected columns for ticker {ticker}.")
        st.dataframe(df.head())
        return pd.DataFrame()

    try:
        df = df.xs(ticker, axis=1, level=1)  # flatten MultiIndex to single columns
    except Exception as e:
        st.error(f"Could not extract data for {ticker}: {e}")
        return pd.DataFrame()

    required_cols = {"Close", "Volume"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing required columns in downloaded data: {required_cols - set(df.columns)}")
        st.dataframe(df.head())  # Show what was returned
        return pd.DataFrame()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    return df

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 Real-Time Stock + Politician Trading Insights")

# Manual scraper run button
if st.button("🔄 Manually Update Politician Trades"):
    with st.spinner("Running scraper..."):
        try:
            subprocess.run(["python", "scraper_politicians.py"], check=True)
            st.success("Politician trade data updated!")
        except Exception as e:
            st.error(f"Failed to run scraper: {e}")
            st.write(e)

ticker = st.text_input("Enter stock ticker:", "AAPL").upper()
if ticker:
    stock_data = get_stock_data(ticker)
    if not stock_data.empty:
        st.subheader(f"📈 Price & Indicators for {ticker}")
        st.dataframe(stock_data.tail())

        try:
            if {"Close", "SMA_20", "SMA_50"}.issubset(stock_data.columns):
                st.line_chart(stock_data[["Close", "SMA_20", "SMA_50"]].dropna())
            if "RSI" in stock_data.columns:
                st.line_chart(stock_data[["RSI"]].dropna())
            if "MACD" in stock_data.columns:
                st.line_chart(stock_data[["MACD"]].dropna())
            if "Volume" in stock_data.columns:
                st.bar_chart(stock_data[["Volume"]].dropna())
        except Exception as e:
            st.error(f"Chart rendering failed: {e}")
            st.dataframe(stock_data.head())

    # Load and display relevant politician trades
    pol_data = load_politician_data()
    if not pol_data.empty:
        pol_data_ticker = pol_data[pol_data["ticker"] == ticker]
        st.subheader(f"🧑‍⚖️ Trades by Top 10 Politicians in {ticker}")
        st.dataframe(pol_data_ticker.sort_values(by="transaction_date", ascending=False))
    else:
        st.warning("No politician trade data found.")
