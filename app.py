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

# Get stock data
def get_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period)
        if "Close" in df.columns:
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            df["RSI"] = compute_rsi(df["Close"], 14)
            df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
            return df
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
    return pd.DataFrame()

# Streamlit app
st.set_page_config(layout="wide")
st.title("📊 Real-Time Stock + Politician Trading Insights")

# ✅ Debug marker
st.markdown("**DEBUG: UI rendered successfully**")

# 🔄 Manual update button
if st.button("🔄 Manually Update Politician Trades"):
    with st.spinner("Running scraper..."):
        try:
            subprocess.run(["python", "scraper_politicians.py"], check=True)
            st.success("✅ Politician trade data updated!")
        except Exception as e:
            st.error("❌ Failed to run scraper.")
            st.text(str(e))

# Ticker input
ticker = st.text_input("Enter stock ticker:", "AAPL").upper()

if ticker:
    stock_data = get_stock_data(ticker)

    if not stock_data.empty:
        st.subheader(f"📈 Price & Indicators for {ticker}")

        # 📉 Closing price with SMAs
        try:
            st.line_chart(stock_data[["Close", "SMA_20", "SMA_50"]].dropna())
        except KeyError as e:
            st.error(f"Missing SMA columns: {e}")

        # 📊 RSI
        try:
            st.line_chart(stock_data[["RSI"]].dropna())
        except KeyError as e:
            st.error(f"Missing RSI column: {e}")

        # 📊 MACD
        try:
            st.line_chart(stock_data[["MACD"]].dropna())
        except KeyError as e:
            st.error(f"Missing MACD column: {e}")

        # 📊 Volume
        try:
            st.bar_chart(stock_data[["Volume"]].dropna())
        except KeyError as e:
            st.error(f"Missing Volume column: {e}")

        # 🧑‍⚖️ Politician trades
        pol_data = load_politician_data()
        if not pol_data.empty:
            pol_data_ticker = pol_data[pol_data["ticker"].str.upper() == ticker]
            st.subheader(f"🧑‍⚖️ Trades by Top 10 Politicians in {ticker}")
            st.dataframe(pol_data_ticker.sort_values(by="transaction_date", ascending=False))
        else:
            st.warning("⚠️ No politician trade data found.")
    else:
        st.warning("⚠️ No stock data found for that ticker.")
