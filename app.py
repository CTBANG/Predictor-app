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
    df = yf.download(ticker, period=period, auto_adjust=True)
    
    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Confirm expected columns exist
    if "Close" in df.columns and "Volume" in df.columns:
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = compute_rsi(df["Close"], 14)
        df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    else:
        st.error("Required columns (Close, Volume) not found in stock data.")
    return df

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 Real-Time Stock + Politician Trading Insights")

# Debug marker
st.markdown("**DEBUG: UI loaded, button rendered**")

# Manual scraper run button
if st.button("🔄 Manually Update Politician Trades"):
    with st.spinner("Running scraper..."):
        try:
            subprocess.run(["python", "scraper_politicians.py"], check=True)
            st.success("Politician trade data updated!")
        except Exception as e:
            st.error(f"Failed to run scraper: {e}")
            st.exception(e)

ticker = st.text_input("Enter stock ticker:", "AAPL").upper()

if ticker:
    stock_data = get_stock_data(ticker)
    if not stock_data.empty:
        st.subheader(f"📈 Price & Indicators for {ticker}")
        try:
            st.line_chart(stock_data[["Close", "SMA_20", "SMA_50"]].dropna())
        except KeyError as e:
            st.error(f"Missing SMA columns: {e}")

        try:
            st.line_chart(stock_data[["RSI"]].dropna())
        except KeyError as e:
            st.error(f"Missing RSI column: {e}")

        try:
            st.line_chart(stock_data[["MACD"]].dropna())
        except KeyError as e:
            st.error(f"Missing MACD column: {e}")

        try:
            st.bar_chart(stock_data[["Volume"]].dropna())
        except KeyError as e:
            st.error(f"Missing Volume column: {e}")

    # Load and display relevant politician trades
    pol_data = load_politician_data()
    if not pol_data.empty:
        pol_data_ticker = pol_data[pol_data["ticker"].str.upper() == ticker]
        st.subheader(f"🧑‍⚖️ Trades by Top 10 Politicians in {ticker}")
        if not pol_data_ticker.empty:
            st.dataframe(pol_data_ticker.sort_values(by="transaction_date", ascending=False))
        else:
            st.info("No recent trades in this ticker by top politicians.")
    else:
        st.warning("⚠️ No politician trade data found.")
