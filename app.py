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
    df = yf.download(ticker, period=period)
    if df.empty:
        return df

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

# Input ticker
ticker = st.text_input("Enter stock ticker:", "AAPL").upper()

if ticker:
    stock_data = get_stock_data(ticker)
    
    if stock_data.empty:
        st.error("Failed to load stock data.")
    else:
        st.subheader(f"📈 Price & Indicators for {ticker}")
        
        # Check and plot Close + SMA
        required_sma_cols = ["Close", "SMA_20", "SMA_50"]
        if all(col in stock_data.columns for col in required_sma_cols):
            st.line_chart(stock_data[required_sma_cols].dropna())
        else:
            st.error(f"Missing SMA columns: {[col for col in required_sma_cols if col not in stock_data.columns]}")

        # Check and plot RSI
        if "RSI" in stock_data.columns:
            st.line_chart(stock_data[["RSI"]].dropna())
        else:
            st.error("Missing RSI column.")

        # Check and plot MACD
        if "MACD" in stock_data.columns:
            st.line_chart(stock_data[["MACD"]].dropna())
        else:
            st.error("Missing MACD column.")

        # Check and plot Volume
        if "Volume" in stock_data.columns:
            st.bar_chart(stock_data[["Volume"]].dropna())
        else:
            st.error("Missing Volume column.")

        # Load and display politician data
        pol_data = load_politician_data()
        if not pol_data.empty:
            pol_data_ticker = pol_data[pol_data["ticker"].str.upper() == ticker]
            st.subheader(f"🧑‍⚖️ Trades by Top 10 Politicians in {ticker}")
            st.dataframe(pol_data_ticker.sort_values(by="transaction_date", ascending=False))
        else:
            st.warning("⚠️ No politician trade data found.")
