import streamlit as st
import yfinance as yf
import pandas as pd

# App Title
st.set_page_config(page_title="Stock Ticker App", page_icon="📈")
st.title("📈 Simple Stock Ticker App")

# Input
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):")

if ticker:
    # Download historical market data
    data = yf.download(ticker, period="6mo", interval="1d")

    if data.empty:
        st.error("No data found. Please enter a valid stock ticker.")
    else:
        # Calculate indicators
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()

        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100 / (1 + rs))

        ema12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = ema12 - ema26
        data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Output
        st.subheader(f"📊 Price and Volume for {ticker.upper()}")
        st.line_chart(data["Close"])
        st.line_chart(data["Volume"])

        st.subheader("📈 SMA 20 vs SMA 50")
        st.line_chart(data[["SMA_20", "SMA_50"]])

        st.subheader("📉 RSI (Relative Strength Index)")
        if "RSI" in data.columns and data["RSI"].notnull().any():
            st.line_chart(data["RSI"])
        else:
            st.warning("RSI data not available for this ticker.")

        st.subheader("📊 MACD vs Signal")
        if all(col in data.columns for col in ["MACD", "Signal"]) and data[["MACD", "Signal"]].notnull().any().any():
            st.line_chart(data[["MACD", "Signal"]])
        else:
            st.warning("MACD and Signal data not available.")
