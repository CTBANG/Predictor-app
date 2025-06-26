import streamlit as st
import yfinance as yf
import pandas as pd

# Page setup
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Simple Stock Ticker App")

# Input
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):").upper()

if ticker:
    try:
        # Fetch data (force flat columns)
        data = yf.download(ticker, period="6mo", interval="1d", group_by='column')

        # Check if data is empty
        if data.empty:
            st.error("No data returned. Ticker may be invalid or unavailable.")
            st.stop()

        # Check if 'Close' exists
        if "Close" not in data.columns:
            st.error("'Close' price column not found.")
            st.dataframe(data.head())  # show what's there for debugging
            st.stop()

        # Debug preview
        st.write("Raw Data Preview:")
        st.dataframe(data.head())

        # Indicators
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Charts
        st.subheader("📊 Closing Price")
        st.line_chart(data["Close"])

        st.subheader("📈 SMA 20 vs SMA 50")
        sma_df = data[["SMA_20", "SMA_50"]].dropna()
        st.line_chart(sma_df)

        st.subheader("📉 MACD & Signal Line")
        macd_df = data[["MACD", "Signal"]].dropna()
        st.line_chart(macd_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
