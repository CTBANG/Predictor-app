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
        # Fetch data
        data = yf.download(ticker, period="6mo", interval="1d")

        # Fix MultiIndex columns (flatten them)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # Ensure 'Close' column exists
        if not any(col.lower() == "close" or col.endswith("_close") for col in data.columns):
            st.error("Could not find 'Close' price in data. Ticker may be invalid or unavailable.")
            st.stop()

        # Find correct close column
        close_col = next((col for col in data.columns if col.lower() == "close" or col.endswith("_close")), None)
        data["Close"] = data[close_col]

        # Show raw preview
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
        sma_cols = ["SMA_20", "SMA_50"]
        if all(col in data.columns for col in sma_cols):
            st.line_chart(data[sma_cols].dropna())
        else:
            st.warning("SMA columns not available.")

        st.subheader("📉 MACD & Signal Line")
        macd_cols = ["MACD", "Signal"]
        if all(col in data.columns for col in macd_cols):
            st.line_chart(data[macd_cols].dropna())
        else:
            st.warning("MACD columns not available.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
