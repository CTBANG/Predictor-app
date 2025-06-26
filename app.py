import streamlit as st
import yfinance as yf
import pandas as pd

# Set Streamlit page config
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Simple Stock Ticker App")

# Input: stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):").upper()

if ticker:
    try:
        # Download data
        data = yf.download(ticker, period="6mo", interval="1d")

        # Check if valid data is returned
        if data.empty or "Close" not in data.columns:
            st.error("Could not find 'Close' price in data. Ticker may be invalid or unavailable.")
            st.stop()

        # Show raw data preview
        st.write("Raw Data Preview:")
        st.dataframe(data.head())

        # === Technical Indicators ===
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # === Charts ===
        st.subheader("📊 Closing Price")
        st.line_chart(data["Close"])

        st.subheader("📈 SMA 20 vs SMA 50")
        sma_df = data[["SMA_20", "SMA_50"]].dropna()
        if not sma_df.empty:
            st.line_chart(sma_df)
        else:
            st.warning("SMA values not yet available — wait for more data to accumulate.")

        st.subheader("📉 MACD & Signal Line")
        macd_df = data[["MACD", "Signal"]].dropna()
        if not macd_df.empty:
            st.line_chart(macd_df)
        else:
            st.warning("MACD values not yet available — wait for more data to accumulate.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
