import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Simple Stock Ticker App")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL):").upper()

if ticker:
    try:
        # Download stock data
        data = yf.download(ticker, period="6mo", interval="1d")

        # Flatten multi-level columns if necessary
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(filter(None, col)).strip() for col in data.columns.values]

        # Debug: show raw data
        st.write("Raw Data Preview:")
        st.dataframe(data.head())

        # Check for 'Close' column
        close_col = [col for col in data.columns if 'Close' in col]
        if not close_col:
            st.error("Could not find 'Close' price in data. Ticker may be invalid or unavailable.")
            st.stop()

        close_col = close_col[0]  # use the first 'Close' column

        # Calculate indicators
        data["SMA_20"] = data[close_col].rolling(window=20).mean()
        data["SMA_50"] = data[close_col].rolling(window=50).mean()
        data["EMA_20"] = data[close_col].ewm(span=20, adjust=False).mean()

        exp1 = data[close_col].ewm(span=12, adjust=False).mean()
        exp2 = data[close_col].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2
        data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # Chart 1: Closing Price
        st.subheader("📊 Closing Price")
        st.line_chart(data[close_col])

        # Chart 2: SMA 20 vs SMA 50
        st.subheader("📈 SMA 20 vs SMA 50")
        sma_df = data[["SMA_20", "SMA_50"]].dropna()
        if not sma_df.empty:
            st.line_chart(sma_df)
        else:
            st.warning("SMA values not yet available — wait for more data.")

        # Chart 3: MACD
        st.subheader("📉 MACD & Signal Line")
        macd_df = data[["MACD", "Signal"]].dropna()
        if not macd_df.empty:
            st.line_chart(macd_df)
        else:
            st.warning("MACD values not yet available — wait for more data.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
