import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Simple Stock Ticker App")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL").upper()

if ticker:
    try:
        data = yf.download(ticker, period="6mo", interval="1d")

        if data.empty or 'Close' not in data.columns:
            st.error("Could not find 'Close' price in data. Ticker may be invalid or unavailable.")
        else:
            # Compute Technical Indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Charts
            st.subheader("📊 Closing Price")
            st.line_chart(data['Close'])

            st.subheader("🟣 SMA 20 vs SMA 50")
            sma_df = data[['SMA_20', 'SMA_50']].dropna()
            if not sma_df.empty:
                st.line_chart(sma_df)
            else:
                st.warning("SMA values not yet available — wait for more data.")

            st.subheader("📉 RSI (Relative Strength Index)")
            rsi = data[['RSI']].dropna()
            if not rsi.empty:
                st.line_chart(rsi)
            else:
                st.warning("RSI values not available.")

            st.subheader("📈 MACD (Moving Average Convergence Divergence)")
            macd_df = data[['MACD', 'Signal_Line']].dropna()
            if not macd_df.empty:
                st.line_chart(macd_df)
            else:
                st.warning("MACD values not available.")

            st.subheader("📊 Volume")
            volume_df = data[['Volume']].dropna()
            st.line_chart(volume_df)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
