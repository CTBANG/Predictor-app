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
            # Indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

            delta = data['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Closing Price
            st.subheader("📊 Closing Price")
            st.line_chart(data['Close'])

            # SMA
            st.subheader("🟣 SMA 20 vs SMA 50")
            try:
                sma_data = data[['SMA_20', 'SMA_50']].copy()
                if sma_data.dropna().empty:
                    st.warning("SMA data not available yet.")
                else:
                    st.line_chart(sma_data.dropna())
            except KeyError as e:
                st.error(f"Missing SMA columns: {e}")

            # RSI
            st.subheader("📉 RSI (Relative Strength Index)")
            if 'RSI' in data.columns and not data['RSI'].dropna().empty:
                st.line_chart(data['RSI'].dropna())
            else:
                st.warning("RSI data not available yet.")

            # MACD
            st.subheader("📈 MACD vs Signal Line")
            if all(col in data.columns for col in ['MACD', 'Signal_Line']):
                macd_data = data[['MACD', 'Signal_Line']].dropna()
                if not macd_data.empty:
                    st.line_chart(macd_data)
                else:
                    st.warning("MACD values not available yet.")
            else:
                st.warning("MACD columns missing.")

            # Volume
            st.subheader("🔊 Volume")
            if 'Volume' in data.columns and not data['Volume'].dropna().empty:
                st.line_chart(data['Volume'])
            else:
                st.warning("Volume data not available.")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
