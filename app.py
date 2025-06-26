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
            # Add indicators
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

            # --- Closing Price ---
            st.subheader("📊 Closing Price")
            st.line_chart(data['Close'])

            # --- SMA ---
            st.subheader("🟣 SMA 20 vs SMA 50")
            try:
                if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                    sma_df = pd.DataFrame({
                        'SMA_20': data['SMA_20'],
                        'SMA_50': data['SMA_50']
                    }).dropna()
                    if not sma_df.empty:
                        st.line_chart(sma_df)
                    else:
                        st.warning("SMA values are not available yet (too few data points).")
                else:
                    st.warning("SMA columns not found in data.")
            except Exception as e:
                st.error(f"Unexpected error plotting SMA: {e}")

            # --- RSI ---
            st.subheader("📉 RSI (Relative Strength Index)")
            if 'RSI' in data.columns:
                st.line_chart(data['RSI'].dropna())

            # --- MACD ---
            st.subheader("📈 MACD vs Signal Line")
            if 'MACD' in data.columns and 'Signal_Line' in data.columns:
                macd_df = pd.DataFrame({
                    'MACD': data['MACD'],
                    'Signal_Line': data['Signal_Line']
                }).dropna()
                if not macd_df.empty:
                    st.line_chart(macd_df)
                else:
                    st.warning("MACD values not available yet.")
            else:
                st.warning("MACD or Signal Line column missing.")

            # --- Volume ---
            st.subheader("🔊 Volume")
            st.line_chart(data['Volume'])

    except Exception as e:
        st.error(f"Unexpected error: {e}")
