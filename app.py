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
            # Compute Indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()

            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Plot 1: Closing Price
            st.subheader("📊 Closing Price")
            st.line_chart(data['Close'])

            # Plot 2: SMA
            st.subheader("🟣 SMA 20 vs SMA 50")
            sma_df = data[['SMA_20', 'SMA_50']].dropna()
            if not sma_df.empty:
                st.line_chart(sma_df)
            else:
                st.warning("SMA data is not available yet (insufficient historical data).")

            # Plot 3: RSI
            st.subheader("📉 RSI (Relative Strength Index)")
            if 'RSI' in data.columns and not data['RSI'].dropna().empty:
                st.line_chart(data['RSI'].dropna())
            else:
                st.warning("RSI data not available.")

            # Plot 4: MACD
            st.subheader("📈 MACD vs Signal Line")
            if 'MACD' in data.columns and 'Signal_Line' in data.columns:
                macd_data = data[['MACD', 'Signal_Line']].dropna()
                if not macd_data.empty:
                    st.line_chart(macd_data)
                else:
                    st.warning("MACD values not yet available.")
            else:
                st.warning("MACD calculation failed.")

            # Plot 5: Volume
            st.subheader("🔊 Volume")
            if 'Volume' in data.columns and not data['Volume'].dropna().empty:
                st.line_chart(data['Volume'])
            else:
                st.warning("Volume data not available.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
