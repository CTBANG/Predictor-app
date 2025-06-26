import streamlit as st
import yfinance as yf
import pandas as pd

# App setup
st.set_page_config(page_title="Stock Ticker App", page_icon="📈")
st.title("📈 Simple Stock Ticker App")

# Input box
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):")

if ticker:
    data = yf.download(ticker, period="6mo", interval="1d")

    if data.empty:
        st.error("No data found. Please enter a valid ticker.")
    else:
        # Technical indicators
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

        # Display charts
        st.subheader(f"📊 Price and Volume for {ticker.upper()}")
        st.line_chart(data["Close"])
        st.line_chart(data["Volume"])

        # Safe plot: SMA
        st.subheader("📈 SMA 20 vs SMA 50")
        sma_df = data[["SMA_20", "SMA_50"]].dropna()
        if not sma_df.empty:
            st.line_chart(sma_df)
        else:
            st.warning("SMA data not available yet — not enough history.")

        # Safe plot: RSI
        st.subheader("📉 RSI (Relative Strength Index)")
        rsi_df = data[["RSI"]].dropna()
        if not rsi_df.empty:
            st.line_chart(rsi_df)
        else:
            st.warning("RSI data not available yet.")

        # Safe plot: MACD
        st.subheader("📊 MACD vs Signal")
        macd_df = data[["MACD", "Signal"]].dropna()
        if not macd_df.empty:
            st.line_chart(macd_df)
        else:
            st.warning("MACD data not available yet.")
