import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Stock Ticker Analyzer", layout="centered")
st.title("📈 Simple Stock Ticker App")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL):")

def compute_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

if ticker:
    data = yf.download(ticker, period="6mo", interval="1d")

    if data.empty:
        st.warning("⚠️ No data found for this ticker.")
    elif 'Close' not in data.columns:
        st.warning("⚠️ 'Close' price data is missing from the downloaded dataset.")
        st.write("Here’s what was downloaded:")
        st.dataframe(data)
    else:
        data["RSI"] = compute_rsi(data)
        data["MACD"], data["Signal"] = compute_macd(data)

        st.subheader(f"📊 Price and Volume for {ticker.upper()}")
        st.line_chart(data["Close"])
        st.bar_chart(data["Volume"])

        st.subheader("📉 RSI (Relative Strength Index)")
        st.line_chart(data["RSI"])

        st.subheader("📊 MACD vs Signal")
        st.line_chart(data[["MACD", "Signal"]])
