import streamlit as st
import yfinance as yf

st.title("📈 Simple Stock Ticker App")

ticker = st.text_input("Enter a stock ticker (e.g. AAPL):")

if ticker:
    data = yf.download(ticker, period="7d", interval="1h")
    st.line_chart(data['Close'])
