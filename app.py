import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Simple Stock Ticker App")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL):").upper()

if ticker:
    try:
        data = yf.download(ticker, period="6mo", interval="1d")

        # Add debug printout
        st.write("Raw Data Preview:")
        st.dataframe(data.head())

        if data.empty:
            st.error("No data was returned. Check your internet connection or try a different ticker.")
        else:
            # Flatten columns if MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns.values]

            if "Close" not in data.columns:
                st.error("Could not find 'Close' price in the downloaded data.")
            else:
                # Indicators
                data["SMA_20"] = data["Close"].rolling(window=20).mean()
                data["SMA_50"] = data["Close"].rolling(window=50).mean()
                data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
                exp1 = data["Close"].ewm(span=12, adjust=False).mean()
                exp2 = data["Close"].ewm(span=26, adjust=False).mean()
                data["MACD"] = exp1 - exp2
                data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

                # Charts
                st.subheader(f"📊 Price and Volume for {ticker}")
                st.line_chart(data[["Close"]])
                st.bar_chart(data["Volume"])

                st.subheader("📈 SMA 20 vs SMA 50")
                sma_df = data[["SMA_20", "SMA_50"]].dropna()
                if not sma_df.empty:
                    st.line_chart(sma_df)
                else:
                    st.warning("SMA values not yet available — wait for more data.")

                st.subheader("📉 EMA 20")
                ema_df = data[["EMA_20"]].dropna()
                if not ema_df.empty:
                    st.line_chart(ema_df)

                st.subheader("📊 MACD vs Signal")
                macd_df = data[["MACD", "Signal"]].dropna()
                if not macd_df.empty:
                    st.line_chart(macd_df)
    except Exception as e:
        st.error(f"An error occurred: {e}")
