import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
import os
import subprocess
import plotly.express as px
import plotly.graph_objects as go

# Load politician trade data
def load_politician_data():
    csv_path = "data/politician_trades.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
        return df
    return pd.DataFrame()

# Compute RSI
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load historical stock data
def get_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, group_by='ticker')
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        st.error(f"No data returned for {ticker}. Please check the ticker symbol.")
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    df = df.loc[:, ~df.columns.duplicated()]

    required_cols = {"Close", "Volume"}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing required columns in downloaded data: {required_cols - set(df.columns)}")
        st.dataframe(df.head())
        return pd.DataFrame()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    return df

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 Real-Time Stock + Politician Trading Insights")

if st.button("🔄 Refresh All Data"):
    with st.spinner("Refreshing stock and politician data..."):
        try:
            subprocess.run(["python", "scraper_politicians.py"], check=True)
            st.success("Data refreshed successfully!")
        except Exception as e:
            st.error(f"Failed to refresh data: {e}")

if st.button("🔁 Manually Update Politician Trades"):
    with st.spinner("Running scraper..."):
        try:
            subprocess.run(["python", "scraper_politicians.py"], check=True)
            st.success("Politician trade data updated!")
        except Exception as e:
            st.error(f"Failed to run scraper: {e}")
            st.write(e)

ticker = st.text_input("Enter stock ticker:", "AAPL").upper()
if ticker:
    stock_data = get_stock_data(ticker)
    if not stock_data.empty:
        try:
            info = yf.Ticker(ticker).info
            company_name = info.get("longName", ticker)
        except:
            company_name = ticker

        st.subheader(f"📈 Price & Indicators for {company_name} ({ticker})")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], mode='lines', name='Close Price', line=dict(color='deepskyblue')))
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_20"], mode='lines', name='20-Day SMA', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_50"], mode='lines', name='50-Day SMA', line=dict(color='pink')))
        fig1.update_layout(
            title=f"{ticker} Stock Price with 20 & 50-Day SMAs",
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)",
            legend_title="Indicators"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(stock_data, x=stock_data.index, y="RSI", title=f"{ticker} Relative Strength Index (RSI)")
        fig2.update_layout(xaxis_title="Date", yaxis_title="RSI (14-day)", legend_title_text='')
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(stock_data, x=stock_data.index, y="MACD", title=f"{ticker} MACD (12-26 EMA)")
        fig3.update_layout(xaxis_title="Date", yaxis_title="MACD Value", legend_title_text='')
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.bar(stock_data, x=stock_data.index, y="Volume", title=f"{ticker} Daily Trading Volume")
        fig4.update_layout(xaxis_title="Date", yaxis_title="Shares Traded", legend_title_text='')
        st.plotly_chart(fig4, use_container_width=True)

    pol_data = load_politician_data()
    if not pol_data.empty:
        pol_data_ticker = pol_data[pol_data["ticker"] == ticker]
        st.subheader(f"🧑‍⚖️ Trades by Top 10 Politicians in {ticker}")
        st.dataframe(pol_data_ticker.sort_values(by="transaction_date", ascending=False))
    else:
        st.warning(f"No politician trade data found for {ticker}.")
