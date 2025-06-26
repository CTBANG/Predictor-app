import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_daily_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    feeds = [
        "http://rss.cnn.com/rss/cnn_topstories.rss",         # Left
        "https://feeds.foxnews.com/foxnews/latest",          # Right
        "http://feeds.bbci.co.uk/news/rss.xml",              # Center/Global
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # Left-Center
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best", # Center
        "https://www.aljazeera.com/xml/rss/all.xml",         # Global
        "https://www.marketwatch.com/rss/topstories",        # Finance
        "https://www.npr.org/rss/rss.php?id=1001"            # Center-Left
    ]

    scores = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            text = entry.title + " " + entry.get("summary", "")
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

    return np.mean(scores) if scores else 0.0

def train_and_save_model():
    ticker = "AAPL"
    df = yf.download(ticker, period="1y")

    if df.empty:
        print("No data available for training.")
        return

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()

    # Generate date-indexed sentiment scores (daily)
    sentiment_score = fetch_daily_sentiment()
    df["Sentiment"] = sentiment_score

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    X = df[["SMA_20", "SMA_50", "RSI", "MACD", "Sentiment"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model Accuracy: {acc:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/predictor_model.pkl")
    print("✅ Saved model to models/predictor_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
