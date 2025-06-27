import yfinance as yf
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

def fetch_daily_sentiment():
    analyzer = SentimentIntensityAnalyzer()
    feeds = [
        "http://rss.cnn.com/rss/cnn_topstories.rss",
        "https://feeds.foxnews.com/foxnews/latest",
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.marketwatch.com/rss/topstories",
        "https://www.npr.org/rss/rss.php?id=1001"
    ]

    scores = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            text = entry.title + " " + entry.get("summary", "")
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

    return sum(scores)/len(scores) if scores else 0.0

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def train_and_save_model():
    ticker = "AAPL"
    df = yf.download(ticker, period="1y")

    if df.empty:
        print("No data downloaded.")
        return

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["Sentiment"] = fetch_daily_sentiment()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()
    X = df[["SMA_20", "SMA_50", "RSI", "MACD", "Sentiment"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {accuracy:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/predictor_model.pkl")
    print("Model saved to models/predictor_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
