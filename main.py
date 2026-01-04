from fastapi import FastAPI
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import yfinance as yf
from datetime import date, timedelta

app = FastAPI()

@app.get("/predict/NVDA")
def predict():
    ticker = "NVDA"

    model = load_model(f"{ticker}_lstm_v1.h5")
    feature_scaler = joblib.load(f"{ticker}_feature_scaler.gz")
    target_scaler = joblib.load(f"{ticker}_target_scaler.gz")

    go_back = 180
    start_date = date.today() - timedelta(go_back)
    print(start_date)

    df_stock = yf.download(
        tickers=ticker,
        start=start_date,
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        progress=False
    )

    df_stock = (
        df_stock
        .stack(level=0)
        .rename_axis(['Date', 'Ticker'])
        .reset_index()
    )

    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Feature engineering
    df_stock['Returns'] = df_stock['Close'].pct_change()
    df_stock['SMA_20'] = df_stock['Close'].rolling(window=20).mean()
    df_stock['SMA_10'] = df_stock['Close'].rolling(window=10).mean()
    df_stock['SMA_5'] = df_stock['Close'].rolling(window=5).mean()
    df_stock['Volitility'] = df_stock['Returns'].rolling(window=10).std()
    df_stock['RSI'] = calculate_rsi(df_stock)

    df_stock['Close'] = np.log(df_stock['Close'])
    df_stock['SMA_20'] = np.log(df_stock['SMA_20'])
    df_stock['SMA_10'] = np.log(df_stock['SMA_10'])
    df_stock['SMA_5'] = np.log(df_stock['SMA_5'])

    df_stock.dropna(subset=['SMA_20', 'SMA_10', 'SMA_5', 'Volitility', 'RSI'], inplace=True)

    df_stock = df_stock.tail(60)

    X = df_stock[['Returns', 'SMA_20', 'SMA_10', 'SMA_5', 'Volitility', 'RSI']]

    X_scaled = feature_scaler.transform(X)
    X_final = np.array([X_scaled])

    y_pred_scaled = model.predict(X_final)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    
    return {"ticker": ticker, "predicted_price": float(np.exp(y_pred[0][0]))}