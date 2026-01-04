from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import pandas as pd

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_stock_data(ticker: str):
    go_back = 180
    start_date = date.today() - timedelta(go_back)

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
    return df_stock

@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        model = load_model(f"{ticker}_lstm_v1.h5")
        feature_scaler = joblib.load(f"{ticker}_feature_scaler.gz")
        target_scaler = joblib.load(f"{ticker}_target_scaler.gz")
    except FileNotFoundError:
        return {"error": f"Model for {ticker} not found."}

    df_stock = get_stock_data(ticker)
    if df_stock.empty:
        return {"error": f"Could not download data for ticker {ticker}."}

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

    df_stock_log = df_stock.copy()
    df_stock_log['Close'] = np.log(df_stock_log['Close'])
    df_stock_log['SMA_20'] = np.log(df_stock_log['SMA_20'])
    df_stock_log['SMA_10'] = np.log(df_stock_log['SMA_10'])
    df_stock_log['SMA_5'] = np.log(df_stock_log['SMA_5'])

    df_stock_log.dropna(subset=['SMA_20', 'SMA_10', 'SMA_5', 'Volitility', 'RSI'], inplace=True)

    df_stock_log = df_stock_log.tail(60)

    X = df_stock_log[['Returns', 'SMA_20', 'SMA_10', 'SMA_5', 'Volitility', 'RSI']]

    X_scaled = feature_scaler.transform(X)
    X_final = np.array([X_scaled])

    y_pred_scaled = model.predict(X_final)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    prediction_date = df_stock['Date'].iloc[-1] + timedelta(days=1)
    
    return {"ticker": ticker, "predicted_price": float(np.exp(y_pred[0][0])), "prediction_date": prediction_date.strftime('%Y-%m-%d')}

@app.get("/historical/{ticker}")
def get_historical_data(ticker: str):
    df_stock = get_stock_data(ticker)
    df_stock['Date'] = df_stock['Date'].dt.strftime('%Y-%m-%d')
    return df_stock[['Date', 'Close']].to_dict('records')