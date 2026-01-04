import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['Volitility'] = df['Returns'].rolling(window=10).std()
    df['RSI'] = calculate_rsi(df)
    return df

def apply_log_transformation(df):
    df = df.copy()
    df['Close'] = np.log(df['Close'])
    df['SMA_20'] = np.log(df['SMA_20'])
    df['SMA_10'] = np.log(df['SMA_10'])
    df['SMA_5'] = np.log(df['SMA_5'])
    return df