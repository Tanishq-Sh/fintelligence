import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import numpy as np
import random


# Pulling data from yfinance API
ticker = 'AAPL'
start_date = datetime(year=2010, month=1, day=1)

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

# Feature engineering
df_stock['Returns'] = df_stock['Close'].pct_change()
df_stock['SMA'] = df_stock['Close'].rolling(window=20).mean()
df_stock['Volitility'] = df_stock['Returns'].rolling(window=20).std()
df_stock['Target'] = df_stock['Close'].shift(-1)
df_stock.dropna(subset=['SMA', 'Volitility', 'Target'], inplace=True)

# Split train-test data
X = df_stock[['Returns', 'SMA', 'Volitility']]
y = df_stock['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Scaling the features and targets
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1,1))

# Generating Data for Looking back 60 days for prediction via LSTM
X_train_final, y_train_final = [], []
for i in range(60, len(X_train_scaled)):
    X_train_final.append(X_train_scaled[i-60:i, :])
    y_train_final.append(y_train_scaled[i, 0])
    
X_train_final = np.array(X_train_final)
y_train_final = np.array(y_train_final)

# Building the Model
def make_results_deterministic():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

input_shape = (60, 3)

model = Sequential([
    tf.keras.Input(input_shape),
    LSTM(units=32, return_sequences=True),
    LSTM(units=16),
    Dense(units=1, activation='linear')
])

model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(0.001)
)

model.fit(
    X_train_final,
    y_train_final,
    epochs = 10,
    batch_size = 32
)

print(X_train_scaled.shape)
print("==========")
print(X_test_scaled.shape)
X_test_scaled = np.concatenate([X_train_scaled[-60:], np.array(X_test_scaled)])
X_test_final = []
for i in range(60, len(X_test_scaled)):
    X_test_final.append(X_test_scaled[i-60:i, :])
    
X_test_final = np.array(X_test_final)

y_test_pred = model.predict(X_test_final)
y_test_pred = target_scaler.inverse_transform(y_test_pred)

mse = mean_squared_error(np.array(y_test), y_test_pred[:, 0])

plt.figure()
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, y_test_pred[:,0], label='Predicted Price')
plt.legend()
plt.show()






