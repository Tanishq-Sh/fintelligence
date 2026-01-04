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
import joblib
from src.features import calculate_rsi
from src.features import add_technical_indicators
from src.features import apply_log_transformation
from src.config import MODELS_DIR, MODEL_CONFIG, RANDOM_SEED
from src.utils import save_training_results

# Pulling data from yfinance API
# ticker = 'AAPL'
# ticker = 'TSLA'
ticker = 'NVDA'

config = MODEL_CONFIG.get(ticker, MODEL_CONFIG["DEFAULT"])

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
df_stock = add_technical_indicators(df_stock)
df_stock = apply_log_transformation(df_stock)


df_stock['Target'] = df_stock['Close'].shift(-1)

df_stock.dropna(subset=['SMA_20', 'SMA_10', 'SMA_5', 'Volitility', 'Target', 'RSI'], inplace=True)

# Split train-test data
# X = df_stock[['Returns', 'SMA', 'Volitility']]
X = df_stock[['Returns', 'SMA_20', 'SMA_10', 'SMA_5', 'Volitility', 'RSI']]
y = df_stock['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config["train_test_split"], shuffle=False)

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

# input_shape = (60, 3)
# input_shape = (60, 4)
input_shape = (60, 6)

model = Sequential([
    tf.keras.Input(input_shape),
    LSTM(units=config["lstm_layer_1_units"], return_sequences=True),
    LSTM(units=config["lstm_layer_2_units"]),
    Dense(units=config["output_linear_layer"], activation='linear')
])

model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
)

history = model.fit(
    X_train_final,
    y_train_final,
    epochs = config["epochs"],
    batch_size = config["batch_size"]
)

final_loss = history.history['loss'][-1]

metrics = {
    "loss": float(final_loss),
    "epochs_trained": len(history.history["loss"])
}

is_production_ready = True

print(f"Final Loss: {final_loss}")

save_training_results(
    model=model,
    feature_scaler=feature_scaler,
    target_scaler=target_scaler,
    ticker=ticker,
    metrics=metrics,
    config=config,
    is_best=is_production_ready
)

X_test_scaled = np.concatenate([X_train_scaled[-60:], np.array(X_test_scaled)])
X_test_final = []
for i in range(60, len(X_test_scaled)):
    X_test_final.append(X_test_scaled[i-60:i, :])
    
X_test_final = np.array(X_test_final)

y_test_pred = model.predict(X_test_final)
y_test_pred = target_scaler.inverse_transform(y_test_pred)

joblib.dump(target_scaler, f'{ticker}_target_scaler.gz')