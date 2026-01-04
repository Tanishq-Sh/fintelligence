import os

RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_CONFIG = {
    "NVDA": {
        "lookback_window": 60,
        "train_test_split": 0.8,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "lstm_layer_1_units": 128,
        "lstm_layer_2_units": 64,
        "output_linear_layer": 1,
    },
    "AAPL": {
        "lookback_window": 60,
        "train_test_split": 0.8,
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "lstm_layer_1_units": 128,
        "lstm_layer_2_units": 64,
        "output_linear_layer": 1,
    },
    "TSLA": {
        "lookback_window": 60,
        "train_test_split": 0.8,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "lstm_layer_1_units": 128,
        "lstm_layer_2_units": 64,
        "output_linear_layer": 1,
    },
    "DEFAULT": {
        "lookback_window": 60,
        "train_test_split": 0.8,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "lstm_layer_1_units": 128,
        "lstm_layer_2_units": 64,
        "output_linear_layer": 1,
    }
}