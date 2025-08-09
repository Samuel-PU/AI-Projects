"""
models.py
---------
Keras models for univariate forecasting.
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

def build_lstm(win: int, units: int = 64, dropout: float = 0.2):
    model = Sequential([
        LSTM(units, input_shape=(win, 1)),
        Dropout(dropout),
        Dense(1)
    ])
    return model

def build_gru(win: int, units: int = 64, dropout: float = 0.2):
    model = Sequential([
        GRU(units, input_shape=(win, 1)),
        Dropout(dropout),
        Dense(1)
    ])
    return model
