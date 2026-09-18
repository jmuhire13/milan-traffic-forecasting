"""
Shared constants and helper functions reused across the model scripts
(SARIMA, LSTM, XGBoost - training, tuning, and audit scripts alike), so
each piece of logic exists in exactly one place instead of being
copy-pasted per script. Every function here is a straight extraction of
code that was previously duplicated identically (or only cosmetically
differently) across up to 10 separate files - this is a pure
reorganization, not a behavior change; every script that imports from
here was re-verified afterward to still produce identical results.

Import what you need, e.g.:
    from common import mae, mape, rmse, fourier_features, make_windows, build_lstm_model
"""
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

# --- Evaluation metrics, shared by every model/audit script ---------------

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


# --- SARIMA: Fourier-term seasonal regressors ------------------------------

DAILY_PERIOD = 144   # 10-minute steps in a day
WEEKLY_PERIOD = 1008  # 10-minute steps in a week
K_DAILY = 4    # number of daily harmonics
K_WEEKLY = 2   # number of weekly harmonics


def fourier_features(t: np.ndarray) -> np.ndarray:
    """t = absolute step index (continuous across train/test boundary)."""
    cols = []
    for k in range(1, K_DAILY + 1):
        cols.append(np.sin(2 * np.pi * k * t / DAILY_PERIOD))
        cols.append(np.cos(2 * np.pi * k * t / DAILY_PERIOD))
    for k in range(1, K_WEEKLY + 1):
        cols.append(np.sin(2 * np.pi * k * t / WEEKLY_PERIOD))
        cols.append(np.cos(2 * np.pi * k * t / WEEKLY_PERIOD))
    return np.column_stack(cols)


# --- LSTM: windowing and model architecture --------------------------------

def make_windows(series: np.ndarray, lookback: int):
    """series -> (X, y) where X[i] = series[i:i+lookback], y[i] = series[i+lookback]."""
    n = len(series) - lookback
    X = np.zeros((n, lookback, 1), dtype="float32")
    y = np.zeros(n, dtype="float32")
    for i in range(n):
        X[i, :, 0] = series[i:i + lookback]
        y[i] = series[i + lookback]
    return X, y


def build_lstm_model(lookback: int, units: int):
    """Single LSTM layer + Dense(1) output, seeded for reproducibility."""
    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# --- XGBoost: lag/calendar feature engineering -----------------------------

LAGS = [1, 2, 3, 144, 1008]


def build_feature_table(full_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": full_series.values}, index=full_series.index)
    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["ten_min_of_day"] = df.index.hour * 6 + df.index.minute // 10
    df["day_of_week"] = df.index.dayofweek
    return df.dropna()  # drops the first max(LAGS) rows where lags aren't available yet
