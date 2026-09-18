"""
LSTM model.

Input representation: sliding windows of the last `lookback` scaled
internet_traffic values predict the next single value - a direct,
standard sequence-to-one-step setup. Uses the MinMax-SCALED series (fit
on training data only, built by prepare_forecasting_data.py), unlike
SARIMA which used original units - neural network training is sensitive
to input scale, so this is a deliberate, model-specific choice,
documented as such.

`lookback=144` (one full day at 10-minute resolution) is not an arbitrary
default - it's chosen directly from the exploratory analysis's finding
that daily periodicity is the single strongest, most consistent structure
in this data (ACF ~0.88 at the 1-day lag), so giving the network a full day of
context as input is the natural starting point.

One-step-ahead evaluation, consistent with SARIMA's approach: predictions
over the test week use TRUE observed history (train tail + true test
values as they "arrive"), not the model's own earlier predictions -
built by windowing the concatenated (train + test) scaled series and
only scoring the windows whose target falls in the test week.

Run: python src/model_lstm.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent))
from common import build_lstm_model as build_model
from common import mae, make_windows, mape, rmse

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_STEPS = 1008  # same held-out validation slice used for SARIMA, for consistency


def run_one_square(square_id: int, lookback=144, units=64, epochs=50, batch_size=64, min_max=None):
    train = pd.read_parquet(DATA_DIR / str(square_id) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(square_id) / "test.parquet")

    y_train_scaled = train["internet_traffic_scaled"].to_numpy()
    y_test_scaled = test["internet_traffic_scaled"].to_numpy()
    y_test_orig = test["internet_traffic"].to_numpy()

    # inner train/val split (mirrors SARIMA's tuning split) for early stopping
    fit_scaled = y_train_scaled[:-VAL_STEPS]
    val_scaled = y_train_scaled[-VAL_STEPS:]

    X_fit, y_fit = make_windows(np.concatenate([fit_scaled]), lookback)
    # validation windows need fit-tail context to fill the first `lookback` targets
    X_val, y_val = make_windows(np.concatenate([fit_scaled[-lookback:], val_scaled]), lookback)

    print(f"\n=== Square {square_id}: LSTM (lookback={lookback}, units={units}) ===")
    print(f"  train windows: {X_fit.shape}, val windows: {X_val.shape}")

    model = build_model(lookback, units)
    t0 = time.perf_counter()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=epochs,
                      batch_size=batch_size, callbacks=[es], verbose=0)
    fit_time = time.perf_counter() - t0
    n_epochs_run = len(hist.history["loss"])
    best_val_loss = min(hist.history["val_loss"])
    print(f"  fit time: {fit_time:.1f}s over {n_epochs_run} epochs (early-stopped), "
          f"best val_loss={best_val_loss:.5f}")

    # Final one-step-ahead predictions over the TEST week, using TRUE history
    # (train tail + true test values), matching SARIMA's evaluation approach.
    full_scaled = np.concatenate([y_train_scaled[-lookback:], y_test_scaled])
    X_test, _ = make_windows(full_scaled, lookback)
    t0 = time.perf_counter()
    yhat_scaled = model.predict(X_test, verbose=0).flatten()
    predict_time = time.perf_counter() - t0

    lo, hi = min_max[str(square_id)]["min"], min_max[str(square_id)]["max"]
    yhat_orig = yhat_scaled * (hi - lo) + lo

    metrics = {
        "square_id": square_id, "model": "LSTM", "lookback": lookback, "units": units,
        "epochs_run": n_epochs_run, "best_val_loss": best_val_loss,
        "MAE": mae(y_test_orig, yhat_orig), "MAPE": mape(y_test_orig, yhat_orig),
        "RMSE": rmse(y_test_orig, yhat_orig), "fit_time_s": fit_time, "predict_time_s": predict_time,
    }
    print(f"  MAE={metrics['MAE']:.2f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.2f}")

    preds_df = pd.DataFrame({"timestamp": test["timestamp"], "actual": y_test_orig, "predicted": yhat_orig})
    return metrics, preds_df


if __name__ == "__main__":
    import json
    with open(DATA_DIR / "scalers.json") as f:
        min_max = json.load(f)

    metrics, preds_df = run_one_square(5161, min_max=min_max)
    out_path = RESULTS_DIR / "lstm_5161_sanity_check.csv"
    preds_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(json.dumps(metrics, indent=2))
