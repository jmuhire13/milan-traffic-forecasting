"""
Systematic LSTM hyperparameter tuning (deliberately small grid - an
earlier single-run timing check showed a single LSTM fit costing several
times a single SARIMA fit on this CPU-only machine, motivating a lean,
well-reasoned grid rather than an exhaustive one).

Grid: lookback in {72, 144} (half a day vs. a full day of context),
units in {32, 64} (network size / compute cost per epoch) - 4
configurations. Epoch cap reduced to 20 (from the sanity check's 50) with
earlier-triggering patience, to bound worst-case cost while still letting
genuinely-still-improving configs train long enough.

Same methodology as SARIMA's tuning, for consistency: each config is
scored on a validation slice (last 1,008 training steps) that mirrors the
real test week but is never the real test week; the winning config per
square is then refit on the full training data before the one real
evaluation against the untouched Dec 16-22 test week.

Run: python src/tune_lstm.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SQUARES = [5161, 5059, 5259]
LOOKBACK_GRID = [72, 144]
UNITS_GRID = [32, 64]
VAL_STEPS = 1008
EPOCHS_CAP = 20
PATIENCE = 3
BATCH_SIZE = 64


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def make_windows(series: np.ndarray, lookback: int):
    n = len(series) - lookback
    X = np.zeros((n, lookback, 1), dtype="float32")
    y = np.zeros(n, dtype="float32")
    for i in range(n):
        X[i, :, 0] = series[i:i + lookback]
        y[i] = series[i + lookback]
    return X, y


def build_model(lookback: int, units: int):
    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(units),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_and_eval(fit_scaled, eval_scaled, lookback, units):
    X_fit, y_fit = make_windows(fit_scaled, lookback)
    X_eval, y_eval = make_windows(np.concatenate([fit_scaled[-lookback:], eval_scaled]), lookback)
    model = build_model(lookback, units)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    t0 = time.perf_counter()
    hist = model.fit(X_fit, y_fit, validation_data=(X_eval, y_eval), epochs=EPOCHS_CAP,
                      batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    fit_time = time.perf_counter() - t0
    yhat = model.predict(X_eval, verbose=0).flatten()
    return model, yhat, y_eval, fit_time, len(hist.history["loss"]), min(hist.history["val_loss"])


def tune_one_square(square_id: int, min_max: dict):
    train = pd.read_parquet(DATA_DIR / str(square_id) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(square_id) / "test.parquet")
    y_train_scaled = train["internet_traffic_scaled"].to_numpy()
    y_test_scaled = test["internet_traffic_scaled"].to_numpy()
    y_test_orig = test["internet_traffic"].to_numpy()

    fit_scaled = y_train_scaled[:-VAL_STEPS]
    val_scaled = y_train_scaled[-VAL_STEPS:]

    print(f"\n{'=' * 70}\nSquare {square_id}: LSTM grid search "
          f"({len(LOOKBACK_GRID)}x{len(UNITS_GRID)} configs)")

    experiments = []
    for lookback in LOOKBACK_GRID:
        for units in UNITS_GRID:
            t0 = time.perf_counter()
            _, yhat_scaled, y_val_scaled, fit_time, n_epochs, best_val_loss = fit_and_eval(
                fit_scaled, val_scaled, lookback, units
            )
            lo, hi = min_max[str(square_id)]["min"], min_max[str(square_id)]["max"]
            yhat_orig = yhat_scaled * (hi - lo) + lo
            y_val_orig = y_val_scaled * (hi - lo) + lo
            m = {
                "lookback": lookback, "units": units, "epochs_run": n_epochs,
                "best_val_loss": best_val_loss, "val_MAE": mae(y_val_orig, yhat_orig),
                "val_RMSE": rmse(y_val_orig, yhat_orig), "fit_time_s": fit_time,
            }
            experiments.append(m)
            print(f"  lookback={lookback:3d} units={units:2d}: val_RMSE={m['val_RMSE']:.2f}  "
                  f"val_MAE={m['val_MAE']:.2f}  epochs_run={n_epochs}  fit_time={fit_time:.1f}s")

    exp_df = pd.DataFrame(experiments)
    best = exp_df.loc[exp_df["val_RMSE"].idxmin()]
    best_lookback, best_units = int(best["lookback"]), int(best["units"])
    print(f"\nBest on validation: lookback={best_lookback} units={best_units}  "
          f"val_RMSE={best['val_RMSE']:.2f}")

    # Refit winning config on FULL training data, evaluate on the real test week.
    _, yhat_test_scaled, y_test_scaled_check, final_fit_time, final_epochs, final_val_loss = fit_and_eval(
        y_train_scaled, y_test_scaled, best_lookback, best_units
    )
    lo, hi = min_max[str(square_id)]["min"], min_max[str(square_id)]["max"]
    yhat_test_orig = yhat_test_scaled * (hi - lo) + lo

    final_metrics = {
        "square_id": square_id, "model": "LSTM", "lookback": best_lookback, "units": best_units,
        "epochs_run": final_epochs,
        "MAE": mae(y_test_orig, yhat_test_orig), "MAPE": mape(y_test_orig, yhat_test_orig),
        "RMSE": rmse(y_test_orig, yhat_test_orig), "fit_time_s": final_fit_time,
    }
    print(f"Final (refit on full train, evaluated on real Dec 16-22 test week): "
          f"MAE={final_metrics['MAE']:.2f}  MAPE={final_metrics['MAPE']:.2f}%  RMSE={final_metrics['RMSE']:.2f}")

    exp_df.to_csv(RESULTS_DIR / f"lstm_tuning_log_{square_id}.csv", index=False)
    preds_df = pd.DataFrame({"timestamp": test["timestamp"], "actual": y_test_orig, "predicted": yhat_test_orig})
    preds_df.to_csv(RESULTS_DIR / f"lstm_predictions_{square_id}.csv", index=False)
    return final_metrics


if __name__ == "__main__":
    with open(DATA_DIR / "scalers.json") as f:
        min_max = json.load(f)

    all_metrics = [tune_one_square(sq, min_max) for sq in SQUARES]
    summary = pd.DataFrame(all_metrics)
    summary.to_csv(RESULTS_DIR / "lstm_final_metrics.csv", index=False)
    print(f"\n{'=' * 70}\nFinal LSTM metrics, all squares:")
    print(summary.to_string(index=False))
