"""
Fix for a test-set leakage bug found in tune_lstm.py's "final refit" step:
that step passed the real Dec 16-22 test week in as `validation_data` to
model.fit(), with EarlyStopping(restore_best_weights=True) - meaning the
model's final weights were chosen to minimize loss ON THE TEST SET
ITSELF. That's leakage: the reported LSTM test metrics were optimistically
biased, not a fair blind evaluation.

Correct approach: the grid-search/tuning phase itself was NOT leaky (it
validated against an internal slice of training data, never the test
week) - so the chosen (lookback, units) per square, and the epoch count
that config trained for under legitimate validation, are both trustworthy.
This script reuses those, but retrains the final model on the FULL
training set for that FIXED number of epochs, with NO validation split
and NO early-stopping callback at all in this run - so no data outside
the training set (validation or test) ever influences which weights are
kept. Prediction on the test week is then a pure inference step on an
already-fixed model.

Run: python src/fix_lstm_final.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from common import build_lstm_model as build_model
from common import mae, make_windows, mape, rmse

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")

# From the LEGITIMATE tuning-phase logs (validated against internal val split only)
WINNING_CONFIGS = {
    5161: {"lookback": 72, "units": 32, "epochs": 20},
    5059: {"lookback": 72, "units": 64, "epochs": 20},
    5259: {"lookback": 144, "units": 64, "epochs": 20},
}
BATCH_SIZE = 64


def main():
    with open(DATA_DIR / "scalers.json") as f:
        min_max = json.load(f)

    all_metrics = []
    for sq, cfg in WINNING_CONFIGS.items():
        lookback, units, n_epochs = cfg["lookback"], cfg["units"], cfg["epochs"]
        train = pd.read_parquet(DATA_DIR / str(sq) / "train.parquet")
        test = pd.read_parquet(DATA_DIR / str(sq) / "test.parquet")
        y_train_scaled = train["internet_traffic_scaled"].to_numpy()
        y_test_scaled = test["internet_traffic_scaled"].to_numpy()
        y_test_orig = test["internet_traffic"].to_numpy()

        print(f"\n=== Square {sq}: CORRECTED final LSTM (lookback={lookback}, units={units}, "
              f"fixed {n_epochs} epochs, no validation data used in this training run) ===")

        X_fit, y_fit = make_windows(y_train_scaled, lookback)
        model = build_model(lookback, units)
        t0 = time.perf_counter()
        model.fit(X_fit, y_fit, epochs=n_epochs, batch_size=BATCH_SIZE, verbose=0)  # no validation_data at all
        fit_time = time.perf_counter() - t0

        # Pure inference on the test week - true history context, no further training
        X_test, _ = make_windows(np.concatenate([y_train_scaled[-lookback:], y_test_scaled]), lookback)
        t0 = time.perf_counter()
        yhat_scaled = model.predict(X_test, verbose=0).flatten()
        predict_time = time.perf_counter() - t0

        lo, hi = min_max[str(sq)]["min"], min_max[str(sq)]["max"]
        yhat_orig = yhat_scaled * (hi - lo) + lo

        metrics = {
            "square_id": sq, "model": "LSTM", "lookback": lookback, "units": units, "epochs_run": n_epochs,
            "MAE": mae(y_test_orig, yhat_orig), "MAPE": mape(y_test_orig, yhat_orig),
            "RMSE": rmse(y_test_orig, yhat_orig), "fit_time_s": fit_time, "predict_time_s": predict_time,
        }
        print(f"  CORRECTED: MAE={metrics['MAE']:.2f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.2f}  "
              f"(fit_time={fit_time:.1f}s)")
        all_metrics.append(metrics)

        preds_df = pd.DataFrame({"timestamp": test["timestamp"], "actual": y_test_orig, "predicted": yhat_orig})
        preds_df.to_csv(RESULTS_DIR / f"lstm_predictions_{sq}.csv", index=False)  # overwrite the leaky version

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(RESULTS_DIR / "lstm_final_metrics.csv", index=False)  # overwrite the leaky version
    print(f"\n{'=' * 70}\nCorrected final LSTM metrics, all squares:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
