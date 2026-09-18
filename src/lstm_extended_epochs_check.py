"""
Follow-up check: were the LSTM winning configs actually done improving at
the 20-epoch cap, or was that cap cutting them off early?

Re-runs each square's winning (lookback, units) config through the SAME
leakage-safe validation procedure as the original tuning phase (train on
the fit slice, validate on the held-out internal slice - never the test
week), but with a much higher epoch cap and more patience, so early
stopping gets a real chance to trigger on its own rather than being
capped. If validation performance meaningfully improves, the new epoch
count gets carried into a corrected, still-leakage-free final refit (full
training data, fixed epochs, no validation split) exactly like
fix_lstm_final.py did before. If it doesn't improve, that's useful
evidence too - it means 20 epochs was already enough.

Run: python src/lstm_extended_epochs_check.py
"""
import json
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
tf.get_logger().setLevel("ERROR")

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")

WINNING_CONFIGS = {
    5161: {"lookback": 72, "units": 32},
    5059: {"lookback": 72, "units": 64},
    5259: {"lookback": 144, "units": 64},
}
VAL_STEPS = 1008
BATCH_SIZE = 64
EXTENDED_EPOCHS_CAP = 50
EXTENDED_PATIENCE = 6


def main():
    with open(DATA_DIR / "scalers.json") as f:
        min_max = json.load(f)

    old_final = pd.read_csv(RESULTS_DIR / "lstm_final_metrics.csv").set_index("square_id")
    updated_rows = []

    for sq, cfg in WINNING_CONFIGS.items():
        lookback, units = cfg["lookback"], cfg["units"]
        train = pd.read_parquet(DATA_DIR / str(sq) / "train.parquet")
        test = pd.read_parquet(DATA_DIR / str(sq) / "test.parquet")
        y_train_scaled = train["internet_traffic_scaled"].to_numpy()
        y_test_scaled = test["internet_traffic_scaled"].to_numpy()
        y_test_orig = test["internet_traffic"].to_numpy()

        fit_scaled = y_train_scaled[:-VAL_STEPS]
        val_scaled = y_train_scaled[-VAL_STEPS:]
        X_fit, y_fit = make_windows(fit_scaled, lookback)
        X_val, y_val = make_windows(np.concatenate([fit_scaled[-lookback:], val_scaled]), lookback)

        print(f"\n=== Square {sq}: extended-epoch validation check "
              f"(lookback={lookback}, units={units}, cap={EXTENDED_EPOCHS_CAP}, patience={EXTENDED_PATIENCE}) ===")
        model = build_model(lookback, units)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=EXTENDED_PATIENCE, restore_best_weights=True)
        hist = model.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=EXTENDED_EPOCHS_CAP,
                          batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
        n_epochs = len(hist.history["loss"])
        lo, hi = min_max[str(sq)]["min"], min_max[str(sq)]["max"]
        yhat_val = model.predict(X_val, verbose=0).flatten() * (hi - lo) + lo
        y_val_orig = y_val * (hi - lo) + lo
        new_val_rmse = rmse(y_val_orig, yhat_val)

        old_tuning_log = pd.read_csv(RESULTS_DIR / f"lstm_tuning_log_{sq}.csv")
        old_row = old_tuning_log[(old_tuning_log.lookback == lookback) & (old_tuning_log.units == units)].iloc[0]
        old_val_rmse = old_row["val_RMSE"]

        print(f"  stopped after {n_epochs} epochs (cap was {EXTENDED_EPOCHS_CAP}) "
              f"{'[early-stopped before cap]' if n_epochs < EXTENDED_EPOCHS_CAP else '[hit the cap again]'}")
        print(f"  val_RMSE: old(cap=20)={old_val_rmse:.2f} -> new(cap=50)={new_val_rmse:.2f}  "
              f"({'IMPROVED' if new_val_rmse < old_val_rmse - 1 else 'no meaningful change'})")

        if new_val_rmse < old_val_rmse - 1:
            # Meaningful improvement -> redo the leakage-free final refit with this epoch count.
            print(f"  -> re-running leakage-free final refit with {n_epochs} epochs...")
            X_full, y_full = make_windows(y_train_scaled, lookback)
            final_model = build_model(lookback, units)
            t0 = time.perf_counter()
            final_model.fit(X_full, y_full, epochs=n_epochs, batch_size=BATCH_SIZE, verbose=0)
            fit_time = time.perf_counter() - t0
            X_test, _ = make_windows(np.concatenate([y_train_scaled[-lookback:], y_test_scaled]), lookback)
            yhat_test = final_model.predict(X_test, verbose=0).flatten() * (hi - lo) + lo
            new_metrics = {
                "square_id": sq, "model": "LSTM", "lookback": lookback, "units": units, "epochs_run": n_epochs,
                "MAE": mae(y_test_orig, yhat_test), "MAPE": mape(y_test_orig, yhat_test),
                "RMSE": rmse(y_test_orig, yhat_test), "fit_time_s": fit_time,
            }
            print(f"  UPDATED final test metrics: MAE={new_metrics['MAE']:.2f} "
                  f"MAPE={new_metrics['MAPE']:.2f}% RMSE={new_metrics['RMSE']:.2f}")
            preds_df = pd.DataFrame({"timestamp": test["timestamp"], "actual": y_test_orig, "predicted": yhat_test})
            preds_df.to_csv(RESULTS_DIR / f"lstm_predictions_{sq}.csv", index=False)
            updated_rows.append(new_metrics)
        else:
            print(f"  -> no meaningful improvement, keeping existing 20-epoch result "
                  f"(RMSE={old_final.loc[sq, 'RMSE']:.2f}) unchanged.")
            row = old_final.loc[sq].to_dict()
            row["square_id"] = sq
            updated_rows.append(row)

    summary = pd.DataFrame(updated_rows)
    summary.to_csv(RESULTS_DIR / "lstm_final_metrics.csv", index=False)
    print(f"\n{'=' * 70}\nFinal LSTM metrics after extended-epoch check:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
