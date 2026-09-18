"""
Forecasting experiments deliverable: training/execution time statistics
for all three models, measured properly rather than scavenged from
whatever timestamps happened to be left over from earlier tuning runs.

Process (stated explicitly, per the brief's own requirement to document
how timing was measured): each model's already-known winning
configuration per square (found during hyperparameter tuning - this
script does NOT re-tune anything) is retrained from scratch, back-to-
back, in this one clean run, so all three models are timed under the
same conditions in the same session rather than pieced together from
measurements taken at different points across the project, under
different machine load. Fit time and one-step-ahead predict time (over
the full 1,008-step test week) are both timed. Reported per model as the
average across the three squares, with the per-square range shown too,
consistent with the brief's allowance to report "an average across the
three areas, provided the method used is clearly stated."

Hardware: 8-core CPU, 17GB RAM, no GPU (see memory_benchmark.py /
WORK_LOG.md for where this was established).

Run: python src/build_forecast_timing.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from common import build_feature_table, build_lstm_model, fourier_features, make_windows

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")
SQUARES = [5161, 5059, 5259]

# Already-known winning configurations (found during tuning - not re-searched here)
SARIMA_ORDERS = {5161: (3, 0, 0), 5059: (3, 0, 0), 5259: (3, 0, 3)}
LSTM_CONFIGS = {
    5161: {"lookback": 72, "units": 32, "epochs": 50},
    5059: {"lookback": 72, "units": 64, "epochs": 37},
    5259: {"lookback": 144, "units": 64, "epochs": 43},
}
XGB_PARAMS = {
    5161: {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    5059: {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    5259: {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
}


def time_sarima(sq):
    train = pd.read_parquet(DATA_DIR / str(sq) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(sq) / "test.parquet")
    y_train, y_test = train["internet_traffic"].to_numpy(), test["internet_traffic"].to_numpy()
    x_train = fourier_features(np.arange(len(y_train)))
    x_test = fourier_features(np.arange(len(y_train), len(y_train) + len(y_test)))

    t0 = time.perf_counter()
    mod = SARIMAX(y_train, order=SARIMA_ORDERS[sq], exog=x_train,
                   enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_ext = res.append(y_test, exog=x_test, refit=False)
    res_ext.get_prediction(start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False)
    predict_time = time.perf_counter() - t0
    return fit_time, predict_time


def time_lstm(sq):
    cfg = LSTM_CONFIGS[sq]
    train = pd.read_parquet(DATA_DIR / str(sq) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(sq) / "test.parquet")
    y_train_scaled = train["internet_traffic_scaled"].to_numpy()
    y_test_scaled = test["internet_traffic_scaled"].to_numpy()
    X_fit, y_fit = make_windows(y_train_scaled, cfg["lookback"])

    model = build_lstm_model(cfg["lookback"], cfg["units"])

    t0 = time.perf_counter()
    model.fit(X_fit, y_fit, epochs=cfg["epochs"], batch_size=64, verbose=0)
    fit_time = time.perf_counter() - t0

    X_test, _ = make_windows(np.concatenate([y_train_scaled[-cfg["lookback"]:], y_test_scaled]), cfg["lookback"])
    t0 = time.perf_counter()
    model.predict(X_test, verbose=0)
    predict_time = time.perf_counter() - t0
    return fit_time, predict_time


def time_xgboost(sq):
    train = pd.read_parquet(DATA_DIR / str(sq) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(sq) / "test.parquet")
    full = pd.concat([train[["timestamp", "internet_traffic"]], test[["timestamp", "internet_traffic"]]])
    full = full.set_index("timestamp")["internet_traffic"].sort_index()
    feats = build_feature_table(full)
    feature_cols = [c for c in feats.columns if c != "y"]
    test_start = test["timestamp"].min()
    train_feats, test_feats = feats[feats.index < test_start], feats[feats.index >= test_start]

    t0 = time.perf_counter()
    model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **XGB_PARAMS[sq])
    model.fit(train_feats[feature_cols], train_feats["y"])
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.predict(test_feats[feature_cols])
    predict_time = time.perf_counter() - t0
    return fit_time, predict_time


def main():
    results = {"SARIMA": [], "LSTM": [], "XGBoost": []}
    timers = {"SARIMA": time_sarima, "LSTM": time_lstm, "XGBoost": time_xgboost}

    for model_name, timer_fn in timers.items():
        print(f"\n=== Timing {model_name} (fresh fit, known winning config, all 3 squares) ===")
        for sq in SQUARES:
            fit_t, predict_t = timer_fn(sq)
            results[model_name].append((fit_t, predict_t))
            print(f"  square {sq}: fit={fit_t:.3f}s  predict={predict_t:.3f}s")

    rows = {}
    for model_name, times in results.items():
        fit_times = [t[0] for t in times]
        predict_times = [t[1] for t in times]
        rows[model_name] = {
            "avg_fit_s": round(np.mean(fit_times), 2),
            "min_fit_s": round(np.min(fit_times), 2),
            "max_fit_s": round(np.max(fit_times), 2),
            "avg_predict_1008steps_s": round(np.mean(predict_times), 3),
        }

    summary = pd.DataFrame(rows).T
    summary.index.name = "Model"
    print("\n=== Final timing summary (fresh, controlled, single-session measurement) ===")
    print(summary.to_string())

    out_path = RESULTS_DIR / "forecast_timing_summary.csv"
    summary.to_csv(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
