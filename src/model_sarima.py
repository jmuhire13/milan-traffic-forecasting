"""
SARIMA model (ARIMA with Fourier-term seasonal regressors, instead of a
literal seasonal_order=144, for computational tractability - a literal
seasonal_order=144 was timed and found to take 67.9s to fit on just 1,000
points, which motivated this approach instead).

Fit on the ORIGINAL (unscaled) internet_traffic values, not the MinMax-
scaled column - ARIMA's linear structure is unaffected by scaling (a
scaled fit is just a linear reparameterisation of the same model), and
working in original units avoids an unnecessary inverse-transform step
before computing MAE/RMSE. LSTM will use the scaled series instead (next
model) since neural network training benefits from small input ranges;
this per-model difference is deliberate and will be documented as such.

One-step-ahead evaluation over the test week: the model is fit on
training data only, then `append(test, refit=False)` folds the true test
observations into the fitted state-space model WITHOUT re-estimating
parameters, and `get_prediction(..., dynamic=False)` then produces
genuine one-step-ahead forecasts - each prediction at time t uses true
observed data through t-1, matching the assignment's formal definition
(x_hat(t+1) from history x_t).

Run: python src/model_sarima.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DAILY_PERIOD = 144
WEEKLY_PERIOD = 1008
K_DAILY = 4   # number of daily harmonics
K_WEEKLY = 2  # number of weekly harmonics


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


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def run_one_square(square_id: int, order=(2, 0, 2)):
    train = pd.read_parquet(DATA_DIR / str(square_id) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(square_id) / "test.parquet")

    y_train = train["internet_traffic"].to_numpy()
    y_test = test["internet_traffic"].to_numpy()
    n_train, n_test = len(y_train), len(y_test)

    t_train = np.arange(n_train)
    t_test = np.arange(n_train, n_train + n_test)
    x_train = fourier_features(t_train)
    x_test = fourier_features(t_test)

    print(f"\n=== Square {square_id}: SARIMA (ARIMA{order} + Fourier terms, "
          f"{K_DAILY} daily / {K_WEEKLY} weekly harmonics) ===")

    t0 = time.perf_counter()
    mod = SARIMAX(y_train, order=order, exog=x_train,
                   enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    fit_time = time.perf_counter() - t0
    print(f"Fit time: {fit_time:.2f}s  (AIC={res.aic:.1f})")

    t0 = time.perf_counter()
    res_ext = res.append(y_test, exog=x_test, refit=False)
    pred = res_ext.get_prediction(start=n_train, end=n_train + n_test - 1, dynamic=False)
    yhat = pred.predicted_mean
    predict_time = time.perf_counter() - t0
    print(f"One-step-ahead prediction time over {n_test} test steps: {predict_time:.3f}s")

    metrics = {
        "square_id": square_id,
        "model": "SARIMA",
        "order": str(order),
        "k_daily": K_DAILY,
        "k_weekly": K_WEEKLY,
        "MAE": mae(y_test, yhat),
        "MAPE": mape(y_test, yhat),
        "RMSE": rmse(y_test, yhat),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "aic": float(res.aic),
    }
    print(f"MAE={metrics['MAE']:.2f}  MAPE={metrics['MAPE']:.2f}%  RMSE={metrics['RMSE']:.2f}")

    preds_df = pd.DataFrame({
        "timestamp": test["timestamp"],
        "actual": y_test,
        "predicted": yhat,
    })
    return metrics, preds_df


if __name__ == "__main__":
    # Sanity-check on one square first before running all three / tuning.
    metrics, preds_df = run_one_square(5161)
    print("\nFirst 5 predictions vs actual:")
    print(preds_df.head().to_string(index=False))
    out_path = RESULTS_DIR / "sarima_5161_sanity_check.csv"
    preds_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(json.dumps(metrics, indent=2))
