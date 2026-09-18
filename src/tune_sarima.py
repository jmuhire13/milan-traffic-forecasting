"""
Systematic SARIMA (ARIMA + Fourier) hyperparameter tuning.

Grid search over ARIMA(p, 0, q) orders - d fixed at 0, justified by the
exploratory analysis's ADF test already confirming the series is
stationary, so this degree of freedom doesn't need to be searched
blindly. Fourier harmonics (4 daily, 2 weekly) are kept fixed at the
values validated in the sanity check, for the same reason - the ACF
analysis already told us how much daily/weekly structure exists.

IMPORTANT methodological point: tuning must NOT touch the Dec 16-22 test
week at all, or model selection would be indirectly fit to the exact data
it's later "evaluated" on. So each square's training data (Nov 1 - Dec 15)
is itself split into an inner train_fit (Nov 1 - Dec 8) and a validation
slice shaped exactly like the real test week (Dec 9-15, 1008 steps). Best
config is chosen by validation performance only; the winning order is then
refit on the FULL training data before the one real evaluation against
the untouched test week.

Run: python src/tune_sarima.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).parent))
from common import fourier_features, mae, mape, rmse

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SQUARES = [5161, 5059, 5259]
P_GRID = [0, 1, 2, 3]
Q_GRID = [0, 1, 2, 3]
VAL_STEPS = 1008  # last 7 days of the training period, held out for tuning only


def one_step_ahead(y_fit, x_fit, y_eval, x_eval, order):
    mod = SARIMAX(y_fit, order=order, exog=x_fit,
                   enforce_stationarity=False, enforce_invertibility=False)
    t0 = time.perf_counter()
    res = mod.fit(disp=False)
    fit_time = time.perf_counter() - t0
    res_ext = res.append(y_eval, exog=x_eval, refit=False)
    pred = res_ext.get_prediction(start=len(y_fit), end=len(y_fit) + len(y_eval) - 1, dynamic=False)
    yhat = pred.predicted_mean
    return yhat, fit_time, res.aic


def tune_one_square(square_id: int):
    train = pd.read_parquet(DATA_DIR / str(square_id) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(square_id) / "test.parquet")
    y_train_full = train["internet_traffic"].to_numpy()
    y_test = test["internet_traffic"].to_numpy()

    y_fit = y_train_full[:-VAL_STEPS]
    y_val = y_train_full[-VAL_STEPS:]
    t_fit = np.arange(len(y_fit))
    t_val = np.arange(len(y_fit), len(y_fit) + len(y_val))
    x_fit, x_val = fourier_features(t_fit), fourier_features(t_val)

    print(f"\n{'=' * 70}\nSquare {square_id}: tuning ARIMA(p,0,q) via {len(P_GRID)}x{len(Q_GRID)} grid "
          f"on validation slice ({len(y_fit)} fit / {len(y_val)} val steps)")

    experiments = []
    for p in P_GRID:
        for q in Q_GRID:
            if p == 0 and q == 0:
                continue  # no autoregressive structure at all - not worth trying
            order = (p, 0, q)
            try:
                yhat, fit_time, aic = one_step_ahead(y_fit, x_fit, y_val, x_val, order)
                m = {"order": str(order), "val_MAE": mae(y_val, yhat), "val_RMSE": rmse(y_val, yhat),
                     "aic": float(aic), "fit_time_s": fit_time, "status": "ok"}
            except Exception as e:
                m = {"order": str(order), "val_MAE": None, "val_RMSE": None,
                     "aic": None, "fit_time_s": None, "status": f"failed: {e}"}
            experiments.append(m)
            print(f"  ARIMA{order}: val_RMSE={m['val_RMSE']}  val_MAE={m['val_MAE']}  "
                  f"aic={m['aic']}  fit_time={m['fit_time_s']}  [{m['status']}]")

    exp_df = pd.DataFrame(experiments)
    ok = exp_df[exp_df["status"] == "ok"].copy()
    best_row = ok.loc[ok["val_RMSE"].idxmin()]
    best_order = eval(best_row["order"])
    print(f"\nBest on validation: ARIMA{best_order}  val_RMSE={best_row['val_RMSE']:.2f}  "
          f"val_MAE={best_row['val_MAE']:.2f}")

    # Refit the winning order on the FULL training set, evaluate on the real,
    # untouched test week.
    t_train_full = np.arange(len(y_train_full))
    t_test = np.arange(len(y_train_full), len(y_train_full) + len(y_test))
    x_train_full, x_test = fourier_features(t_train_full), fourier_features(t_test)
    yhat_test, final_fit_time, final_aic = one_step_ahead(
        y_train_full, x_train_full, y_test, x_test, best_order
    )
    final_metrics = {
        "square_id": square_id, "model": "SARIMA", "order": str(best_order),
        "MAE": mae(y_test, yhat_test), "MAPE": mape(y_test, yhat_test), "RMSE": rmse(y_test, yhat_test),
        "fit_time_s": final_fit_time, "aic": float(final_aic),
    }
    print(f"Final (refit on full train, evaluated on real Dec 16-22 test week): "
          f"MAE={final_metrics['MAE']:.2f}  MAPE={final_metrics['MAPE']:.2f}%  RMSE={final_metrics['RMSE']:.2f}")

    exp_df.to_csv(RESULTS_DIR / f"sarima_tuning_log_{square_id}.csv", index=False)
    preds_df = pd.DataFrame({"timestamp": test["timestamp"], "actual": y_test, "predicted": yhat_test})
    preds_df.to_csv(RESULTS_DIR / f"sarima_predictions_{square_id}.csv", index=False)
    return final_metrics, exp_df


if __name__ == "__main__":
    all_metrics = []
    for sq in SQUARES:
        m, _ = tune_one_square(sq)
        all_metrics.append(m)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(RESULTS_DIR / "sarima_final_metrics.csv", index=False)
    print(f"\n{'=' * 70}\nFinal SARIMA metrics, all squares:")
    print(summary.to_string(index=False))
