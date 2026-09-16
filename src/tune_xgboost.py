"""
Stage 4 - XGBoost model: systematic hyperparameter tuning via Grid Search
(explicitly suggested by the brief for cases not requiring manual
experimentation - XGBoost trains fast enough on this data that a real
grid is practical, unlike LSTM).

Input representation: turns the forecasting problem into tabular
regression using engineered features, not a raw sequence - a genuinely
different paradigm from SARIMA and LSTM (see Stage 3 justification):
- lag_1, lag_2, lag_3: short-term persistence (SARIMA's tuning found
  AR terms dominate; these give XGBoost the same short-term signal
  directly as features)
- lag_144: same time yesterday (the single strongest structure found in
  Stage 2, ACF ~0.88 at 1-day lag)
- lag_1008: same time last week (Stage 2 found this beats the 2-3 day
  lags - a genuine weekly effect, not just decay)
- ten_min_of_day (0-143) and day_of_week (0-6): explicit calendar
  features, given Stage 2's weekday/weekend behavioural split

Uses ORIGINAL units (like SARIMA, unlike LSTM) - tree splits are scale-
invariant, so scaling would change nothing about performance, and raw
units keep the engineered features directly interpretable.

Because lag_1008 needs a full week of prior history, the first 1,008 rows
of the entire per-square series (which starts right at the beginning of
the dataset, Nov 1) can't have that feature computed and are dropped -
disclosed, not hidden.

Same validation methodology as SARIMA/LSTM: grid-searched on a held-out
slice of training data shaped like the test week; winning config refit on
full usable training data; evaluated once on the real, untouched test
week.

Run: python src/tune_xgboost.py
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DATA_DIR = Path("data/processed/stage4")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SQUARES = [5161, 5059, 5259]
LAGS = [1, 2, 3, 144, 1008]
VAL_STEPS = 1008

PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
}


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def build_feature_table(full_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": full_series.values}, index=full_series.index)
    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["ten_min_of_day"] = (df.index.hour * 6 + df.index.minute // 10)
    df["day_of_week"] = df.index.dayofweek
    df = df.dropna()  # drops the first max(LAGS) rows where lags aren't available yet
    return df


def load_full_series(square_id: int) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    train = pd.read_parquet(DATA_DIR / str(square_id) / "train.parquet")
    test = pd.read_parquet(DATA_DIR / str(square_id) / "test.parquet")
    full = pd.concat([train[["timestamp", "internet_traffic"]], test[["timestamp", "internet_traffic"]]])
    s = full.set_index("timestamp")["internet_traffic"].sort_index()
    return s, train["timestamp"].min(), test["timestamp"].min()


def fit_eval(X_fit, y_fit, X_eval, y_eval, params):
    model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **params)
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    yhat = model.predict(X_eval)
    predict_time = time.perf_counter() - t0
    return model, yhat, fit_time, predict_time


def tune_one_square(square_id: int):
    full_series, train_start, test_start = load_full_series(square_id)
    feats = build_feature_table(full_series)
    feature_cols = [c for c in feats.columns if c != "y"]

    train_feats = feats[feats.index < test_start]
    test_feats = feats[feats.index >= test_start]
    fit_feats = train_feats.iloc[:-VAL_STEPS]
    val_feats = train_feats.iloc[-VAL_STEPS:]

    print(f"\n{'=' * 70}\nSquare {square_id}: XGBoost grid search "
          f"({len(list(itertools.product(*PARAM_GRID.values())))} configs)")
    print(f"  usable rows after dropping first {max(LAGS)} (no lag_1008 history): "
          f"fit={len(fit_feats)}, val={len(val_feats)}, test={len(test_feats)}")

    experiments = []
    keys = list(PARAM_GRID.keys())
    for combo in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(keys, combo))
        _, yhat, fit_time, _ = fit_eval(
            fit_feats[feature_cols], fit_feats["y"], val_feats[feature_cols], val_feats["y"], params
        )
        m = {**params, "val_MAE": mae(val_feats["y"], yhat), "val_RMSE": rmse(val_feats["y"], yhat),
             "fit_time_s": fit_time}
        experiments.append(m)
        print(f"  {params}: val_RMSE={m['val_RMSE']:.2f}  val_MAE={m['val_MAE']:.2f}  fit_time={fit_time:.2f}s")

    exp_df = pd.DataFrame(experiments)
    best = exp_df.loc[exp_df["val_RMSE"].idxmin()]
    best_params = {k: (int(best[k]) if k != "learning_rate" else float(best[k])) for k in keys}
    print(f"\nBest on validation: {best_params}  val_RMSE={best['val_RMSE']:.2f}")

    # Refit on full usable training data, evaluate on the real test week.
    model, yhat_test, final_fit_time, predict_time = fit_eval(
        train_feats[feature_cols], train_feats["y"], test_feats[feature_cols], test_feats["y"], best_params
    )
    final_metrics = {
        "square_id": square_id, "model": "XGBoost", **best_params,
        "MAE": mae(test_feats["y"], yhat_test), "MAPE": mape(test_feats["y"], yhat_test),
        "RMSE": rmse(test_feats["y"], yhat_test), "fit_time_s": final_fit_time, "predict_time_s": predict_time,
    }
    print(f"Final (refit on full usable train, evaluated on real Dec 16-22 test week): "
          f"MAE={final_metrics['MAE']:.2f}  MAPE={final_metrics['MAPE']:.2f}%  RMSE={final_metrics['RMSE']:.2f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Feature importances:")
    print(importances.to_string())

    exp_df.to_csv(RESULTS_DIR / f"xgboost_tuning_log_{square_id}.csv", index=False)
    preds_df = pd.DataFrame({"timestamp": test_feats.index, "actual": test_feats["y"].values, "predicted": yhat_test})
    preds_df.to_csv(RESULTS_DIR / f"xgboost_predictions_{square_id}.csv", index=False)
    importances.to_csv(RESULTS_DIR / f"xgboost_feature_importance_{square_id}.csv")
    return final_metrics


if __name__ == "__main__":
    all_metrics = [tune_one_square(sq) for sq in SQUARES]
    summary = pd.DataFrame(all_metrics)
    summary.to_csv(RESULTS_DIR / "xgboost_final_metrics.csv", index=False)
    print(f"\n{'=' * 70}\nFinal XGBoost metrics, all squares:")
    print(summary.to_string(index=False))
