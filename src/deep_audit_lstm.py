"""
Deep audit of the LSTM results - same structural/numerical checks as the
SARIMA audit, plus LSTM-specific ones: that the currently-saved files are
genuinely the corrected (leakage-free, extended-epoch) version and not a
stale leftover from the buggy run, and that the scaled-to-original-units
inverse transform used the right per-square scaler (a plausible mix-up
risk since all three squares' scalers are similar-looking small dicts).

Run: python src/deep_audit_lstm.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/tables")
DATA_DIR = Path("data/processed/stage4")
SQUARES = [5161, 5059, 5259]

# From the extended-epoch follow-up (the final, correct results)
EXPECTED_RMSE_AFTER_FIX = {5161: 126.61, 5059: 100.00, 5259: 97.04}

checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status, detail))
    print(f"[{status}] {name}  {detail}")


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    reported = pd.read_csv(RESULTS_DIR / "lstm_final_metrics.csv").set_index("square_id")
    with open(DATA_DIR / "scalers.json") as f:
        scalers = json.load(f)

    for sq in SQUARES:
        print(f"\n=== Square {sq} ===")
        df = pd.read_csv(RESULTS_DIR / f"lstm_predictions_{sq}.csv", parse_dates=["timestamp"])

        check(f"[{sq}] exactly 1008 rows", len(df) == 1008, f"actual={len(df)}")
        check(f"[{sq}] no NaN/Inf in predictions", np.isfinite(df["predicted"]).all(),
              f"n_bad={(~np.isfinite(df['predicted'])).sum()}")
        expected_index = pd.date_range("2013-12-16 00:00:00", "2013-12-22 23:50:00", freq="10min")
        check(f"[{sq}] timestamps exactly match Dec16-22 grid, in order, no dupes",
              list(df["timestamp"]) == list(expected_index),
              f"first={df['timestamp'].iloc[0]}, last={df['timestamp'].iloc[-1]}")

        recomputed = {"MAE": mae(df["actual"], df["predicted"]), "MAPE": mape(df["actual"], df["predicted"]),
                      "RMSE": rmse(df["actual"], df["predicted"])}
        rep = reported.loc[sq]
        for metric in ["MAE", "MAPE", "RMSE"]:
            match = abs(recomputed[metric] - rep[metric]) < 0.01
            check(f"[{sq}] reported {metric} matches independent recomputation from saved predictions",
                  match, f"reported={rep[metric]:.3f}, recomputed={recomputed[metric]:.3f}")

        # LSTM-specific: confirm the file on disk is the CORRECTED (post-fix,
        # post-extended-epoch) version, not a stale leaky leftover.
        check(f"[{sq}] saved file matches the known-correct post-fix RMSE (not a stale/leaky version)",
              abs(recomputed["RMSE"] - EXPECTED_RMSE_AFTER_FIX[sq]) < 0.5,
              f"expected~={EXPECTED_RMSE_AFTER_FIX[sq]}, got={recomputed['RMSE']:.2f}")

        # LSTM-specific: verify the inverse-scaling used the correct per-square
        # scaler, not a mixed-up one - actual/predicted values should sit in a
        # sane range relative to this square's own known training range.
        train = pd.read_parquet(DATA_DIR / str(sq) / "train.parquet")
        train_min, train_max = train["internet_traffic"].min(), train["internet_traffic"].max()
        pred_min, pred_max = df["predicted"].min(), df["predicted"].max()
        sane_range = (pred_min > -0.2 * train_max) and (pred_max < 1.5 * train_max)
        check(f"[{sq}] predicted values fall in a sane range for this square's own scaler "
              f"(not mixed up with another square's min/max)",
              sane_range, f"train_range=[{train_min:.1f},{train_max:.1f}], "
                          f"predicted_range=[{pred_min:.1f},{pred_max:.1f}]")

        # Same one-step-ahead amplitude-retention check as SARIMA's audit.
        first_2d, last_2d = df.iloc[:288], df.iloc[-288:]
        amp_ratio_first = first_2d["predicted"].std() / first_2d["actual"].std()
        amp_ratio_last = last_2d["predicted"].std() / last_2d["actual"].std()
        print(f"  predicted/actual amplitude ratio: first 2 days={amp_ratio_first:.3f}, "
              f"last 2 days={amp_ratio_last:.3f}")
        check(f"[{sq}] no severe amplitude collapse late in the week (multi-step-forecast red flag)",
              amp_ratio_last > 0.7, f"ratio={amp_ratio_last:.3f}")

    print("\n=== Summary ===")
    n_pass = sum(1 for _, s, _ in checks if s == "PASS")
    print(f"{n_pass}/{len(checks)} checks passed")
    if n_pass != len(checks):
        for name, s, d in checks:
            if s == "FAIL":
                print(f"  FAIL: {name}: {d}")


if __name__ == "__main__":
    main()
