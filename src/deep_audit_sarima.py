"""
Deep audit of the SARIMA results, beyond the leakage check already done -
verifying the saved predictions are structurally sound and genuinely
behave like one-step-ahead forecasts, not an accidental multi-step/
"dynamic" forecast that would look plausible but be answering a different
question than the assignment asks.

Run: python src/deep_audit_sarima.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/tables")
SQUARES = [5161, 5059, 5259]

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
    reported = pd.read_csv(RESULTS_DIR / "sarima_final_metrics.csv").set_index("square_id")

    for sq in SQUARES:
        print(f"\n=== Square {sq} ===")
        df = pd.read_csv(RESULTS_DIR / f"sarima_predictions_{sq}.csv", parse_dates=["timestamp"])

        check(f"[{sq}] exactly 1008 rows", len(df) == 1008, f"actual={len(df)}")
        check(f"[{sq}] no NaN/Inf in predictions", np.isfinite(df["predicted"]).all(),
              f"n_bad={(~np.isfinite(df['predicted'])).sum()}")
        expected_index = pd.date_range("2013-12-16 00:00:00", "2013-12-22 23:50:00", freq="10min")
        check(f"[{sq}] timestamps exactly match Dec16-22 grid, in order, no dupes",
              list(df["timestamp"]) == list(expected_index),
              f"first={df['timestamp'].iloc[0]}, last={df['timestamp'].iloc[-1]}")

        # Recompute metrics independently from the saved actual/predicted columns
        # and compare against what was reported/logged, to catch any transcription
        # mismatch between what was computed and what got written down.
        recomputed = {
            "MAE": mae(df["actual"], df["predicted"]),
            "MAPE": mape(df["actual"], df["predicted"]),
            "RMSE": rmse(df["actual"], df["predicted"]),
        }
        rep = reported.loc[sq]
        for metric in ["MAE", "MAPE", "RMSE"]:
            match = abs(recomputed[metric] - rep[metric]) < 0.01
            check(f"[{sq}] reported {metric} matches independent recomputation from saved predictions",
                  match, f"reported={rep[metric]:.3f}, recomputed={recomputed[metric]:.3f}")

        # Check this is genuinely ONE-STEP-AHEAD, not an accidental multi-step/
        # dynamic forecast. A multi-step forecast starting fresh at Dec 16 would
        # typically degrade (lose daily amplitude, drift toward a flat mean) as
        # the horizon grows across the week. Compare error and predicted-amplitude
        # in the first vs. last 2 days of the test week - true one-step-ahead
        # forecasts should NOT show a clear degrading trend, since each prediction
        # is re-grounded in the true previous value regardless of how far into the
        # week it is.
        first_2d = df.iloc[:288]
        last_2d = df.iloc[-288:]
        rmse_first = rmse(first_2d["actual"], first_2d["predicted"])
        rmse_last = rmse(last_2d["actual"], last_2d["predicted"])
        amp_actual_first, amp_pred_first = first_2d["actual"].std(), first_2d["predicted"].std()
        amp_actual_last, amp_pred_last = last_2d["actual"].std(), last_2d["predicted"].std()
        pred_amp_ratio_first = amp_pred_first / amp_actual_first
        pred_amp_ratio_last = amp_pred_last / amp_actual_last
        print(f"  RMSE first 2 days={rmse_first:.2f} vs last 2 days={rmse_last:.2f}")
        print(f"  predicted/actual amplitude ratio: first 2 days={pred_amp_ratio_first:.3f}, "
              f"last 2 days={pred_amp_ratio_last:.3f} (should both stay close to 1.0, "
              f"not shrink toward 0 - shrinking would mean the forecast is 'flattening out' "
              f"like a multi-step forecast, not a true one-step-ahead one)")
        check(f"[{sq}] no severe amplitude collapse late in the week (multi-step-forecast red flag)",
              pred_amp_ratio_last > 0.7,
              f"ratio={pred_amp_ratio_last:.3f}")

    print("\n=== Summary ===")
    n_pass = sum(1 for _, s, _ in checks if s == "PASS")
    print(f"{n_pass}/{len(checks)} checks passed")
    if n_pass != len(checks):
        for name, s, d in checks:
            if s == "FAIL":
                print(f"  FAIL: {name}: {d}")


if __name__ == "__main__":
    main()
