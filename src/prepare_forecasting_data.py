"""
Shared data preparation pipeline for the forecasting experiments (Section
4 of the assignment) - builds the train/test split and scalers for the
three forecasting-target squares (5161, 5059, 5259).

Produces, per square, one chronological train/test split (train = before
2013-12-16; test = the week 2013-12-16 to 2013-12-22 inclusive) and a
MinMax scaler fit ONLY on the training portion (no leakage from the test
week). Every model (SARIMA, LSTM, XGBoost) reads from this same split and
scaler, so preprocessing is identical across all three - each model then
does its own further shaping (windowing, lag features, etc.) on top of
this common foundation.

Run: python src/prepare_forecasting_data.py
"""
import json
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed/internet_traffic_by_square_10min.parquet")
OUT_DIR = Path("data/processed/stage4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SQUARES = [5161, 5059, 5259]
TRAIN_START = pd.Timestamp("2013-11-01 00:00:00")
TEST_START = pd.Timestamp("2013-12-16 00:00:00")
TEST_END_EXCLUSIVE = pd.Timestamp("2013-12-23 00:00:00")  # covers Dec 16-22 inclusive


def load_square_series(df_all: pd.DataFrame, square_id: int) -> pd.Series:
    sub = df_all[df_all["square_id"] == square_id]
    s = sub.set_index("timestamp")["internet_traffic"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    n_missing = len(full_index) - len(s)
    s = s.reindex(full_index, fill_value=0.0)
    return s, n_missing


def main():
    print(f"Loading squares {TARGET_SQUARES} from {PROCESSED} ...")
    df_all = pd.read_parquet(PROCESSED, filters=[("square_id", "in", TARGET_SQUARES)])

    scalers = {}
    for sq in TARGET_SQUARES:
        s, n_missing = load_square_series(df_all, sq)
        print(f"\nSquare {sq}: {len(s)} steps loaded, {n_missing} filled with zero (gaps)")

        train = s[(s.index >= TRAIN_START) & (s.index < TEST_START)]
        test = s[(s.index >= TEST_START) & (s.index < TEST_END_EXCLUSIVE)]
        print(f"  train: {len(train)} steps ({train.index.min()} -> {train.index.max()})")
        print(f"  test:  {len(test)} steps ({test.index.min()} -> {test.index.max()})")
        assert len(test) == 1008, f"expected 1008 test steps (7 days x 144), got {len(test)}"

        # MinMax scaler fit on TRAIN ONLY - test week must never influence the scaler
        train_min, train_max = float(train.min()), float(train.max())
        span = train_max - train_min
        scalers[sq] = {"min": train_min, "max": train_max}

        def scale(x):
            return (x - train_min) / span

        train_df = pd.DataFrame({
            "timestamp": train.index,
            "internet_traffic": train.values,
            "internet_traffic_scaled": scale(train.values),
        })
        test_df = pd.DataFrame({
            "timestamp": test.index,
            "internet_traffic": test.values,
            "internet_traffic_scaled": scale(test.values),
        })

        sq_dir = OUT_DIR / str(sq)
        sq_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(sq_dir / "train.parquet", index=False)
        test_df.to_parquet(sq_dir / "test.parquet", index=False)
        print(f"  train range (original units): [{train.min():.1f}, {train.max():.1f}]")
        print(f"  test range  (original units): [{test.min():.1f}, {test.max():.1f}]"
              f"{'  <-- exceeds train max, scaled values will be >1.0' if test.max() > train_max else ''}")
        print(f"  saved: {sq_dir / 'train.parquet'}, {sq_dir / 'test.parquet'}")

    scalers_path = OUT_DIR / "scalers.json"
    with open(scalers_path, "w") as f:
        json.dump(scalers, f, indent=2)
    print(f"\nSaved scaler parameters: {scalers_path}")
    print(json.dumps(scalers, indent=2))


if __name__ == "__main__":
    main()
