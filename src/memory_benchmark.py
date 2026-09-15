"""
Compare a naive full-column, inferred-dtype CSV load against a targeted,
typed, aggregated load for a single day of the Milano Grid dataset.

Run: python src/memory_benchmark.py
"""
import gc
import time
from pathlib import Path

import pandas as pd
import psutil

DATA_FILE = Path("data/raw/batch_1/sms-call-internet-mi-2013-11-01.txt")
COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_traffic",
]
PROC = psutil.Process()


def rss_mb():
    return PROC.memory_info().rss / 1e6


def measure(label, fn):
    gc.collect()
    before = rss_mb()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    after = rss_mb()
    print(f"[{label}] rss_before={before:8.1f}MB  rss_after={after:8.1f}MB  "
          f"delta={after - before:8.1f}MB  time={elapsed:6.2f}s")
    return result, after - before, elapsed


def naive_load():
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=COLUMNS)
    return df


def optimized_load_aggregated():
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=COLUMNS,
        usecols=["square_id", "time_interval", "internet_traffic"],
        dtype={
            "square_id": "int32",
            "time_interval": "int64",
            "internet_traffic": "float32",
        },
        engine="c",
    )
    agg = (
        df.groupby(["square_id", "time_interval"], as_index=False)[
            "internet_traffic"
        ].sum()
    )
    return agg


if __name__ == "__main__":
    print(f"File: {DATA_FILE} ({DATA_FILE.stat().st_size / 1e6:.1f} MB on disk)")
    print()

    naive_df, naive_delta, naive_time = measure("naive (8 cols, inferred dtypes)", naive_load)
    naive_shape = naive_df.shape
    print(f"  naive shape={naive_shape}, dtypes=\n{naive_df.dtypes.to_string()}")
    print(f"  naive df.memory_usage(deep=True).sum() = {naive_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    del naive_df
    gc.collect()
    print()

    opt_df, opt_delta, opt_time = measure(
        "optimized (3 cols, typed, aggregated over country_code)", optimized_load_aggregated
    )
    print(f"  optimized shape={opt_df.shape}, dtypes=\n{opt_df.dtypes.to_string()}")
    print(f"  optimized df.memory_usage(deep=True).sum() = {opt_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print()

    print(" Summary (single day file) ")
    print(f"Naive rows -> Optimized rows: {naive_shape[0]:,} -> {opt_df.shape[0]:,}")
    print(f"Naive peak RSS delta:     {naive_delta:8.1f} MB   ({naive_time:.2f}s)")
    print(f"Optimized peak RSS delta: {opt_delta:8.1f} MB   ({opt_time:.2f}s)")
    print(f"Reduction: {(1 - opt_delta / naive_delta) * 100:.1f}%")
