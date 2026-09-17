"""
Full dataset build - data handling and memory management.

Streams all 62 daily raw files one at a time (never holding more than one
day's data in memory), keeps only the 3 columns needed (square_id,
time_interval, internet_traffic), collapses the country_code breakdown by
summing it away, and combines the result into one compact Parquet file
that every later step (EDA, modeling) reads instead of the raw text.

Also tracks memory (RSS = Resident Set Size, i.e. how much real RAM the
Python process is holding) and wall-clock time throughout, as evidence for
the memory-management write-up.

Run: python src/build_dataset.py
"""
import gc
import time
from pathlib import Path

import pandas as pd
import psutil

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_NAMES = [
    "square_id", "time_interval", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet_traffic",
]
USECOLS = ["square_id", "time_interval", "internet_traffic"]
DTYPES = {"square_id": "int32", "time_interval": "int64", "internet_traffic": "float32"}

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / 1e6


def process_file(path: Path) -> tuple[pd.DataFrame, float]:
    """Read one day, aggregate away country_code, return (result, peak_rss_mb)."""
    df = pd.read_csv(
        path, sep="\t", header=None, names=COLUMN_NAMES,
        usecols=USECOLS, dtype=DTYPES, engine="c",
    )
    peak = rss_mb()  # measured right after the raw read - the true local peak
    agg = (
        df.groupby(["square_id", "time_interval"], as_index=False)["internet_traffic"]
        .sum()
    )
    del df
    return agg, peak


def main():
    files = sorted(RAW_DIR.glob("batch_*/sms-call-internet-mi-*.txt"))
    print(f"Found {len(files)} daily files")
    if not files:
        raise SystemExit("No raw files found under data/raw/batch_*/")

    overall_peak_rss = rss_mb()
    t_start = time.perf_counter()
    daily_frames = []
    per_file_naive_mb_estimate = []  # on-disk size, used later to sanity check the naive extrapolation

    for i, path in enumerate(files, 1):
        t0 = time.perf_counter()
        agg, file_peak = process_file(path)
        daily_frames.append(agg)
        per_file_naive_mb_estimate.append(path.stat().st_size / 1e6)
        gc.collect()
        overall_peak_rss = max(overall_peak_rss, file_peak, rss_mb())
        print(
            f"[{i:2d}/{len(files)}] {path.name}  rows={len(agg):>9,}  "
            f"rss_after_read={file_peak:7.1f}MB  time={time.perf_counter() - t0:5.2f}s"
        )

    print("\nConcatenating all daily aggregates into one table...")
    full = pd.concat(daily_frames, ignore_index=True)
    del daily_frames
    gc.collect()
    overall_peak_rss = max(overall_peak_rss, rss_mb())
    print(f"  rss after concat: {rss_mb():.1f}MB")

    # Raw time_interval is ms-since-epoch, read by pandas as UTC by default.
    # Milan runs on CET (UTC+1) for this entire Nov-Jan window (no DST change
    # in between), so convert to Europe/Rome local time and drop the tz info
    # to get plain local wall-clock timestamps - this is a fixed +1h shift,
    # not a change to any measured value.
    full["timestamp"] = (
        pd.to_datetime(full["time_interval"], unit="ms", utc=True)
        .dt.tz_convert("Europe/Rome")
        .dt.tz_localize(None)
    )
    full = full.drop(columns=["time_interval"])[["square_id", "timestamp", "internet_traffic"]]

    out_path = OUT_DIR / "internet_traffic_by_square_10min.parquet"
    full.to_parquet(out_path, index=False)
    overall_peak_rss = max(overall_peak_rss, rss_mb())
    print(f"  rss after writing parquet: {rss_mb():.1f}MB")

    totals = (
        full.groupby("square_id", as_index=False)["internet_traffic"]
        .sum()
        .rename(columns={"internet_traffic": "total_internet_traffic"})
        .sort_values("total_internet_traffic", ascending=False)
    )
    totals_path = OUT_DIR / "total_internet_traffic_by_square.csv"
    totals.to_csv(totals_path, index=False)

    elapsed = time.perf_counter() - t_start
    naive_extrapolated_gb = sum(
        # single-file benchmark showed naive in-memory delta ~= on-disk size,
        # so extrapolate from raw file sizes rather than actually attempting
        # a 62-file naive load (that risks exhausting RAM on this machine)
        s * 0.965  # naive/on-disk ratio observed in the single-file benchmark (312.2/322.9)
        for s in per_file_naive_mb_estimate
    ) / 1e3

    print("\n Data pipeline summary")
    print(f"Files processed:              {len(files)}")
    print(f"Combined rows:                {len(full):,}")
    print(f"Distinct squares:             {full['square_id'].nunique():,}")
    print(f"Date range:                   {full['timestamp'].min()} -> {full['timestamp'].max()}")
    print(f"Output parquet:               {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB on disk)")
    print(f"Output per-square totals:     {totals_path}")
    print(f"Estimated naive full-load RAM (extrapolated, NOT attempted): ~{naive_extrapolated_gb:.1f} GB")
    print(f"Actual measured peak RSS this run:                           {overall_peak_rss:.1f} MB "
          f"({overall_peak_rss / 1e3:.2f} GB)")
    print(f"Total wall time:               {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("\nTop 5 squares by total internet traffic:")
    print(totals.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
