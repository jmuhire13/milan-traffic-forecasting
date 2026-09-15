"""
Stage 1 review: sanity-check the processed Parquet output for integrity,
and cross-check it against an independent re-computation for one day.

Run: python src/validate_processed_dataset.py
"""
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed/internet_traffic_by_square_10min.parquet")
RAW_NOV1 = Path("data/raw/batch_1/sms-call-internet-mi-2013-11-01.txt")

COLUMN_NAMES = [
    "square_id", "time_interval", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet_traffic",
]


def main():
    print(f"Loading {PROCESSED} ...")
    df = pd.read_parquet(PROCESSED)
    print(f"  shape={df.shape}, dtypes:\n{df.dtypes.to_string()}\n")

    checks = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        checks.append((name, status, detail))
        print(f"[{status}] {name}  {detail}")

    # 1. No duplicate (square_id, timestamp) pairs
    dup_count = df.duplicated(subset=["square_id", "timestamp"]).sum()
    check("no duplicate (square_id, timestamp) rows", dup_count == 0, f"duplicates={dup_count}")

    # 2. All timestamps fall exactly on a 10-minute boundary
    off_grid = (~((df["timestamp"].dt.minute % 10 == 0) & (df["timestamp"].dt.second == 0))).sum()
    check("all timestamps on 10-minute boundary", off_grid == 0, f"off-grid rows={off_grid}")

    # 3. No negative internet traffic
    neg_count = (df["internet_traffic"] < 0).sum()
    check("no negative internet_traffic values", neg_count == 0, f"negative rows={neg_count}")

    # 4. Distinct timestamps: expect 62 days x 144 ten-minute slots = 8928
    n_ts = df["timestamp"].nunique()
    expected_ts = 62 * 144
    check("distinct timestamps == 62 days x 144 slots", n_ts == expected_ts,
          f"actual={n_ts}, expected={expected_ts}")

    # 5. Distinct squares == 10,000
    n_sq = df["square_id"].nunique()
    check("distinct squares == 10000", n_sq == 10000, f"actual={n_sq}")

    # 6. Date range covers exactly Nov 1 00:00 -> Jan 1 23:50 local time
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    check(
        "date range == 2013-11-01 00:00 -> 2014-01-01 23:50",
        (tmin == pd.Timestamp("2013-11-01 00:00:00")) and (tmax == pd.Timestamp("2014-01-01 23:50:00")),
        f"actual: {tmin} -> {tmax}",
    )

    # 7. Cross-check Nov 1 total against an independent re-computation straight
    #    from the raw file, using the same read+aggregate logic but run here
    #    fresh, as a check that build_dataset.py's pipeline did the same thing.
    raw = pd.read_csv(
        RAW_NOV1, sep="\t", header=None, names=COLUMN_NAMES,
        usecols=["square_id", "time_interval", "internet_traffic"],
        dtype={"square_id": "int32", "time_interval": "int64", "internet_traffic": "float32"},
    )
    raw_nov1_total = raw["internet_traffic"].sum()

    nov1_mask = (df["timestamp"] >= "2013-11-01") & (df["timestamp"] < "2013-11-02")
    processed_nov1_total = df.loc[nov1_mask, "internet_traffic"].sum()

    rel_diff = abs(raw_nov1_total - processed_nov1_total) / raw_nov1_total
    check(
        "Nov-1 total internet_traffic matches independent recomputation",
        rel_diff < 1e-4,
        f"raw_sum={raw_nov1_total:,.1f}  processed_sum={processed_nov1_total:,.1f}  rel_diff={rel_diff:.2e}",
    )

    # 8. Memory to read the processed file back, vs the original naive full-file cost
    import psutil
    proc = psutil.Process()
    print(f"\nMemory to hold the full processed dataset in memory: "
          f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB "
          f"(process RSS at this point: {proc.memory_info().rss / 1e6:.1f} MB)")

    print("\n=== Summary ===")
    n_pass = sum(1 for _, s, _ in checks if s == "PASS")
    print(f"{n_pass}/{len(checks)} checks passed")
    if n_pass != len(checks):
        print("FAILURES:")
        for name, status, detail in checks:
            if status == "FAIL":
                print(f"  - {name}: {detail}")


if __name__ == "__main__":
    main()
