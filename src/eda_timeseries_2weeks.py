"""
Stage 2, step 2 - Exploratory Analysis.

Internet traffic time series for the first two weeks (2013-11-01 to
2013-11-14 inclusive, Milan local time) for the 5 areas the brief asks
for: the top-3 busiest squares + Square 4159 + Square 4556.

Run: python src/eda_timeseries_2weeks.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED = Path("data/processed/internet_traffic_by_square_10min.parquet")
TOTALS = Path("data/processed/total_internet_traffic_by_square.csv")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2013-11-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2013-11-15 00:00:00")  # 14 full days: Nov 1 - Nov 14
FIXED_SQUARES = [4159, 4556]


def main():
    totals = pd.read_csv(TOTALS).sort_values("total_internet_traffic", ascending=False).reset_index(drop=True)
    totals["rank"] = totals.index + 1
    totals["percentile"] = (1 - totals["rank"] / len(totals)) * 100

    top3_ids = totals.head(3)["square_id"].tolist()
    target_ids = top3_ids + FIXED_SQUARES
    print(f"Target squares: top-3 = {top3_ids}, fixed = {FIXED_SQUARES} -> {target_ids}")

    subset = pd.read_parquet(PROCESSED, filters=[("square_id", "in", target_ids)])
    subset = subset[(subset["timestamp"] >= START) & (subset["timestamp"] < END_EXCLUSIVE)]
    print(f"Loaded {len(subset):,} rows for the first-two-weeks window across {len(target_ids)} squares")

    full_index = pd.date_range(START, END_EXCLUSIVE, freq="10min", inclusive="left")
    expected_n = len(full_index)
    print(f"Expected time slots per square in this window: {expected_n}")

    series_by_square = {}
    for sq in target_ids:
        s = subset.loc[subset["square_id"] == sq].set_index("timestamp")["internet_traffic"]
        n_present = len(s)
        s = s.reindex(full_index, fill_value=0.0)
        n_missing = expected_n - n_present
        row = totals.loc[totals["square_id"] == sq].iloc[0]
        print(
            f"  square {sq:5d}: rank #{int(row['rank']):5d} of 10000 "
            f"(top {100 - row['percentile']:.2f}% busiest)  "
            f"present={n_present}/{expected_n}  filled_with_zero={n_missing}  "
            f"mean={s.mean():8.1f}  max={s.max():9.1f}"
        )
        series_by_square[sq] = s

    fig, axes = plt.subplots(len(target_ids), 1, figsize=(11, 2.4 * len(target_ids)), sharex=True)
    weekend_spans = [
        (pd.Timestamp("2013-11-02 00:00"), pd.Timestamp("2013-11-04 00:00")),  # Sat-Sun
        (pd.Timestamp("2013-11-09 00:00"), pd.Timestamp("2013-11-11 00:00")),  # Sat-Sun
    ]

    labels = {sq: ("busiest area" if sq == top3_ids[0] else
                    "2nd busiest" if sq == top3_ids[1] else
                    "3rd busiest" if sq == top3_ids[2] else
                    f"Square {sq} (assignment-specified)")
               for sq in target_ids}

    for ax, sq in zip(axes, target_ids):
        ax.plot(series_by_square[sq].index, series_by_square[sq].values, linewidth=0.8, color="#4C72B0")
        for w_start, w_end in weekend_spans:
            ax.axvspan(w_start, w_end, color="grey", alpha=0.15, lw=0)
        row = totals.loc[totals["square_id"] == sq].iloc[0]
        ax.set_title(f"Square {sq} — {labels[sq]} (rank #{int(row['rank'])}/10000)", fontsize=10, loc="left")
        ax.set_ylabel("Internet\ntraffic")

    axes[-1].set_xlabel("Date (Milan local time)")
    fig.suptitle("Internet traffic, first two weeks (2013-11-01 to 2013-11-14)\ngrey bands = weekends (Sat-Sun)")
    fig.tight_layout()
    out_path = FIG_DIR / "timeseries_first_two_weeks_5squares.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
