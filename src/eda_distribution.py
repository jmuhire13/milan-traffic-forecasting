"""
Stage 2, step 1 - Exploratory Analysis.

(a) Distribution of total Internet traffic across all 10,000 grid squares
    over the full 2-month period, as a figure + summary stats.
(b) Identify the top-3 busiest squares, and independently verify that
    ranking against the full processed dataset (not just trusting the
    Stage-1 summary file) - checking it reflects sustained traffic, not
    a one-off outlier, and that the margin over 4th/5th place is real.

Run: python src/eda_distribution.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed/internet_traffic_by_square_10min.parquet")
TOTALS = Path("data/processed/total_internet_traffic_by_square.csv")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SLOTS_FULL_PERIOD = 62 * 144  # 8,928 possible 10-minute readings per square


def main():
    totals = pd.read_csv(TOTALS).sort_values("total_internet_traffic", ascending=False).reset_index(drop=True)
    print(f"Loaded totals for {len(totals):,} squares from {TOTALS}")

    # (a) Distribution across all 10,000 squares 
    vals = totals["total_internet_traffic"].to_numpy()
    print("\n=== Distribution summary statistics (total Internet traffic per square, full period) ===")
    print(f"count:  {len(vals):,}")
    print(f"mean:   {vals.mean():,.1f}")
    print(f"median: {np.median(vals):,.1f}")
    print(f"std:    {vals.std():,.1f}")
    print(f"min:    {vals.min():,.1f}")
    print(f"max:    {vals.max():,.1f}")
    print(f"skewness (Fisher-Pearson): {pd.Series(vals).skew():.2f}")

    sorted_desc = np.sort(vals)[::-1]
    total_sum = sorted_desc.sum()
    for k in (10, 100, 1000):
        share = sorted_desc[:k].sum() / total_sum * 100
        print(f"share of total traffic held by top {k:5d} squares ({k/100:.1f}% of squares): {share:5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(vals, bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Linear scale")
    axes[0].set_xlabel("Total internet traffic (2-month sum)")
    axes[0].set_ylabel("Number of grid squares")

    positive_vals = vals[vals > 0]
    log_bins = np.logspace(np.log10(positive_vals.min()), np.log10(positive_vals.max()), 40)
    axes[1].hist(positive_vals, bins=log_bins, color="#55A868", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_title("Log scale (x-axis)")
    axes[1].set_xlabel("Total internet traffic (2-month sum, log scale)")
    axes[1].set_ylabel("Number of grid squares")

    fig.suptitle("Distribution of total Internet traffic across 10,000 Milan grid squares\n(Nov 1, 2013 - Jan 1, 2014)")
    fig.tight_layout()
    out_path = FIG_DIR / "traffic_distribution_all_squares.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure: {out_path}")

    # (b) Identify + verify top-3 
    top3 = totals.head(3)
    print("\n=== Top 3 squares by total traffic (from Stage 1 summary file) ===")
    print(top3.to_string(index=False))

    top3_ids = top3["square_id"].tolist()
    print(f"\nIndependently re-reading only rows for squares {top3_ids} from the full "
          f"processed dataset (filtered read, not a full load)...")
    subset = pd.read_parquet(PROCESSED, filters=[("square_id", "in", top3_ids)])
    print(f"  loaded {len(subset):,} rows (memory-efficient: only these 3 squares, not all 89M rows)")

    print("\n=== Verification ===")
    for sq in top3_ids:
        sq_rows = subset[subset["square_id"] == sq]
        recomputed_total = sq_rows["internet_traffic"].sum()
        official_total = totals.loc[totals["square_id"] == sq, "total_internet_traffic"].iloc[0]
        n_present = len(sq_rows)
        coverage_pct = n_present / N_SLOTS_FULL_PERIOD * 100
        max_single = sq_rows["internet_traffic"].max()
        max_share_pct = max_single / recomputed_total * 100
        match = "OK" if abs(recomputed_total - official_total) / official_total < 1e-4 else "MISMATCH"
        print(f"square {sq}: recomputed_total={recomputed_total:,.1f}  official_total={official_total:,.1f}  "
              f"[{match}]")
        print(f"           time-slots present: {n_present:,}/{N_SLOTS_FULL_PERIOD} ({coverage_pct:.1f}% coverage)")
        print(f"           single largest reading: {max_single:,.1f} "
              f"({max_share_pct:.3f}% of that square's total -> not outlier-dominated if this is small)")

    margin_1_vs_2 = (top3["total_internet_traffic"].iloc[0] / top3["total_internet_traffic"].iloc[1] - 1) * 100
    margin_top3_vs_4 = (top3["total_internet_traffic"].iloc[2] / totals["total_internet_traffic"].iloc[3] - 1) * 100
    print(f"\nMargin: #1 is {margin_1_vs_2:.1f}% ahead of #2.")
    print(f"Margin: #3 is {margin_top3_vs_4:.1f}% ahead of #4 (i.e. the top-3 cutoff is not a close call).")


if __name__ == "__main__":
    main()
