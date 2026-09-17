"""
Exploratory analysis review: independently re-check every number/claim
produced during the EDA (distribution stats, top-3 ranking, time-series
window, ACF values, STL decomposition, ADF test, anomaly list) before
signing off.

Run: python src/validate_eda_findings.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller

PROCESSED = Path("data/processed/internet_traffic_by_square_10min.parquet")
TOTALS = Path("data/processed/total_internet_traffic_by_square.csv")

checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status, detail))
    print(f"[{status}] {name}  {detail}")


def main():
    # ---- 1. Full re-derivation of total_internet_traffic_by_square.csv ----
    print("Re-deriving per-square totals from scratch, for ALL 10,000 squares "
          "(not just the top few) - this file has been relied on throughout Stage 2...")
    full = pd.read_parquet(PROCESSED, columns=["square_id", "internet_traffic"])
    recomputed_totals = (
        full.groupby("square_id", as_index=False)["internet_traffic"]
        .sum()
        .rename(columns={"internet_traffic": "recomputed_total"})
    )
    del full

    official_totals = pd.read_csv(TOTALS)
    merged = official_totals.merge(recomputed_totals, on="square_id", how="outer")
    merged["diff"] = (merged["total_internet_traffic"] - merged["recomputed_total"]).abs()
    max_diff = merged["diff"].max()
    n_mismatch = (merged["diff"] / merged["recomputed_total"].clip(lower=1e-9) > 1e-4).sum()
    check(
        "ALL 10,000 squares' totals match an independent full recomputation",
        n_mismatch == 0,
        f"max_abs_diff={max_diff:.4f}, mismatches(>0.01% rel diff)={n_mismatch}",
    )
    check("no square present in only one of the two sources", merged.isna().sum().sum() == 0,
          f"na_count={merged.isna().sum().sum()}")

    # ---- 2. Distribution stats re-derived independently -------------------
    vals = official_totals["total_internet_traffic"].to_numpy()
    mean_, median_, std_ = vals.mean(), np.median(vals), vals.std()
    check("distribution stats reproducible (mean~555289, median~277871)",
          abs(mean_ - 555289.4) < 1 and abs(median_ - 277871.1) < 1,
          f"mean={mean_:.1f}, median={median_:.1f}, std={std_:.1f}")

    sorted_desc = np.sort(vals)[::-1]
    top10_share = sorted_desc[:10].sum() / sorted_desc.sum() * 100
    check("top-10 share of total traffic reproducible (~1.7%)", abs(top10_share - 1.7) < 0.2,
          f"top10_share={top10_share:.2f}%")

    # ---- 3. Top-3 identification, consistent everywhere -------------------
    top3 = official_totals.sort_values("total_internet_traffic", ascending=False).head(3)
    top3_ids = top3["square_id"].tolist()
    check("top-3 squares are exactly [5161, 5059, 5259]", top3_ids == [5161, 5059, 5259],
          f"actual={top3_ids}")

    # ---- 4. Two-week time-series window integrity for the 5 target squares
    target_ids = top3_ids + [4159, 4556]
    start = pd.Timestamp("2013-11-01 00:00:00")
    end_excl = pd.Timestamp("2013-11-15 00:00:00")
    subset = pd.read_parquet(PROCESSED, filters=[("square_id", "in", target_ids)])
    subset = subset[(subset["timestamp"] >= start) & (subset["timestamp"] < end_excl)]
    counts = subset.groupby("square_id").size()
    check("all 5 target squares have exactly 2016 rows in the 2-week window",
          (counts == 2016).all(), f"counts={counts.to_dict()}")
    check("no negative traffic values in the 2-week window", (subset["internet_traffic"] >= 0).all(),
          f"min={subset['internet_traffic'].min()}")
    dup = subset.duplicated(subset=["square_id", "timestamp"]).sum()
    check("no duplicate (square_id, timestamp) in the 2-week window", dup == 0, f"duplicates={dup}")

    # ---- 5. ACF cross-check for square 5161 (independent of statsmodels) --
    sq_df = pd.read_parquet(PROCESSED, filters=[("square_id", "==", 5161)])
    s = sq_df.set_index("timestamp")["internet_traffic"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    check("square 5161 series has no gaps (8928 steps)", len(s) == len(full_index) == 8928,
          f"len(s)={len(s)}, expected=8928")
    s = s.reindex(full_index, fill_value=0.0)

    sm_acf = acf(s, nlags=1008, fft=True)
    x = s.to_numpy()
    xbar = x.mean()

    # Manual lag-k autocorrelation via numpy, avoiding statsmodels entirely.
    # Must match the textbook/Box-Jenkins definition: global mean, denominator
    # over ALL n points (not n-lag). np.corrcoef() is NOT equivalent here -
    # it re-centers each shifted segment by its own local mean/variance,
    # which is a different (also valid, but different) statistic.
    def manual_acf(x, lag, xbar):
        num = np.sum((x[:-lag] - xbar) * (x[lag:] - xbar))
        den = np.sum((x - xbar) ** 2)
        return num / den

    manual_1day = manual_acf(x, 144, xbar)
    manual_7day = manual_acf(x, 1008, xbar)
    check("ACF at 1-day lag matches independent numpy computation (~0.878)",
          abs(sm_acf[144] - manual_1day) < 1e-6 and abs(manual_1day - 0.878) < 0.01,
          f"statsmodels={sm_acf[144]:.4f}, numpy={manual_1day:.4f}")
    check("ACF at 7-day lag matches independent numpy computation (~0.838)",
          abs(sm_acf[1008] - manual_7day) < 1e-6 and abs(manual_7day - 0.838) < 0.01,
          f"statsmodels={sm_acf[1008]:.4f}, numpy={manual_7day:.4f}")
    check("weekly effect confirmed: ACF(7-day) > ACF(2-day) and ACF(3-day)",
          sm_acf[1008] > sm_acf[288] and sm_acf[1008] > sm_acf[432],
          f"acf_2d={sm_acf[288]:.3f}, acf_3d={sm_acf[432]:.3f}, acf_7d={sm_acf[1008]:.3f}")

    # 6. STL decomposition identity: observed == trend + seasonal + resid
    stl = STL(s, period=144, robust=True)
    res = stl.fit()
    reconstructed = res.trend + res.seasonal + res.resid
    max_reconstruction_error = (s - reconstructed).abs().max()
    check("STL decomposition reconstructs the observed series exactly (trend+seasonal+resid)",
          max_reconstruction_error < 1e-6, f"max_abs_error={max_reconstruction_error:.2e}")

    # 7. ADF test reproducibility 
    adf_result = adfuller(s, autolag="AIC")
    adf_stat, adf_p = adf_result[0], adf_result[1]
    check("ADF test reproducible (stat~-19.03, p~0)", abs(adf_stat - (-19.032)) < 0.01 and adf_p < 0.01,
          f"stat={adf_stat:.3f}, p={adf_p:.4g}")

    # 8. Anomaly count + top anomaly reproducibility 
    resid = res.resid.dropna()
    z = (resid - resid.mean()) / resid.std()
    anomalies = z[z.abs() > 4]
    top_anomaly_ts = z.abs().idxmax()
    check("124 anomalies at |z|>4 threshold, reproducible", len(anomalies) == 124,
          f"count={len(anomalies)}")
    check("single largest anomaly is 2013-11-02 13:50, reproducible",
          top_anomaly_ts == pd.Timestamp("2013-11-02 13:50:00"),
          f"actual={top_anomaly_ts}, z={z.loc[top_anomaly_ts]:.2f}")

    print("\n=== Summary ===")
    n_pass = sum(1 for _, st, _ in checks if st == "PASS")
    print(f"{n_pass}/{len(checks)} checks passed")
    if n_pass != len(checks):
        print("FAILURES:")
        for name, st, detail in checks:
            if st == "FAIL":
                print(f"  - {name}: {detail}")


if __name__ == "__main__":
    main()
