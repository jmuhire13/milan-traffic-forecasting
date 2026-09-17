"""
Exploratory analysis, part 3 - two additional analyses on the busiest square (5161),
using the full 2-month series (not just the first two weeks).

(a) Autocorrelation (ACF) - quantifies how strongly traffic depends on its
    own recent past, and pinpoints periodicity (daily/weekly cycles)
    numerically rather than just by eye.
(b) STL seasonal decomposition + an Augmented Dickey-Fuller stationarity
    test - splits the series into trend / daily-seasonal / residual parts,
    checks whether the raw series' statistical behaviour is stable over
    time, and uses the leftover residual to flag anomalies across the
    *whole* period (not just the Nov 2-3 spike already spotted visually).

Run: python src/eda_square5161_analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller

PROCESSED = Path("data/processed/internet_traffic_by_square_10min.parquet")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SQUARE = 5161
PERIOD_DAILY = 144  # number of 10-minute steps in 24 hours


def load_square_series(square_id: int) -> tuple[pd.Series, int]:
    df = pd.read_parquet(PROCESSED, filters=[("square_id", "==", square_id)])
    s = df.set_index("timestamp")["internet_traffic"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    n_missing = len(full_index) - len(s)
    s = s.reindex(full_index, fill_value=0.0)
    return s, n_missing


def main():
    s, n_missing = load_square_series(TARGET_SQUARE)
    print(f"Square {TARGET_SQUARE}: {len(s)} time steps loaded, {n_missing} filled with zero (gaps)")

    # (a) Autocorrelation 
    max_lag = 144 * 10  # 10 days of lags - enough to see both daily and weekly structure
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_acf(s, lags=max_lag, ax=ax)
    ax.set_title(f"Square {TARGET_SQUARE}: Autocorrelation, full 2-month series "
                 f"(up to {max_lag} lags = {max_lag / 144:.0f} days)")
    ax.set_xlabel("Lag (10-minute steps)")
    fig.tight_layout()
    out_acf = FIG_DIR / "square5161_acf.png"
    fig.savefig(out_acf, dpi=150)
    print(f"Saved: {out_acf}")

    acf_arr = acf(s, nlags=max_lag, fft=True)
    print("\nAutocorrelation at exact daily/weekly multiples:")
    for days in (1, 2, 3, 7):
        lag = days * 144
        print(f"  lag = {days} day(s) ({lag} steps): ACF = {acf_arr[lag]:.3f}")

    #  (b) STL decomposition + stationarity + residual-based anomalies 
    stl = STL(s, period=PERIOD_DAILY, robust=True)
    res = stl.fit()

    fig2, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(s.index, s.values, linewidth=0.5, color="#4C72B0")
    axes[0].set_ylabel("Observed")
    axes[1].plot(s.index, res.trend, linewidth=0.9, color="tab:orange")
    axes[1].set_ylabel("Trend")
    axes[2].plot(s.index, res.seasonal, linewidth=0.4, color="tab:green")
    axes[2].set_ylabel("Daily\nseasonal")
    axes[3].plot(s.index, res.resid, linewidth=0.4, color="tab:red")
    axes[3].set_ylabel("Residual")
    axes[-1].set_xlabel("Date")
    fig2.suptitle(f"Square {TARGET_SQUARE}: STL decomposition (daily period = 144 steps)")
    fig2.tight_layout()
    out_stl = FIG_DIR / "square5161_stl_decomposition.png"
    fig2.savefig(out_stl, dpi=150)
    print(f"\nSaved: {out_stl}")

    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    print(f"\nADF stationarity test on the raw series: statistic={adf_stat:.3f}, "
          f"p-value={adf_p:.4g} -> "
          f"{'stationary' if adf_p < 0.05 else 'NOT stationary'} at the 5% significance level")

    resid = res.resid.dropna()
    z = (resid - resid.mean()) / resid.std()
    anomalies = z[z.abs() > 4].reindex(z[z.abs() > 4].abs().sort_values(ascending=False).index)
    print(f"\nResidual-based anomalies (|z-score| > 4) across the full 2-month period: "
          f"{len(anomalies)} found")
    print(anomalies.head(15).to_string())


if __name__ == "__main__":
    main()
