"""
Forecasting experiments deliverable: the 9 required plots (3 models x 3
areas) of actual vs. predicted Internet traffic for the Dec 16-22 test
week, laid out as one 3x3 grid (rows = models, columns = areas) so all 9
required comparisons are visible together, plus saved as individual PNGs
too in case the report needs them separately.

Run: python src/build_forecast_plots.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["sarima", "lstm", "xgboost"]
MODEL_LABELS = {"sarima": "SARIMA", "lstm": "LSTM", "xgboost": "XGBoost"}
SQUARES = [5161, 5059, 5259]
SQUARE_LABELS = {5161: "Square 5161 (busiest)", 5059: "Square 5059 (2nd)", 5259: "Square 5259 (3rd)"}


def main():
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)

    for row, model in enumerate(MODELS):
        for col, sq in enumerate(SQUARES):
            ax = axes[row, col]
            df = pd.read_csv(RESULTS_DIR / f"{model}_predictions_{sq}.csv", parse_dates=["timestamp"])
            ax.plot(df["timestamp"], df["actual"], label="actual", linewidth=0.8, color="#4C72B0")
            ax.plot(df["timestamp"], df["predicted"], label="predicted", linewidth=0.8,
                     color="#DD8452", alpha=0.85)
            if row == 0:
                ax.set_title(SQUARE_LABELS[sq], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nInternet traffic", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")

            # also save each cell as its own individual plot
            fig_i, ax_i = plt.subplots(figsize=(10, 3.5))
            ax_i.plot(df["timestamp"], df["actual"], label="actual", linewidth=0.9, color="#4C72B0")
            ax_i.plot(df["timestamp"], df["predicted"], label="predicted", linewidth=0.9,
                       color="#DD8452", alpha=0.85)
            ax_i.set_title(f"{MODEL_LABELS[model]} - {SQUARE_LABELS[sq]} - Dec 16-22 (one-step-ahead)")
            ax_i.set_xlabel("Date")
            ax_i.set_ylabel("Internet traffic")
            ax_i.legend(fontsize=8)
            fig_i.tight_layout()
            fig_i.savefig(FIG_DIR / f"forecast_{model}_{sq}.png", dpi=150)
            plt.close(fig_i)

    for ax in axes[-1, :]:
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Forecasting experiments: actual vs. one-step-ahead predicted Internet traffic, Dec 16-22\n"
                  "rows = model, columns = area", fontsize=13)
    fig.tight_layout()
    out_path = FIG_DIR / "forecast_all_models_all_areas_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved grid: {out_path}")
    print(f"Saved 9 individual plots: forecast_{{model}}_{{square}}.png in {FIG_DIR}")


if __name__ == "__main__":
    main()
