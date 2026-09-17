"""
Forecasting experiments deliverable: three tables (one per area)
reporting MAE, MAPE, and RMSE for all three models - the assignment's
required format.

Run: python src/build_forecast_tables.py
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/tables")
SQUARES = [5161, 5059, 5259]
MODELS = ["sarima", "lstm", "xgboost"]
MODEL_LABELS = {"sarima": "SARIMA", "lstm": "LSTM", "xgboost": "XGBoost"}


def main():
    metrics = {}
    for model in MODELS:
        df = pd.read_csv(RESULTS_DIR / f"{model}_final_metrics.csv").set_index("square_id")
        metrics[model] = df

    for sq in SQUARES:
        rows = []
        for model in MODELS:
            row = metrics[model].loc[sq]
            rows.append({
                "Model": MODEL_LABELS[model],
                "MAE": round(row["MAE"], 2),
                "MAPE (%)": round(row["MAPE"], 2),
                "RMSE": round(row["RMSE"], 2),
            })
        table = pd.DataFrame(rows).set_index("Model")
        table = table.sort_values("RMSE")  # best model first
        out_path = RESULTS_DIR / f"forecast_metrics_table_square_{sq}.csv"
        table.to_csv(out_path)
        print(f"\n=== Square {sq} ===")
        print(table.to_string())
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
