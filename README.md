# Milan Mobile Traffic Forecasting

This repository holds the code for a comparative study of sequential forecasting models on mobile network traffic in Milan. The underlying question is simple to state and harder to answer well: given ten-minute-resolution Internet traffic readings across a grid of 10,000 city cells, how do different modeling approaches compare at predicting the next reading, and does the answer change depending on which part of the city you're looking at?

Three models are built and evaluated on equal footing: SARIMA, an LSTM recurrent network, and XGBoost. Each takes a genuinely different approach to the same problem like classical statistical decomposition, sequence learning, and gradient-boosted trees on engineered features. The project spends as much effort explaining *why* each one behaves the way it does as it does reporting error numbers.

## Dataset

The data comes from the Telecom Italia "Milano Grid" Call Detail Record dataset (the Telecom Italia Big Data Challenge), covering November 1, 2013 through January 1, 2014 at ten-minute intervals. It is not included in this repository due to roughly 20GB across 62 daily files, it's both too large for git and easy enough to fetch yourself from the original source.

Once downloaded, place the raw `.txt` files under `data/raw/`, inside one or more subfolders matching the pattern `batch_*`. A single folder works fine:

```
data/raw/batch_1/sms-call-internet-mi-2013-11-01.txt
data/raw/batch_1/sms-call-internet-mi-2013-11-02.txt
...
```

Everything downstream is generated from this raw data by the scripts in `src/`, so nothing else needs to be downloaded or configured.

## Setup

The project targets Python 3.11+. Install the dependencies with:

```
pip install -r requirements.txt
```

Training runs entirely on CPU. There's no CUDA/GPU dependency anywhere in this codebase, which also means the LSTM model is noticeably the slowest of the three to train and that's expected, not a bug, and it's discussed directly in the results.

## Project Structure

`src/` contains the full pipeline as a series of standalone scripts, meant to be run in order rather than as an installable package. `common.py` is the one exception, it's not a pipeline step itself, just the shared module the model scripts import their metric functions, Fourier-term logic, and windowing/feature-engineering helpers from, so that logic exists in one place instead of being repeated across every script that needs it. `results/` holds every figure and metrics table the pipeline produces, organized into `figures/` and `tables/`.

## Running the Pipeline

The scripts are ordered by what they depend on, not alphabetically, so running them in this exact sequence matters and several of them read files that an earlier script has to produce first. Every command below is meant to be run from the repository root.

Start with the data pipeline. `build_dataset.py` reads the raw daily files, aggregates them into a single compact Parquet file, and reports memory usage as it goes:

```
python src/build_dataset.py
```

`validate_processed_dataset.py` re-derives the same numbers independently and checks them against what the pipeline produced:

```
python src/validate_processed_dataset.py
```

Exploratory analysis comes next. Each of these four scripts is self-contained and only needs the processed dataset from the step above:

```
python src/eda_distribution.py
python src/eda_timeseries_2weeks.py
python src/eda_square5161_analysis.py
python src/eda_spatial_heatmap.py
```

`validate_eda_findings.py` cross-checks the findings from all four:

```
python src/validate_eda_findings.py
```

Before touching any models, `prepare_forecasting_data.py` builds the train/validation/test split and per-square scalers that every model below reads from, so it has to run first:

```
python src/prepare_forecasting_data.py
```

Each model has its own tuning script, which searches hyperparameters, picks a winner per grid square, and writes predictions and metrics out to `results/tables/`:

```
python src/tune_sarima.py
python src/tune_lstm.py
python src/tune_xgboost.py
```

`fix_lstm_final.py` and `lstm_extended_epochs_check.py` are follow-up corrections to the LSTM run specifically — a test-set leakage bug was found and fixed during review, then a follow-up check confirmed the model needed more training epochs than it was originally given. Both must run after `tune_lstm.py` and in this order:

```
python src/fix_lstm_final.py
python src/lstm_extended_epochs_check.py
```

Finally, these three assemble the comparison figures, per-area metrics tables, and timing statistics from everything produced above. The timing script retrains each model's already-tuned configuration fresh, back-to-back, in one clean run, so the comparison is measured under the same conditions rather than pieced together from earlier tuning runs:

```
python src/build_forecast_plots.py
python src/build_forecast_tables.py
python src/build_forecast_timing.py
```

A few scripts sit outside this required sequence and don't need to run for the results in `results/` to be reproduced, they're supporting checks, not steps that produce or change anything. `memory_benchmark.py`, `model_sarima.py`, and `model_lstm.py` were early, standalone sanity checks written before the full tuning scripts existed; `deep_audit_sarima.py`, `deep_audit_lstm.py`, and `deep_audit_xgboost.py` are read-only checks that independently re-verify each model's already-saved predictions and metrics. All six can be run at any point after the step they depend on (`build_dataset.py` for the first three, `model_lstm.py` additionally needs `prepare_forecasting_data.py`; the three audit scripts need their corresponding model's tuning and for LSTM, the fix and follow-up is already run):

```
python src/memory_benchmark.py
python src/model_sarima.py
python src/model_lstm.py
python src/deep_audit_sarima.py
python src/deep_audit_lstm.py
python src/deep_audit_xgboost.py
```

## Results

The headline result is that no single model wins everywhere. SARIMA takes two of the three grid squares tested, largely because its Fourier-based seasonal terms line up well with how regular that traffic is. XGBoost wins the third, specifically because it leans heavily on the most recent observed value, which makes it faster to react when that square's traffic does something unusual. The full breakdown, including per-model metrics tables, timing comparisons, and a look at where each model actually fails, is in `results/tables/` and `results/figures/`.
