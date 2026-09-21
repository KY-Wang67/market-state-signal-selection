# Reproduction Code

This repository contains the main empirical workflow for the CSI 300 and
S&P 500 applications. The implementation is flat: each Python
file corresponds to one research stage, with `run_reproduction.py` as the single command-line entry point.

Market data and risk-factor data are not distributed with the code. Users
must obtain data from sources they are authorized to use and convert them to
the structures described in `DATA_STRUCTURE.md`.

## Workflow

The program runs the following sequence for either market:

1. Estimate a two-state Gaussian HMM using training observations only.
2. Produce recursively filtered state probabilities.
3. Construct the State K3, Global K3 and Full 101 signal specifications. State K3 selects three Alpha101 signals per state using training and validation Rank IC.
4. Construct probability-gated Alpha channels and combine them with the five fixed stock-level characteristics.
5. Fit one pooled XGBoost pairwise ranker per rolling window for each model-based specification.
6. Rank stocks and apply the buffered Top-24 rule with at most three voluntary replacements to the ranked portfolio specifications.
7. Construct the reported portfolio benchmarks and apply market-specific historical transaction costs where applicable.
8. Compute continuous-wealth performance statistics across the rolling test windows.
9. Optionally estimate A-share FF3/CH3 or U.S. FF5 alpha by OLS.

## Files

- `run_reproduction.py`: integrated command-line entry point.
- `common.py`: configuration, input, weighting, and performance utilities.
- `hmm.py`: training-only HMM estimation and recursive filtering.
- `signals.py`: cross-sectional Rank IC and signal selection.
- `ranking.py`: probability gating, XGBoost ranking, and buffered holdings.
- `portfolio_a_share.py`: A-share execution and transaction costs.
- `portfolio_us_stock.py`: US execution and regulatory fees.
- `risk_adjustment.py`: common FF3, CH3, and FF5 OLS regressions.
- `config.json`: rolling windows and model parameters.
- `DATA_STRUCTURE.md`: required user-supplied data fields.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Data check

After preparing the four required tables for a market, check their paths:

```bash
python run_reproduction.py --market a_share --check-data
python run_reproduction.py --market us_stock --check-data
```

This check only reports whether the configured files exist. It does not
download, alter, or redistribute any data.

## Run

```bash
python run_reproduction.py --market a_share
python run_reproduction.py --market us_stock
```

Run both markets with:

```bash
python run_reproduction.py --market all
```

Existing market output directories are not overwritten unless
`--overwrite` is supplied.

## Risk adjustment

Risk adjustment is optional. Set the appropriate path in `config.json` only
after obtaining an authorized factor dataset:

- A shares: `ff3` and/or `ch3`
- US stocks: `ff5`

If these paths remain `null`, the portfolio pipeline completes normally and
the risk-adjustment stage is skipped. The public implementation reports OLS
alpha, factor loadings, and R-squared.

## Main outputs

Each market writes to `outputs/<market>/`:

- `hmm_probabilities.parquet`
- `hmm_states.csv`
- `hmm_initializations.csv`
- `selected_signals.csv`
- `stock_scores.parquet`
- `target_holdings.parquet`
- `portfolio_returns.csv`
- `portfolio_holdings.parquet`
- `performance_summary.csv`
- `risk_adjustment.csv`, when factor data are configured
- `run_summary.json`

Random seeds are specified in `config.json`.
