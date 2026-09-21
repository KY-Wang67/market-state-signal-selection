# Data Structure

The repository contains no market data, factor data, credentials, or download
scripts. Users supply their own authorized datasets. Paths are configured in
`config.json`.

The CSI 300 and S&P 500 use the same four required filenames and the same
column definitions:

```text
data/
├── a_share/
│   ├── daily_panel.parquet
│   ├── weekly_panel.parquet
│   ├── market_observations.parquet
│   └── benchmark_total_return.parquet
└── us_stock/
    ├── daily_panel.parquet
    ├── weekly_panel.parquet
    ├── market_observations.parquet
    └── benchmark_total_return.parquet
```

These directories and files are user-created and are ignored by Git.

## 1. Daily stock panel

File: `daily_panel.parquet`

Unique key: `(date, symbol)`

Required columns:

| Column                 | Definition                                                         |
| ---------------------- | ------------------------------------------------------------------ |
| `date`               | Trading date.                                                      |
| `symbol`             | Stable security identifier stored as text.                         |
| `raw_close`          | Unadjusted closing price used for trade-value calculations.        |
| `adj_close`          | Adjusted closing price used for return and predictor construction. |
| `volume`             | Raw trading volume.                                                |
| `amount`             | Daily trading value.                                               |
| `total_market_value` | Total market capitalization.                                       |
| `float_market_value` | Float market capitalization.                                       |
| `is_member`          | Point-in-time index membership, encoded as 0 or 1.                 |

`amount` and `float_market_value` must use a common monetary scale within each
market. The code uses ratios, so the scale may differ between markets but may
not vary within a market.

The panel should start sufficiently early to provide the required lookback history, including 120 trading days before the first evaluated signal date.

## 2. Weekly signal panel

File: `weekly_panel.parquet`

Unique key: `(signal_date, symbol)`

Required columns:

| Column                                | Definition                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `signal_date`                       | Date on which signals are observed after the close.                                       |
| `execution_date`                    | Next trading-day close used for execution.                                                |
| `next_execution_date`               | End of the holding interval and next rebalance date.                                      |
| `symbol`                            | Security identifier matching the daily panel.                                             |
| `is_member`                         | Point-in-time membership on the signal date.                                              |
| `forward_return`                    | Total return from the close on`execution_date` to the close on `next_execution_date`. |
| `alpha101_001` ... `alpha101_101` | Alpha101 signals known at the signal close.                                               |

The forward return must include distributions, corporate-action effects, and
delisting or terminal returns where applicable in both markets.

The weekly panel contains the execution schedule, so neither market requires
a separate trading-calendar or weekly-schedule file.

## 3. Market observations

File: `market_observations.parquet`

Unique key: `date`

Required columns:

| Column                       | Definition                                                     |
| ---------------------------- | -------------------------------------------------------------- |
| `date`                     | Trading date.                                                 |
| `mkt_drawdown_60`          | Own-market 60-day drawdown.                                    |
| `mkt_downside_vol_20`      | Own-market 20-day downside volatility.                         |
| `cs_ret_dispersion_5_ma20` | Twenty-day mean of five-day cross-sectional return dispersion. |
| `mkt_ret_20`               | Own-market 20-day return used only to label the fitted states. |

The first three fields enter the HMM. `mkt_ret_20` labels the estimated states
as higher- and lower-return states; it is not an HMM input.

## 4. Official total-return benchmark

File: `benchmark_total_return.parquet`

Unique key: `date`

Required columns:

| Column                 | Definition                         |
| ---------------------- | ---------------------------------- |
| `date`               | Trading date.                      |
| `total_return_index` | Official total-return index level. |

Use the CSI 300 total-return index for the A-share application and the S&P 500
total-return index for the US application.

## Optional corporate-action table

Corporate actions are optional because `adj_close` and `forward_return`
already incorporate economic return effects. A separate table is needed only
when identifiers change across holding periods.

Configured default filename: `corporate_actions.csv`

Required columns when supplied:

| Column                 | Definition                        |
| ---------------------- | --------------------------------- |
| `effective_date`     | First date of the new identifier. |
| `source_symbol`      | Identifier before the event.      |
| `destination_symbol` | Identifier after the event.       |

The same structure applies to both markets.

## Optional risk-factor data

A user may store an authorized CSV or Parquet file anywhere and place its path in `config.json`. All factor
returns must be decimal simple returns, not percentages.

A-share FF3 fields:

```text
date, rf, mkt_rf, smb, hml
```

A-share CH3 fields:

```text
date, rf, mkt_rf, smb, vmg
```

US FF5 fields:

```text
date, rf, mkt_rf, smb, hml, rmw, cma
```

The code aggregates daily factors over the exact open-closed interval
`(execution_date, next_execution_date]`.

## General requirements

- Dates must be parseable by pandas and use one consistent timezone convention.
- Returns must be decimal simple returns.
- Each declared key must be unique.
- Membership must be historical and point-in-time.
- Signals and control variables may use only information available at the
  signal-date close.
- The two markets may have different symbols and currencies, but their table
  structures remain the same.
