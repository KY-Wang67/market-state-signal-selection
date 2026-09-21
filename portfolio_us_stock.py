"""US-stock portfolio construction with historical regulatory fees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    backtest_weight_schedule,
    benchmark_path,
    build_weight_schedule,
    performance_summary,
)


def _section_31_rate(date: pd.Timestamp) -> float:
    """Historical SEC Section 31 covered-sale rate."""

    date = pd.Timestamp(date)
    if date < pd.Timestamp("2023-02-27"):
        return 22.90 / 1_000_000.0
    if date < pd.Timestamp("2024-05-22"):
        return 8.00 / 1_000_000.0
    if date < pd.Timestamp("2025-05-14"):
        return 27.80 / 1_000_000.0
    return 0.0


def _taf_schedule(date: pd.Timestamp) -> tuple[float, float]:
    """Historical FINRA TAF dollars per share and cap per sale."""

    year = pd.Timestamp(date).year
    if year <= 2022:
        return 0.000130, 6.49
    if year == 2023:
        return 0.000145, 7.27
    return 0.000166, 8.30


def _cost_model():
    def calculate(
        execution_date: pd.Timestamp,
        trades: pd.DataFrame,
        portfolio_value: float,
    ) -> tuple[float, dict]:
        sales = trades.loc[trades["delta_weight"].lt(0.0)].copy()
        sales["sale_value"] = -sales["delta_weight"] * portfolio_value
        section_31 = float(sales["sale_value"].sum()) * _section_31_rate(execution_date)
        per_share, maximum = _taf_schedule(execution_date)
        taf = 0.0
        for row in sales.itertuples(index=False):
            price = float(row.raw_price)
            if not np.isfinite(price) or price <= 0.0:
                raise RuntimeError(
                    f"Missing positive execution price for {row.symbol} on {execution_date.date()}"
                )
            shares = float(row.sale_value) / price
            taf += min(shares * per_share, maximum)
        total = section_31 + taf
        return total / portfolio_value, {
            "section_31_fee": section_31,
            "finra_taf": taf,
        }

    return calculate


def build_us_stock_portfolios(
    targets: pd.DataFrame,
    daily_panel: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    method: dict,
    market: dict,
    output_dir: str | Path,
    corporate_actions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build model paths and the common external benchmark paths."""

    output = Path(output_dir)
    weights = build_weight_schedule(targets, weekly_panel, daily_panel, method["portfolio"])
    nav, holdings = backtest_weight_schedule(
        weights,
        weekly_panel,
        daily_panel,
        _cost_model(),
        initial_capital=float(market["costs"]["initial_capital"]),
        corporate_actions=corporate_actions,
    )
    official = benchmark_path(
        benchmark,
        targets[["signal_date", "execution_date", "next_execution_date"]],
        value_column=str(market.get("benchmark_value_column", "total_return_index")),
    )
    nav = pd.concat([nav, official], ignore_index=True)
    summaries = []
    summary_nav = nav.loc[nav["strategy"].ne("Fixed")]
    for strategy, frame in summary_nav.groupby("strategy", sort=True):
        summaries.append({"strategy": strategy, **performance_summary(frame)})
    nav.to_csv(output / "portfolio_returns.csv", index=False)
    holdings.to_parquet(output / "portfolio_holdings.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output / "performance_summary.csv", index=False)
    return nav, pd.DataFrame(summaries)
