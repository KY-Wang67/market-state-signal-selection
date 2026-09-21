"""A-share portfolio construction with historical transaction costs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    backtest_weight_schedule,
    benchmark_path,
    build_weight_schedule,
    performance_summary,
)


def _transfer_fee_rate(date: pd.Timestamp, symbol: str) -> float:
    """Historical securities-transfer fee as a fraction of trade value."""

    if str(symbol) == "689009":
        return 0.2 / 10_000.0
    return (0.2 if pd.Timestamp(date) < pd.Timestamp("2022-04-29") else 0.1) / 10_000.0


def _cost_model(config: dict):
    commission_rate = float(config["commission_bps"]) / 10_000.0
    minimum_commission = float(config["minimum_commission"])

    def calculate(
        execution_date: pd.Timestamp,
        trades: pd.DataFrame,
        portfolio_value: float,
    ) -> tuple[float, dict]:
        total_commission = 0.0
        transfer_fee = 0.0
        stamp_duty = 0.0
        stamp_rate = (
            10.0 / 10_000.0
            if execution_date < pd.Timestamp("2023-08-28")
            else 5.0 / 10_000.0
        )
        for row in trades.loc[trades["delta_weight"].ne(0.0)].itertuples(index=False):
            notional = abs(float(row.delta_weight)) * portfolio_value
            total_commission += max(notional * commission_rate, minimum_commission)
            transfer_fee += notional * _transfer_fee_rate(execution_date, row.symbol)
            if row.delta_weight < 0.0:
                stamp_duty += notional * stamp_rate
        total = total_commission + transfer_fee + stamp_duty
        return total / portfolio_value, {
            "commission": total_commission,
            "transfer_fee": transfer_fee,
            "stamp_duty": stamp_duty,
        }

    return calculate


def build_a_share_portfolios(
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
        _cost_model(market["costs"]),
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
