#!/usr/bin/env python3
"""Run the integrated A-share or US-stock reproduction pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import (
    load_config,
    parse_windows,
    read_table,
    resolve_path,
    set_random_seed,
    write_json,
)


CORE_FILES = {
    "daily_panel": "daily_panel.parquet",
    "weekly_panel": "weekly_panel.parquet",
    "market_observations": "market_observations.parquet",
    "benchmark_total_return": "benchmark_total_return.parquet",
}


def _market_paths(config: dict, market_name: str) -> dict[str, Path]:
    market = config["markets"][market_name]
    data_dir = resolve_path(config, market["data_dir"])
    return {role: data_dir / filename for role, filename in CORE_FILES.items()}


def inspect_data(config: dict, markets: list[str]) -> dict:
    """Report data availability without downloading or changing anything."""

    report = {}
    for market_name in markets:
        paths = _market_paths(config, market_name)
        market = config["markets"][market_name]
        factor_status = {}
        for model, value in market.get("factor_files", {}).items():
            path = resolve_path(config, value)
            factor_status[model] = {
                "configured": path is not None,
                "path": str(path) if path is not None else None,
                "exists": bool(path is not None and path.exists()),
            }
        report[market_name] = {
            "ready": all(path.exists() for path in paths.values()),
            "required_files": {
                role: {"path": str(path), "exists": path.exists()}
                for role, path in paths.items()
            },
            "optional_factor_files": factor_status,
        }
    return report


def run_market(config: dict, market_name: str, overwrite: bool) -> dict:
    """Run all main empirical stages for one market."""

    from hmm import estimate_regimes
    from ranking import rank_stocks
    from risk_adjustment import run_risk_adjustment
    from signals import select_signals

    market = config["markets"][market_name]
    paths = _market_paths(config, market_name)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required {market_name} data: {missing}")

    output_root = resolve_path(config, config.get("output_dir", "outputs"))
    output = output_root / market_name
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    seed = int(config["random_seed"])
    set_random_seed(seed)
    windows = parse_windows(config)
    daily = read_table(paths["daily_panel"])
    weekly = read_table(paths["weekly_panel"])
    market_observations = paths["market_observations"]
    benchmark = read_table(paths["benchmark_total_return"])

    probabilities, _ = estimate_regimes(
        market_observations,
        windows,
        config["method"]["hmm"],
        output,
    )
    selected = select_signals(
        weekly,
        probabilities,
        windows,
        config["method"]["signal_selection"],
        output,
    )
    _, targets = rank_stocks(
        weekly,
        daily,
        probabilities,
        selected,
        windows,
        config["method"],
        market["xgboost"],
        seed,
        output,
    )

    action_path = resolve_path(config, market.get("corporate_actions"))
    actions = read_table(action_path) if action_path is not None and action_path.exists() else None
    if market_name == "a_share":
        from portfolio_a_share import build_a_share_portfolios

        nav, performance = build_a_share_portfolios(
            targets, daily, weekly, benchmark, config["method"], market, output, actions
        )
    else:
        from portfolio_us_stock import build_us_stock_portfolios

        nav, performance = build_us_stock_portfolios(
            targets, daily, weekly, benchmark, config["method"], market, output, actions
        )

    factor_files = {
        model: resolve_path(config, path)
        for model, path in market.get("factor_files", {}).items()
        if path not in (None, "")
    }
    missing_factors = [str(path) for path in factor_files.values() if not path.exists()]
    if missing_factors:
        raise FileNotFoundError(f"Configured factor files do not exist: {missing_factors}")
    risk_adjustment = run_risk_adjustment(nav, factor_files, output)
    summary = {
        "market": market_name,
        "random_seed": seed,
        "rolling_windows": int(len(windows)),
        "strategies": performance["strategy"].tolist(),
        "risk_models": (
            sorted(risk_adjustment["market_model"].unique().tolist())
            if not risk_adjustment.empty
            else []
        ),
        "output_directory": str(output),
    }
    write_json(output / "run_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--market", choices=("a_share", "us_stock", "all"), default="all"
    )
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    markets = list(config["markets"]) if args.market == "all" else [args.market]
    if args.check_data:
        report = inspect_data(config, markets)
        print(json.dumps(report, indent=2))
        return 0 if all(item["ready"] for item in report.values()) else 2
    summaries = [run_market(config, market, args.overwrite) for market in markets]
    print(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
