"""Shared utilities used by the reproduction pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATE_COLUMNS = (
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
)
ALPHA_COLUMNS = tuple(f"alpha101_{i:03d}" for i in range(1, 102))


def load_config(path: str | Path) -> dict:
    """Read the public JSON configuration."""

    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_root"] = str(config_path.parent)
    return config


def resolve_path(config: dict, value: str | Path | None) -> Path | None:
    """Resolve a configured path relative to the repository root."""

    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    root = Path(config["_root"])
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def set_random_seed(seed: int) -> None:
    """Set the shared Python and NumPy random seed."""

    random.seed(int(seed))
    np.random.seed(int(seed))


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported table format: {file_path}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Fail early when a required input field is absent."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def parse_windows(config: dict) -> pd.DataFrame:
    """Return the common rolling windows as a typed table."""

    windows = pd.DataFrame(config["windows"]).copy()
    require_columns(windows, ("window_id", *DATE_COLUMNS), "windows")
    for column in DATE_COLUMNS:
        windows[column] = pd.to_datetime(windows[column], errors="raise")
    return windows.sort_values("test_start").reset_index(drop=True)


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common identifier and date fields without changing values."""

    output = frame.copy()
    for column in ("date", "signal_date", "execution_date", "next_execution_date"):
        if column in output:
            output[column] = pd.to_datetime(output[column], errors="raise")
    if "symbol" in output:
        output["symbol"] = output["symbol"].astype(str).str.strip()
    return output


def compound_return(values: Iterable[float]) -> float:
    """Compound simple returns."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.prod(1.0 + array) - 1.0) if len(array) else np.nan


def capped_weights(values: pd.Series, cap: float) -> pd.Series:
    """Normalize positive values while enforcing a feasible long-only cap."""

    raw = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if raw.isna().any() or raw.le(0.0).any():
        raise ValueError("Portfolio weights require finite positive inputs")
    effective_cap = max(float(cap), 1.0 / len(raw))
    output = pd.Series(0.0, index=raw.index, dtype=float)
    active = pd.Series(True, index=raw.index)
    remaining = 1.0
    while active.any():
        proposal = raw.loc[active] / raw.loc[active].sum() * remaining
        breached = proposal.gt(effective_cap + 1e-14)
        if not breached.any():
            output.loc[proposal.index] = proposal
            break
        capped = proposal.index[breached]
        output.loc[capped] = effective_cap
        active.loc[capped] = False
        remaining = 1.0 - float(output.sum())
    return output / output.sum()


def trailing_volatility(
    daily: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    lookback: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """Compute stock volatility known at each signal-date close."""

    require_columns(daily, ("date", "symbol", "adj_close"), "daily_panel")
    data = normalize_panel(daily[["date", "symbol", "adj_close"]])
    data["adj_close"] = pd.to_numeric(data["adj_close"], errors="coerce")
    data = data.sort_values(["symbol", "date"])
    data["daily_return"] = data.groupby("symbol", sort=False)["adj_close"].pct_change(
        fill_method=None
    )
    rolling = data.groupby("symbol", sort=False)["daily_return"].rolling(
        int(lookback), min_periods=int(minimum_observations)
    )
    data["volatility"] = rolling.std().reset_index(level=0, drop=True)
    keep_dates = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).unique()
    return data.loc[
        data["date"].isin(keep_dates), ["date", "symbol", "volatility"]
    ].rename(columns={"date": "signal_date"})


def inverse_volatility_weights(
    symbols: list[str],
    volatility: pd.DataFrame,
    cap: float,
) -> pd.Series:
    """Return capped inverse-volatility weights, with equal-weight fallback."""

    index = pd.Index(symbols, name="symbol")
    if volatility.empty or not {"symbol", "volatility"}.issubset(volatility.columns):
        return pd.Series(1.0 / len(index), index=index)
    vol = volatility.set_index("symbol")["volatility"].reindex(index)
    valid = vol.notna() & np.isfinite(vol) & vol.gt(0.0)
    if not valid.all():
        return pd.Series(1.0 / len(index), index=index)
    return capped_weights(1.0 / vol, cap)


def buffered_names(
    scores: pd.Series,
    current: list[str],
    *,
    holdings: int,
    entry_rank: int,
    exit_rank: int,
    maximum_replacements: int,
) -> list[str]:
    """Apply the buffered ranking rule and separate forced departures."""

    ranking = scores.dropna().sort_values(ascending=False)
    ranks = {symbol: rank for rank, symbol in enumerate(ranking.index, start=1)}
    if not current:
        return ranking.head(holdings).index.tolist()

    # Names absent from the current eligible universe are forced departures.
    selected = [symbol for symbol in dict.fromkeys(current) if symbol in ranks]
    voluntary_exits = sorted(
        [symbol for symbol in selected if ranks[symbol] > exit_rank],
        key=lambda symbol: ranks[symbol],
        reverse=True,
    )
    entrants = [
        symbol for symbol in ranking.head(entry_rank).index if symbol not in selected
    ]
    replacements = min(
        int(maximum_replacements), len(voluntary_exits), len(entrants)
    )
    for symbol in voluntary_exits[:replacements]:
        selected.remove(symbol)
    selected.extend(entrants[:replacements])
    for symbol in ranking.index:
        if symbol not in selected:
            selected.append(symbol)
        if len(selected) == holdings:
            break
    return selected[:holdings]


def build_weight_schedule(
    targets: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    daily_panel: pd.DataFrame,
    portfolio: dict,
) -> pd.DataFrame:
    """Build model and common benchmark target weights."""

    weekly = normalize_panel(weekly_panel)
    daily = normalize_panel(daily_panel)
    target_panel = normalize_panel(targets)
    require_columns(
        weekly,
        (
            "signal_date",
            "execution_date",
            "next_execution_date",
            "symbol",
            "forward_return",
            "is_member",
        ),
        "weekly_panel",
    )
    target_intervals = target_panel[
        ["signal_date", "execution_date", "next_execution_date"]
    ].drop_duplicates()
    signal_dates = target_intervals["signal_date"].drop_duplicates()
    volatility = trailing_volatility(
        daily,
        signal_dates,
        int(portfolio["volatility_lookback"]),
        int(portfolio["minimum_volatility_observations"]),
    )
    volatility_by_date = {
        date: frame[["symbol", "volatility"]]
        for date, frame in volatility.groupby("signal_date", sort=False)
    }
    cap = float(portfolio["maximum_stock_weight"])
    rows: list[dict] = []

    def append_group(
        strategy: str,
        signal_date: pd.Timestamp,
        execution_date: pd.Timestamp,
        next_execution_date: pd.Timestamp,
        weights: pd.Series,
        weighting_method: str,
    ) -> None:
        for symbol, weight in weights.items():
            rows.append(
                {
                    "strategy": strategy,
                    "weighting_method": weighting_method,
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "next_execution_date": next_execution_date,
                    "symbol": str(symbol),
                    "target_weight": float(weight),
                }
            )

    for (strategy, signal_date), group in target_panel.groupby(
        ["strategy", "signal_date"], sort=True
    ):
        symbols = group["symbol"].drop_duplicates().tolist()
        vol = volatility_by_date.get(pd.Timestamp(signal_date), pd.DataFrame())
        weights = inverse_volatility_weights(symbols, vol, cap)
        append_group(
            str(strategy),
            pd.Timestamp(signal_date),
            pd.Timestamp(group["execution_date"].iloc[0]),
            pd.Timestamp(group["next_execution_date"].iloc[0]),
            weights,
            "inverse_volatility_60d",
        )

    eligible_weekly = weekly.loc[
        pd.to_numeric(weekly["is_member"], errors="coerce").eq(1)
    ].merge(
        target_intervals,
        on=["signal_date", "execution_date", "next_execution_date"],
        how="inner",
        validate="many_to_one",
    )
    for signal_date, group in eligible_weekly.groupby("signal_date", sort=True):
        symbols = group["symbol"].drop_duplicates().tolist()
        if not symbols:
            continue
        execution_date = pd.Timestamp(group["execution_date"].iloc[0])
        next_execution_date = pd.Timestamp(group["next_execution_date"].iloc[0])
        equal = pd.Series(1.0 / len(symbols), index=pd.Index(symbols, name="symbol"))
        append_group(
            "AllConstituentsEqualWeight",
            pd.Timestamp(signal_date),
            execution_date,
            next_execution_date,
            equal,
            "equal_weight",
        )
        vol = volatility_by_date.get(pd.Timestamp(signal_date), pd.DataFrame())
        inverse = inverse_volatility_weights(symbols, vol, cap)
        append_group(
            "AllConstituentsInverseVolatility",
            pd.Timestamp(signal_date),
            execution_date,
            next_execution_date,
            inverse,
            "inverse_volatility_60d",
        )
    return pd.DataFrame(rows)


def remap_weights_for_actions(
    weights: dict[str, float],
    actions: pd.DataFrame | None,
    previous_date: pd.Timestamp | None,
    execution_date: pd.Timestamp,
) -> dict[str, float]:
    """Carry weights across identifier changes recorded by the user."""

    if actions is None or actions.empty:
        return weights
    lower = pd.Timestamp.min if previous_date is None else pd.Timestamp(previous_date)
    active = actions.loc[
        actions["effective_date"].gt(lower)
        & actions["effective_date"].le(pd.Timestamp(execution_date))
    ]
    remapped = dict(weights)
    for row in active.sort_values("effective_date").itertuples(index=False):
        source, destination = str(row.source_symbol), str(row.destination_symbol)
        if source in remapped:
            remapped[destination] = remapped.get(destination, 0.0) + remapped.pop(source)
    return remapped


def backtest_weight_schedule(
    schedule: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    daily_panel: pd.DataFrame,
    cost_function,
    *,
    initial_capital: float,
    corporate_actions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run continuous-wealth portfolios from target weights and interval returns."""

    weekly = normalize_panel(weekly_panel)
    daily = normalize_panel(daily_panel)
    if corporate_actions is not None and not corporate_actions.empty:
        corporate_actions = normalize_panel(corporate_actions)
        require_columns(
            corporate_actions,
            ("effective_date", "source_symbol", "destination_symbol"),
            "corporate_actions",
        )
        corporate_actions["effective_date"] = pd.to_datetime(
            corporate_actions["effective_date"], errors="raise"
        )
        corporate_actions["source_symbol"] = corporate_actions["source_symbol"].astype(str)
        corporate_actions["destination_symbol"] = corporate_actions[
            "destination_symbol"
        ].astype(str)
    return_lookup = weekly.set_index(["signal_date", "symbol"])["forward_return"]
    raw_price = (
        daily.set_index(["date", "symbol"])["raw_close"]
        if "raw_close" in daily
        else daily.set_index(["date", "symbol"])["adj_close"]
    )
    nav_rows: list[dict] = []
    holding_rows: list[dict] = []

    for strategy, path in schedule.groupby("strategy", sort=True):
        wealth = 1.0
        pretrade: dict[str, float] = {}
        previous_execution: pd.Timestamp | None = None
        for signal_date, group in path.groupby("signal_date", sort=True):
            execution_date = pd.Timestamp(group["execution_date"].iloc[0])
            next_execution_date = pd.Timestamp(group["next_execution_date"].iloc[0])
            pretrade = remap_weights_for_actions(
                pretrade, corporate_actions, previous_execution, execution_date
            )
            target = group.set_index("symbol")["target_weight"].astype(float).to_dict()
            names = sorted(set(pretrade) | set(target))
            trades = pd.DataFrame(
                {
                    "symbol": names,
                    "pretrade_weight": [pretrade.get(name, 0.0) for name in names],
                    "target_weight": [target.get(name, 0.0) for name in names],
                }
            )
            trades["delta_weight"] = trades["target_weight"] - trades["pretrade_weight"]
            trades["raw_price"] = [
                raw_price.get((execution_date, name), np.nan) for name in names
            ]
            portfolio_value = float(initial_capital) * wealth
            cost_fraction, cost_detail = cost_function(
                execution_date, trades, portfolio_value
            )

            asset_returns = {}
            for symbol in target:
                key = (pd.Timestamp(signal_date), symbol)
                if key not in return_lookup.index or pd.isna(return_lookup.loc[key]):
                    raise RuntimeError(
                        f"Missing interval return for {symbol} at {pd.Timestamp(signal_date).date()}"
                    )
                asset_returns[symbol] = float(return_lookup.loc[key])
            gross_return = float(
                sum(target[symbol] * asset_returns[symbol] for symbol in target)
            )
            net_return = (1.0 - cost_fraction) * (1.0 + gross_return) - 1.0
            wealth *= 1.0 + net_return
            denominator = max(1.0 + gross_return, 1e-12)
            pretrade = {
                symbol: target[symbol] * (1.0 + asset_returns[symbol]) / denominator
                for symbol in target
            }
            nav_rows.append(
                {
                    "strategy": strategy,
                    "weighting_method": group["weighting_method"].iloc[0],
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "next_execution_date": next_execution_date,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "wealth": wealth,
                    "one_way_turnover": 0.5 * float(trades["delta_weight"].abs().sum()),
                    "cost_fraction": cost_fraction,
                    **cost_detail,
                }
            )
            holding_rows.extend(
                {
                    "strategy": strategy,
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "symbol": symbol,
                    "target_weight": weight,
                }
                for symbol, weight in target.items()
            )
            previous_execution = execution_date
    return pd.DataFrame(nav_rows), pd.DataFrame(holding_rows)


def benchmark_path(
    benchmark: pd.DataFrame,
    intervals: pd.DataFrame,
    value_column: str = "total_return_index",
) -> pd.DataFrame:
    """Convert a total-return index level into the portfolio holding intervals."""

    data = normalize_panel(benchmark)
    require_columns(data, ("date", value_column), "benchmark_total_return")
    values = data[["date", value_column]].dropna().sort_values("date")
    rows: list[dict] = []
    for interval in intervals.drop_duplicates(
        ["execution_date", "next_execution_date"]
    ).itertuples(index=False):
        start = values.loc[values["date"].le(interval.execution_date), value_column]
        end = values.loc[values["date"].le(interval.next_execution_date), value_column]
        if start.empty or end.empty:
            raise RuntimeError("Benchmark does not cover all portfolio intervals")
        interval_return = float(end.iloc[-1] / start.iloc[-1] - 1.0)
        rows.append(
            {
                "strategy": "OfficialTotalReturnIndex",
                "weighting_method": "official_benchmark",
                "signal_date": interval.signal_date,
                "execution_date": interval.execution_date,
                "next_execution_date": interval.next_execution_date,
                "gross_return": interval_return,
                "net_return": interval_return,
                "one_way_turnover": 0.0,
                "cost_fraction": 0.0,
            }
        )
    output = pd.DataFrame(rows).sort_values("execution_date")
    output["wealth"] = (1.0 + output["net_return"]).cumprod()
    return output


def performance_summary(nav: pd.DataFrame, periods_per_year: int = 52) -> dict:
    """Summarize a weekly net-return path."""

    returns = pd.to_numeric(nav["net_return"], errors="coerce").dropna()
    if returns.empty:
        return {}
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    years = len(returns) / periods_per_year
    return {
        "observations": int(len(returns)),
        "annual_return": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "annual_volatility": volatility,
        "sharpe_ratio": (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))
            if volatility > 0.0
            else np.nan
        ),
        "maximum_drawdown": float(drawdown.min()),
        "final_wealth": float(wealth.iloc[-1]),
        "average_one_way_turnover": (
            float(nav["one_way_turnover"].mean())
            if "one_way_turnover" in nav
            else 0.0
        ),
        "total_cost_fraction": (
            float(nav["cost_fraction"].sum()) if "cost_fraction" in nav else 0.0
        ),
    }


def write_json(path: str | Path, payload: dict | list) -> None:
    """Write readable JSON with NumPy values converted to Python scalars."""

    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        return value

    Path(path).write_text(
        json.dumps(convert(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
