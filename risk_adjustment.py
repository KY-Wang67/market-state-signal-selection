"""Common OLS risk adjustment for A-share FF3/CH3 and US FF5 models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import compound_return, normalize_panel, read_table, require_columns


MODEL_FACTORS = {
    "ff3": ("mkt_rf", "smb", "hml"),
    "ch3": ("mkt_rf", "smb", "vmg"),
    "ff5": ("mkt_rf", "smb", "hml", "rmw", "cma"),
}


def _load_factors(path: str | Path, model: str) -> pd.DataFrame:
    factors = read_table(path)
    factors.columns = [
        str(column).strip().lower().replace("-", "_") for column in factors.columns
    ]
    aliases = {"mktrf": "mkt_rf", "mkt_rf_": "mkt_rf", "rf_dly": "rf"}
    factors = factors.rename(columns={key: value for key, value in aliases.items() if key in factors})
    factors = normalize_panel(factors)
    required = ("date", "rf", *MODEL_FACTORS[model])
    require_columns(factors, required, f"{model}_factors")
    for column in required[1:]:
        factors[column] = pd.to_numeric(factors[column], errors="raise")
    if factors["date"].duplicated().any():
        raise ValueError(f"{model} factor data contain duplicate dates")
    return factors.sort_values("date")


def _aggregate_intervals(
    intervals: pd.DataFrame,
    daily: pd.DataFrame,
    factor_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    for interval in intervals.itertuples(index=False):
        selected = daily.loc[
            daily["date"].gt(interval.execution_date)
            & daily["date"].le(interval.next_execution_date)
        ]
        if selected.empty:
            raise RuntimeError(
                "Factor data do not cover interval "
                f"({interval.execution_date}, {interval.next_execution_date}]"
            )
        row = {
            "execution_date": interval.execution_date,
            "next_execution_date": interval.next_execution_date,
            "rf": compound_return(selected["rf"]),
        }
        for factor in factor_columns:
            row[factor] = compound_return(selected[factor])
        rows.append(row)
    return pd.DataFrame(rows)


def _ols(dependent: np.ndarray, factors: np.ndarray) -> dict:
    design = np.column_stack([np.ones(len(dependent)), factors])
    coefficients, _, _, _ = np.linalg.lstsq(design, dependent, rcond=None)
    fitted = design @ coefficients
    residual = dependent - fitted
    total = float(np.sum((dependent - dependent.mean()) ** 2))
    residual_sum = float(residual @ residual)
    r_squared = 1.0 - residual_sum / total if total > 0.0 else np.nan
    return {
        "alpha_weekly": float(coefficients[0]),
        "alpha_annualized": float(52.0 * coefficients[0]),
        "r_squared": r_squared,
        "coefficients": coefficients[1:],
    }


def run_risk_adjustment(
    portfolio_returns: pd.DataFrame,
    factor_files: dict[str, str | Path | None],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Run every configured model and silently skip unconfigured factor files."""

    nav = normalize_panel(portfolio_returns)
    require_columns(
        nav,
        ("strategy", "execution_date", "next_execution_date", "net_return"),
        "portfolio_returns",
    )
    intervals = nav[["execution_date", "next_execution_date"]].drop_duplicates().sort_values(
        "execution_date"
    )
    rows: list[dict] = []
    for model, path in factor_files.items():
        if path in (None, ""):
            continue
        model_name = str(model).lower()
        if model_name not in MODEL_FACTORS:
            raise ValueError(f"Unsupported risk model: {model}")
        factors = MODEL_FACTORS[model_name]
        daily = _load_factors(path, model_name)
        aggregated = _aggregate_intervals(intervals, daily, factors)
        for strategy, frame in nav.groupby("strategy", sort=True):
            merged = frame.merge(
                aggregated,
                on=["execution_date", "next_execution_date"],
                how="inner",
                validate="one_to_one",
            ).dropna(subset=["net_return", "rf", *factors])
            result = _ols(
                (merged["net_return"] - merged["rf"]).to_numpy(dtype=float),
                merged[list(factors)].to_numpy(dtype=float),
            )
            row = {
                "market_model": model_name.upper(),
                "strategy": strategy,
                "observations": int(len(merged)),
                "alpha_weekly": result["alpha_weekly"],
                "alpha_annualized": result["alpha_annualized"],
                "r_squared": result["r_squared"],
            }
            row.update(
                {
                    f"beta_{factor}": float(value)
                    for factor, value in zip(factors, result["coefficients"])
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(Path(output_dir) / "risk_adjustment.csv", index=False)
    return result
