"""Probability-gated XGBoost ranking and buffered stock selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from common import (
    ALPHA_COLUMNS,
    buffered_names,
    normalize_panel,
    require_columns,
)


CONTROL_FEATURES = (
    "momentum_120_20",
    "short_return_5",
    "volatility_20",
    "log_float_market_value",
    "turnover_value_20",
)


def build_control_features(
    daily_panel: pd.DataFrame, signal_dates: pd.Series
) -> pd.DataFrame:
    """Create five common price-volume controls at the signal close."""

    require_columns(
        daily_panel,
        ("date", "symbol", "adj_close", "amount", "float_market_value"),
        "daily_panel",
    )
    data = normalize_panel(
        daily_panel[["date", "symbol", "adj_close", "amount", "float_market_value"]]
    ).sort_values(["symbol", "date"])
    for column in ("adj_close", "amount", "float_market_value"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    grouped = data.groupby("symbol", sort=False)
    data["daily_return"] = grouped["adj_close"].pct_change(fill_method=None)
    data["momentum_120_20"] = grouped["adj_close"].shift(20) / grouped[
        "adj_close"
    ].shift(120) - 1.0
    data["short_return_5"] = data["adj_close"] / grouped["adj_close"].shift(5) - 1.0
    data["volatility_20"] = (
        data.groupby("symbol", sort=False)["daily_return"]
        .rolling(20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )
    data["log_float_market_value"] = np.log(
        data["float_market_value"].where(data["float_market_value"].gt(0.0))
    )
    data["turnover_value_20"] = (
        data.groupby("symbol", sort=False)["amount"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
        / data["float_market_value"].where(data["float_market_value"].gt(0.0))
    )
    dates = pd.DatetimeIndex(pd.to_datetime(signal_dates)).unique()
    return data.loc[
        data["date"].isin(dates), ["date", "symbol", *CONTROL_FEATURES]
    ].rename(columns={"date": "signal_date"})


def _mean_rank_ic(panel: pd.DataFrame, features: list[str]) -> pd.Series:
    rows: list[pd.Series] = []
    for _, cross_section in panel.groupby("signal_date", sort=True):
        future = pd.to_numeric(cross_section["forward_return"], errors="coerce").rank(
            pct=True
        )
        ranked = cross_section[features].apply(pd.to_numeric, errors="coerce").rank(
            pct=True
        )
        rows.append(ranked.corrwith(future))
    if not rows:
        return pd.Series(1.0, index=features)
    return pd.DataFrame(rows).mean().reindex(features)


def _channels(
    factor_set: str,
    selected: pd.DataFrame,
    train: pd.DataFrame,
    state_labels: list[str],
) -> list[dict]:
    """Build the Alpha probability channels plus five common controls."""

    channel_rows: list[dict] = []
    if factor_set == "State":
        bundles = {
            state: selected.loc[selected["state_label"].eq(state)]
            for state in state_labels
        }
    elif factor_set in {"Global", "Fixed"}:
        pooled = selected.loc[selected["factor_set"].eq(factor_set)]
        bundles = {state: pooled for state in state_labels}
    elif factor_set == "Full101":
        orientation = np.sign(_mean_rank_ic(train, list(ALPHA_COLUMNS))).replace(
            0.0, 1.0
        )
        full = pd.DataFrame(
            {"feature": ALPHA_COLUMNS, "orientation": orientation.to_numpy()}
        )
        bundles = {state: full for state in state_labels}
    else:
        raise ValueError(f"Unknown factor set: {factor_set}")

    for state in state_labels:
        for row in bundles[state].itertuples(index=False):
            channel_rows.append(
                {
                    "name": f"{state}__{row.feature}",
                    "feature": str(row.feature),
                    "orientation": float(row.orientation),
                    "probability": f"probability__{state}",
                }
            )
    control_orientation = np.sign(_mean_rank_ic(train, list(CONTROL_FEATURES))).replace(
        0.0, 1.0
    )
    for feature in CONTROL_FEATURES:
        channel_rows.append(
            {
                "name": f"control__{feature}",
                "feature": feature,
                "orientation": float(control_orientation.loc[feature]),
                "probability": None,
            }
        )
    return channel_rows


def _channel_matrix(panel: pd.DataFrame, channels: list[dict]) -> pd.DataFrame:
    raw_features = list(dict.fromkeys(row["feature"] for row in channels))
    ranks = panel[raw_features].apply(pd.to_numeric, errors="coerce").groupby(
        panel["signal_date"], sort=False
    ).rank(pct=True, method="average")
    matrix = {}
    for channel in channels:
        oriented = ranks[channel["feature"]]
        if channel["orientation"] < 0.0:
            oriented = 1.0 - oriented
        if channel["probability"] is not None:
            probability = pd.to_numeric(panel[channel["probability"]], errors="coerce")
            oriented = 0.5 + probability * (oriented - 0.5)
        matrix[channel["name"]] = oriented.astype("float32")
    return pd.DataFrame(matrix, index=panel.index)


def _relevance_labels(panel: pd.DataFrame) -> pd.Series:
    """Map each weekly return cross-section to relevance labels 0--4."""

    labels = pd.Series(np.nan, index=panel.index)
    for _, cross_section in panel.groupby("signal_date", sort=True):
        returns = pd.to_numeric(cross_section["forward_return"], errors="coerce")
        valid = returns.dropna()
        percentile = valid.rank(method="average", pct=True, ascending=True)
        values = np.clip(np.floor(5.0 * percentile), 0, 4).astype(int)
        labels.loc[values.index] = values
    return labels


def _fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    channels: list[dict],
    parameters: dict,
    seed: int,
) -> np.ndarray:
    labels = _relevance_labels(train)
    valid = labels.notna()
    ordered = train.loc[valid].sort_values(["signal_date", "symbol"]).index
    train_matrix = _channel_matrix(train, channels).loc[ordered]
    train_labels = labels.loc[ordered].astype(int).to_numpy()
    qid = pd.factorize(train.loc[ordered, "signal_date"], sort=True)[0]
    model_parameters = dict(parameters)
    model_parameters.update(
        {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@24",
            "tree_method": "hist",
            "random_state": int(seed),
            # Every channel has already been oriented so that larger is better.
            "monotone_constraints": tuple(1 for _ in train_matrix.columns),
        }
    )
    model = XGBRanker(**model_parameters)
    model.fit(train_matrix, train_labels, qid=qid, verbose=False)
    return np.asarray(model.predict(_channel_matrix(test, channels)), dtype=float)


def rank_stocks(
    weekly_panel: pd.DataFrame,
    daily_panel: pd.DataFrame,
    probabilities: pd.DataFrame,
    selected_signals: pd.DataFrame,
    windows: pd.DataFrame,
    method: dict,
    xgboost_parameters: dict,
    random_seed: int,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one pooled ranker per window and strategy, then buffer holdings."""

    weekly = normalize_panel(weekly_panel)
    require_columns(
        weekly,
        (
            "signal_date",
            "execution_date",
            "next_execution_date",
            "symbol",
            "forward_return",
            "is_member",
            *ALPHA_COLUMNS,
        ),
        "weekly_panel",
    )
    controls = build_control_features(daily_panel, weekly["signal_date"])
    weekly = weekly.merge(
        controls, on=["signal_date", "symbol"], how="left", validate="one_to_one"
    )
    weekly = weekly.loc[pd.to_numeric(weekly["is_member"], errors="coerce").eq(1)]

    prediction_frames: list[pd.DataFrame] = []
    factor_sets = ("State", "Global", "Fixed", "Full101")
    for window_index, window in enumerate(windows.itertuples(index=False)):
        window_id = str(window.window_id)
        probability = probabilities.loc[
            probabilities["window_id"].astype(str).eq(window_id)
        ]
        state_labels = sorted(
            column.replace("probability__", "")
            for column in probability
            if column.startswith("probability__")
        )
        working = weekly.merge(
            probability[["date", *[f"probability__{label}" for label in state_labels]]],
            left_on="signal_date",
            right_on="date",
            how="inner",
            validate="many_to_one",
        ).drop(columns="date")
        train = working.loc[
            working["signal_date"].between(window.train_start, window.train_end)
            & working["next_execution_date"].le(window.train_end)
        ].copy()
        test = working.loc[
            working["signal_date"].between(window.test_start, window.test_end)
        ].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"{window_id}: empty XGBoost train or test sample")

        window_selection = selected_signals.loc[
            selected_signals["window_id"].astype(str).eq(window_id)
        ]
        for branch, factor_set in enumerate(factor_sets):
            relevant = window_selection.loc[
                window_selection["factor_set"].eq(factor_set)
            ]
            channels = _channels(factor_set, relevant, train, state_labels)
            score = _fit_and_predict(
                train,
                test,
                channels,
                xgboost_parameters,
                int(random_seed) + 100 * window_index + branch,
            )
            item = test[
                [
                    "signal_date",
                    "execution_date",
                    "next_execution_date",
                    "symbol",
                    "forward_return",
                ]
            ].copy()
            item.insert(0, "window_id", window_id)
            item.insert(1, "strategy", factor_set)
            item["score"] = score
            prediction_frames.append(item)

    scores = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["strategy", "signal_date", "score"], ascending=[True, True, False]
    )
    portfolio = method["portfolio"]
    target_rows: list[dict] = []
    for strategy, strategy_scores in scores.groupby("strategy", sort=True):
        current: list[str] = []
        for signal_date, cross_section in strategy_scores.groupby(
            "signal_date", sort=True
        ):
            ranked = cross_section.drop_duplicates("symbol").set_index("symbol")
            current = buffered_names(
                ranked["score"],
                current,
                holdings=int(portfolio["target_holdings"]),
                entry_rank=int(portfolio["entry_rank"]),
                exit_rank=int(portfolio["exit_rank"]),
                maximum_replacements=int(portfolio["maximum_replacements"]),
            )
            ranks = ranked["score"].rank(method="first", ascending=False).astype(int)
            for symbol in current:
                source = ranked.loc[symbol]
                target_rows.append(
                    {
                        "window_id": source["window_id"],
                        "strategy": strategy,
                        "signal_date": signal_date,
                        "execution_date": source["execution_date"],
                        "next_execution_date": source["next_execution_date"],
                        "symbol": symbol,
                        "score": source["score"],
                        "rank": int(ranks.loc[symbol]),
                    }
                )

    targets = pd.DataFrame(target_rows)
    output = Path(output_dir)
    scores.to_parquet(output / "stock_scores.parquet", index=False)
    targets.to_parquet(output / "target_holdings.parquet", index=False)
    return scores, targets
