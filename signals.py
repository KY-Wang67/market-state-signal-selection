"""Alpha101 signal evaluation and state-dependent signal selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import ALPHA_COLUMNS, normalize_panel, require_columns


def _rank_ic(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute one cross-sectional Rank IC per signal date and factor."""

    rows: list[dict] = []
    for date, cross_section in panel.groupby("signal_date", sort=True):
        future = pd.to_numeric(cross_section["forward_return"], errors="coerce").rank(
            pct=True
        )
        factors = cross_section[list(ALPHA_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        ).rank(pct=True)
        values = factors.corrwith(future, method="pearson")
        rows.extend(
            {
                "signal_date": date,
                "state_label": (
                    str(cross_section["state_label"].iloc[0])
                    if "state_label" in cross_section
                    else "pooled"
                ),
                "feature": feature,
                "rank_ic": value,
            }
            for feature, value in values.items()
        )
    return pd.DataFrame(rows)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
    clean = clean.loc[clean["weight"].gt(0.0)]
    return (
        float(np.average(clean["value"], weights=clean["weight"]))
        if not clean.empty
        else np.nan
    )


def _factor_summary(
    rank_ic: pd.DataFrame,
    probabilities: pd.DataFrame,
    state_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_probabilities = probabilities[
        ["date", *[f"probability__{label}" for label in state_labels]]
    ].drop_duplicates("date")
    merged = rank_ic.merge(
        date_probabilities, left_on="signal_date", right_on="date", how="left"
    )
    state_rows: list[dict] = []
    for state_label in state_labels:
        probability = f"probability__{state_label}"
        for feature, group in merged.groupby("feature", sort=True):
            state_rows.append(
                {
                    "state_label": state_label,
                    "feature": feature,
                    "mean_ic": _weighted_mean(group["rank_ic"], group[probability]),
                }
            )
    pooled = (
        merged.groupby("feature", as_index=False)["rank_ic"]
        .mean()
        .rename(columns={"rank_ic": "mean_ic"})
    )
    return pd.DataFrame(state_rows), pooled


def _select(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    rank_ic: pd.DataFrame,
    *,
    state_label: str | None,
    count: int,
    correlation_cap: float,
) -> pd.DataFrame:
    """Apply the stable-mean score and a training-only redundancy cap."""

    if state_label is None:
        train_values = train.set_index("feature")["mean_ic"].reindex(ALPHA_COLUMNS)
        valid_values = valid.set_index("feature")["mean_ic"].reindex(ALPHA_COLUMNS)
        redundancy_source = rank_ic
    else:
        train_values = (
            train.loc[train["state_label"].eq(state_label)]
            .set_index("feature")["mean_ic"]
            .reindex(ALPHA_COLUMNS)
        )
        valid_values = (
            valid.loc[valid["state_label"].eq(state_label)]
            .set_index("feature")["mean_ic"]
            .reindex(ALPHA_COLUMNS)
        )
        redundancy_source = rank_ic
        redundancy_source = redundancy_source.loc[
            redundancy_source["state_label"].eq(state_label)
        ]

    orientation = np.sign(train_values).replace(0.0, 1.0)
    candidates = pd.DataFrame(
        {
            "feature": ALPHA_COLUMNS,
            "orientation": orientation.to_numpy(),
            "training_ic": train_values.to_numpy(),
            "validation_ic": valid_values.to_numpy(),
        }
    ).dropna()
    candidates["oriented_training_ic"] = (
        candidates["orientation"] * candidates["training_ic"]
    )
    candidates["oriented_validation_ic"] = (
        candidates["orientation"] * candidates["validation_ic"]
    )
    candidates = candidates.loc[candidates["oriented_validation_ic"].gt(0.0)]
    candidates["selection_score"] = 0.5 * (
        candidates["oriented_training_ic"] + candidates["oriented_validation_ic"]
    )
    candidates = candidates.sort_values(
        ["selection_score", "feature"], ascending=[False, True]
    )

    ic_matrix = redundancy_source.pivot(
        index="signal_date", columns="feature", values="rank_ic"
    )
    redundancy = (
        ic_matrix.corr(method="spearman")
        .reindex(index=ALPHA_COLUMNS, columns=ALPHA_COLUMNS)
        .fillna(0.0)
    )
    selected: list[pd.Series] = []
    for _, candidate in candidates.iterrows():
        if all(
            abs(float(redundancy.at[candidate["feature"], prior["feature"]]))
            <= correlation_cap
            for prior in selected
        ):
            selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Could not select {count} signals under the correlation cap")
    output = pd.DataFrame(selected).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    return output


def select_signals(
    weekly_panel: pd.DataFrame,
    probabilities: pd.DataFrame,
    windows: pd.DataFrame,
    method: dict,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Select State, Global and first-window Fixed signal sets."""

    weekly = normalize_panel(weekly_panel)
    require_columns(
        weekly,
        ("signal_date", "next_execution_date", "forward_return", *ALPHA_COLUMNS),
        "weekly_panel",
    )
    rows: list[pd.DataFrame] = []
    fixed: pd.DataFrame | None = None
    factor_count = int(method["factors_per_state"])
    correlation_cap = float(method["maximum_abs_training_ic_correlation"])

    for window in windows.itertuples(index=False):
        window_probabilities = probabilities.loc[
            probabilities["window_id"].astype(str).eq(str(window.window_id))
        ]
        state_labels = sorted(
            column.replace("probability__", "")
            for column in window_probabilities
            if column.startswith("probability__")
        )
        segments = {}
        for name, start, end in (
            ("train", window.train_start, window.train_end),
            ("valid", window.valid_start, window.valid_end),
        ):
            subset = weekly.loc[
                weekly["signal_date"].between(start, end)
                & weekly["next_execution_date"].le(end)
            ].copy()
            subset = subset.merge(
                window_probabilities[["date", "state_label"]],
                left_on="signal_date",
                right_on="date",
                how="inner",
                validate="many_to_one",
            ).drop(columns="date")
            ic = _rank_ic(subset)
            state_summary, pooled_summary = _factor_summary(
                ic, window_probabilities, state_labels
            )
            segments[name] = {
                "rank_ic": ic,
                "state": state_summary,
                "pooled": pooled_summary,
            }

        for state_label in state_labels:
            selected = _select(
                segments["train"]["state"],
                segments["valid"]["state"],
                segments["train"]["rank_ic"],
                state_label=state_label,
                count=factor_count,
                correlation_cap=correlation_cap,
            )
            selected.insert(0, "state_label", state_label)
            selected.insert(0, "factor_set", "State")
            selected.insert(0, "window_id", str(window.window_id))
            rows.append(selected)

        global_selected = _select(
            segments["train"]["pooled"],
            segments["valid"]["pooled"],
            segments["train"]["rank_ic"],
            state_label=None,
            count=factor_count,
            correlation_cap=correlation_cap,
        )
        if fixed is None:
            fixed = global_selected.copy()
        for factor_set, selected in (("Global", global_selected), ("Fixed", fixed)):
            item = selected.copy()
            item.insert(0, "state_label", "pooled")
            item.insert(0, "factor_set", factor_set)
            item.insert(0, "window_id", str(window.window_id))
            rows.append(item)

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(Path(output_dir) / "selected_signals.csv", index=False)
    return result
