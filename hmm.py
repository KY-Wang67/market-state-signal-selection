"""Two-state Gaussian HMM estimation with recursive filtered probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

from common import normalize_panel, read_table, require_columns


def _filtered_probabilities(model: GaussianHMM, values: np.ndarray) -> np.ndarray:
    """Run the forward filter; no future observation enters a probability."""

    emission = model._compute_log_likelihood(values)  # hmmlearn emission density
    log_start = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transition = np.log(np.clip(model.transmat_, 1e-300, None))
    output = np.empty_like(emission)
    current = log_start + emission[0]
    current -= logsumexp(current)
    output[0] = np.exp(current)
    for row in range(1, len(values)):
        current = emission[row] + logsumexp(
            current[:, None] + log_transition, axis=0
        )
        current -= logsumexp(current)
        output[row] = np.exp(current)
    return output


def _fit_best_model(
    train_values: np.ndarray,
    *,
    states: int,
    covariance_type: str,
    initializations: int,
    seed_start: int,
    max_iterations: int,
) -> tuple[GaussianHMM, int, pd.DataFrame]:
    """Choose the fixed-seed initialization with highest training likelihood."""

    models: list[tuple[float, int, GaussianHMM]] = []
    audit: list[dict] = []
    for offset in range(int(initializations)):
        seed = int(seed_start) + offset
        model = GaussianHMM(
            n_components=int(states),
            covariance_type=str(covariance_type),
            n_iter=int(max_iterations),
            random_state=seed,
            min_covar=1e-6,
        )
        try:
            model.fit(train_values)
            likelihood = float(model.score(train_values))
            converged = bool(model.monitor_.converged)
            finite = bool(np.isfinite(likelihood))
        except (ValueError, np.linalg.LinAlgError):
            likelihood, converged, finite = -np.inf, False, False
        audit.append(
            {
                "seed": seed,
                "converged": converged,
                "train_log_likelihood": likelihood if finite else np.nan,
            }
        )
        if converged and finite:
            models.append((likelihood, seed, model))
    if not models:
        raise RuntimeError("No converged finite HMM initialization was found")
    _, selected_seed, selected_model = max(models, key=lambda item: item[0])
    seed_audit = pd.DataFrame(audit)
    seed_audit["selected"] = seed_audit["seed"].eq(selected_seed)
    return selected_model, selected_seed, seed_audit


def estimate_regimes(
    observations_path: str | Path,
    windows: pd.DataFrame,
    method: dict,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate one training-only HMM for every rolling window."""

    observations = normalize_panel(read_table(observations_path))
    features = list(method["features"])
    label_column = str(method.get("state_label_column", "mkt_ret_20"))
    require_columns(observations, ("date", label_column, *features), "market_observations")
    if int(method["states"]) != 2:
        raise ValueError("The public implementation supports the paper's two-state HMM")

    probability_frames: list[pd.DataFrame] = []
    profile_rows: list[dict] = []
    seed_frames: list[pd.DataFrame] = []
    output = Path(output_dir)

    for window in windows.itertuples(index=False):
        sample = observations.loc[
            observations["date"].between(window.train_start, window.test_end)
        ].dropna(subset=[*features, label_column]).copy()
        train_mask = sample["date"].between(window.train_start, window.train_end)
        if train_mask.sum() < 20:
            raise RuntimeError(f"{window.window_id}: too few HMM training observations")

        scaler = StandardScaler().fit(sample.loc[train_mask, features])
        all_values = scaler.transform(sample[features])
        train_values = all_values[train_mask.to_numpy()]
        model, selected_seed, seed_audit = _fit_best_model(
            train_values,
            states=int(method["states"]),
            covariance_type=str(method.get("covariance_type", "full")),
            initializations=int(method["initializations"]),
            seed_start=int(method["seed_start"]),
            max_iterations=int(method.get("max_iterations", 500)),
        )
        probabilities = _filtered_probabilities(model, all_values)

        train_probabilities = probabilities[train_mask.to_numpy()]
        train_returns = pd.to_numeric(
            sample.loc[train_mask, label_column], errors="raise"
        ).to_numpy(dtype=float)
        state_means = []
        for state_id in range(model.n_components):
            weights = train_probabilities[:, state_id]
            state_means.append(float(np.average(train_returns, weights=weights)))
        high_state = int(np.argmax(state_means))
        label_by_state = {
            state_id: ("high_market_return" if state_id == high_state else "low_market_return")
            for state_id in range(model.n_components)
        }

        frame = sample[["date"]].copy()
        for state_id, label in label_by_state.items():
            frame[f"probability__{label}"] = probabilities[:, state_id]
            profile_rows.append(
                {
                    "window_id": str(window.window_id),
                    "state_id": state_id,
                    "state_label": label,
                    "training_mean_market_return": state_means[state_id],
                    "selected_seed": selected_seed,
                }
            )
        probability_columns = sorted(
            column for column in frame if column.startswith("probability__")
        )
        frame["state_label"] = frame[probability_columns].idxmax(axis=1).str.replace(
            "probability__", "", regex=False
        )
        frame["segment"] = np.select(
            [
                frame["date"].between(window.train_start, window.train_end),
                frame["date"].between(window.valid_start, window.valid_end),
                frame["date"].between(window.test_start, window.test_end),
            ],
            ["train", "valid", "test"],
            default="unused",
        )
        frame.insert(0, "window_id", str(window.window_id))
        probability_frames.append(frame.loc[frame["segment"].ne("unused")])
        seed_audit.insert(0, "window_id", str(window.window_id))
        seed_frames.append(seed_audit)

    probabilities = pd.concat(probability_frames, ignore_index=True)
    profiles = pd.DataFrame(profile_rows)
    probabilities.to_parquet(output / "hmm_probabilities.parquet", index=False)
    profiles.to_csv(output / "hmm_states.csv", index=False)
    pd.concat(seed_frames, ignore_index=True).to_csv(
        output / "hmm_initializations.csv", index=False
    )
    return probabilities, profiles
