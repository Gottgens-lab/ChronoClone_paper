#!/usr/bin/env python3
"""Corrected scientific contract for the Day0-locked method benchmark.

This module deliberately reuses reviewed low-level exact-removal and metric
utilities from the completed local benchmark, while implementing new noise
stages under the corrected contract:

* the complete Day0 count block is immutable;
* primary eligibility is recalculated for each condition using starting-state
  purity, consecutive temporal persistence, and the total-count floor;
* the clean fixed cohort is exposed only as a secondary sensitivity analysis;
* uniform redistribution must preserve eligibility relative to the Gaussian
  realization from which it starts; and
* ChronoClone's only post-partition exclusion is the inclusive, unrounded
  Euclidean sample-silhouette rule ``>= -0.1``.

The legacy benchmark module is loaded by an explicit immutable source path.
No legacy result or source file is modified.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)


ANALYSES_ROOT = Path(__file__).resolve().parent.parent
LEGACY_CORE_PATH = (
    ANALYSES_ROOT
    / "native_method_noise_benchmark_local_20260803"
    / "benchmark_package"
    / "scripts"
    / "benchmark_core.py"
)
LEGACY_MODULE_NAME = "_chronoclone_locked_legacy_benchmark_core"
START_STATES = ("HSC", "HSPC")
MIN_PERSISTENCE = 2
MIN_TOTAL_COUNT = 3
SILHOUETTE_CUTOFF = -0.1
DAY0_LOCKED_GAUSSIAN_STREAM = 111
DAY0_LOCKED_UNIFORM_STREAM = 122


def _load_legacy_core() -> ModuleType:
    """Load the reviewed legacy core from its explicit repository path."""

    if not LEGACY_CORE_PATH.is_file():
        raise FileNotFoundError(f"Legacy benchmark core is missing: {LEGACY_CORE_PATH}")
    existing = sys.modules.get(LEGACY_MODULE_NAME)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(
        LEGACY_MODULE_NAME, LEGACY_CORE_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load legacy benchmark core: {LEGACY_CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    # Dataclass decoration consults sys.modules while the module is executing.
    sys.modules[LEGACY_MODULE_NAME] = module
    specification.loader.exec_module(module)
    return module


LEGACY_CORE = _load_legacy_core()
TIMES = tuple(LEGACY_CORE.TIMES)
STATES = tuple(LEGACY_CORE.STATES)


@dataclass(frozen=True)
class ConditionEligibility:
    """Primary condition-specific and secondary clean-fixed clone masks."""

    current_lineage_mask: np.ndarray
    clean_fixed_mask: np.ndarray

    @property
    def n_current(self) -> int:
        return int(self.current_lineage_mask.sum())

    @property
    def n_clean_fixed(self) -> int:
        return int(self.clean_fixed_mask.sum())

    @property
    def n_mask_disagreements(self) -> int:
        return int(np.count_nonzero(self.current_lineage_mask != self.clean_fixed_mask))


@dataclass(frozen=True)
class ChronoClonePostfilter:
    """Frozen ChronoClone partition plus its sample-level retention decision."""

    labels: np.ndarray
    sample_silhouette: np.ndarray
    retained_mask: np.ndarray
    cutoff: float = SILHOUETTE_CUTOFF

    @property
    def n_fitted(self) -> int:
        return int(len(self.labels))

    @property
    def n_retained(self) -> int:
        return int(self.retained_mask.sum())

    @property
    def retained_fraction_of_fitted(self) -> float:
        return float(self.retained_mask.mean())


def _validate_count_tensor(counts: np.ndarray, *, name: str) -> np.ndarray:
    """Validate and return a non-negative integer clone/time/state tensor."""

    array = np.asarray(counts)
    expected_tail = (len(TIMES), len(STATES))
    if array.ndim != 3 or array.shape[1:] != expected_tail:
        raise ValueError(
            f"{name} must have shape clones x {expected_tail[0]} times x "
            f"{expected_tail[1]} states; found {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer counts")
    if (array < 0).any():
        raise ValueError(f"{name} must contain non-negative counts")
    return array


def assert_day0_frozen(reference_counts: np.ndarray, counts: np.ndarray) -> None:
    """Raise unless every clone's entire Day0 state vector is unchanged."""

    reference = _validate_count_tensor(reference_counts, name="reference_counts")
    observed = _validate_count_tensor(counts, name="counts")
    if reference.shape != observed.shape:
        raise ValueError("reference_counts and counts have different shapes")
    if not np.array_equal(reference[:, 0, :], observed[:, 0, :]):
        differing = int(
            np.count_nonzero(
                np.any(reference[:, 0, :] != observed[:, 0, :], axis=1)
            )
        )
        raise AssertionError(f"Day0 changed for {differing} clone(s)")


def gaussian_stage_day0_locked(
    clean_means: np.ndarray,
    gaussian_pct: int,
    repeat: int,
    master_seed: int,
) -> tuple[np.ndarray, int]:
    """Apply relative Gaussian noise only to later times; never draw for Day0."""

    clean_array = np.asarray(clean_means)
    if clean_array.ndim != 3 or clean_array.shape[1:] != (
        len(TIMES),
        len(STATES),
    ):
        raise ValueError("clean_means must have shape clones x 4 times x 13 states")
    if not np.isfinite(clean_array).all() or (clean_array < 0).any():
        raise ValueError("clean_means must be finite and non-negative")
    if not isinstance(gaussian_pct, (int, np.integer)):
        raise TypeError("gaussian_pct must be an integer")
    if not 0 <= int(gaussian_pct) <= 100:
        raise ValueError("gaussian_pct must be between 0 and 100")
    if not isinstance(repeat, (int, np.integer)) or int(repeat) < 0:
        raise ValueError("repeat must be a non-negative integer")

    seed_sequence = LEGACY_CORE._seed_sequence(
        master_seed,
        DAY0_LOCKED_GAUSSIAN_STREAM,
        repeat,
        gaussian_pct,
    )
    seed_id = LEGACY_CORE.seed_fingerprint(seed_sequence)
    noisy = np.empty(clean_array.shape, dtype=np.int64)
    # The locked deterministic profiles are expected counts, including thirds
    # at Day0. The legacy generator defines its zero-noise count realization by
    # np.rint; that exact integer Day0 realization is frozen at every dose.
    noisy[:, 0, :] = np.rint(clean_array[:, 0, :]).astype(np.int64, copy=False)
    later_means = clean_array[:, 1:, :]
    fraction = int(gaussian_pct) / 100.0
    if fraction == 0:
        later_noisy = np.rint(later_means)
    else:
        rng = np.random.default_rng(seed_sequence)
        later_noisy = later_means + rng.normal(
            loc=0.0,
            scale=later_means * fraction,
            size=later_means.shape,
        )
        later_noisy = np.rint(later_noisy)
    noisy[:, 1:, :] = np.clip(later_noisy, 0, None).astype(
        np.int64, copy=False
    )
    assert_day0_frozen(rounded_clean_counts(clean_array), noisy)
    return noisy, int(seed_id)


def uniform_redistribution_stage_day0_locked(
    gaussian_counts: np.ndarray,
    gaussian_pct: int,
    uniform_pct: int,
    repeat: int,
    master_seed: int,
    *,
    start_states: Sequence[str] = START_STATES,
    min_persistence: int = MIN_PERSISTENCE,
    min_total_count: int = MIN_TOTAL_COUNT,
) -> tuple[np.ndarray, int]:
    """Redistribute later-time cells and enforce eligibility invariance.

    Uniform redistribution preserves every clone/time total. With Day0 frozen,
    it therefore cannot change starting-state purity, persistence, or the
    total-count floor relative to the Gaussian realization.
    """

    gaussian_array = _validate_count_tensor(
        gaussian_counts, name="gaussian_counts"
    )
    if not isinstance(gaussian_pct, (int, np.integer)):
        raise TypeError("gaussian_pct must be an integer")
    if not isinstance(uniform_pct, (int, np.integer)):
        raise TypeError("uniform_pct must be an integer")
    if not 0 <= int(gaussian_pct) <= 100:
        raise ValueError("gaussian_pct must be between 0 and 100")
    if not 0 <= int(uniform_pct) <= 100:
        raise ValueError("uniform_pct must be between 0 and 100")
    if not isinstance(repeat, (int, np.integer)) or int(repeat) < 0:
        raise ValueError("repeat must be a non-negative integer")

    seed_sequence = LEGACY_CORE._seed_sequence(
        master_seed,
        DAY0_LOCKED_UNIFORM_STREAM,
        repeat,
        gaussian_pct,
        uniform_pct,
    )
    seed_id = LEGACY_CORE.seed_fingerprint(seed_sequence)
    redistributed = gaussian_array.copy()
    # Reshape only the later-time view. Day0 is neither sampled nor copied
    # through a noise result: it remains the untouched slice in `redistributed`.
    later_shape = gaussian_array[:, 1:, :].shape
    later_blocks = gaussian_array[:, 1:, :].reshape(-1, len(STATES)).copy()
    totals = later_blocks.sum(axis=1, dtype=np.int64)
    n_move = np.rint(totals * (int(uniform_pct) / 100.0)).astype(np.int64)
    if int(uniform_pct) == 0:
        later_redistributed = later_blocks
    else:
        rng = np.random.default_rng(seed_sequence)
        removals = LEGACY_CORE._sample_exact_removals(
            later_blocks, n_move, rng
        )
        probabilities = np.full(
            len(STATES), 1.0 / len(STATES), dtype=np.float64
        )
        additions = rng.multinomial(n_move, probabilities).astype(
            np.int64, copy=False
        )
        later_redistributed = later_blocks - removals + additions
    if (later_redistributed < 0).any():
        raise AssertionError("Uniform redistribution created a negative count")
    if not np.array_equal(
        later_redistributed.sum(axis=1, dtype=np.int64), totals
    ):
        raise AssertionError("Uniform redistribution changed later clone/time totals")
    redistributed[:, 1:, :] = later_redistributed.reshape(later_shape)
    assert_day0_frozen(gaussian_array, redistributed)
    if not np.array_equal(
        gaussian_array.sum(axis=2, dtype=np.int64),
        redistributed.sum(axis=2, dtype=np.int64),
    ):
        raise AssertionError("Uniform redistribution changed clone/time totals")
    before = lineage_eligibility_mask(
        gaussian_array,
        start_states=start_states,
        min_persistence=min_persistence,
        min_total_count=min_total_count,
    )
    after = lineage_eligibility_mask(
        redistributed,
        start_states=start_states,
        min_persistence=min_persistence,
        min_total_count=min_total_count,
    )
    if not np.array_equal(before, after):
        raise AssertionError(
            "Uniform redistribution changed lineage eligibility relative to "
            "the Gaussian realization"
        )
    return redistributed, int(seed_id)


def rounded_clean_counts(clean_means: np.ndarray) -> np.ndarray:
    """Return the generator's zero-noise integer realization of clean means."""

    clean = np.asarray(clean_means)
    if clean.ndim != 3 or clean.shape[1:] != (len(TIMES), len(STATES)):
        raise ValueError("clean_means must have shape clones x 4 times x 13 states")
    if not np.isfinite(clean).all() or (clean < 0).any():
        raise ValueError("clean_means must be finite and non-negative")
    rounded = np.rint(clean)
    return rounded.astype(np.int64, copy=False)


def lineage_eligibility_mask(
    counts: np.ndarray,
    *,
    start_states: Sequence[str] = START_STATES,
    min_persistence: int = MIN_PERSISTENCE,
    min_total_count: int = MIN_TOTAL_COUNT,
) -> np.ndarray:
    """Implement ChronoClone's pre-fit lineage eligibility on count tensors.

    A retained clone has at least one Day0 cell in an allowed start state, no
    Day0 cell outside those states, and observations on a gap-free initial
    time sequence of length at least ``min_persistence``. The benchmark's
    existing total-count floor is applied in the same mask.
    """

    array = _validate_count_tensor(counts, name="counts")
    if isinstance(start_states, str):
        selected_states = (start_states,)
    else:
        selected_states = tuple(str(value) for value in start_states)
    if not selected_states or len(selected_states) != len(set(selected_states)):
        raise ValueError("start_states must be nonempty and unique")
    unknown = sorted(set(selected_states) - set(STATES))
    if unknown:
        raise ValueError(f"Unknown start states: {unknown}")
    if not isinstance(min_persistence, (int, np.integer)):
        raise TypeError("min_persistence must be an integer")
    if not 1 <= int(min_persistence) <= len(TIMES):
        raise ValueError(f"min_persistence must be in 1..{len(TIMES)}")
    if not isinstance(min_total_count, (int, np.integer)):
        raise TypeError("min_total_count must be an integer")
    if int(min_total_count) < 0:
        raise ValueError("min_total_count must be non-negative")

    start_indices = np.asarray([STATES.index(value) for value in selected_states])
    day0 = array[:, 0, :]
    allowed_count = day0[:, start_indices].sum(axis=1, dtype=np.int64)
    outside_mask = np.ones(len(STATES), dtype=bool)
    outside_mask[start_indices] = False
    forbidden_count = day0[:, outside_mask].sum(axis=1, dtype=np.int64)
    day0_pure = (allowed_count > 0) & (forbidden_count == 0)

    time_present = array.sum(axis=2, dtype=np.int64) > 0
    required_prefix = time_present[:, : int(min_persistence)].all(axis=1)
    # Package semantics require observed timepoints to be exactly the initial
    # sequence through the last observed time: no internal absence is allowed.
    observed_count = time_present.sum(axis=1, dtype=np.int64)
    prefix_matrix = np.arange(len(TIMES))[None, :] < observed_count[:, None]
    gap_free_initial_sequence = np.all(time_present == prefix_matrix, axis=1)
    total_count_ok = (
        array.sum(axis=(1, 2), dtype=np.int64) >= int(min_total_count)
    )
    return day0_pure & required_prefix & gap_free_initial_sequence & total_count_ok


def clean_fixed_eligibility_mask(
    clean_counts: np.ndarray,
    *,
    start_states: Sequence[str] = START_STATES,
    min_persistence: int = MIN_PERSISTENCE,
    min_total_count: int = MIN_TOTAL_COUNT,
) -> np.ndarray:
    """Return the predeclared clean-cohort mask for secondary sensitivity."""

    return lineage_eligibility_mask(
        clean_counts,
        start_states=start_states,
        min_persistence=min_persistence,
        min_total_count=min_total_count,
    )


def condition_eligibility_masks(
    clean_counts: np.ndarray,
    current_counts: np.ndarray,
    *,
    start_states: Sequence[str] = START_STATES,
    min_persistence: int = MIN_PERSISTENCE,
    min_total_count: int = MIN_TOTAL_COUNT,
) -> ConditionEligibility:
    """Return primary current and secondary clean-fixed eligibility masks."""

    clean = _validate_count_tensor(clean_counts, name="clean_counts")
    current = _validate_count_tensor(current_counts, name="current_counts")
    if clean.shape != current.shape:
        raise ValueError("clean_counts and current_counts have different shapes")
    assert_day0_frozen(clean, current)
    current_mask = lineage_eligibility_mask(
        current,
        start_states=start_states,
        min_persistence=min_persistence,
        min_total_count=min_total_count,
    )
    fixed_mask = clean_fixed_eligibility_mask(
        clean,
        start_states=start_states,
        min_persistence=min_persistence,
        min_total_count=min_total_count,
    )
    return ConditionEligibility(
        current_lineage_mask=current_mask,
        clean_fixed_mask=fixed_mask,
    )


def assert_fixed_cohort_mask(
    supplied_mask: np.ndarray,
    clean_counts: np.ndarray,
    *,
    start_states: Sequence[str] = START_STATES,
    min_persistence: int = MIN_PERSISTENCE,
    min_total_count: int = MIN_TOTAL_COUNT,
) -> None:
    """Raise unless a secondary fixed mask exactly equals clean eligibility."""

    observed = np.asarray(supplied_mask)
    expected = clean_fixed_eligibility_mask(
        clean_counts,
        start_states=start_states,
        min_persistence=min_persistence,
        min_total_count=min_total_count,
    )
    if observed.dtype != np.bool_ or observed.shape != expected.shape:
        raise ValueError("supplied_mask must be a Boolean vector over all clones")
    if not np.array_equal(observed, expected):
        raise AssertionError("Supplied fixed cohort differs from clean eligibility")


def chronoclone_silhouette_postfilter(
    normalized_features: np.ndarray,
    labels: Sequence[int],
) -> ChronoClonePostfilter:
    """Apply exactly the inclusive, unrounded sample silhouette >= -0.1 rule.

    The input labels are returned unchanged. This function never refits or
    reselects the partition and has no metaclone-size criterion.
    """

    features = np.asarray(normalized_features, dtype=np.float64)
    label_array = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("normalized_features must be two-dimensional")
    if len(features) != len(label_array):
        raise ValueError("normalized_features and labels have different lengths")
    if len(features) < 3:
        raise ValueError("At least three fitted clones are required")
    if not np.isfinite(features).all():
        raise ValueError("normalized_features must be finite")
    unique_labels = np.unique(label_array)
    if not 1 < len(unique_labels) < len(label_array):
        raise ValueError("Sample silhouette requires 2..n-1 recovered groups")
    sample_values = np.asarray(
        silhouette_samples(features, label_array, metric="euclidean"),
        dtype=np.float64,
    )
    if sample_values.shape != (len(label_array),) or not np.isfinite(
        sample_values
    ).all():
        raise AssertionError("Sample silhouette calculation returned invalid values")
    retained = sample_values >= SILHOUETTE_CUTOFF
    return ChronoClonePostfilter(
        labels=label_array.copy(),
        sample_silhouette=sample_values,
        retained_mask=retained,
    )


def chronoclone_postfilter_metrics(
    normalized_features: np.ndarray,
    labels: Sequence[int],
    truth: Sequence[str],
    *,
    n_input_clones: int | None = None,
) -> tuple[ChronoClonePostfilter, dict[str, float | int]]:
    """Score retained clones and report all required coverage denominators."""

    selection = chronoclone_silhouette_postfilter(normalized_features, labels)
    truth_array = np.asarray(truth)
    if len(truth_array) != selection.n_fitted:
        raise ValueError("truth and labels have different lengths")
    if selection.n_retained < 2:
        raise ValueError("Fewer than two clones survived the silhouette postfilter")
    if n_input_clones is None:
        n_input = selection.n_fitted
    else:
        if not isinstance(n_input_clones, (int, np.integer)):
            raise TypeError("n_input_clones must be an integer")
        n_input = int(n_input_clones)
        if n_input < selection.n_fitted:
            raise ValueError("n_input_clones cannot be smaller than fitted clone count")
    retained_truth = truth_array[selection.retained_mask]
    retained_labels = selection.labels[selection.retained_mask]
    metrics: dict[str, float | int] = {
        "selective_ari": float(
            adjusted_rand_score(retained_truth, retained_labels)
        ),
        "selective_nmi": float(
            normalized_mutual_info_score(retained_truth, retained_labels)
        ),
        "selective_hungarian_macro_f1": float(
            LEGACY_CORE.hungarian_macro_f1(retained_truth, retained_labels)
        ),
        "n_input_clones": n_input,
        "n_fitted_clones": selection.n_fitted,
        "n_retained_clones": selection.n_retained,
        "fitted_fraction_of_input": float(selection.n_fitted / n_input),
        "retained_fraction_of_fitted": selection.retained_fraction_of_fitted,
        "retained_fraction_of_input": float(selection.n_retained / n_input),
        "silhouette_cutoff": SILHOUETTE_CUTOFF,
    }
    return selection, metrics


__all__ = [
    "START_STATES",
    "MIN_PERSISTENCE",
    "MIN_TOTAL_COUNT",
    "SILHOUETTE_CUTOFF",
    "DAY0_LOCKED_GAUSSIAN_STREAM",
    "DAY0_LOCKED_UNIFORM_STREAM",
    "STATES",
    "TIMES",
    "ConditionEligibility",
    "ChronoClonePostfilter",
    "assert_day0_frozen",
    "assert_fixed_cohort_mask",
    "chronoclone_postfilter_metrics",
    "chronoclone_silhouette_postfilter",
    "clean_fixed_eligibility_mask",
    "condition_eligibility_masks",
    "gaussian_stage_day0_locked",
    "lineage_eligibility_mask",
    "rounded_clean_counts",
    "uniform_redistribution_stage_day0_locked",
]
