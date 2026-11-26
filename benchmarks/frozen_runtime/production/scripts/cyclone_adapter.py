#!/usr/bin/env python3
"""Source-faithful Cyclone abundance-correlation adapter.

The reviewed legacy adapter used :func:`numpy.corrcoef` after Cyclone's
timepoint-wise scaling.  That is equivalent to base R only when the scaled
matrix is completely finite.  In the corrected benchmark Day0 is immutable
and clone-invariant, so scaling its abundance row produces ``NaN``.  Cyclone's
published implementation deliberately calls::

    cor(mat, method = "pearson", use = "pairwise.complete.obs")

and therefore omits that row separately for every clone pair.  ``np.corrcoef``
instead propagates the missing Day0 value through the complete correlation
matrix.  This module implements the R pairwise-complete intermediate while
retaining the already cross-language-validated legacy R-compatible distance,
Ward linkage, cut-tree operation, return type, and method metadata.
"""

from __future__ import annotations

import time
import warnings
from typing import Sequence

import numpy as np

try:  # Direct script-path import used by the benchmark runner and tests.
    from source_runtime import activate_source_imports
except ImportError:  # pragma: no cover - supports an eventual package import.
    from .source_runtime import activate_source_imports


SOURCE_PROVENANCE = activate_source_imports()

import benchmark_core as legacy_core  # noqa: E402


def r_pairwise_complete_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return Pearson correlations with base-R pairwise-complete semantics.

    ``matrix`` follows R's ``cor`` convention: rows are observations and
    columns are variables.  For each column pair, rows containing ``NA``/NaN
    in either column are omitted.  Fewer than two complete observations or a
    zero-variance member yields NaN for that pair, matching
    ``cor(..., use='pairwise.complete.obs')``.  Infinite values are not treated
    as missing (as in R); their non-finite arithmetic consequently yields NaN.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Cyclone correlation input must be a two-dimensional matrix")
    n_variables = int(values.shape[1])
    correlation = np.full((n_variables, n_variables), np.nan, dtype=np.float64)

    # Pairwise completion depends only on each variable's observation mask.
    # Cyclone has four time rows, so there are at most 2**4=16 mask groups and
    # 136 unordered group pairs.  Correlating whole mask-group blocks replaces
    # the infeasible per-clone-pair Python loop with small matrix products.
    # np.isnan removes both R NA and NaN; +/-Inf deliberately remain present.
    present_by_variable = (~np.isnan(values)).T
    unique_masks, group_for_variable = np.unique(
        present_by_variable, axis=0, return_inverse=True
    )
    group_columns = [
        np.flatnonzero(group_for_variable == group)
        for group in range(len(unique_masks))
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for left_group, left_columns in enumerate(group_columns):
            for right_group in range(left_group, len(group_columns)):
                right_columns = group_columns[right_group]
                complete_rows = np.flatnonzero(
                    unique_masks[left_group] & unique_masks[right_group]
                )
                if len(complete_rows) < 2:
                    continue

                left_values = values[np.ix_(complete_rows, left_columns)]
                right_values = values[np.ix_(complete_rows, right_columns)]
                left_centered = left_values - left_values.mean(axis=0)
                right_centered = right_values - right_values.mean(axis=0)
                left_ss = np.einsum(
                    "ij,ij->j", left_centered, left_centered
                )
                right_ss = np.einsum(
                    "ij,ij->j", right_centered, right_centered
                )
                numerator = left_centered.T @ right_centered
                denominator = np.sqrt(left_ss[:, None] * right_ss[None, :])
                block = numerator / denominator
                block[
                    (~np.isfinite(block))
                    | (~np.isfinite(denominator))
                    | (denominator <= 0.0)
                ] = np.nan
                # Numerical round-off may put a mathematically bounded Pearson
                # coefficient a few ulps outside [-1, 1].  R returns the bound.
                np.clip(block, -1.0, 1.0, out=block)
                correlation[np.ix_(left_columns, right_columns)] = block
                if right_group != left_group:
                    correlation[np.ix_(right_columns, left_columns)] = block.T

    return correlation


def cyclone_correlation_core(
    counts: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Cyclone's published abundance/correlation/Ward clustering core.

    The input is the common eligible clone-by-time-by-state count tensor.  The
    method sees only state-summed clone abundance.  A clone-invariant Day0 row
    is allowed to become missing during row scaling and is then omitted by the
    pairwise-complete correlation, exactly as in the published R source.
    """

    count_array = np.asarray(counts)
    if count_array.ndim != 3:
        raise ValueError("Cyclone counts must be clone x time x state")
    if len(count_array) < 2:
        raise legacy_core.NativeSelectionFailure(
            "Cyclone requires at least two eligible clones"
        )
    if not np.issubdtype(count_array.dtype, np.integer):
        raise TypeError("Cyclone counts must contain integer values")
    if (count_array < 0).any():
        raise ValueError("Cyclone counts must be non-negative")

    abundance = count_array.sum(axis=2, dtype=np.int64).astype(np.float64)
    time_totals = abundance.sum(axis=0, keepdims=True)
    if (time_totals <= 0).any():
        raise legacy_core.NativeSelectionFailure("Cyclone has a zero-total time point")
    time_by_clone = (abundance / time_totals * 100.0).T
    row_means = time_by_clone.mean(axis=1, keepdims=True)
    row_sd = time_by_clone.std(axis=1, ddof=1, keepdims=True)

    # Detect mathematical constancy from the integer abundance counts.  This
    # avoids accepting a tiny floating SD caused only by division round-off and
    # reproduces R's division of a constant row by zero.
    constant_rows = np.ptp(abundance, axis=0) == 0
    row_sd = row_sd.copy()
    row_sd[constant_rows] = 0.0
    nonconstant_sd = row_sd[~constant_rows]
    if (~np.isfinite(nonconstant_sd)).any() or (nonconstant_sd <= 0).any():
        raise legacy_core.NativeSelectionFailure(
            "Cyclone nonconstant row scaling is non-finite"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scaled = (time_by_clone - row_means) / row_sd
    correlation = r_pairwise_complete_correlation(scaled)

    distance = legacy_core.cyclone_r_pairwise_euclidean(correlation)
    condensed = distance[np.triu_indices(len(distance), k=1)]
    tree = legacy_core.linkage(condensed, method="ward")
    labels = legacy_core.cut_tree(
        tree, n_clusters=[int(n_clusters)]
    ).reshape(-1)
    return np.asarray(labels, dtype=np.int64), correlation


def fit_cyclone_method(
    counts: np.ndarray,
    clone_ids: Sequence[str],
    config: dict,
) -> legacy_core.MethodFit:
    """Return the legacy ``MethodFit`` for the corrected Cyclone adapter."""

    if len(clone_ids) != len(counts):
        raise ValueError("Cyclone clone_ids and count rows have different lengths")
    started = time.perf_counter()
    requested_k = int(config["n_clusters"])
    labels, correlation = cyclone_correlation_core(counts, requested_k)
    recovered_k = int(len(np.unique(labels)))
    if not 1 <= recovered_k <= len(labels):
        raise AssertionError("Cyclone returned an impossible cluster count")
    abundance = np.asarray(counts).sum(axis=2, dtype=np.int64)
    return legacy_core.MethodFit(
        method="Cyclone abundance-only",
        labels=labels,
        recovered_k=recovered_k,
        selection_parameter=float(requested_k),
        selection_description="published function default n_clust",
        selected_silhouette=np.nan,
        runtime_seconds=float(time.perf_counter() - started),
        adapter_scope=(
            "Cyclone abundance-correlation core; exact R pairwise-complete "
            "correlation; dynamic filter disabled"
        ),
        diagnostic={
            "information_used": "state-summed clone abundance at four times",
            "dynamic_clone_filter": bool(config["dynamic_clone_filter"]),
            "correlation_min": float(np.nanmin(correlation)),
            "correlation_max": float(np.nanmax(correlation)),
            "nonfinite_correlation_entries": int(
                (~np.isfinite(correlation)).sum()
            ),
            "constant_timepoint_count": int(
                np.sum(np.ptp(abundance, axis=0) == 0)
            ),
            "pairwise_complete_correlation": True,
            "uniform_noise_invariance_expected": True,
        },
    )


__all__ = [
    "SOURCE_PROVENANCE",
    "cyclone_correlation_core",
    "fit_cyclone_method",
    "legacy_core",
    "r_pairwise_complete_correlation",
]
