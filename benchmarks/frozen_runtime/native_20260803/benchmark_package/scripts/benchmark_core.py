#!/usr/bin/env python3
"""Core simulation and native ChronoClone benchmark functions.

The fitting function never receives the number or names of the true programs.
Truth is introduced only after native partition selection, for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import SparseEfficiencyWarning
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


TIMES = (0, 4, 9, 15)
STATES = (
    "HSC",
    "HSPC",
    "MEP",
    "CMP",
    "Mk_prog",
    "MEMP",
    "GMP",
    "GMDP",
    "Mast_prog",
    "Baso_prog",
    "Neu_prog",
    "DC_prog",
    "Mono_prog",
)
WIDE_COLUMNS = tuple(
    f"Day{time}_Cluster{state}" for time in TIMES for state in STATES
)
EXPECTED_PROFILE_SHA256 = (
    "e4275d94c1513026dc684bcca44703135bba4af4694ec55cd714d3c0fa21bf6f"
)
EXPECTED_MANIFEST_SHA256 = (
    "ba890ce14f052113d203904aa158cab79c36794c4edaf638ee0afd5b85063d11"
)
GAUSSIAN_STREAM = 11
UNIFORM_STREAM = 22


class NativeSelectionFailure(RuntimeError):
    """A valid input was produced, but native selection returned no partition."""


@dataclass(frozen=True)
class ReferencePrograms:
    """Locked deterministic programs and their simulation-only truth labels."""

    profiles: np.ndarray
    class_sizes: np.ndarray
    truth_levels: np.ndarray
    clean_means: np.ndarray
    truth: np.ndarray
    clone_ids: np.ndarray
    profile_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class NativeFit:
    """Native ChronoClone fit before truth-based evaluation."""

    labels: np.ndarray
    eligible_mask: np.ndarray
    selected_threshold: float
    selected_silhouette: float
    recovered_k: int
    runtime_seconds: float
    consensus_diagonal_before_fix: float
    local_diagonal_fix_applied: bool


@dataclass(frozen=True)
class MethodFit:
    """One truth-blind method result on the benchmark's common clone set."""

    method: str
    labels: np.ndarray
    recovered_k: int
    selection_parameter: float
    selection_description: str
    selected_silhouette: float
    runtime_seconds: float
    adapter_scope: str
    diagnostic: dict


def load_json(path: Path) -> dict:
    """Load a JSON object from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config_path: Path) -> str:
    """Return the exact configuration-file digest recorded in every result row."""

    return sha256_file(config_path)


def load_reference_programs(reference_dir: Path) -> ReferencePrograms:
    """Load and validate the 12 locked time-major deterministic programs."""

    profile_path = reference_dir / "deterministic_profiles.csv"
    manifest_path = reference_dir / "program_manifest.csv"
    observed_sha256 = sha256_file(profile_path)
    if observed_sha256 != EXPECTED_PROFILE_SHA256:
        raise ValueError(
            "Deterministic profile checksum mismatch: "
            f"{observed_sha256} != {EXPECTED_PROFILE_SHA256}"
        )
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "Program manifest checksum mismatch: "
            f"{observed_manifest_sha256} != {EXPECTED_MANIFEST_SHA256}"
        )

    profiles = np.loadtxt(profile_path, delimiter=",", dtype=np.float64)
    manifest = pd.read_csv(manifest_path)
    expected_manifest_columns = {"truth_metaclone", "n_clones", "description"}
    if set(manifest.columns) != expected_manifest_columns:
        raise ValueError(
            f"Unexpected program manifest columns: {manifest.columns.tolist()}"
        )
    if profiles.shape != (12, len(TIMES) * len(STATES)):
        raise ValueError(f"Unexpected deterministic profile shape: {profiles.shape}")
    if len(manifest) != 12:
        raise ValueError(f"Expected 12 manifest rows, found {len(manifest)}")
    if not np.isfinite(profiles).all() or (profiles < 0).any():
        raise ValueError("Deterministic profiles must be finite and non-negative")

    profiles = profiles.reshape(12, len(TIMES), len(STATES))
    class_sizes = manifest["n_clones"].to_numpy(dtype=np.int64)
    truth_levels = manifest["truth_metaclone"].astype(str).to_numpy()
    clean_means = np.repeat(profiles, class_sizes, axis=0)
    truth = np.repeat(truth_levels, class_sizes)
    clone_ids = np.asarray(
        [f"clone_{index:04d}" for index in range(len(clean_means))], dtype=str
    )

    if clean_means.shape != (517, 4, 13):
        raise ValueError(f"Expected a 517 x 4 x 13 tensor, found {clean_means.shape}")
    if len(np.unique(truth)) != 12:
        raise ValueError("Expected 12 simulation truth programs")
    if any("MC" in clone_id.upper() for clone_id in clone_ids):
        raise ValueError("Neutral clone identifiers unexpectedly contain truth labels")

    return ReferencePrograms(
        profiles=profiles,
        class_sizes=class_sizes,
        truth_levels=truth_levels,
        clean_means=clean_means,
        truth=truth,
        clone_ids=clone_ids,
        profile_sha256=observed_sha256,
        manifest_sha256=observed_manifest_sha256,
    )


def _seed_sequence(master_seed: int, *parts: int) -> np.random.SeedSequence:
    """Construct a stable seed sequence without Python's randomized hash()."""

    return np.random.SeedSequence([int(master_seed), *(int(part) for part in parts)])


def seed_fingerprint(seed_sequence: np.random.SeedSequence) -> int:
    """Return a compact, reproducible unsigned 64-bit seed identifier."""

    state = seed_sequence.generate_state(2, dtype=np.uint32)
    return (int(state[0]) << 32) | int(state[1])


def gaussian_stage(
    clean_means: np.ndarray,
    gaussian_pct: int,
    repeat: int,
    master_seed: int,
) -> tuple[np.ndarray, int]:
    """Apply relative Gaussian SD noise and return non-negative integer counts.

    ``gaussian_pct`` is a coefficient of variation, not a percentage of cells
    corrupted. Structural zero means have zero Gaussian variance; state
    misassignment is handled separately by uniform redistribution.
    """

    if not isinstance(gaussian_pct, (int, np.integer)):
        raise TypeError("gaussian_pct must be an integer")
    if not isinstance(repeat, (int, np.integer)) or repeat < 0:
        raise ValueError("repeat must be a non-negative integer")
    clean_means = np.asarray(clean_means, dtype=np.float64)
    if clean_means.ndim != 3 or clean_means.shape[1:] != (
        len(TIMES),
        len(STATES),
    ):
        raise ValueError(
            "clean_means must have shape clones x 4 times x 13 states"
        )
    if not np.isfinite(clean_means).all() or (clean_means < 0).any():
        raise ValueError("clean_means must be finite and non-negative")
    if gaussian_pct < 0 or gaussian_pct > 100:
        raise ValueError("gaussian_pct must be between 0 and 100")
    seed_sequence = _seed_sequence(
        master_seed, GAUSSIAN_STREAM, repeat, gaussian_pct
    )
    seed_id = seed_fingerprint(seed_sequence)
    fraction = gaussian_pct / 100.0

    if fraction == 0:
        noisy = np.rint(clean_means)
    else:
        rng = np.random.default_rng(seed_sequence)
        standard_deviation = clean_means * fraction
        noisy = clean_means + rng.normal(
            loc=0.0, scale=standard_deviation, size=clean_means.shape
        )
        noisy = np.rint(noisy)

    noisy = np.clip(noisy, 0, None).astype(np.int64, copy=False)
    return noisy, seed_id


def _sample_exact_removals(
    blocks: np.ndarray,
    n_move: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized multivariate-hypergeometric removal from many state blocks."""

    removals = np.zeros_like(blocks, dtype=np.int64)
    remaining_draws = n_move.astype(np.int64, copy=True)
    remaining_total = blocks.sum(axis=1, dtype=np.int64)

    for state_index in range(blocks.shape[1] - 1):
        active = (remaining_draws > 0) & (remaining_total > 0)
        if not active.any():
            break
        good = blocks[active, state_index]
        bad = remaining_total[active] - good
        draws = remaining_draws[active]
        sampled = rng.hypergeometric(good, bad, draws)
        removals[active, state_index] = sampled
        remaining_draws[active] -= sampled
        remaining_total[active] -= good

    removals[:, -1] = remaining_draws
    if not np.array_equal(removals.sum(axis=1), n_move):
        raise AssertionError("Exact uniform-noise removal count was not preserved")
    if (removals > blocks).any():
        raise AssertionError("Uniform-noise removal exceeded an observed state count")
    return removals


def uniform_redistribution_stage(
    gaussian_counts: np.ndarray,
    gaussian_pct: int,
    uniform_pct: int,
    repeat: int,
    master_seed: int,
) -> tuple[np.ndarray, int]:
    """Redistribute an exact percentage of cells within each clone-time block.

    Cells are removed without replacement from their observed states, then
    assigned independently and uniformly to one of the 13 states. Clone-time
    totals after Gaussian noise are preserved exactly.
    """

    if not isinstance(gaussian_pct, (int, np.integer)):
        raise TypeError("gaussian_pct must be an integer")
    if not isinstance(uniform_pct, (int, np.integer)):
        raise TypeError("uniform_pct must be an integer")
    if not isinstance(repeat, (int, np.integer)) or repeat < 0:
        raise ValueError("repeat must be a non-negative integer")
    if gaussian_pct < 0 or gaussian_pct > 100:
        raise ValueError("gaussian_pct must be between 0 and 100")
    if uniform_pct < 0 or uniform_pct > 100:
        raise ValueError("uniform_pct must be between 0 and 100")
    gaussian_counts = np.asarray(gaussian_counts)
    if gaussian_counts.ndim != 3 or gaussian_counts.shape[1:] != (
        len(TIMES),
        len(STATES),
    ):
        raise ValueError(
            "gaussian_counts must have shape clones x 4 times x 13 states"
        )
    if not np.issubdtype(gaussian_counts.dtype, np.integer):
        raise TypeError("gaussian_counts must contain integer counts")
    if (gaussian_counts < 0).any():
        raise ValueError("gaussian_counts must be non-negative")
    seed_sequence = _seed_sequence(
        master_seed,
        UNIFORM_STREAM,
        repeat,
        gaussian_pct,
        uniform_pct,
    )
    seed_id = seed_fingerprint(seed_sequence)
    if uniform_pct == 0:
        return gaussian_counts.copy(), seed_id

    rng = np.random.default_rng(seed_sequence)
    original_shape = gaussian_counts.shape
    blocks = gaussian_counts.reshape(-1, len(STATES)).astype(np.int64, copy=True)
    totals = blocks.sum(axis=1, dtype=np.int64)
    n_move = np.rint(totals * (uniform_pct / 100.0)).astype(np.int64)

    removals = _sample_exact_removals(blocks, n_move, rng)
    probabilities = np.full(len(STATES), 1.0 / len(STATES), dtype=np.float64)
    additions = rng.multinomial(n_move, probabilities).astype(np.int64, copy=False)
    redistributed = blocks - removals + additions

    if (redistributed < 0).any():
        raise AssertionError("Uniform redistribution created a negative count")
    if not np.array_equal(redistributed.sum(axis=1), totals):
        raise AssertionError("Uniform redistribution changed clone-time totals")
    return redistributed.reshape(original_shape), seed_id


def _agglomerative_precomputed(
    distance: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Fit average-linkage clustering across supported scikit-learn versions."""

    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(threshold),
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(threshold),
            affinity="precomputed",
            linkage="average",
        )
    return model.fit_predict(distance)


def fit_native_chronoclone(
    counts: np.ndarray,
    clone_ids: Sequence[str],
    *,
    min_total_count: int,
    n_neighbors: int,
    resolutions: Iterable[float],
    thresholds: Iterable[float],
    algorithm_seed: int,
) -> NativeFit:
    """Fit only ChronoClone's native, silhouette-selected partition."""

    import anndata as ad
    import chronoclone as cc
    import scanpy as sc

    started = time.perf_counter()
    counts = np.asarray(counts)
    if counts.ndim != 3 or counts.shape[1:] != (len(TIMES), len(STATES)):
        raise ValueError(f"Expected counts shaped clones x 4 x 13, found {counts.shape}")
    if len(clone_ids) != len(counts):
        raise ValueError("clone_ids and counts have different lengths")
    if (counts < 0).any() or not np.isfinite(counts).all():
        raise ValueError("Counts must be finite and non-negative")
    clone_ids_array = np.asarray(clone_ids, dtype=str)
    if len(np.unique(clone_ids_array)) != len(clone_ids_array):
        raise ValueError("clone_ids must be unique")
    if min_total_count < 0:
        raise ValueError("min_total_count must be non-negative")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if algorithm_seed < 0:
        raise ValueError("algorithm_seed must be non-negative")
    resolution_values = np.asarray(list(resolutions), dtype=float)
    threshold_values = np.asarray(list(thresholds), dtype=float)
    for name, values in (
        ("resolutions", resolution_values),
        ("thresholds", threshold_values),
    ):
        if (
            len(values) == 0
            or not np.isfinite(values).all()
            or (values <= 0).any()
            or len(np.unique(values)) != len(values)
        ):
            raise ValueError(
                f"{name} must be nonempty, finite, positive, and unique"
            )

    eligible_mask = counts.sum(axis=(1, 2)) >= int(min_total_count)
    eligible_counts = counts[eligible_mask]
    eligible_ids = clone_ids_array[eligible_mask]
    if len(eligible_counts) < 3:
        raise NativeSelectionFailure("Fewer than three eligible clones remained")
    if n_neighbors >= len(eligible_counts):
        raise NativeSelectionFailure(
            "Fewer eligible clones remained than required by n_neighbors"
        )

    raw = pd.DataFrame(
        eligible_counts.reshape(len(eligible_counts), -1),
        index=pd.Index(eligible_ids, name="clone_id"),
        columns=WIDE_COLUMNS,
    )
    normalized = cc.normalize_clone_table(
        raw,
        times_order=[f"Day{time}" for time in TIMES],
        features_order=list(STATES),
    )

    sc.settings.verbosity = 0
    trajectory = ad.AnnData(X=normalized.to_numpy(dtype=float))
    trajectory.obs_names = normalized.index.astype(str)
    trajectory.var_names = normalized.columns.astype(str)
    consensus = np.zeros(
        (len(eligible_counts), len(eligible_counts)), dtype=np.float64
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="In the future, the default backend for leiden.*",
            category=FutureWarning,
        )
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        sc.pp.neighbors(
            trajectory,
            n_neighbors=int(n_neighbors),
            use_rep="X",
            random_state=int(algorithm_seed),
        )
        for resolution in resolution_values:
            key = f"leiden_res{resolution:.1f}"
            sc.tl.leiden(
                trajectory,
                resolution=float(resolution),
                key_added=key,
                random_state=int(algorithm_seed),
                flavor="leidenalg",
                n_iterations=-1,
            )
            labels = trajectory.obs[key].cat.codes.to_numpy(dtype=np.int64)
            consensus += labels[:, None] == labels[None, :]
    consensus /= len(resolution_values)
    observed_diagonal = np.diag(consensus).astype(float, copy=True)
    if not np.allclose(observed_diagonal, 1.0):
        raise AssertionError("Locally constructed consensus diagonal is not one")

    distance = 1.0 - consensus
    np.fill_diagonal(distance, 0.0)
    if not np.allclose(distance, distance.T):
        raise AssertionError("Consensus distance is not symmetric")

    features = normalized.to_numpy(dtype=float)
    candidates: list[tuple[float, int, float, np.ndarray]] = []
    label_cache: dict[tuple[int, ...], float] = {}
    for threshold in threshold_values:
        labels = _agglomerative_precomputed(distance, threshold=float(threshold))
        recovered_k = int(len(np.unique(labels)))
        if 1 < recovered_k < len(features):
            key = tuple(int(value) for value in labels)
            if key not in label_cache:
                label_cache[key] = float(silhouette_score(features, labels))
            candidates.append(
                (label_cache[key], -recovered_k, -float(threshold), labels)
            )

    if not candidates:
        raise NativeSelectionFailure(
            "No valid native partition was produced by the threshold sweep"
        )
    best_silhouette, negative_k, negative_threshold, labels = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return NativeFit(
        labels=np.asarray(labels, dtype=np.int64),
        eligible_mask=eligible_mask,
        selected_threshold=float(-negative_threshold),
        selected_silhouette=float(best_silhouette),
        recovered_k=int(-negative_k),
        runtime_seconds=float(time.perf_counter() - started),
        consensus_diagonal_before_fix=float(observed_diagonal[0]),
        local_diagonal_fix_applied=False,
    )


def common_eligible_mask(counts: np.ndarray, min_total_count: int) -> np.ndarray:
    """Return the one clone filter shared by all five methods."""

    counts = np.asarray(counts)
    if counts.ndim != 3 or counts.shape[1:] != (len(TIMES), len(STATES)):
        raise ValueError(f"Expected counts shaped clones x 4 x 13, found {counts.shape}")
    if not np.issubdtype(counts.dtype, np.integer):
        raise TypeError("Benchmark counts must be integers")
    if (counts < 0).any():
        raise ValueError("Benchmark counts must be non-negative")
    if min_total_count < 0:
        raise ValueError("min_total_count must be non-negative")
    return counts.sum(axis=(1, 2), dtype=np.int64) >= int(min_total_count)


def tensor_to_proportions(counts: np.ndarray) -> np.ndarray:
    """Flatten within-clone/time state proportions to 52 features."""

    totals = counts.sum(axis=2, keepdims=True)
    proportions = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals > 0,
    )
    return proportions.reshape(len(counts), -1)


def _valid_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    recovered_k = len(np.unique(labels))
    if not 1 < recovered_k < len(features):
        return None
    return float(silhouette_score(features, labels, metric="euclidean"))


def _select_silhouette_k(
    features: np.ndarray,
    fitter,
    k_values: Iterable[int],
) -> tuple[np.ndarray, int, float, list[dict]]:
    """Select requested K without truth; smallest requested K breaks ties."""

    candidates: list[tuple[float, int, int, np.ndarray]] = []
    diagnostics: list[dict] = []
    score_cache: dict[tuple[int, ...], float | None] = {}
    for requested_k in k_values:
        labels = np.asarray(fitter(int(requested_k)), dtype=np.int64)
        recovered_k = int(len(np.unique(labels)))
        canonical_map: dict[int, int] = {}
        canonical = tuple(
            canonical_map.setdefault(int(label), len(canonical_map)) for label in labels
        )
        if canonical not in score_cache:
            score_cache[canonical] = _valid_silhouette(features, labels)
        score = score_cache[canonical]
        diagnostics.append(
            {
                "requested_k": int(requested_k),
                "recovered_k": recovered_k,
                "silhouette": score,
            }
        )
        if score is not None and np.isfinite(score):
            candidates.append(
                (score, -recovered_k, -int(requested_k), labels)
            )
    if not candidates:
        raise NativeSelectionFailure("No valid silhouette-selected K partition")
    score, _negative_recovered_k, negative_requested_k, labels = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return labels, -negative_requested_k, float(score), diagnostics


def fit_chronoclone_method(
    counts: np.ndarray,
    clone_ids: Sequence[str],
    config: dict,
) -> MethodFit:
    """Run the native ChronoClone selector on an already filtered tensor."""

    fit = fit_native_chronoclone(
        counts,
        clone_ids,
        min_total_count=0,
        n_neighbors=int(config["n_neighbors"]),
        resolutions=config["resolutions"],
        thresholds=config["thresholds"],
        algorithm_seed=int(config["algorithm_seed"]),
    )
    if not fit.eligible_mask.all():
        raise AssertionError("ChronoClone changed the common eligible clone set")
    return MethodFit(
        method="ChronoClone",
        labels=fit.labels,
        recovered_k=fit.recovered_k,
        selection_parameter=fit.selected_threshold,
        selection_description="silhouette-selected consensus distance threshold",
        selected_silhouette=fit.selected_silhouette,
        runtime_seconds=fit.runtime_seconds,
        adapter_scope="native ChronoClone consensus partition",
        diagnostic={
            "consensus_diagonal": fit.consensus_diagonal_before_fix,
            "local_diagonal_fix_applied": fit.local_diagonal_fix_applied,
            "n_neighbors": int(config["n_neighbors"]),
            "resolution_count": len(config["resolutions"]),
            "threshold_count": len(config["thresholds"]),
        },
    )


def fit_clades_preprocessing_method(
    counts: np.ndarray,
    clone_ids: Sequence[str],
    config: dict,
) -> MethodFit:
    """Reimplement the published CLADES mouse preprocessing partition."""

    import anndata as ad
    import scanpy as sc

    started = time.perf_counter()
    matrix = counts.reshape(len(counts), -1).astype(np.float64)
    adata = ad.AnnData(matrix)
    adata.obs_names = pd.Index(clone_ids, dtype=str)
    adata.var_names = pd.Index(WIDE_COLUMNS, dtype=str)
    sc.settings.verbosity = 0
    sc.pp.filter_cells(adata, min_counts=3)
    sc.pp.filter_genes(adata, min_counts=1)
    if adata.n_obs != len(counts):
        raise AssertionError("CLADES adapter changed the common clone set")
    n_comps = min(
        int(config["max_pcs"]),
        int(adata.n_obs) - 1,
        int(adata.n_vars) - 1,
    )
    if n_comps < 2:
        raise NativeSelectionFailure("CLADES PCA has fewer than two components")
    n_neighbors = int(config["n_neighbors"])
    if not 1 < n_neighbors < adata.n_obs:
        raise NativeSelectionFailure("CLADES neighbour count is invalid")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="In the future, the default backend for leiden.*",
            category=FutureWarning,
        )
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        sc.tl.pca(
            adata,
            n_comps=n_comps,
            svd_solver="arpack",
            random_state=int(config["algorithm_seed"]),
        )
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_comps,
            random_state=int(config["algorithm_seed"]),
        )
        sc.tl.leiden(
            adata,
            resolution=float(config["resolution"]),
            key_added="native",
            random_state=int(config["algorithm_seed"]),
            flavor="leidenalg",
            n_iterations=-1,
        )
    labels = adata.obs["native"].cat.codes.to_numpy(dtype=np.int64)
    recovered_k = int(len(np.unique(labels)))
    if not 1 <= recovered_k <= len(labels):
        raise AssertionError("CLADES returned an impossible cluster count")
    return MethodFit(
        method="CLADES preprocessing",
        labels=labels,
        recovered_k=recovered_k,
        selection_parameter=float(config["resolution"]),
        selection_description="published mouse preprocessing Leiden resolution",
        selected_silhouette=np.nan,
        runtime_seconds=float(time.perf_counter() - started),
        adapter_scope="CLADES preprocessing partition; excludes NeuralODE/Gillespie stages",
        diagnostic={
            "n_pcs": n_comps,
            "n_neighbors": n_neighbors,
            "nonzero_features": int(adata.n_vars),
            "implementation": "documented Python reimplementation",
        },
    )


def cyclone_correlation_core(
    counts: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the source-faithful Python equivalent of Cyclone's base-R core."""

    abundance = counts.sum(axis=2, dtype=np.int64).astype(np.float64)
    time_totals = abundance.sum(axis=0, keepdims=True)
    if (time_totals <= 0).any():
        raise NativeSelectionFailure("Cyclone has a zero-total time point")
    time_by_clone = (abundance / time_totals * 100.0).T
    row_means = time_by_clone.mean(axis=1, keepdims=True)
    row_sd = time_by_clone.std(axis=1, ddof=1, keepdims=True)
    # Detect mathematical constancy from integer abundance rather than trusting
    # a tiny floating-point SD from identical division results. R divides that
    # row by zero; the downstream correlation/distance becomes non-finite and
    # hclust cannot fit. We reproduce that outcome as a method failure.
    constant_rows = np.ptp(abundance, axis=0) == 0
    row_sd = row_sd.copy()
    row_sd[constant_rows] = 0.0
    if (~np.isfinite(row_sd[~constant_rows])).any() or (row_sd[~constant_rows] <= 0).any():
        raise NativeSelectionFailure("Cyclone nonconstant row scaling is non-finite")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scaled = (time_by_clone - row_means) / row_sd
        correlation = np.corrcoef(scaled, rowvar=False)
    distance = cyclone_r_pairwise_euclidean(correlation)
    condensed = distance[np.triu_indices(len(distance), k=1)]
    tree = linkage(condensed, method="ward")
    labels = cut_tree(tree, n_clusters=[int(n_clusters)]).reshape(-1)
    return np.asarray(labels, dtype=np.int64), correlation


def cyclone_r_pairwise_euclidean(correlation: np.ndarray) -> np.ndarray:
    """Reproduce base-R ``dist(..., method='euclidean')`` with missing values.

    R excludes coordinates missing in either row and scales the squared sum by
    ``n_columns / n_observed``. If a row pair shares no finite coordinates, the
    source pipeline cannot produce a finite clustering distance and this helper
    raises an explicit method failure.
    """

    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("Cyclone correlation matrix must be square")
    n = len(correlation)
    if np.isfinite(correlation).all():
        return squareform(pdist(correlation, metric="euclidean"))
    finite = np.isfinite(correlation)
    distance = np.zeros((n, n), dtype=np.float64)
    for index in range(n):
        valid = finite & finite[index][None, :]
        n_observed = valid.sum(axis=1)
        if (n_observed == 0).any():
            raise NativeSelectionFailure(
                "Cyclone R-compatible distance has a pair with no finite coordinates"
            )
        differences = np.where(valid, correlation - correlation[index], 0.0)
        squared = np.sum(differences * differences, axis=1)
        distance[index] = np.sqrt(squared * (n / n_observed))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    if not np.isfinite(distance).all():
        raise NativeSelectionFailure(
            "Cyclone R-compatible correlation-profile distances are non-finite"
        )
    return distance


def fit_cyclone_method(
    counts: np.ndarray,
    clone_ids: Sequence[str],
    config: dict,
) -> MethodFit:
    """Run Cyclone's abundance-only core with its published default K=6."""

    del clone_ids
    started = time.perf_counter()
    requested_k = int(config["n_clusters"])
    labels, correlation = cyclone_correlation_core(counts, requested_k)
    recovered_k = int(len(np.unique(labels)))
    if not 1 <= recovered_k <= len(labels):
        raise AssertionError("Cyclone returned an impossible cluster count")
    return MethodFit(
        method="Cyclone abundance-only",
        labels=labels,
        recovered_k=recovered_k,
        selection_parameter=float(requested_k),
        selection_description="published function default n_clust",
        selected_silhouette=np.nan,
        runtime_seconds=float(time.perf_counter() - started),
        adapter_scope="Cyclone abundance-correlation core; dynamic filter disabled",
        diagnostic={
            "information_used": "state-summed clone abundance at four times",
            "dynamic_clone_filter": bool(config["dynamic_clone_filter"]),
            "correlation_min": float(np.nanmin(correlation)),
            "correlation_max": float(np.nanmax(correlation)),
            "nonfinite_correlation_entries": int((~np.isfinite(correlation)).sum()),
            "constant_timepoint_count": int(
                np.sum(np.ptp(counts.sum(axis=2, dtype=np.int64), axis=0) == 0)
            ),
            "uniform_noise_invariance_expected": True,
        },
    )


def fit_kmeans_method(
    counts: np.ndarray,
    clone_ids: Sequence[str],
    config: dict,
) -> MethodFit:
    """Run K-means with a truth-blind silhouette choice of requested K."""

    del clone_ids
    started = time.perf_counter()
    features = tensor_to_proportions(counts)
    k_values = list(range(int(config["min_k"]), int(config["max_k"]) + 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        labels, requested_k, selected_score, diagnostics = _select_silhouette_k(
            features,
            lambda k: KMeans(
                n_clusters=k,
                n_init=int(config["kmeans_n_init"]),
                random_state=int(config["algorithm_seed"]),
                algorithm="lloyd",
            ).fit_predict(features),
            k_values,
        )
    recovered_k = int(len(np.unique(labels)))
    return MethodFit(
        method="K-means",
        labels=labels,
        recovered_k=recovered_k,
        selection_parameter=float(requested_k),
        selection_description="silhouette-selected requested K",
        selected_silhouette=selected_score,
        runtime_seconds=float(time.perf_counter() - started),
        adapter_scope="generic full-tensor proportion baseline",
        diagnostic={
            "feature_space": "within-clone/time state proportions",
            "selected_requested_k": requested_k,
            "selected_recovered_k": recovered_k,
            "candidate_count": len(diagnostics),
            "unique_recovered_k_count": len(
                {int(row["recovered_k"]) for row in diagnostics}
            ),
            "requested_recovered_mismatch_count": sum(
                int(row["requested_k"] != row["recovered_k"])
                for row in diagnostics
            ),
            "selected_at_grid_boundary": requested_k in {k_values[0], k_values[-1]},
            "k_grid_sha256": hashlib.sha256(
                json.dumps(k_values, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "convergence_warning_count": sum(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ),
        },
    )


def fit_ward_method(
    counts: np.ndarray,
    clone_ids: Sequence[str],
    config: dict,
) -> MethodFit:
    """Run Ward clustering with a truth-blind silhouette choice of K."""

    del clone_ids
    started = time.perf_counter()
    features = tensor_to_proportions(counts)
    k_values = list(range(int(config["min_k"]), int(config["max_k"]) + 1))
    labels, requested_k, selected_score, diagnostics = _select_silhouette_k(
        features,
        lambda k: AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
            metric="euclidean",
        ).fit_predict(features),
        k_values,
    )
    recovered_k = int(len(np.unique(labels)))
    return MethodFit(
        method="Ward",
        labels=labels,
        recovered_k=recovered_k,
        selection_parameter=float(requested_k),
        selection_description="silhouette-selected K",
        selected_silhouette=selected_score,
        runtime_seconds=float(time.perf_counter() - started),
        adapter_scope="generic full-tensor proportion baseline",
        diagnostic={
            "feature_space": "within-clone/time state proportions",
            "selected_requested_k": requested_k,
            "selected_recovered_k": recovered_k,
            "candidate_count": len(diagnostics),
            "unique_recovered_k_count": len(
                {int(row["recovered_k"]) for row in diagnostics}
            ),
            "selected_at_grid_boundary": requested_k in {k_values[0], k_values[-1]},
            "k_grid_sha256": hashlib.sha256(
                json.dumps(k_values, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
        },
    )


def fit_named_method(
    method: str,
    counts: np.ndarray,
    clone_ids: Sequence[str],
    config: dict,
) -> MethodFit:
    """Dispatch one predeclared truth-blind method adapter."""

    adapters = {
        "ChronoClone": lambda: fit_chronoclone_method(
            counts, clone_ids, config["chronoclone"]
        ),
        "CLADES preprocessing": lambda: fit_clades_preprocessing_method(
            counts, clone_ids, config["clades"]
        ),
        "Cyclone abundance-only": lambda: fit_cyclone_method(
            counts, clone_ids, config["cyclone"]
        ),
        "K-means": lambda: fit_kmeans_method(
            counts, clone_ids, config["generic_baselines"]
        ),
        "Ward": lambda: fit_ward_method(
            counts, clone_ids, config["generic_baselines"]
        ),
    }
    if method not in adapters:
        raise KeyError(f"Unknown benchmark method: {method}")
    fit = adapters[method]()
    if fit.method != method:
        raise AssertionError("Method adapter returned a mismatched method label")
    if len(fit.labels) != len(counts):
        raise AssertionError("Method adapter changed the common eligible clone set")
    return fit


def score_method_fit(fit: MethodFit, truth: Sequence[str]) -> dict[str, float]:
    """Evaluate a completed method fit after all truth-blind fitting."""

    truth_array = np.asarray(truth)
    if len(truth_array) != len(fit.labels):
        raise ValueError("Truth and method labels have different lengths")
    return {
        "ari": float(adjusted_rand_score(truth_array, fit.labels)),
        "nmi": float(normalized_mutual_info_score(truth_array, fit.labels)),
        "hungarian_macro_f1": hungarian_macro_f1(truth_array, fit.labels),
    }


def hungarian_macro_f1(y_true: Sequence[str], y_pred: Sequence[int]) -> float:
    """Macro-F1 after one-to-one Hungarian alignment of arbitrary cluster IDs."""

    true_levels = np.unique(y_true)
    pred_levels = np.unique(y_pred)
    true_codes = pd.Categorical(y_true, categories=true_levels).codes
    pred_codes = pd.Categorical(y_pred, categories=pred_levels).codes
    contingency = np.zeros((len(true_levels), len(pred_levels)), dtype=np.int64)
    np.add.at(contingency, (true_codes, pred_codes), 1)

    size = max(contingency.shape)
    padded = np.zeros((size, size), dtype=np.int64)
    padded[: contingency.shape[0], : contingency.shape[1]] = contingency
    rows, columns = linear_sum_assignment(-padded)
    scores = np.zeros(len(true_levels), dtype=float)
    for row, column in zip(rows, columns):
        if row >= len(true_levels) or column >= len(pred_levels):
            continue
        true_positive = contingency[row, column]
        denominator = (
            contingency[row, :].sum() + contingency[:, column].sum()
        )
        scores[row] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    return float(scores.mean())


def score_native_fit(fit: NativeFit, truth: Sequence[str]) -> dict[str, float]:
    """Evaluate a native fit; truth is not used anywhere in fitting."""

    truth_array = np.asarray(truth)[fit.eligible_mask]
    if len(truth_array) != len(fit.labels):
        raise ValueError("Truth and native labels have different eligible lengths")
    return {
        "ari": float(adjusted_rand_score(truth_array, fit.labels)),
        "nmi": float(normalized_mutual_info_score(truth_array, fit.labels)),
        "hungarian_macro_f1": hungarian_macro_f1(truth_array, fit.labels),
    }


def scenario_name(gaussian_pct: int, uniform_pct: int) -> str:
    """Return the human-readable grid region for one condition."""

    if gaussian_pct == 0 and uniform_pct == 0:
        return "baseline"
    if uniform_pct == 0:
        return "gaussian_only"
    if gaussian_pct == 0:
        return "uniform_only"
    return "gaussian_then_uniform"
