#!/usr/bin/env python3
"""Run one corrected Day0-locked five-method benchmark shard.

Array task ``t`` owns ``repeat=t//6`` and ``gaussian_block=t%6``.  Day0 is
never sampled or redistributed; lineage eligibility is recalculated from each
realized tensor and the exact same ordered cohort is passed to all five
truth-blind adapters.  ChronoClone's fitted partition is additionally scored
after the sole inclusive sample-silhouette rule ``>= -0.1`` (no size rule and
no refit).

Checkpoints are compressed and atomically replaced after each Gaussian level.
A completion sidecar is written only after every expected key is terminal.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from result_schema import (  # noqa: E402
    GAUSSIAN_BLOCK_COUNT,
    KEY_COLUMNS,
    METHODS,
    REQUIRED_COLUMNS,
    TERMINAL_STATUSES,
    array_sha256,
    atomic_write_csv_gz,
    atomic_write_json,
    canonical_json_sha256,
    expected_keys,
    gaussian_levels_for_block,
    ordered_table,
    parse_level_subset,
    scenario_name,
    sha256_file,
    strings_sha256,
    validate_existing,
)
from source_runtime import (  # noqa: E402
    SOURCE_MANIFEST,
    activate_source_imports,
)

SOURCE_PROVENANCE = activate_source_imports()

import benchmark_core as legacy_core  # noqa: E402
import corrected_core  # noqa: E402
import cyclone_adapter  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "benchmark_config.json"
DEFAULT_REFERENCE = ROOT / "reference"
DEFAULT_OUTPUT = ROOT / "results" / "chunks"
DEFAULT_ENVIRONMENT_MANIFEST = SOURCE_MANIFEST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--gaussian-block", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gaussian-levels",
        help="comma-separated in-block subset, intended only for preflight",
    )
    parser.add_argument(
        "--uniform-levels",
        help="comma-separated configured subset, intended only for preflight",
    )
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=DEFAULT_ENVIRONMENT_MANIFEST,
    )
    return parser.parse_args()


def resolve_task(args: argparse.Namespace) -> tuple[int, int, int]:
    task_id = args.task_id
    if task_id is None:
        text = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
        task_id = int(text) if text else None
    explicit = args.repeat is not None or args.gaussian_block is not None
    if explicit:
        if args.repeat is None or args.gaussian_block is None:
            raise ValueError("Provide both --repeat and --gaussian-block")
        derived = int(args.repeat) * GAUSSIAN_BLOCK_COUNT + int(
            args.gaussian_block
        )
        if task_id is not None and int(task_id) != derived:
            raise ValueError("Explicit repeat/block disagree with task id")
        task_id = derived
    if task_id is None or int(task_id) < 0:
        raise ValueError("Provide a non-negative --task-id")
    task_id = int(task_id)
    return task_id, task_id // GAUSSIAN_BLOCK_COUNT, task_id % GAUSSIAN_BLOCK_COUNT


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another process owns {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} task={os.environ.get('SLURM_ARRAY_TASK_ID', '')}\n")
    handle.flush()
    return handle


def conda_records_sha256(prefix: Path) -> str:
    records = []
    retained = (
        "name",
        "version",
        "build",
        "build_number",
        "channel",
        "url",
        "sha256",
        "md5",
    )
    paths = sorted((prefix / "conda-meta").glob("*.json"))
    if not paths:
        raise RuntimeError(f"No Conda records under {prefix}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append({name: payload.get(name) for name in retained})
    return canonical_json_sha256(records)


def package_tree_sha256(package_dir: Path) -> str:
    digest = __import__("hashlib").sha256()
    paths = sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not paths:
        raise RuntimeError(f"No package files under {package_dir}")
    for path in paths:
        digest.update(path.relative_to(package_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_runtime_environment(manifest_path: Path) -> dict[str, str]:
    """Require the exact MacBook environment locked by the source benchmark."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected_versions = {
        str(name): str(value) for name, value in manifest["versions"].items()
    }
    observed_versions = {"python": platform.python_version()}
    for distribution in sorted(set(expected_versions) - {"python"}):
        observed_versions[distribution] = importlib.metadata.version(distribution)
    mismatches = {
        name: {
            "expected": expected_versions.get(name),
            "observed": observed_versions.get(name),
        }
        for name in sorted(set(expected_versions) | set(observed_versions))
        if expected_versions.get(name) != observed_versions.get(name)
    }
    if mismatches:
        raise RuntimeError(f"Runtime package versions changed: {mismatches}")
    prefix = Path(sys.prefix).resolve()
    if prefix != Path(manifest["python_prefix"]).resolve():
        raise RuntimeError(
            f"Python prefix changed: {prefix} != {manifest['python_prefix']}"
        )
    observed_conda = conda_records_sha256(prefix)
    if observed_conda != str(manifest["conda_records_sha256"]):
        raise RuntimeError("Solved Conda records changed")

    import chronoclone

    module_path = Path(chronoclone.__file__).resolve()
    if module_path != Path(manifest["chronoclone_module_path"]).resolve():
        raise RuntimeError("Imported ChronoClone path changed")
    observed_tree = package_tree_sha256(module_path.parent)
    if observed_tree != str(manifest["chronoclone_package_tree_sha256"]):
        raise RuntimeError("Locked ChronoClone package tree changed")
    return {
        "runtime_versions_sha256": canonical_json_sha256(observed_versions),
        "conda_records_sha256": observed_conda,
        "chronoclone_package_tree_sha256": observed_tree,
        "python_version": observed_versions["python"],
        "chronoclone_commit": str(manifest["chronoclone_commit"]),
    }


def validate_config(config: dict) -> None:
    levels = [int(value) for value in config["noise_levels_pct"]]
    if levels != list(range(0, 101, 2)):
        raise ValueError("Noise levels must be 0..100 in 2% increments")
    if tuple(config["method_order"]) != METHODS:
        raise ValueError("Method order differs from the locked five-method set")
    if config["primary_eligibility_track"] != "realized_tensor_lineage_continuity":
        raise ValueError("Primary eligibility track changed")
    if config["freeze_starting_timepoint"] is not True:
        raise ValueError("Starting-timepoint lock is disabled")
    if int(config["starting_timepoint"]) != 0:
        raise ValueError("Starting timepoint must be Day0")
    if tuple(config["allowed_start_states"]) != corrected_core.START_STATES:
        raise ValueError("Allowed starting states changed")
    if int(config["min_persistence"]) != corrected_core.MIN_PERSISTENCE:
        raise ValueError("Minimum persistence changed")
    if int(config["min_total_count"]) != corrected_core.MIN_TOTAL_COUNT:
        raise ValueError("Minimum total count changed")
    expected_noise_model = {
        "clean_count_realization": "round_to_nearest_even_then_clip_nonnegative",
        "gaussian_scope": "Day4_Day9_Day15_only",
        "gaussian_stream_id": corrected_core.DAY0_LOCKED_GAUSSIAN_STREAM,
        "gaussian_definition": "round(max(0, count + Normal(0, gaussian_fraction * count)))",
        "gaussian_structural_zero_sd": 0,
        "uniform_scope": "Day4_Day9_Day15_only_after_gaussian",
        "uniform_stream_id": corrected_core.DAY0_LOCKED_UNIFORM_STREAM,
        "uniform_move_count": "round_to_nearest_even(clone_time_total * uniform_fraction)",
        "uniform_reassignment": "independent_uniform_over_13_states",
        "uniform_preserves_clone_time_totals": True,
    }
    if config.get("noise_model") != expected_noise_model:
        raise ValueError("Noise-model contract changed")
    expected_cyclone = {
        "n_clusters": 6,
        "dynamic_clone_filter": False,
        "pairwise_complete_correlation": True,
        "interpretation": (
            "common-cohort applicability and method-failure rate; accuracy only when estimable"
        ),
    }
    if config.get("cyclone") != expected_cyclone:
        raise ValueError("Cyclone adapter contract changed")
    filter_config = config["chronoclone"]["postfit_silhouette_filter"]
    expected_filter = {
        "operator": ">=",
        "cutoff": -0.1,
        "metric": "euclidean",
        "group_size_filter_enabled": False,
        "refit_after_filter": False,
    }
    if filter_config != expected_filter:
        raise ValueError(f"ChronoClone post-fit rule changed: {filter_config}")


def bounded_metric(value: float, low: float, high: float) -> float:
    """Clamp only floating-point ULP spillover outside a metric's bounds."""

    observed = float(value)
    if not np.isfinite(observed):
        raise ValueError("Metric is non-finite")
    if observed < low - 1e-12 or observed > high + 1e-12:
        raise ValueError(f"Metric {observed} outside [{low}, {high}]")
    return float(min(high, max(low, observed)))


def score_labels(truth: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if len(truth) != len(labels) or len(truth) == 0:
        raise ValueError("Truth and labels must have equal nonzero length")
    return {
        "ari": bounded_metric(adjusted_rand_score(truth, labels), -1, 1),
        "nmi": bounded_metric(normalized_mutual_info_score(truth, labels), 0, 1),
        "hungarian_macro_f1": bounded_metric(
            legacy_core.hungarian_macro_f1(truth, labels), 0, 1
        ),
    }


def chronoclone_normalized_features(
    counts: np.ndarray, clone_ids: np.ndarray
) -> np.ndarray:
    """Reconstruct the exact normalized matrix used inside the locked adapter."""

    import chronoclone as cc

    raw = pd.DataFrame(
        np.asarray(counts).reshape(len(counts), -1),
        index=pd.Index(np.asarray(clone_ids, dtype=str), name="clone_id"),
        columns=legacy_core.WIDE_COLUMNS,
    )
    normalized = cc.normalize_clone_table(
        raw,
        times_order=[f"Day{time}" for time in legacy_core.TIMES],
        features_order=list(legacy_core.STATES),
    )
    expected_columns = list(legacy_core.WIDE_COLUMNS)
    if normalized.columns.tolist() != expected_columns:
        raise AssertionError("ChronoClone normalized feature order changed")
    if normalized.index.astype(str).tolist() != np.asarray(
        clone_ids, dtype=str
    ).tolist():
        raise AssertionError("ChronoClone normalized clone order changed")
    features = normalized.to_numpy(dtype=np.float64)
    if features.shape != (len(counts), len(legacy_core.WIDE_COLUMNS)):
        raise AssertionError("ChronoClone normalized feature shape changed")
    if not np.isfinite(features).all():
        raise AssertionError("ChronoClone normalized features are non-finite")
    return features


def abstention_aware_hungarian_macro_f1(
    truth_all: np.ndarray,
    labels_retained: np.ndarray,
    keep_mask: np.ndarray,
) -> float:
    """Macro-F1 with clones rejected by ChronoClone counted as false negatives."""

    truth_all = np.asarray(truth_all)
    labels_retained = np.asarray(labels_retained)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if len(truth_all) != len(keep_mask):
        raise ValueError("Truth and retention mask differ in length")
    if len(labels_retained) != int(keep_mask.sum()):
        raise ValueError("Retained labels and retention mask disagree")
    true_levels = np.unique(truth_all)
    if len(labels_retained) == 0:
        return 0.0
    retained_truth = truth_all[keep_mask]
    pred_levels = np.unique(labels_retained)
    true_codes = pd.Categorical(retained_truth, categories=true_levels).codes
    pred_codes = pd.Categorical(labels_retained, categories=pred_levels).codes
    contingency = np.zeros((len(true_levels), len(pred_levels)), dtype=np.int64)
    np.add.at(contingency, (true_codes, pred_codes), 1)
    size = max(contingency.shape)
    padded = np.zeros((size, size), dtype=np.int64)
    padded[: contingency.shape[0], : contingency.shape[1]] = contingency
    rows, columns = linear_sum_assignment(-padded)
    true_totals = np.asarray(
        [(truth_all == level).sum() for level in true_levels], dtype=np.int64
    )
    predicted_totals = contingency.sum(axis=0)
    scores = np.zeros(len(true_levels), dtype=np.float64)
    for row, column in zip(rows, columns):
        if row >= len(true_levels) or column >= len(pred_levels):
            continue
        true_positive = contingency[row, column]
        denominator = true_totals[row] + predicted_totals[column]
        scores[row] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    return bounded_metric(float(scores.mean()), 0, 1)


def empty_result_fields() -> dict:
    fields = {name: np.nan for name in REQUIRED_COLUMNS}
    fields.update(
        method="",
        adapter_scope="",
        selection_description="",
        method_diagnostic_json="{}",
        postfilter_status="not_evaluated",
        program_retention_json="{}",
        status="",
        error_type="",
        error_message="",
        completed_at_utc="",
    )
    return fields


def base_row(
    *,
    config: dict,
    provenance: Mapping[str, str],
    repeat: int,
    block: int,
    gaussian_pct: int,
    uniform_pct: int,
    gaussian_seed: int,
    uniform_seed: int,
    clean_counts: np.ndarray,
    gaussian_counts: np.ndarray,
    counts: np.ndarray,
    current_mask: np.ndarray,
    clean_fixed_mask: np.ndarray,
    clone_ids: np.ndarray,
) -> dict:
    row = empty_result_fields()
    changed_day0 = counts[:, 0, :] != clean_counts[:, 0, :]
    later_uniform_delta = counts[:, 1:, :] - gaussian_counts[:, 1:, :]
    later_totals = gaussian_counts[:, 1:, :].sum(axis=2, dtype=np.int64)
    requested_moves = np.rint(
        later_totals * (int(uniform_pct) / 100.0)
    ).astype(np.int64)
    row.update(
        schema_version=str(config["schema_version"]),
        benchmark_name=str(config["benchmark_name"]),
        **provenance,
        input_tensor_sha256=array_sha256(counts),
        day0_tensor_sha256=array_sha256(counts[:, 0, :]),
        clean_day0_tensor_sha256=array_sha256(clean_counts[:, 0, :]),
        day0_changed_entries=int(changed_day0.sum()),
        day0_changed_clones=int(np.any(changed_day0, axis=1).sum()),
        eligibility_mask_sha256=array_sha256(
            current_mask.astype(np.uint8, copy=False)
        ),
        eligible_clone_ids_sha256=strings_sha256(clone_ids[current_mask]),
        clean_fixed_mask_sha256=array_sha256(
            clean_fixed_mask.astype(np.uint8, copy=False)
        ),
        clean_fixed_clone_ids_sha256=strings_sha256(clone_ids[clean_fixed_mask]),
        selection_track="condition_specific_lineage_eligibility",
        scenario=scenario_name(gaussian_pct, uniform_pct),
        repeat=int(repeat),
        gaussian_block=int(block),
        gaussian_pct=int(gaussian_pct),
        uniform_pct=int(uniform_pct),
        master_seed=int(config["master_seed"]),
        gaussian_seed=f"{int(gaussian_seed):016x}",
        uniform_seed=f"{int(uniform_seed):016x}",
        n_input_clones=int(len(counts)),
        n_eligible_clones=int(current_mask.sum()),
        eligible_fraction=float(current_mask.mean()),
        n_clean_fixed_eligible_clones=int(clean_fixed_mask.sum()),
        clean_fixed_eligible_fraction=float(clean_fixed_mask.mean()),
        n_eligibility_mask_disagreements=int(
            np.count_nonzero(current_mask != clean_fixed_mask)
        ),
        input_total_count=int(counts.sum(dtype=np.int64)),
        input_zero_fraction=float(np.mean(counts == 0)),
        gaussian_changed_later_entries=int(
            np.count_nonzero(gaussian_counts[:, 1:, :] != clean_counts[:, 1:, :])
        ),
        uniform_changed_later_entries=int(np.count_nonzero(later_uniform_delta)),
        uniform_requested_move_count=int(requested_moves.sum(dtype=np.int64)),
        uniform_l1_half_count=int(
            np.abs(later_uniform_delta).sum(dtype=np.int64) // 2
        ),
        slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    )
    return row


def postfilter_fields_without_selection(status: str) -> dict:
    return {
        "postfilter_status": status,
        "silhouette_cutoff": corrected_core.SILHOUETTE_CUTOFF,
        "n_postfit_retained_clones": np.nan,
        "postfit_retained_fraction_of_eligible": np.nan,
        "postfit_retained_fraction_of_all_input": np.nan,
        "postfit_retained_k": np.nan,
        "postfit_clusters_lost": np.nan,
        "postfit_retention_mask_sha256": "",
        "postfit_retention_mask_hex": "",
        "postfit_sample_silhouette_sha256": "",
        "postfit_silhouette_min": np.nan,
        "postfit_silhouette_mean": np.nan,
        "postfit_silhouette_median": np.nan,
        "postfit_silhouette_max": np.nan,
        "program_retention_json": "{}",
        "shared_retained_subset_ari": np.nan,
        "shared_retained_subset_nmi": np.nan,
        "shared_retained_subset_hungarian_macro_f1": np.nan,
        "shared_retained_subset_abstention_aware_hungarian_macro_f1": np.nan,
    }


def postfilter_structural_fields(
    selection: corrected_core.ChronoClonePostfilter,
    *,
    recovered_k: int,
    n_input: int,
) -> dict:
    keep = selection.retained_mask
    retained_k = int(len(np.unique(selection.labels[keep]))) if keep.any() else 0
    if selection.n_retained == 0:
        status = "all_abstained"
    elif selection.n_retained == 1:
        status = "insufficient"
    else:
        status = "ok"
    return {
        "postfilter_status": status,
        "silhouette_cutoff": corrected_core.SILHOUETTE_CUTOFF,
        "n_postfit_retained_clones": int(selection.n_retained),
        "postfit_retained_fraction_of_eligible": float(
            selection.n_retained / selection.n_fitted
        ),
        "postfit_retained_fraction_of_all_input": float(
            selection.n_retained / n_input
        ),
        "postfit_retained_k": retained_k,
        "postfit_clusters_lost": int(recovered_k - retained_k),
        "postfit_retention_mask_sha256": array_sha256(
            keep.astype(np.uint8, copy=False)
        ),
        "postfit_retention_mask_hex": np.packbits(
            keep.astype(np.uint8, copy=False), bitorder="little"
        ).tobytes().hex(),
        "postfit_sample_silhouette_sha256": array_sha256(
            selection.sample_silhouette.astype(np.float64, copy=False)
        ),
        "postfit_silhouette_min": float(selection.sample_silhouette.min()),
        "postfit_silhouette_mean": float(selection.sample_silhouette.mean()),
        "postfit_silhouette_median": float(
            np.median(selection.sample_silhouette)
        ),
        "postfit_silhouette_max": float(selection.sample_silhouette.max()),
    }


def add_truth_dependent_postfilter_fields(
    fields: dict,
    *,
    selection: corrected_core.ChronoClonePostfilter | None,
    eligible_truth: np.ndarray,
    method_labels: np.ndarray | None,
) -> dict:
    result = dict(fields)
    if selection is None:
        return result
    keep = selection.retained_mask
    program_retention = {
        str(level): float(np.sum(eligible_truth[keep] == level) / np.sum(eligible_truth == level))
        for level in np.unique(eligible_truth)
    }
    result["program_retention_json"] = json.dumps(
        program_retention, sort_keys=True, separators=(",", ":")
    )
    if method_labels is None:
        return result
    retained_labels = np.asarray(method_labels)[keep]
    result[
        "shared_retained_subset_abstention_aware_hungarian_macro_f1"
    ] = abstention_aware_hungarian_macro_f1(
        eligible_truth, retained_labels, keep
    )
    if int(keep.sum()) >= 2:
        subset = score_labels(eligible_truth[keep], retained_labels)
        result.update(
            shared_retained_subset_ari=subset["ari"],
            shared_retained_subset_nmi=subset["nmi"],
            shared_retained_subset_hungarian_macro_f1=subset[
                "hungarian_macro_f1"
            ],
        )
    return result


def provenance_hashes(
    config_path: Path,
    reference,
    environment_manifest: Path,
) -> dict[str, str]:
    runtime = validate_runtime_environment(environment_manifest)
    observed = {
        "config_sha256": sha256_file(config_path),
        "profile_sha256": reference.profile_sha256,
        "manifest_sha256": reference.manifest_sha256,
        "corrected_core_sha256": sha256_file(ROOT / "corrected_core.py"),
        "source_core_sha256": SOURCE_PROVENANCE["source_core_sha256"],
        "source_runtime_sha256": sha256_file(SCRIPT_DIR / "source_runtime.py"),
        "cyclone_adapter_sha256": sha256_file(SCRIPT_DIR / "cyclone_adapter.py"),
        "result_schema_sha256": sha256_file(SCRIPT_DIR / "result_schema.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "environment_manifest_sha256": sha256_file(environment_manifest),
        "runtime_versions_sha256": runtime["runtime_versions_sha256"],
        "conda_records_sha256": runtime["conda_records_sha256"],
        "chronoclone_package_tree_sha256": runtime[
            "chronoclone_package_tree_sha256"
        ],
        "chronoclone_commit": runtime["chronoclone_commit"],
        "python_version": runtime["python_version"],
    }
    return observed


def complete_sidecar_payload(
    *,
    destination: Path,
    task_id: int,
    repeat: int,
    block: int,
    row_count: int,
    provenance: Mapping[str, str],
) -> dict:
    return {
        "schema": "chronoclone_day0_locked_chunk_completion_v1",
        "task_id": int(task_id),
        "repeat": int(repeat),
        "gaussian_block": int(block),
        "status": "complete",
        "row_count": int(row_count),
        "terminal_row_count": int(row_count),
        "chunk_path": str(destination.resolve()),
        "chunk_sha256": sha256_file(destination),
        **{
            name: str(provenance[name])
            for name in (
                "config_sha256",
                "profile_sha256",
                "manifest_sha256",
                "corrected_core_sha256",
                "source_core_sha256",
                "source_runtime_sha256",
                "cyclone_adapter_sha256",
                "result_schema_sha256",
                "runner_sha256",
                "environment_manifest_sha256",
                "runtime_versions_sha256",
                "conda_records_sha256",
                "chronoclone_package_tree_sha256",
                "chronoclone_commit",
                "python_version",
            )
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_chunk(args: argparse.Namespace) -> Path:
    task_id, repeat, block = resolve_task(args)
    config_path = args.config.resolve()
    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    environment_manifest = args.environment_manifest.resolve()
    config = legacy_core.load_json(config_path)
    validate_config(config)
    if not 0 <= repeat < int(config["repeats"]):
        raise ValueError(f"Task {task_id} maps to invalid repeat {repeat}")
    configured_levels = [int(value) for value in config["noise_levels_pct"]]
    block_levels = gaussian_levels_for_block(configured_levels, block)
    gaussian_levels = parse_level_subset(
        args.gaussian_levels, block_levels, "--gaussian-levels"
    )
    uniform_levels = parse_level_subset(
        args.uniform_levels, configured_levels, "--uniform-levels"
    )

    reference = legacy_core.load_reference_programs(reference_dir)
    provenance = provenance_hashes(config_path, reference, environment_manifest)
    if provenance["profile_sha256"] != str(config["reference_profile_sha256"]):
        raise ValueError("Configuration/reference profile hashes disagree")
    if provenance["manifest_sha256"] != str(config["reference_manifest_sha256"]):
        raise ValueError("Configuration/reference manifest hashes disagree")
    if provenance["chronoclone_commit"] != str(config["chronoclone_git_commit"]):
        raise ValueError("ChronoClone commit differs from configuration")
    if provenance["chronoclone_commit"] != SOURCE_PROVENANCE["chronoclone_commit"]:
        raise ValueError("ChronoClone commit differs from locked source runtime")

    clean_counts = corrected_core.rounded_clean_counts(reference.clean_means)
    clean_fixed_mask = corrected_core.clean_fixed_eligibility_mask(
        clean_counts,
        start_states=config["allowed_start_states"],
        min_persistence=int(config["min_persistence"]),
        min_total_count=int(config["min_total_count"]),
    )
    corrected_core.assert_fixed_cohort_mask(
        clean_fixed_mask,
        clean_counts,
        start_states=config["allowed_start_states"],
        min_persistence=int(config["min_persistence"]),
        min_total_count=int(config["min_total_count"]),
    )

    destination = output_dir / f"repeat_{repeat:03d}__gblock_{block:02d}.csv.gz"
    sidecar = output_dir / f"repeat_{repeat:03d}__gblock_{block:02d}.complete.json"
    lock_handle = acquire_lock(
        output_dir.parent / "locks" / f"repeat_{repeat:03d}__gblock_{block:02d}.lock"
    )
    hashes_for_resume = {
        name: provenance[name]
        for name in (
            "config_sha256",
            "profile_sha256",
            "manifest_sha256",
            "corrected_core_sha256",
            "source_core_sha256",
            "source_runtime_sha256",
            "cyclone_adapter_sha256",
            "result_schema_sha256",
            "runner_sha256",
            "environment_manifest_sha256",
            "runtime_versions_sha256",
            "conda_records_sha256",
            "chronoclone_package_tree_sha256",
            "chronoclone_commit",
            "python_version",
        )
    }
    try:
        if destination.is_file():
            existing = pd.read_csv(
                destination,
                dtype={"gaussian_seed": str, "uniform_seed": str},
            )
            validate_existing(
                existing,
                repeat=repeat,
                block=block,
                gaussian_levels=gaussian_levels,
                uniform_levels=uniform_levels,
                hashes=hashes_for_resume,
            )
        else:
            existing = pd.DataFrame(columns=REQUIRED_COLUMNS)
        rows_by_key = {
            (
                int(row["repeat"]),
                int(row["gaussian_pct"]),
                int(row["uniform_pct"]),
                str(row["method"]),
            ): row.to_dict()
            for _, row in existing.iterrows()
            if str(row["status"]) in TERMINAL_STATUSES
        }
        expected = expected_keys(repeat, gaussian_levels, uniform_levels)
        if set(rows_by_key) == expected:
            payload = complete_sidecar_payload(
                destination=destination,
                task_id=task_id,
                repeat=repeat,
                block=block,
                row_count=len(rows_by_key),
                provenance=provenance,
            )
            atomic_write_json(payload, sidecar)
            print(f"ALREADY_COMPLETE {destination} rows={len(rows_by_key)}", flush=True)
            return destination

        had_error = False
        for gaussian_pct in gaussian_levels:
            gaussian_counts, gaussian_seed = corrected_core.gaussian_stage_day0_locked(
                reference.clean_means,
                gaussian_pct,
                repeat,
                int(config["master_seed"]),
            )
            corrected_core.assert_day0_frozen(clean_counts, gaussian_counts)
            for uniform_pct in uniform_levels:
                condition_keys = [
                    (repeat, gaussian_pct, uniform_pct, method) for method in METHODS
                ]
                missing_methods = [
                    method
                    for method, key in zip(METHODS, condition_keys)
                    if key not in rows_by_key
                ]
                if not missing_methods:
                    continue
                # If an earlier attempt could not construct the shared
                # ChronoClone retention mask, replace every row for this
                # condition together after the retry succeeds.  This prevents
                # terminal comparator rows from retaining a stale "error"
                # postfilter signature.
                existing_condition = [
                    rows_by_key[key] for key in condition_keys if key in rows_by_key
                ]
                if any(
                    str(row.get("postfilter_status", "")) == "error"
                    for row in existing_condition
                ):
                    missing_methods = list(METHODS)

                counts, uniform_seed = (
                    corrected_core.uniform_redistribution_stage_day0_locked(
                        gaussian_counts,
                        gaussian_pct,
                        uniform_pct,
                        repeat,
                        int(config["master_seed"]),
                        start_states=config["allowed_start_states"],
                        min_persistence=int(config["min_persistence"]),
                        min_total_count=int(config["min_total_count"]),
                    )
                )
                corrected_core.assert_day0_frozen(clean_counts, counts)
                eligibility = corrected_core.condition_eligibility_masks(
                    clean_counts,
                    counts,
                    start_states=config["allowed_start_states"],
                    min_persistence=int(config["min_persistence"]),
                    min_total_count=int(config["min_total_count"]),
                )
                if not np.array_equal(
                    eligibility.clean_fixed_mask, clean_fixed_mask
                ):
                    raise AssertionError("Clean-fixed eligibility changed")
                current_mask = eligibility.current_lineage_mask
                eligible_counts = counts[current_mask]
                eligible_ids = reference.clone_ids[current_mask]
                common = base_row(
                    config=config,
                    provenance=provenance,
                    repeat=repeat,
                    block=block,
                    gaussian_pct=gaussian_pct,
                    uniform_pct=uniform_pct,
                    gaussian_seed=gaussian_seed,
                    uniform_seed=uniform_seed,
                    clean_counts=clean_counts,
                    gaussian_counts=gaussian_counts,
                    counts=counts,
                    current_mask=current_mask,
                    clean_fixed_mask=clean_fixed_mask,
                    clone_ids=reference.clone_ids,
                )
                if common["day0_changed_entries"] != 0:
                    raise AssertionError("Day0 changed before fitting")

                # Fit every adapter before truth is accessed.  When resuming a
                # condition with one retryable error, deterministic successful
                # fits are harmlessly recomputed so the shared ChronoClone
                # retention mask is always available for subset scoring.
                fits: dict[str, object] = {}
                failures: dict[str, BaseException] = {}
                for method in METHODS:
                    try:
                        if len(eligible_counts) < 3:
                            raise legacy_core.NativeSelectionFailure(
                                "Fewer than three lineage-eligible clones remained"
                            )
                        if method == "Cyclone abundance-only":
                            fits[method] = cyclone_adapter.fit_cyclone_method(
                                eligible_counts, eligible_ids, config["cyclone"]
                            )
                        else:
                            fits[method] = legacy_core.fit_named_method(
                                method, eligible_counts, eligible_ids, config
                            )
                    except legacy_core.NativeSelectionFailure as exc:
                        failures[method] = exc
                    except Exception as exc:
                        traceback.print_exc()
                        failures[method] = exc
                        had_error = True

                selection = None
                cc_fit = fits.get("ChronoClone")
                if cc_fit is not None:
                    try:
                        features = chronoclone_normalized_features(
                            eligible_counts, eligible_ids
                        )
                        selection = corrected_core.chronoclone_silhouette_postfilter(
                            features, cc_fit.labels
                        )
                        if not np.isclose(
                            selection.sample_silhouette.mean(),
                            float(cc_fit.selected_silhouette),
                            atol=1e-12,
                            rtol=1e-10,
                        ):
                            raise AssertionError(
                                "ChronoClone mean/sample silhouettes disagree"
                            )
                        postfilter_base = postfilter_structural_fields(
                            selection,
                            recovered_k=int(cc_fit.recovered_k),
                            n_input=len(counts),
                        )
                    except Exception as exc:
                        traceback.print_exc()
                        failures["ChronoClone"] = exc
                        fits.pop("ChronoClone", None)
                        selection = None
                        postfilter_base = postfilter_fields_without_selection("error")
                        had_error = True
                elif isinstance(
                    failures.get("ChronoClone"), legacy_core.NativeSelectionFailure
                ):
                    postfilter_base = postfilter_fields_without_selection(
                        "chronoclone_method_failure"
                    )
                else:
                    postfilter_base = postfilter_fields_without_selection("error")

                # Truth enters only here, after all five fits and the
                # truth-independent retention decision have completed.
                eligible_truth = reference.truth[current_mask]
                for method in missing_methods:
                    row = dict(common)
                    row["method"] = method
                    fit = fits.get(method)
                    postfilter_fields = add_truth_dependent_postfilter_fields(
                        postfilter_base,
                        selection=selection,
                        eligible_truth=eligible_truth,
                        method_labels=None if fit is None else fit.labels,
                    )
                    row.update(postfilter_fields)
                    if fit is None:
                        exc = failures[method]
                        unexpected = not isinstance(
                            exc, legacy_core.NativeSelectionFailure
                        )
                        row.update(
                            status="error" if unexpected else "method_failure",
                            error_type=type(exc).__name__,
                            error_message=str(exc)[:1000],
                            completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        )
                        if unexpected:
                            had_error = True
                    else:
                        try:
                            metrics = score_labels(eligible_truth, fit.labels)
                            row.update(
                                adapter_scope=fit.adapter_scope,
                                selection_description=fit.selection_description,
                                recovered_k=int(fit.recovered_k),
                                selection_parameter=float(fit.selection_parameter),
                                selected_silhouette=float(fit.selected_silhouette),
                                ari=metrics["ari"],
                                nmi=metrics["nmi"],
                                hungarian_macro_f1=metrics[
                                    "hungarian_macro_f1"
                                ],
                                runtime_seconds=float(fit.runtime_seconds),
                                method_diagnostic_json=json.dumps(
                                    fit.diagnostic,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                status="ok",
                                completed_at_utc=datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            )
                        except Exception as exc:
                            traceback.print_exc()
                            row.update(
                                status="error",
                                error_type=type(exc).__name__,
                                error_message=str(exc)[:1000],
                                completed_at_utc=datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            )
                            had_error = True
                    rows_by_key[(repeat, gaussian_pct, uniform_pct, method)] = row

                del counts, eligible_counts, eligible_ids, eligible_truth, fits, failures
                gc.collect()

            checkpoint = ordered_table(rows_by_key)
            validate_existing(
                checkpoint,
                repeat=repeat,
                block=block,
                gaussian_levels=gaussian_levels,
                uniform_levels=uniform_levels,
                hashes=hashes_for_resume,
            )
            atomic_write_csv_gz(checkpoint, destination)
            terminal_count = int(checkpoint["status"].isin(TERMINAL_STATUSES).sum())
            print(
                f"repeat={repeat:03d} block={block:02d} g={gaussian_pct:03d} "
                f"terminal_rows={terminal_count}/{len(expected)}",
                flush=True,
            )
            del gaussian_counts, checkpoint
            gc.collect()

        completed = pd.read_csv(
            destination, dtype={"gaussian_seed": str, "uniform_seed": str}
        )
        validate_existing(
            completed,
            repeat=repeat,
            block=block,
            gaussian_levels=gaussian_levels,
            uniform_levels=uniform_levels,
            hashes=hashes_for_resume,
        )
        terminal = completed[completed["status"].isin(TERMINAL_STATUSES)]
        observed = set(terminal[list(KEY_COLUMNS)].itertuples(index=False, name=None))
        if observed != expected:
            missing = sorted(expected - observed)
            raise RuntimeError(f"Shard incomplete; first missing keys: {missing[:10]}")
        if had_error or completed["status"].eq("error").any():
            raise RuntimeError("Shard contains retryable error rows")
        payload = complete_sidecar_payload(
            destination=destination,
            task_id=task_id,
            repeat=repeat,
            block=block,
            row_count=len(completed),
            provenance=provenance,
        )
        atomic_write_json(payload, sidecar)
        print(f"COMPLETE {destination} rows={len(completed)}", flush=True)
        return destination
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def main() -> int:
    started = time.perf_counter()
    try:
        path = run_chunk(parse_args())
    except Exception as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"WALL_SECONDS {time.perf_counter() - started:.3f} path={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
