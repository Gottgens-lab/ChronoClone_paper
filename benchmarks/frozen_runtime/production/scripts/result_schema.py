#!/usr/bin/env python3
"""Result schema and checkpoint validation for the corrected benchmark.

This module is intentionally independent of the fitting implementation so the
controller and later audit can validate a shard without importing Scanpy or
ChronoClone.  A production shard has one row per
``(repeat, gaussian_pct, uniform_pct, method)`` and is resumable only when all
immutable provenance and input-signature fields still agree.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


GAUSSIAN_BLOCK_COUNT = 6
METHODS = (
    "ChronoClone",
    "CLADES preprocessing",
    "Cyclone abundance-only",
    "K-means",
    "Ward",
)
TERMINAL_STATUSES = frozenset({"ok", "method_failure"})
ALLOWED_STATUSES = frozenset({*TERMINAL_STATUSES, "error"})
KEY_COLUMNS = ("repeat", "gaussian_pct", "uniform_pct", "method")
HASH_COLUMNS = (
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
)

REQUIRED_COLUMNS = (
    # Immutable run provenance.
    "schema_version",
    "benchmark_name",
    *HASH_COLUMNS,
    "chronoclone_commit",
    "python_version",
    # Condition input identity and scientific-contract diagnostics.
    "input_tensor_sha256",
    "day0_tensor_sha256",
    "clean_day0_tensor_sha256",
    "day0_changed_entries",
    "day0_changed_clones",
    "eligibility_mask_sha256",
    "eligible_clone_ids_sha256",
    "clean_fixed_mask_sha256",
    "clean_fixed_clone_ids_sha256",
    "method",
    "adapter_scope",
    "selection_track",
    "selection_description",
    "scenario",
    "repeat",
    "gaussian_block",
    "gaussian_pct",
    "uniform_pct",
    "master_seed",
    "gaussian_seed",
    "uniform_seed",
    "n_input_clones",
    "n_eligible_clones",
    "eligible_fraction",
    "n_clean_fixed_eligible_clones",
    "clean_fixed_eligible_fraction",
    "n_eligibility_mask_disagreements",
    "input_total_count",
    "input_zero_fraction",
    "gaussian_changed_later_entries",
    "uniform_changed_later_entries",
    "uniform_requested_move_count",
    "uniform_l1_half_count",
    # Full-coverage method output on the common eligible cohort.
    "recovered_k",
    "selection_parameter",
    "selected_silhouette",
    "ari",
    "nmi",
    "hungarian_macro_f1",
    "runtime_seconds",
    "method_diagnostic_json",
    # ChronoClone's exact >= -0.1 post-fit retention mask.  The same mask and
    # subset scores are repeated across method rows to support fair subset
    # comparison without refitting any method.
    "postfilter_status",
    "silhouette_cutoff",
    "n_postfit_retained_clones",
    "postfit_retained_fraction_of_eligible",
    "postfit_retained_fraction_of_all_input",
    "postfit_retained_k",
    "postfit_clusters_lost",
    "postfit_retention_mask_sha256",
    "postfit_retention_mask_hex",
    "postfit_sample_silhouette_sha256",
    "postfit_silhouette_min",
    "postfit_silhouette_mean",
    "postfit_silhouette_median",
    "postfit_silhouette_max",
    "program_retention_json",
    "shared_retained_subset_ari",
    "shared_retained_subset_nmi",
    "shared_retained_subset_hungarian_macro_f1",
    "shared_retained_subset_abstention_aware_hungarian_macro_f1",
    # Row lifecycle.
    "status",
    "error_type",
    "error_message",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "completed_at_utc",
)


def sha256_file(path: Path) -> str:
    """Return SHA-256 of a file without loading it in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash shape, dtype and bytes of an array."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def strings_sha256(values: Iterable[object]) -> str:
    """Hash an ordered sequence of strings with unambiguous separators."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping | list) -> str:
    """Hash a JSON-compatible object using a canonical serialization."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scenario_name(gaussian_pct: int, uniform_pct: int) -> str:
    if gaussian_pct == 0 and uniform_pct == 0:
        return "baseline"
    if uniform_pct == 0:
        return "gaussian_only"
    if gaussian_pct == 0:
        return "uniform_only"
    return "gaussian_then_uniform"


def gaussian_levels_for_block(levels: list[int], block: int) -> list[int]:
    """Map one of six production blocks to its nine (last: six) doses."""

    if levels != list(range(0, 101, 2)):
        raise ValueError("Noise levels must be exactly 0..100 by 2")
    if not 0 <= int(block) < GAUSSIAN_BLOCK_COUNT:
        raise ValueError("Gaussian block must be in 0..5")
    start = 9 * int(block)
    stop = 9 * (int(block) + 1) if int(block) < 5 else len(levels)
    selected = levels[start:stop]
    expected = 9 if int(block) < 5 else 6
    if len(selected) != expected:
        raise AssertionError("Unexpected Gaussian block size")
    return selected


def parse_level_subset(
    text: str | None, allowed: list[int], option_name: str
) -> list[int]:
    if text is None:
        return list(allowed)
    try:
        selected = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"{option_name} must be comma-separated integers") from exc
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"{option_name} must be nonempty with no duplicates")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"{option_name} contains disallowed levels: {unknown}")
    return sorted(selected)


def ordered_table(rows_by_key: Mapping[tuple[int, int, int, str], dict]) -> pd.DataFrame:
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    return pd.DataFrame(rows).reindex(columns=REQUIRED_COLUMNS)


def atomic_write_csv_gz(table: pd.DataFrame, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    table.to_csv(
        temporary,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    os.replace(temporary, destination)


def atomic_write_json(payload: Mapping, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def _as_finite_float(row: pd.Series, name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value):
        raise ValueError(f"Successful row has non-finite {name}")
    return value


def _validate_metric(value: object, name: str, low: float, high: float) -> None:
    observed = float(value)
    if not np.isfinite(observed) or not low - 1e-12 <= observed <= high + 1e-12:
        raise ValueError(f"{name} is outside [{low}, {high}]: {observed}")


def validate_existing(
    table: pd.DataFrame,
    *,
    repeat: int,
    block: int,
    gaussian_levels: list[int],
    uniform_levels: list[int],
    hashes: Mapping[str, str],
) -> None:
    """Strictly validate a partial or complete shard before resumption."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Existing shard is missing columns: {missing}")
    forbidden = [
        column
        for column in table.columns
        if "matched" in column.lower() or "target_k" in column.lower()
    ]
    if forbidden:
        raise ValueError(f"Matched-K fields are forbidden: {forbidden}")
    if table.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Existing shard has duplicate method-condition keys")
    if table.empty:
        return
    if set(table["repeat"].astype(int)) != {int(repeat)}:
        raise ValueError("Existing shard has the wrong repeat")
    if set(table["gaussian_block"].astype(int)) != {int(block)}:
        raise ValueError("Existing shard has the wrong Gaussian block")
    if not set(table["gaussian_pct"].astype(int)).issubset(set(gaussian_levels)):
        raise ValueError("Existing shard has an out-of-block Gaussian level")
    if not set(table["uniform_pct"].astype(int)).issubset(set(uniform_levels)):
        raise ValueError("Existing shard has an unconfigured uniform level")
    if not set(table["method"].astype(str)).issubset(set(METHODS)):
        raise ValueError("Existing shard has an unknown method")
    if set(table["selection_track"].astype(str)) != {
        "condition_specific_lineage_eligibility"
    }:
        raise ValueError("Existing shard has the wrong selection track")
    if not set(table["status"].astype(str)).issubset(ALLOWED_STATUSES):
        raise ValueError("Existing shard has an unknown status")
    for column, expected in hashes.items():
        if column not in table.columns:
            raise ValueError(f"Unknown requested provenance field: {column}")
        if set(table[column].astype(str)) != {str(expected)}:
            raise ValueError(f"Existing shard has a different {column}")

    condition_signatures = (
        "input_tensor_sha256",
        "day0_tensor_sha256",
        "clean_day0_tensor_sha256",
        "day0_changed_entries",
        "day0_changed_clones",
        "eligibility_mask_sha256",
        "eligible_clone_ids_sha256",
        "clean_fixed_mask_sha256",
        "clean_fixed_clone_ids_sha256",
        "gaussian_seed",
        "uniform_seed",
        "n_input_clones",
        "n_eligible_clones",
        "n_clean_fixed_eligible_clones",
        "input_total_count",
        "postfilter_status",
        "n_postfit_retained_clones",
        "postfit_retention_mask_sha256",
        "postfit_retention_mask_hex",
    )
    for key, group in table.groupby(
        ["repeat", "gaussian_pct", "uniform_pct"], sort=False, dropna=False
    ):
        for column in condition_signatures:
            if group[column].astype(str).nunique(dropna=False) != 1:
                raise ValueError(f"Methods disagree in {column} at {key}")

    for _, row in table.iterrows():
        gaussian_pct = int(row["gaussian_pct"])
        uniform_pct = int(row["uniform_pct"])
        if str(row["scenario"]) != scenario_name(gaussian_pct, uniform_pct):
            raise ValueError("Scenario label does not match noise coordinates")
        n_input = int(row["n_input_clones"])
        n_eligible = int(row["n_eligible_clones"])
        n_clean = int(row["n_clean_fixed_eligible_clones"])
        if n_input < 1 or not 0 <= n_eligible <= n_input or not 0 <= n_clean <= n_input:
            raise ValueError("Invalid clone-count denominator")
        if not np.isclose(float(row["eligible_fraction"]), n_eligible / n_input):
            raise ValueError("Invalid eligible fraction")
        if not np.isclose(
            float(row["clean_fixed_eligible_fraction"]), n_clean / n_input
        ):
            raise ValueError("Invalid clean-fixed eligible fraction")
        if int(row["day0_changed_entries"]) != 0 or int(row["day0_changed_clones"]) != 0:
            raise ValueError("Day0-lock invariant failed in checkpoint")
        if str(row["day0_tensor_sha256"]) != str(row["clean_day0_tensor_sha256"]):
            raise ValueError("Day0 hash differs from clean Day0")
        if int(row["input_total_count"]) < 0:
            raise ValueError("Negative input total")
        _validate_metric(row["input_zero_fraction"], "input_zero_fraction", 0, 1)
        try:
            json.loads(str(row["method_diagnostic_json"]))
            if pd.notna(row["program_retention_json"]):
                json.loads(str(row["program_retention_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON field") from exc

        status = str(row["status"])
        if status == "ok":
            for name in ("recovered_k", "selection_parameter", "runtime_seconds"):
                _as_finite_float(row, name)
            _validate_metric(row["ari"], "ari", -1, 1)
            _validate_metric(row["nmi"], "nmi", 0, 1)
            _validate_metric(row["hungarian_macro_f1"], "hungarian_macro_f1", 0, 1)
            recovered_k = float(row["recovered_k"])
            if recovered_k != round(recovered_k) or not 1 <= recovered_k <= n_eligible:
                raise ValueError("Invalid recovered K")
            if float(row["runtime_seconds"]) < 0:
                raise ValueError("Negative method runtime")
            if pd.notna(row["selected_silhouette"]):
                _validate_metric(row["selected_silhouette"], "selected_silhouette", -1, 1)
            for text_field in ("adapter_scope", "selection_description"):
                if pd.isna(row[text_field]) or not str(row[text_field]).strip():
                    raise ValueError(f"Successful row lacks {text_field}")
            if str(row["postfilter_status"]) in {"ok", "all_abstained", "insufficient"}:
                retained = int(row["n_postfit_retained_clones"])
                if not 0 <= retained <= n_eligible:
                    raise ValueError("Invalid retained-clone count")
                if not np.isclose(
                    float(row["postfit_retained_fraction_of_eligible"]),
                    retained / n_eligible,
                ):
                    raise ValueError("Invalid retained fraction")
        elif status == "method_failure":
            if any(pd.notna(row[name]) for name in ("ari", "nmi", "hungarian_macro_f1")):
                raise ValueError("Method-failure row contains accuracy metrics")
            if not str(row["error_type"]).strip() or not str(row["error_message"]).strip():
                raise ValueError("Method-failure row lacks an error record")


def expected_keys(
    repeat: int,
    gaussian_levels: list[int],
    uniform_levels: list[int],
) -> set[tuple[int, int, int, str]]:
    return {
        (int(repeat), int(g), int(u), method)
        for g in gaussian_levels
        for u in uniform_levels
        for method in METHODS
    }


__all__ = [
    "ALLOWED_STATUSES",
    "GAUSSIAN_BLOCK_COUNT",
    "HASH_COLUMNS",
    "KEY_COLUMNS",
    "METHODS",
    "REQUIRED_COLUMNS",
    "TERMINAL_STATUSES",
    "array_sha256",
    "atomic_write_csv_gz",
    "atomic_write_json",
    "canonical_json_sha256",
    "expected_keys",
    "gaussian_levels_for_block",
    "ordered_table",
    "parse_level_subset",
    "scenario_name",
    "sha256_file",
    "strings_sha256",
    "validate_existing",
]
