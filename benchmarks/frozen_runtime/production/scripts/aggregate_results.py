#!/usr/bin/env python3
"""Strict, memory-bounded aggregation for the corrected production grid.

This program is intentionally separate from method fitting.  It accepts a
result only when all 600 hash-matched completion sidecars and shards exist,
all 1,300,500 method-condition keys are terminal and unique, Day0 remains
frozen, and all five methods received the same ordered eligible clone cohort
at every condition.  Method failures remain explicit applicability outcomes;
they are never converted to accuracy zero.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from result_schema import (  # noqa: E402
    GAUSSIAN_BLOCK_COUNT,
    HASH_COLUMNS,
    KEY_COLUMNS,
    METHODS,
    REQUIRED_COLUMNS,
    TERMINAL_STATUSES,
    array_sha256,
    gaussian_levels_for_block,
    scenario_name,
    sha256_file,
    validate_existing,
)


SOURCE_ROOT = ROOT.parent / "native_method_noise_benchmark_local_20260803"
SOURCE_CORE = SOURCE_ROOT / "benchmark_package" / "scripts" / "benchmark_core.py"
ENVIRONMENT_MANIFEST = SOURCE_ROOT / "results" / "provenance" / "environment_manifest.json"
CONFIG = ROOT / "config" / "benchmark_config.json"
PROFILE = ROOT / "reference" / "deterministic_profiles.csv"
PROGRAM_MANIFEST = ROOT / "reference" / "program_manifest.csv"
CHUNKS = ROOT / "results" / "chunks"
AGGREGATE = ROOT / "results" / "aggregate"
POSTPROCESSOR_SOURCES = (
    ROOT / "scripts" / "aggregate_results.py",
    ROOT / "scripts" / "build_notebooks.py",
    ROOT / "scripts" / "execute_notebook.py",
    ROOT / "scripts" / "validate_postprocessing.py",
)
POSTPROCESSOR_MANIFEST = (
    ROOT / "results" / "provenance" / "postprocessing_source_manifest.sha256"
)

FILENAME_RE = re.compile(r"^repeat_(\d{3})__gblock_(\d{2})\.csv\.gz$")
EXPECTED_REPEATS = 100
EXPECTED_LEVELS = list(range(0, 101, 2))
EXPECTED_CHUNKS = EXPECTED_REPEATS * GAUSSIAN_BLOCK_COUNT
EXPECTED_ROWS = EXPECTED_REPEATS * len(EXPECTED_LEVELS) ** 2 * len(METHODS)

MAIN_METRICS = (
    "ari",
    "nmi",
    "hungarian_macro_f1",
    "recovered_k",
    "selection_parameter",
    "selected_silhouette",
    "runtime_seconds",
)
CONDITION_METRICS = (
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
)
POSTFILTER_METRICS = (
    "n_postfit_retained_clones",
    "postfit_retained_fraction_of_eligible",
    "postfit_retained_fraction_of_all_input",
    "postfit_retained_k",
    "postfit_clusters_lost",
    "postfit_silhouette_min",
    "postfit_silhouette_mean",
    "postfit_silhouette_median",
    "postfit_silhouette_max",
    "shared_retained_subset_ari",
    "shared_retained_subset_nmi",
    "shared_retained_subset_hungarian_macro_f1",
    "shared_retained_subset_abstention_aware_hungarian_macro_f1",
)
SHARED_SUBSET_METRICS = (
    "shared_retained_subset_ari",
    "shared_retained_subset_nmi",
    "shared_retained_subset_hungarian_macro_f1",
    "shared_retained_subset_abstention_aware_hungarian_macro_f1",
)

COMMON_CONDITION_COLUMNS = (
    "input_tensor_sha256",
    "day0_tensor_sha256",
    "clean_day0_tensor_sha256",
    "day0_changed_entries",
    "day0_changed_clones",
    "eligibility_mask_sha256",
    "eligible_clone_ids_sha256",
    "clean_fixed_mask_sha256",
    "clean_fixed_clone_ids_sha256",
    "scenario",
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--input-dir", type=Path, default=CHUNKS)
    parser.add_argument("--output-dir", type=Path, default=AGGREGATE)
    return parser.parse_args()


def canonical_schema_version(value: object) -> str:
    """Normalize numerically equivalent CSV renderings such as 3 and 3.0."""

    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid schema version: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"Invalid schema version: {value!r}")
    return format(parsed.normalize(), "f")


def summarize_vector(
    values: Iterable[float], quantiles: tuple[float, float]
) -> dict[str, float | int]:
    """Return transparent empirical summaries without imputing failures."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "n_finite": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q025": np.nan,
            "q975": np.nan,
        }
    return {
        "n_finite": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "q025": float(np.quantile(finite, quantiles[0])),
        "q975": float(np.quantile(finite, quantiles[1])),
    }


def validate_common_condition_group(group: pd.DataFrame) -> None:
    """Require five rows with identical common input/retention identities."""

    if len(group) != len(METHODS) or set(group["method"].astype(str)) != set(METHODS):
        raise ValueError("A condition does not contain exactly the five locked methods")
    for column in COMMON_CONDITION_COLUMNS:
        if group[column].astype(str).nunique(dropna=False) != 1:
            coordinates = tuple(
                group.iloc[0][["repeat", "gaussian_pct", "uniform_pct"]].tolist()
            )
            raise ValueError(f"Methods disagree in {column} at {coordinates}")


def validate_uniform_diagnostics(row: pd.Series) -> None:
    """Validate saved redistribution diagnostics without reconstructing tensors."""

    requested = int(row["uniform_requested_move_count"])
    realized = int(row["uniform_l1_half_count"])
    changed = int(row["uniform_changed_later_entries"])
    uniform_pct = int(row["uniform_pct"])
    if min(requested, realized, changed) < 0:
        raise ValueError("Uniform-noise diagnostics cannot be negative")
    if realized > requested:
        raise ValueError("Realized uniform L1/2 exceeds requested reassignment count")
    if uniform_pct == 0 and (requested != 0 or realized != 0 or changed != 0):
        raise ValueError("Uniform-zero condition has nonzero redistribution diagnostics")


def _metric_in_range(value: object, low: float, high: float, name: str) -> None:
    observed = float(value)
    if not np.isfinite(observed) or not low - 1e-12 <= observed <= high + 1e-12:
        raise ValueError(f"{name} is outside [{low}, {high}]: {observed}")


def _is_sha256(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None


def validate_postfilter_condition(
    group: pd.DataFrame, *, program_ids: set[str]
) -> None:
    """Validate the lossless >= -0.1 mask and all subset-score domains."""

    cc = group.loc[group["method"].eq("ChronoClone")]
    if len(cc) != 1:
        raise ValueError("A condition lacks exactly one ChronoClone row")
    cc_row = cc.iloc[0]
    status = str(cc_row["postfilter_status"])
    n_input = int(cc_row["n_input_clones"])
    n_eligible = int(cc_row["n_eligible_clones"])
    if float(cc_row["silhouette_cutoff"]) != -0.1:
        raise ValueError("ChronoClone post-fit cutoff differs from -0.1")

    if status == "chronoclone_method_failure":
        if str(cc_row["status"]) != "method_failure":
            raise ValueError("Postfilter method-failure status disagrees with ChronoClone")
        nullable = (
            "n_postfit_retained_clones",
            "postfit_retained_fraction_of_eligible",
            "postfit_retained_fraction_of_all_input",
            "postfit_retained_k",
            "postfit_clusters_lost",
            "postfit_silhouette_min",
            "postfit_silhouette_mean",
            "postfit_silhouette_median",
            "postfit_silhouette_max",
        )
        if any(pd.notna(cc_row[name]) for name in nullable):
            raise ValueError("ChronoClone method-failure has postfilter numeric payload")
        if any(
            pd.notna(cc_row[name]) and str(cc_row[name]).strip()
            for name in (
                "postfit_retention_mask_sha256",
                "postfit_retention_mask_hex",
                "postfit_sample_silhouette_sha256",
            )
        ):
            raise ValueError("ChronoClone method-failure has postfilter hash payload")
        if json.loads(str(cc_row["program_retention_json"])) != {}:
            raise ValueError("ChronoClone method-failure has program-retention payload")
        for _, row in group.iterrows():
            for name in SHARED_SUBSET_METRICS:
                if pd.notna(row[name]):
                    raise ValueError("Method-failure postfilter has subset accuracy")
        return

    if status not in {"ok", "all_abstained", "insufficient"}:
        raise ValueError(f"Unknown terminal postfilter status: {status}")
    if str(cc_row["status"]) != "ok":
        raise ValueError("A populated postfilter requires a successful ChronoClone fit")

    n_retained = int(cc_row["n_postfit_retained_clones"])
    if not 0 <= n_retained <= n_eligible <= n_input:
        raise ValueError("Invalid postfilter clone counts")
    expected_status = (
        "all_abstained" if n_retained == 0 else "insufficient" if n_retained == 1 else "ok"
    )
    if status != expected_status:
        raise ValueError("Postfilter status disagrees with retained-clone count")
    if not np.isclose(
        float(cc_row["postfit_retained_fraction_of_eligible"]),
        n_retained / n_eligible,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("Postfilter eligible-cohort fraction is inconsistent")
    if not np.isclose(
        float(cc_row["postfit_retained_fraction_of_all_input"]),
        n_retained / n_input,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("Postfilter all-input fraction is inconsistent")

    mask_hex = str(cc_row["postfit_retention_mask_hex"])
    try:
        packed = bytes.fromhex(mask_hex)
    except ValueError as exc:
        raise ValueError("Postfilter mask is not valid hexadecimal") from exc
    expected_bytes = (n_eligible + 7) // 8
    if len(packed) != expected_bytes:
        raise ValueError("Postfilter mask byte length differs from eligible-clone count")
    unpacked = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), bitorder="little"
    )
    mask = unpacked[:n_eligible].astype(np.uint8, copy=False)
    if unpacked[n_eligible:].any():
        raise ValueError("Postfilter mask has nonzero padding bits")
    if int(mask.sum()) != n_retained:
        raise ValueError("Postfilter mask count differs from retained-clone count")
    if str(cc_row["postfit_retention_mask_sha256"]) != array_sha256(mask):
        raise ValueError("Postfilter mask hash does not match its decoded bytes")
    if not _is_sha256(cc_row["postfit_sample_silhouette_sha256"]):
        raise ValueError("Postfilter sample-silhouette hash is invalid")

    recovered_k = int(cc_row["recovered_k"])
    retained_k = int(cc_row["postfit_retained_k"])
    expected_k_floor = 0 if n_retained == 0 else 1
    if not expected_k_floor <= retained_k <= min(recovered_k, n_retained):
        raise ValueError("Postfilter retained K is outside its possible range")
    if int(cc_row["postfit_clusters_lost"]) != recovered_k - retained_k:
        raise ValueError("Postfilter clusters-lost value is inconsistent")

    silhouette_names = (
        "postfit_silhouette_min",
        "postfit_silhouette_mean",
        "postfit_silhouette_median",
        "postfit_silhouette_max",
    )
    silhouettes = [float(cc_row[name]) for name in silhouette_names]
    if not all(np.isfinite(value) and -1 - 1e-12 <= value <= 1 + 1e-12 for value in silhouettes):
        raise ValueError("Postfilter silhouette summaries are non-finite or out of range")
    minimum, mean, median, maximum = silhouettes
    if minimum > mean + 1e-12 or minimum > median + 1e-12 or max(mean, median) > maximum + 1e-12:
        raise ValueError("Postfilter silhouette summaries are not ordered")
    if n_retained == n_eligible and minimum < -0.1 - 1e-12:
        raise ValueError("All-retained mask conflicts with the silhouette cutoff")
    if n_retained < n_eligible and minimum >= -0.1:
        raise ValueError("Filtered mask conflicts with the silhouette cutoff")

    retention = json.loads(str(cc_row["program_retention_json"]))
    if not isinstance(retention, dict) or not retention:
        raise ValueError("Program-retention payload must be a nonempty object")
    if not set(map(str, retention)).issubset(program_ids):
        raise ValueError("Program-retention payload contains an unknown program")
    for value in retention.values():
        _metric_in_range(value, 0, 1, "program retention")
    if n_retained == n_eligible and not all(float(value) == 1 for value in retention.values()):
        raise ValueError("All-retained mask has incomplete program retention")
    if n_retained == 0 and not all(float(value) == 0 for value in retention.values()):
        raise ValueError("All-abstained mask has nonzero program retention")

    for _, row in group.iterrows():
        method_ok = str(row["status"]) == "ok"
        metric_contract = (
            ("shared_retained_subset_ari", -1, 1),
            ("shared_retained_subset_nmi", 0, 1),
            ("shared_retained_subset_hungarian_macro_f1", 0, 1),
        )
        for name, low, high in metric_contract:
            if method_ok and n_retained >= 2:
                _metric_in_range(row[name], low, high, name)
            elif pd.notna(row[name]):
                raise ValueError(f"{name} violates its status/nullability contract")
        abstention = "shared_retained_subset_abstention_aware_hungarian_macro_f1"
        if method_ok:
            _metric_in_range(row[abstention], 0, 1, abstention)
        elif pd.notna(row[abstention]):
            raise ValueError("Failed method has abstention-aware subset accuracy")


def atomic_csv(table: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    table.to_csv(temporary, index=False, float_format="%.12g")
    os.replace(temporary, destination)


def atomic_json(payload: Mapping, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def expected_hashes(config_path: Path) -> tuple[dict[str, str], dict]:
    environment = json.loads(ENVIRONMENT_MANIFEST.read_text(encoding="utf-8"))
    hashes = {
        "config_sha256": sha256_file(config_path),
        "profile_sha256": sha256_file(PROFILE),
        "manifest_sha256": sha256_file(PROGRAM_MANIFEST),
        "corrected_core_sha256": sha256_file(ROOT / "corrected_core.py"),
        "source_core_sha256": sha256_file(SOURCE_CORE),
        "source_runtime_sha256": sha256_file(SCRIPT_DIR / "source_runtime.py"),
        "cyclone_adapter_sha256": sha256_file(SCRIPT_DIR / "cyclone_adapter.py"),
        "result_schema_sha256": sha256_file(SCRIPT_DIR / "result_schema.py"),
        "runner_sha256": sha256_file(SCRIPT_DIR / "run_chunk.py"),
        "environment_manifest_sha256": sha256_file(ENVIRONMENT_MANIFEST),
        "runtime_versions_sha256": str(environment["runtime_versions_sha256"]),
        "conda_records_sha256": str(environment["conda_records_sha256"]),
        "chronoclone_package_tree_sha256": str(
            environment["chronoclone_package_tree_sha256"]
        ),
    }
    if set(hashes) != set(HASH_COLUMNS):
        raise AssertionError("Aggregator hash contract differs from result schema")
    return hashes, environment


def expected_postprocessor_hashes() -> dict[str, str]:
    missing = [str(path) for path in POSTPROCESSOR_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing postprocessing source files: {missing}")
    observed = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in POSTPROCESSOR_SOURCES
    }
    if not POSTPROCESSOR_MANIFEST.is_file():
        raise FileNotFoundError(POSTPROCESSOR_MANIFEST)
    locked: dict[str, str] = {}
    for line in POSTPROCESSOR_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        locked[name] = digest
    if locked != observed:
        raise ValueError("Postprocessing sources differ from the predeclared manifest")
    return observed


def validate_sidecar(
    path: Path,
    *,
    repeat: int,
    block: int,
    expected_rows: int,
    hashes: Mapping[str, str],
    environment: Mapping,
) -> dict:
    sidecar = path.with_name(path.name.removesuffix(".csv.gz") + ".complete.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing completion sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    required = {
        "schema": "chronoclone_day0_locked_chunk_completion_v1",
        "status": "complete",
        "task_id": repeat * GAUSSIAN_BLOCK_COUNT + block,
        "repeat": repeat,
        "gaussian_block": block,
        "row_count": expected_rows,
        "terminal_row_count": expected_rows,
        **hashes,
        "chronoclone_commit": str(environment["chronoclone_commit"]),
        "python_version": str(environment["versions"]["python"]),
    }
    mismatches = {
        name: {"expected": expected, "observed": payload.get(name)}
        for name, expected in required.items()
        if payload.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Completion sidecar mismatch for {path.name}: {mismatches}")
    if Path(str(payload.get("chunk_path", ""))).resolve() != path.resolve():
        raise ValueError(f"Completion sidecar points to a different chunk: {path.name}")
    if payload.get("chunk_sha256") != sha256_file(path):
        raise ValueError(f"Completion sidecar chunk hash mismatch: {path.name}")
    return {
        "task_id": required["task_id"],
        "repeat": repeat,
        "gaussian_block": block,
        "chunk": path.name,
        "rows": expected_rows,
        "chunk_sha256": payload["chunk_sha256"],
        "sidecar": sidecar.name,
        "sidecar_sha256": sha256_file(sidecar),
    }


def append_stats(
    row: dict,
    prefix: str,
    values: Iterable[float],
    quantiles: tuple[float, float],
) -> None:
    for suffix, value in summarize_vector(values, quantiles).items():
        row[f"{prefix}_{suffix}"] = value


def build_staged_aggregate(args: argparse.Namespace) -> dict:
    config_path = args.config.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    levels = [int(value) for value in config["noise_levels_pct"]]
    methods = tuple(str(value) for value in config["method_order"])
    if levels != EXPECTED_LEVELS or int(config["repeats"]) != EXPECTED_REPEATS:
        raise ValueError("Strict production design must be 100 repeats and 0..100 by 2%")
    if methods != METHODS:
        raise ValueError("Configured method order differs from the locked schema")
    quantiles = tuple(float(value) for value in config["uncertainty_interval"])
    if quantiles != (0.025, 0.975):
        raise ValueError("Uncertainty interval must be the locked 2.5th--97.5th percentiles")
    hashes, environment = expected_hashes(config_path)
    postprocessor_hashes = expected_postprocessor_hashes()
    program_table = pd.read_csv(PROGRAM_MANIFEST)
    if "truth_metaclone" not in program_table or program_table["truth_metaclone"].duplicated().any():
        raise ValueError("Program manifest has no unique truth_metaclone field")
    program_ids = set(program_table["truth_metaclone"].astype(str))
    if not program_ids:
        raise ValueError("Program manifest contains no truth programs")
    if hashes["profile_sha256"] != str(config["reference_profile_sha256"]):
        raise ValueError("Reference profile does not match configuration")
    if hashes["manifest_sha256"] != str(config["reference_manifest_sha256"]):
        raise ValueError("Program manifest does not match configuration")
    if str(config["chronoclone_git_commit"]) != str(environment["chronoclone_commit"]):
        raise ValueError("ChronoClone commit differs from the locked environment")

    paths = sorted(input_dir.glob("repeat_*__gblock_*.csv.gz"))
    if len(paths) != EXPECTED_CHUNKS:
        raise ValueError(f"Expected {EXPECTED_CHUNKS} chunks, found {len(paths)}")
    unexpected = sorted(
        path.name for path in input_dir.glob("*.csv.gz") if FILENAME_RE.match(path.name) is None
    )
    if unexpected:
        raise ValueError(f"Unexpected compressed chunks: {unexpected[:10]}")

    method_index = {method: index for index, method in enumerate(methods)}
    level_index = {level: index for index, level in enumerate(levels)}
    method_shape = (len(methods), EXPECTED_REPEATS, len(levels), len(levels))
    condition_shape = (EXPECTED_REPEATS, len(levels), len(levels))
    occupied = np.zeros(method_shape, dtype=bool)
    successful = np.zeros(method_shape, dtype=bool)
    failed = np.zeros(method_shape, dtype=bool)
    main = {
        name: np.full(method_shape, np.nan, dtype=np.float64)
        for name in MAIN_METRICS
    }
    shared = {
        name: np.full(method_shape, np.nan, dtype=np.float64)
        for name in SHARED_SUBSET_METRICS
    }
    condition = {
        name: np.full(condition_shape, np.nan, dtype=np.float64)
        for name in CONDITION_METRICS
    }
    postfilter = {
        name: np.full(condition_shape, np.nan, dtype=np.float64)
        for name in POSTFILTER_METRICS
    }
    postfilter_status: Counter = Counter()
    failures: Counter = Counter()
    global_values = {
        "schema_version": set(),
        "benchmark_name": set(),
        "chronoclone_commit": set(),
        "python_version": set(),
        **{name: set() for name in HASH_COLUMNS},
    }
    clean_signatures: set[tuple[str, ...]] = set()
    day0_signatures: set[tuple[str, str]] = set()
    gaussian_signatures: dict[tuple[int, int], tuple[str, int, int, int, str, int]] = {}
    sidecar_rows: list[dict] = []
    observed_chunk_ids: set[int] = set()
    observed_rows = 0

    combined = output_dir / "native_method_noise_runs.csv.gz"
    combined.parent.mkdir(parents=True, exist_ok=True)
    temporary = combined.with_name(f".{combined.name}.tmp.{os.getpid()}")
    raw_handle = temporary.open("wb")
    gzip_handle = gzip.GzipFile(
        filename="", fileobj=raw_handle, mode="wb", compresslevel=6, mtime=0
    )
    text_handle = io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="")
    wrote_header = False
    try:
        for path in paths:
            match = FILENAME_RE.match(path.name)
            if match is None:
                raise ValueError(f"Unexpected chunk filename: {path.name}")
            repeat, block = int(match.group(1)), int(match.group(2))
            chunk_id = repeat * GAUSSIAN_BLOCK_COUNT + block
            if not 0 <= repeat < EXPECTED_REPEATS or not 0 <= block < GAUSSIAN_BLOCK_COUNT:
                raise ValueError(f"Out-of-range chunk coordinates: {path.name}")
            if chunk_id in observed_chunk_ids:
                raise ValueError(f"Duplicate chunk id: {chunk_id}")
            observed_chunk_ids.add(chunk_id)
            gaussian_levels = gaussian_levels_for_block(levels, block)
            expected_chunk_rows = len(gaussian_levels) * len(levels) * len(methods)
            sidecar_rows.append(
                validate_sidecar(
                    path,
                    repeat=repeat,
                    block=block,
                    expected_rows=expected_chunk_rows,
                    hashes=hashes,
                    environment=environment,
                )
            )
            table = pd.read_csv(
                path,
                dtype={
                    "gaussian_seed": str,
                    "uniform_seed": str,
                    "postfit_retention_mask_hex": str,
                },
            )
            if list(table.columns) != list(REQUIRED_COLUMNS):
                raise ValueError(f"Column order/schema differs in {path.name}")
            validate_existing(
                table,
                repeat=repeat,
                block=block,
                gaussian_levels=gaussian_levels,
                uniform_levels=levels,
                hashes=hashes,
            )
            expected_keys = {
                (repeat, gaussian, uniform, method)
                for gaussian in gaussian_levels
                for uniform in levels
                for method in methods
            }
            observed_keys = set(
                table[list(KEY_COLUMNS)].itertuples(index=False, name=None)
            )
            if observed_keys != expected_keys or len(table) != expected_chunk_rows:
                raise ValueError(f"Terminal key coverage differs in {path.name}")
            if not table["status"].isin(TERMINAL_STATUSES).all():
                raise ValueError(f"Non-terminal rows remain in {path.name}")
            if not table["selection_track"].eq(
                "condition_specific_lineage_eligibility"
            ).all():
                raise ValueError(f"Wrong selection track in {path.name}")

            for _, group in table.groupby(
                ["repeat", "gaussian_pct", "uniform_pct"], sort=False, dropna=False
            ):
                validate_common_condition_group(group)
                row = group.loc[group["method"].eq("ChronoClone")].iloc[0]
                gaussian_pct, uniform_pct = int(row["gaussian_pct"]), int(row["uniform_pct"])
                ri, gi, ui = repeat, level_index[gaussian_pct], level_index[uniform_pct]
                validate_uniform_diagnostics(row)
                validate_postfilter_condition(group, program_ids=program_ids)
                if str(row["scenario"]) != scenario_name(gaussian_pct, uniform_pct):
                    raise ValueError("Scenario label disagrees with noise coordinates")
                if float(row["silhouette_cutoff"]) != -0.1:
                    raise ValueError("ChronoClone post-fit cutoff differs from -0.1")
                if int(row["day0_changed_entries"]) or int(row["day0_changed_clones"]):
                    raise ValueError("A production condition changed Day0")
                if str(row["day0_tensor_sha256"]) != str(row["clean_day0_tensor_sha256"]):
                    raise ValueError("A production condition has a changed Day0 hash")
                day0_signatures.add(
                    (str(row["day0_tensor_sha256"]), str(row["clean_day0_tensor_sha256"]))
                )
                for name in CONDITION_METRICS:
                    condition[name][ri, gi, ui] = float(row[name])
                for name in POSTFILTER_METRICS:
                    postfilter[name][ri, gi, ui] = float(row[name]) if pd.notna(row[name]) else np.nan
                postfilter_status[(gaussian_pct, uniform_pct, str(row["postfilter_status"]))] += 1

                gaussian_signature = (
                    str(row["gaussian_seed"]),
                    int(row["gaussian_changed_later_entries"]),
                    int(row["input_total_count"]),
                    int(row["n_eligible_clones"]),
                    str(row["eligible_clone_ids_sha256"]),
                    int(row["n_eligibility_mask_disagreements"]),
                )
                rg = (repeat, gaussian_pct)
                if rg in gaussian_signatures and gaussian_signatures[rg] != gaussian_signature:
                    raise ValueError(
                        "Uniform redistribution altered abundance/eligibility diagnostics "
                        f"at repeat={repeat}, gaussian={gaussian_pct}"
                    )
                gaussian_signatures[rg] = gaussian_signature
                if gaussian_pct == 0 and uniform_pct == 0:
                    clean_signatures.add(
                        (
                            str(row["input_tensor_sha256"]),
                            str(row["eligible_clone_ids_sha256"]),
                            str(row["n_input_clones"]),
                            str(row["n_eligible_clones"]),
                            str(row["input_total_count"]),
                        )
                    )

            for _, row in table.iterrows():
                gaussian_pct, uniform_pct = int(row["gaussian_pct"]), int(row["uniform_pct"])
                method = str(row["method"])
                index = (
                    method_index[method],
                    repeat,
                    level_index[gaussian_pct],
                    level_index[uniform_pct],
                )
                if occupied[index]:
                    raise ValueError(
                        f"Global duplicate key: {(repeat, gaussian_pct, uniform_pct, method)}"
                    )
                occupied[index] = True
                for name, values in global_values.items():
                    value = row[name]
                    if name == "schema_version":
                        value = canonical_schema_version(value)
                    values.add(str(value))
                if str(row["status"]) == "ok":
                    successful[index] = True
                    for name in MAIN_METRICS:
                        main[name][index] = float(row[name]) if pd.notna(row[name]) else np.nan
                    for name in SHARED_SUBSET_METRICS:
                        shared[name][index] = float(row[name]) if pd.notna(row[name]) else np.nan
                else:
                    failed[index] = True
                    failures[
                        (
                            method,
                            gaussian_pct,
                            uniform_pct,
                            str(row["error_type"]),
                            str(row["error_message"]),
                        )
                    ] += 1
            table = table.sort_values(list(KEY_COLUMNS), kind="mergesort")
            table.to_csv(
                text_handle,
                index=False,
                header=not wrote_header,
                float_format="%.12g",
                lineterminator="\n",
            )
            wrote_header = True
            observed_rows += len(table)
        text_handle.flush()
        text_handle.close()
        raw_handle.close()
        os.replace(temporary, combined)
    except Exception:
        try:
            text_handle.close()
        finally:
            raw_handle.close()
            temporary.unlink(missing_ok=True)
        raise

    if observed_chunk_ids != set(range(EXPECTED_CHUNKS)):
        raise ValueError("Chunk-id coverage is incomplete")
    if observed_rows != EXPECTED_ROWS or not occupied.all():
        raise ValueError(f"Expected {EXPECTED_ROWS} unique rows, found {observed_rows}")
    if np.any(successful & failed) or not np.all(successful | failed):
        raise AssertionError("Success/failure arrays do not partition all terminal rows")
    if len(day0_signatures) != 1:
        raise ValueError("Frozen Day0 differs across production conditions")
    if len(clean_signatures) != 1:
        raise ValueError("Deterministic clean input differs across repeat indices")
    if len(gaussian_signatures) != EXPECTED_REPEATS * len(levels):
        raise ValueError("Gaussian-stage invariant coverage is incomplete")

    expected_global = {
        "schema_version": {canonical_schema_version(config["schema_version"])},
        "benchmark_name": {str(config["benchmark_name"])},
        "chronoclone_commit": {str(environment["chronoclone_commit"])},
        "python_version": {str(environment["versions"]["python"])},
        **{name: {value} for name, value in hashes.items()},
    }
    for name, expected in expected_global.items():
        if global_values[name] != expected:
            raise ValueError(
                f"Rows do not share the current {name}: {sorted(global_values[name])[:5]}"
            )

    # The abundance-only Cyclone core must be invariant to state redistribution.
    cyclone_i = method_index["Cyclone abundance-only"]
    for repeat in range(EXPECTED_REPEATS):
        for gi, gaussian_pct in enumerate(levels):
            if not np.array_equal(
                successful[cyclone_i, repeat, gi, :],
                np.repeat(successful[cyclone_i, repeat, gi, 0], len(levels)),
            ):
                raise ValueError(
                    f"Cyclone status changes across uniform noise at repeat={repeat}, "
                    f"gaussian={gaussian_pct}"
                )
            for name in (
                "ari",
                "nmi",
                "hungarian_macro_f1",
                "recovered_k",
                "selection_parameter",
                "selected_silhouette",
            ):
                vector = main[name][cyclone_i, repeat, gi, :]
                if not np.allclose(vector, vector[0], atol=0, rtol=0, equal_nan=True):
                    raise ValueError(
                        f"Cyclone {name} changes across uniform noise at repeat={repeat}, "
                        f"gaussian={gaussian_pct}"
                    )

    for method in methods:
        mi = method_index[method]
        for name in ("ari", "nmi", "hungarian_macro_f1", "recovered_k"):
            baseline = main[name][mi, :, 0, 0]
            finite = baseline[np.isfinite(baseline)]
            if finite.size and not np.allclose(finite, finite[0], atol=0, rtol=0):
                raise ValueError(f"Deterministic clean baseline varies for {method}/{name}")

    summary_rows: list[dict] = []
    completeness_rows: list[dict] = []
    same_retained_rows: list[dict] = []
    for method in methods:
        mi = method_index[method]
        for gaussian_pct in levels:
            gi = level_index[gaussian_pct]
            for uniform_pct in levels:
                ui = level_index[uniform_pct]
                ok = successful[mi, :, gi, ui]
                method_failed = failed[mi, :, gi, ui]
                base = {
                    "method": method,
                    "selection_track": "condition_specific_lineage_eligibility",
                    "scenario": scenario_name(gaussian_pct, uniform_pct),
                    "gaussian_pct": gaussian_pct,
                    "uniform_pct": uniform_pct,
                    "n_expected": EXPECTED_REPEATS,
                    "n_success": int(ok.sum()),
                    "n_method_failure": int(method_failed.sum()),
                    "success_fraction": float(ok.mean()),
                    "method_failure_fraction": float(method_failed.mean()),
                }
                completeness_rows.append(dict(base))
                summary = dict(base)
                for name in MAIN_METRICS:
                    append_stats(summary, name, main[name][mi, :, gi, ui], quantiles)
                summary_rows.append(summary)

                sensitivity = dict(base)
                for name in SHARED_SUBSET_METRICS:
                    append_stats(
                        sensitivity,
                        name,
                        shared[name][mi, :, gi, ui],
                        quantiles,
                    )
                full_ari = main["ari"][mi, :, gi, ui]
                subset_ari = shared["shared_retained_subset_ari"][mi, :, gi, ui]
                append_stats(sensitivity, "full_cohort_ari", full_ari, quantiles)
                append_stats(
                    sensitivity,
                    "shared_retained_minus_full_ari",
                    subset_ari - full_ari,
                    quantiles,
                )
                same_retained_rows.append(sensitivity)

    eligibility_rows: list[dict] = []
    selective_rows: list[dict] = []
    for gaussian_pct in levels:
        gi = level_index[gaussian_pct]
        for uniform_pct in levels:
            ui = level_index[uniform_pct]
            base = {
                "scenario": scenario_name(gaussian_pct, uniform_pct),
                "gaussian_pct": gaussian_pct,
                "uniform_pct": uniform_pct,
                "n_expected": EXPECTED_REPEATS,
            }
            eligibility = dict(base)
            for name in CONDITION_METRICS:
                append_stats(eligibility, name, condition[name][:, gi, ui], quantiles)
            eligibility_rows.append(eligibility)

            selective = dict(base)
            statuses = {
                status: int(postfilter_status[(gaussian_pct, uniform_pct, status)])
                for status in ("ok", "all_abstained", "insufficient")
            }
            selective.update({f"n_postfilter_{key}": value for key, value in statuses.items()})
            selective["n_postfilter_other"] = EXPECTED_REPEATS - sum(statuses.values())
            for name in POSTFILTER_METRICS:
                append_stats(selective, name, postfilter[name][:, gi, ui], quantiles)
            selective_rows.append(selective)

    failure_rows = [
        {
            "method": key[0],
            "gaussian_pct": key[1],
            "uniform_pct": key[2],
            "error_type": key[3],
            "error_message": key[4],
            "count": count,
        }
        for key, count in sorted(failures.items())
    ]
    outputs = {
        "native_method_noise_summary.csv": pd.DataFrame(summary_rows),
        "result_completeness.csv": pd.DataFrame(completeness_rows),
        "eligibility_summary.csv": pd.DataFrame(eligibility_rows),
        "chronoclone_selective_summary.csv": pd.DataFrame(selective_rows),
        "same_retained_subset_summary.csv": pd.DataFrame(same_retained_rows),
        "method_failure_counts.csv": pd.DataFrame(
            failure_rows,
            columns=(
                "method",
                "gaussian_pct",
                "uniform_pct",
                "error_type",
                "error_message",
                "count",
            ),
        ),
        "chunk_manifest.csv": pd.DataFrame(sidecar_rows),
    }
    for name, table in outputs.items():
        atomic_csv(table, output_dir / name)

    artifact_names = ["native_method_noise_runs.csv.gz", *outputs.keys()]
    metadata = {
        "schema": "chronoclone_day0_locked_aggregation_v1",
        "status": "pass",
        "aggregation_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_chunks": EXPECTED_CHUNKS,
        "observed_chunks": len(paths),
        "expected_rows": EXPECTED_ROWS,
        "observed_rows": observed_rows,
        "summary_rows": len(summary_rows),
        "eligibility_summary_rows": len(eligibility_rows),
        "chronoclone_selective_summary_rows": len(selective_rows),
        "same_retained_subset_summary_rows": len(same_retained_rows),
        "method_failure_rows": int(failed.sum()),
        "quantiles": list(quantiles),
        "hash_contract": hashes,
        "day0_lock_validated": True,
        "five_method_common_cohort_validated": True,
        "uniform_total_and_eligibility_invariance_validated": True,
        "cyclone_uniform_invariance_validated": True,
        "deterministic_clean_baseline_validated": True,
        "silhouette_cutoff": -0.1,
        "postfilter_payload_validated": True,
        "postprocessor_sha256": postprocessor_hashes,
        "postprocessor_manifest_sha256": sha256_file(POSTPROCESSOR_MANIFEST),
        "artifact_sha256": {
            name: sha256_file(output_dir / name) for name in artifact_names
        },
    }
    atomic_json(metadata, output_dir / "aggregation_metadata.json")
    return metadata


def aggregate(args: argparse.Namespace) -> dict:
    """Build in isolation and publish metadata last as the commit marker."""

    destination = args.output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".day0_locked_aggregate_", dir=destination.parent))
    staged_args = argparse.Namespace(**vars(args))
    staged_args.output_dir = staging
    try:
        metadata = build_staged_aggregate(staged_args)
        expected = set(metadata["artifact_sha256"])
        staged_artifacts = {
            path.name for path in staging.iterdir() if path.name != "aggregation_metadata.json"
        }
        if staged_artifacts != expected:
            raise ValueError(
                f"Unexpected staged artifact set: {sorted(staged_artifacts ^ expected)}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for name in sorted(expected):
            os.replace(staging / name, destination / name)
        os.replace(
            staging / "aggregation_metadata.json",
            destination / "aggregation_metadata.json",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return metadata


def main() -> int:
    metadata = aggregate(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
