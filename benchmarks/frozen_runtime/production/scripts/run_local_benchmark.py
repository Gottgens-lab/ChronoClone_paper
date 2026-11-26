#!/usr/bin/env python3
"""Run the 600 corrected benchmark shards with exactly eight local workers.

The scientific runner owns row-level validation and checkpoint resumption.  This
controller adds process-level durability: an exclusive controller lock, bounded
concurrency, one-thread numerical children, retry bookkeeping, terminal shard
sidecars, atomic status JSON, and optional post-processing after all 600 shards
have passed a structural terminal check.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
CHUNKS = RESULTS / "chunks"
LOCKS = RESULTS / "locks"
LOGS = ROOT / "logs"
CACHE = ROOT / "cache"
CONFIG = ROOT / "config" / "benchmark_config.json"
RUNNER = SCRIPTS / "run_chunk.py"
SOURCE_ROOT = ROOT.parent / "native_method_noise_benchmark_local_20260803"
CORRECTED_CORE = ROOT / "corrected_core.py"
RESULT_SCHEMA = SCRIPTS / "result_schema.py"
SOURCE_RUNTIME = SCRIPTS / "source_runtime.py"
CYCLONE_ADAPTER = SCRIPTS / "cyclone_adapter.py"
SOURCE_CORE = SOURCE_ROOT / "benchmark_package" / "scripts" / "benchmark_core.py"
SOURCE_ENVIRONMENT_MANIFEST = SOURCE_ROOT / "results" / "provenance" / "environment_manifest.json"
REFERENCE_PROFILE = ROOT / "reference" / "deterministic_profiles.csv"
REFERENCE_MANIFEST = ROOT / "reference" / "program_manifest.csv"
PYTHON = Path("/path/to/ChronoClone/.runtime/chronoclone/bin/python")
SOURCE_SITE_PACKAGES = SOURCE_ROOT / "runtime" / "site-packages"
SOURCE_SCRIPTS = SOURCE_ROOT / "benchmark_package" / "scripts"

STATE_PATH = RESULTS / "controller_state.json"
PID_PATH = RESULTS / "controller.pid"
LOCK_PATH = RESULTS / "controller.lock"
EXPECTED_TASKS = 600
GAUSSIAN_BLOCK_COUNT = 6
MAX_WORKERS = 8
METHODS_PER_CONDITION = 5
TERMINAL_STATUSES = {"ok", "method_failure"}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
STOP_REQUESTED = threading.Event()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_coordinates(task_id: int) -> tuple[int, int]:
    if not 0 <= task_id < EXPECTED_TASKS:
        raise ValueError(f"task_id must be in 0..{EXPECTED_TASKS - 1}")
    return task_id // GAUSSIAN_BLOCK_COUNT, task_id % GAUSSIAN_BLOCK_COUNT


def chunk_path(task_id: int) -> Path:
    repeat, block = task_coordinates(task_id)
    return CHUNKS / f"repeat_{repeat:03d}__gblock_{block:02d}.csv.gz"


def sidecar_path(task_id: int) -> Path:
    path = chunk_path(task_id)
    return path.with_name(path.name.removesuffix(".csv.gz") + ".complete.json")


def expected_gaussian_levels(config: dict[str, Any], block: int) -> list[int]:
    levels = [int(value) for value in config["noise_levels_pct"]]
    if len(levels) != 51:
        raise ValueError("Production configuration must contain 51 noise levels")
    start = block * 9
    stop = min(start + 9, len(levels))
    selected = levels[start:stop]
    expected = 9 if block < 5 else 6
    if len(selected) != expected:
        raise AssertionError(f"Gaussian block {block} has an unexpected size")
    return selected


def expected_rows(config: dict[str, Any], block: int) -> int:
    return (
        len(expected_gaussian_levels(config, block))
        * len(config["noise_levels_pct"])
        * METHODS_PER_CONDITION
    )


def matching_sidecar(
    task_id: int, config_sha256: str, config: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a trusted runner completion marker when its chunk hash matches."""

    chunk = chunk_path(task_id)
    marker = sidecar_path(task_id)
    if not chunk.is_file() or not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        environment_manifest = json.loads(
            SOURCE_ENVIRONMENT_MANIFEST.read_text(encoding="utf-8")
        )
        repeat, block = task_coordinates(task_id)
        row_count = expected_rows(config, block)
        required = {
            "schema": "chronoclone_day0_locked_chunk_completion_v1",
            "status": "complete",
            "task_id": task_id,
            "repeat": repeat,
            "gaussian_block": block,
            "row_count": row_count,
            "terminal_row_count": row_count,
            "config_sha256": config_sha256,
            "runner_sha256": sha256_file(RUNNER),
            "corrected_core_sha256": sha256_file(CORRECTED_CORE),
            "result_schema_sha256": sha256_file(RESULT_SCHEMA),
            "source_runtime_sha256": sha256_file(SOURCE_RUNTIME),
            "cyclone_adapter_sha256": sha256_file(CYCLONE_ADAPTER),
            "source_core_sha256": sha256_file(SOURCE_CORE),
            "environment_manifest_sha256": sha256_file(SOURCE_ENVIRONMENT_MANIFEST),
            "profile_sha256": sha256_file(REFERENCE_PROFILE),
            "manifest_sha256": sha256_file(REFERENCE_MANIFEST),
            "runtime_versions_sha256": str(environment_manifest["runtime_versions_sha256"]),
            "conda_records_sha256": str(environment_manifest["conda_records_sha256"]),
            "chronoclone_package_tree_sha256": str(
                environment_manifest["chronoclone_package_tree_sha256"]
            ),
            "chronoclone_commit": str(environment_manifest["chronoclone_commit"]),
            "python_version": "3.11.15",
        }
        recorded_path = Path(str(payload.get("chunk_path", "")))
        if (
            all(payload.get(key) == value for key, value in required.items())
            and recorded_path.resolve() == chunk.resolve()
            and payload.get("chunk_sha256") == sha256_file(chunk)
        ):
            return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def validate_terminal_chunk(
    task_id: int, config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    """Perform a strict, truth-independent terminal structure check."""

    repeat, block = task_coordinates(task_id)
    path = chunk_path(task_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    gaussian_levels = expected_gaussian_levels(config, block)
    uniform_levels = [int(value) for value in config["noise_levels_pct"]]
    method_order = [str(value) for value in config["method_order"]]
    expected_keys = {
        (repeat, gaussian, uniform, method)
        for gaussian in gaussian_levels
        for uniform in uniform_levels
        for method in method_order
    }
    observed_keys: set[tuple[int, int, int, str]] = set()
    status_counts = {status: 0 for status in sorted(TERMINAL_STATUSES)}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "repeat",
            "gaussian_block",
            "gaussian_pct",
            "uniform_pct",
            "method",
            "status",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path.name} lacks required columns: "
                f"{sorted(required_columns - set(reader.fieldnames or []))}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                row_repeat = int(row["repeat"])
                row_block = int(row["gaussian_block"])
                gaussian = int(row["gaussian_pct"])
                uniform = int(row["uniform_pct"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid integer coordinate at row {row_number}") from exc
            if row_repeat != repeat or row_block != block:
                raise ValueError(f"Foreign task coordinate at row {row_number}")
            status = str(row["status"])
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"Non-terminal status {status!r} at row {row_number}")
            key = (row_repeat, gaussian, uniform, str(row["method"]))
            if key in observed_keys:
                raise ValueError(f"Duplicate condition/method key at row {row_number}: {key}")
            observed_keys.add(key)
            status_counts[status] += 1
    if observed_keys != expected_keys:
        missing = len(expected_keys - observed_keys)
        foreign = len(observed_keys - expected_keys)
        raise ValueError(
            f"{path.name} is not terminal: missing_keys={missing}, foreign_keys={foreign}"
        )
    if len(observed_keys) != expected_rows(config, block):
        raise AssertionError("Validated key count disagrees with expected row count")
    stat = path.stat()
    payload = {
        "schema": "chronoclone_day0_locked_chunk_completion_v1",
        "status": "complete",
        "task_id": task_id,
        "repeat": repeat,
        "gaussian_block": block,
        "chunk_name": path.name,
        "chunk_size_bytes": stat.st_size,
        "chunk_mtime_ns": stat.st_mtime_ns,
        "chunk_sha256": sha256_file(path),
        "config_sha256": config_sha256,
        "terminal_rows": len(observed_keys),
        "status_counts": status_counts,
        "validated_at_utc": utc_now(),
    }
    return payload


def scan_pending_tasks(
    config: dict[str, Any], config_sha256: str
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Find incomplete tasks, validating unmarked complete-looking shards once."""

    complete: list[int] = []
    pending: list[int] = []
    scan_errors: list[dict[str, Any]] = []
    for task_id in range(EXPECTED_TASKS):
        if matching_sidecar(task_id, config_sha256, config) is not None:
            complete.append(task_id)
            continue
        pending.append(task_id)
        if chunk_path(task_id).exists():
            scan_errors.append(
                {
                    "task_id": task_id,
                    "error": "missing_or_invalid_runner_completion_sidecar",
                }
            )
    return complete, pending, scan_errors


def runtime_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [ROOT, SCRIPTS, SOURCE_SITE_PACKAGES, SOURCE_SCRIPTS]
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "NUMBA_CACHE_DIR": str(CACHE / "numba"),
            "MPLCONFIGDIR": str(CACHE / "matplotlib"),
            "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
        }
    )
    for variable in THREAD_VARIABLES:
        environment[variable] = "1"
    return environment


def run_task(
    task_id: int,
    attempt: int,
    environment: dict[str, str],
    config: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    log_path = LOGS / f"task_{task_id:03d}.log"
    command = [
        str(PYTHON),
        str(RUNNER),
        "--task-id",
        str(task_id),
        "--output-dir",
        str(CHUNKS),
    ]
    started_monotonic = time.monotonic()
    started_utc = utc_now()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {started_utc} attempt={attempt} command={command!r} ===\n")
        handle.flush()
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
        validation_error = None
        if process.returncode == 0:
            try:
                terminal_validation = validate_terminal_chunk(
                    task_id, config, config_sha256
                )
                marker = matching_sidecar(task_id, config_sha256, config)
                if marker is None:
                    raise ValueError(
                        "Runner exited zero without a hash-matched terminal sidecar"
                    )
            except Exception as exc:  # written to the task log and retried by controller
                validation_error = f"{type(exc).__name__}: {exc}"
                handle.write(f"POST-RUN TERMINAL VALIDATION FAILED: {validation_error}\n")
                handle.flush()
            else:
                returncode = 0
        if process.returncode != 0:
            returncode = process.returncode
            marker = None
        elif validation_error is not None:
            returncode = 90
            marker = None
    return {
        "task_id": task_id,
        "attempt": attempt,
        "returncode": returncode,
        "started_at_utc": started_utc,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
        "terminal_rows": (
            terminal_validation.get("terminal_rows")
            if marker and terminal_validation
            else None
        ),
        "validation_error": validation_error,
        "log": str(log_path),
    }


def run_logged(command: list[str], environment: dict[str, str], log_name: str) -> None:
    log_path = LOGS / log_name
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {utc_now()} command={command!r} ===\n")
        handle.flush()
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Post-processing command failed ({process.returncode}); see {log_path}"
        )


def update_progress(state: dict[str, Any]) -> None:
    complete = int(state.get("complete_tasks", 0))
    state["updated_at_utc"] = utc_now()
    state["progress_percent"] = round(100.0 * complete / EXPECTED_TASKS, 3)
    durations = [
        float(row["elapsed_seconds"])
        for row in state.get("completed_this_run", [])
        if float(row.get("elapsed_seconds", 0.0)) > 0
    ]
    if durations and state.get("status") == "running":
        mean_seconds = sum(durations) / len(durations)
        remaining = max(0, EXPECTED_TASKS - complete)
        state["mean_task_seconds_this_run"] = round(mean_seconds, 3)
        state["estimated_remaining_seconds"] = round(
            remaining * mean_seconds / MAX_WORKERS, 1
        )
    atomic_json(state, STATE_PATH)


def handle_signal(signum: int, _frame: Any) -> None:
    STOP_REQUESTED.set()
    print(
        f"Received signal {signum}; active shards may checkpoint, and no new shards "
        "will be submitted.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Stop after all 600 shards pass terminal validation.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen setup and pilot evidence, then exit before production.",
    )
    args = parser.parse_args()
    if args.workers != MAX_WORKERS:
        raise ValueError("This production controller is fixed at exactly 8 workers")
    if not 0 <= args.max_retries <= 3:
        raise ValueError("--max-retries must be between 0 and 3")
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Required Python is missing: {PYTHON}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"Benchmark runner is missing: {RUNNER}")
    if not CONFIG.is_file():
        raise FileNotFoundError(f"Benchmark configuration is missing: {CONFIG}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if int(config["repeats"]) != 100:
        raise ValueError("Production controller requires exactly 100 repeats")
    if len(config["method_order"]) != METHODS_PER_CONDITION:
        raise ValueError("Production controller requires exactly five methods")

    structural_report_path = RESULTS / "preflight" / "structural_report.json"
    setup_notebook_path = ROOT / "notebooks" / "0001_day0_locked_setup_and_preflight.ipynb"
    behavioral_notebook_path = ROOT / "notebooks" / "0002_day0_locked_behavioral_pilot.ipynb"
    behavioral_report_path = RESULTS / "preflight" / "behavioral_report.json"
    behavioral_dir = RESULTS / "preflight" / "behavioral_chunks"
    if not structural_report_path.is_file() or not setup_notebook_path.is_file():
        raise RuntimeError("Executed structural preflight evidence is missing")
    structural_report = json.loads(
        structural_report_path.read_text(encoding="utf-8")
    )
    if structural_report.get("status") != "pass":
        raise RuntimeError("Structural preflight did not pass")
    if structural_report.get("config_sha256") != sha256_file(CONFIG):
        raise RuntimeError("Structural preflight is stale relative to the config")
    if structural_report.get("corrected_core_sha256") != sha256_file(
        ROOT / "corrected_core.py"
    ):
        raise RuntimeError("Structural preflight is stale relative to corrected_core.py")
    notebook_payload = json.loads(setup_notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook_payload.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    if notebook_payload.get("metadata", {}).get("execution", {}).get("status") != "ok":
        raise RuntimeError("Setup/preflight notebook is not marked executed")
    if [cell.get("execution_count") for cell in code_cells] != list(
        range(1, len(code_cells) + 1)
    ):
        raise RuntimeError("Setup/preflight notebook was not sequentially executed")
    if any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    ):
        raise RuntimeError("Setup/preflight notebook contains an error output")
    expected_runner_sha = sha256_file(RUNNER)
    behavioral_paths = sorted(behavioral_dir.glob("repeat_*__gblock_00.csv.gz"))
    if len(behavioral_paths) != 3:
        raise RuntimeError("Three behavioral pilot shards are required")
    for path in behavioral_paths:
        repeat = int(path.name.split("__", 1)[0].split("_")[-1])
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 360:
            raise RuntimeError(f"Behavioral pilot has wrong row count: {path}")
        if {row.get("runner_sha256") for row in rows} != {expected_runner_sha}:
            raise RuntimeError(f"Behavioral pilot is stale relative to runner: {path}")
        pilot_hashes = {
            "config_sha256": sha256_file(CONFIG),
            "corrected_core_sha256": sha256_file(CORRECTED_CORE),
            "result_schema_sha256": sha256_file(RESULT_SCHEMA),
            "source_runtime_sha256": sha256_file(SOURCE_RUNTIME),
            "cyclone_adapter_sha256": sha256_file(CYCLONE_ADAPTER),
            "source_core_sha256": sha256_file(SOURCE_CORE),
            "environment_manifest_sha256": sha256_file(SOURCE_ENVIRONMENT_MANIFEST),
            "profile_sha256": sha256_file(REFERENCE_PROFILE),
            "manifest_sha256": sha256_file(REFERENCE_MANIFEST),
        }
        for name, expected_value in pilot_hashes.items():
            if {row.get(name) for row in rows} != {expected_value}:
                raise RuntimeError(f"Behavioral pilot has stale {name}: {path}")
        if any(int(row.get("day0_changed_entries", -1)) != 0 for row in rows):
            raise RuntimeError(f"Behavioral pilot changed Day0: {path}")
        expected_pilot_keys = {
            (repeat, gaussian, uniform, method)
            for gaussian in (0, 2, 4, 6, 8, 10, 12, 16)
            for uniform in (0, 2, 4, 6, 8, 10, 12, 16, 20)
            for method in config["method_order"]
        }
        observed_pilot_keys = {
            (
                int(row["repeat"]),
                int(row["gaussian_pct"]),
                int(row["uniform_pct"]),
                row["method"],
            )
            for row in rows
        }
        if observed_pilot_keys != expected_pilot_keys:
            raise RuntimeError(f"Behavioral pilot key set differs: {path}")
        if any(row.get("status") not in TERMINAL_STATUSES for row in rows):
            raise RuntimeError(f"Behavioral pilot contains nonterminal rows: {path}")
        sidecar_path = path.with_name(
            path.name.removesuffix(".csv.gz") + ".complete.json"
        )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("chunk_sha256") != sha256_file(path):
            raise RuntimeError(f"Behavioral pilot sidecar hash differs: {path}")
        for name, expected_value in {**pilot_hashes, "runner_sha256": expected_runner_sha}.items():
            if sidecar.get(name) != expected_value:
                raise RuntimeError(f"Behavioral sidecar has stale {name}: {path}")
        chrono = [row for row in rows if row.get("method") == "ChronoClone"]
        if len(chrono) != 72 or any(row.get("status") != "ok" for row in chrono):
            raise RuntimeError(f"ChronoClone behavioral pilot is incomplete: {path}")

    if not behavioral_notebook_path.is_file() or not behavioral_report_path.is_file():
        raise RuntimeError("Executed behavioral-pilot evidence is missing")
    behavioral_notebook = json.loads(
        behavioral_notebook_path.read_text(encoding="utf-8")
    )
    behavioral_code = [
        cell for cell in behavioral_notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    if behavioral_notebook.get("metadata", {}).get("execution", {}).get("status") != "ok":
        raise RuntimeError("Behavioral-pilot notebook is not marked executed")
    if [cell.get("execution_count") for cell in behavioral_code] != list(
        range(1, len(behavioral_code) + 1)
    ):
        raise RuntimeError("Behavioral-pilot notebook was not sequentially executed")
    if any(
        output.get("output_type") == "error"
        for cell in behavioral_code
        for output in cell.get("outputs", [])
    ):
        raise RuntimeError("Behavioral-pilot notebook contains an error output")
    behavioral_report = json.loads(
        behavioral_report_path.read_text(encoding="utf-8")
    )
    expected_report = {
        "status": "pass",
        "behavioral_rows": 1080,
        "chronoclone_rows": 216,
        "day0_changed_entries_total": 0,
        "runner_sha256": expected_runner_sha,
        "config_sha256": sha256_file(CONFIG),
        "result_schema_sha256": sha256_file(RESULT_SCHEMA),
        "cyclone_adapter_sha256": sha256_file(CYCLONE_ADAPTER),
    }
    if any(behavioral_report.get(name) != value for name, value in expected_report.items()):
        raise RuntimeError("Behavioral-pilot report is stale or failed")

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "structural_conditions": 243,
                    "behavioral_shards": len(behavioral_paths),
                    "behavioral_rows": 1080,
                    "runner_sha256": expected_runner_sha,
                    "config_sha256": sha256_file(CONFIG),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for path in (CHUNKS, LOCKS, LOGS, CACHE / "numba", CACHE / "matplotlib"):
        path.mkdir(parents=True, exist_ok=True)
    config_sha256 = sha256_file(CONFIG)
    environment = runtime_environment()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    lock_handle = LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError("Another corrected benchmark controller is already active") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started={utc_now()}\n")
    lock_handle.flush()
    PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")

    state: dict[str, Any] = {
        "schema": "chronoclone_day0_locked_local_controller_v1",
        "status": "scanning",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "controller_pid": os.getpid(),
        "workers": MAX_WORKERS,
        "threads_per_worker": 1,
        "expected_tasks": EXPECTED_TASKS,
        "expected_rows": 1_300_500,
        "config_sha256": config_sha256,
        "task_order": "gaussian_block_major_then_repeat",
        "active_tasks": [],
        "completed_this_run": [],
        "failed_tasks": [],
        "scan_warnings": [],
    }
    atomic_json(state, STATE_PATH)
    try:
        complete, pending, scan_errors = scan_pending_tasks(config, config_sha256)
        state["initial_complete_tasks"] = len(complete)
        state["initial_pending_tasks"] = len(pending)
        state["complete_tasks"] = len(complete)
        state["pending_tasks"] = len(pending)
        state["scan_warnings"] = scan_errors
        state["status"] = "running" if pending else "post_run_audit"
        update_progress(state)

        # Prioritize one shard from every Gaussian block for repeat 0, then use
        # block-major order. This exposes both one-dimensional axes early while
        # retaining the fixed task mapping and full-grid estimand.
        first_repeat = [task for task in range(6) if task in pending]
        first_repeat_set = set(first_repeat)
        remaining = first_repeat + sorted(
            (task for task in pending if task not in first_repeat_set),
            key=lambda task: (task % 6, task // 6),
        )
        attempts = {task_id: 0 for task_id in remaining}
        exhausted_failures: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures: dict[Future[dict[str, Any]], int] = {}

            def submit(task_id: int) -> None:
                attempts[task_id] += 1
                future = executor.submit(
                    run_task,
                    task_id,
                    attempts[task_id],
                    environment,
                    config,
                    config_sha256,
                )
                futures[future] = task_id

            while (remaining or futures) and not exhausted_failures:
                while (
                    remaining
                    and len(futures) < MAX_WORKERS
                    and not STOP_REQUESTED.is_set()
                ):
                    submit(remaining.pop(0))
                state["active_tasks"] = sorted(futures.values())
                state["pending_tasks"] = len(remaining) + len(futures)
                update_progress(state)
                if not futures:
                    break
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    task_id = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "task_id": task_id,
                            "attempt": attempts[task_id],
                            "returncode": 99,
                            "elapsed_seconds": 0.0,
                            "exception": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    if int(result["returncode"]) == 0:
                        state["completed_this_run"].append(result)
                        state["complete_tasks"] = int(state["complete_tasks"]) + 1
                    elif attempts[task_id] <= args.max_retries:
                        # Keep the same block-major priority on retry.
                        remaining.insert(0, task_id)
                    else:
                        exhausted_failures.append(result)
                        state["failed_tasks"].append(result)
                state["active_tasks"] = sorted(futures.values())
                state["pending_tasks"] = len(remaining) + len(futures)
                update_progress(state)

        if STOP_REQUESTED.is_set():
            state["status"] = "stopped_safely"
            state["active_tasks"] = []
            state["stop_note"] = (
                "No new tasks were submitted after the signal; restart the launcher "
                "to validate and resume checkpoints."
            )
            update_progress(state)
            return 130
        if exhausted_failures:
            state["status"] = "failed"
            state["active_tasks"] = []
            state["controller_error"] = (
                "One or more shards exhausted retries; inspect failed_tasks and logs."
            )
            update_progress(state)
            return 1

        state["status"] = "post_run_audit"
        state["active_tasks"] = []
        update_progress(state)
        complete_after, pending_after, scan_errors_after = scan_pending_tasks(
            config, config_sha256
        )
        state["complete_tasks"] = len(complete_after)
        state["pending_tasks"] = len(pending_after)
        state["post_run_scan_warnings"] = scan_errors_after
        update_progress(state)
        if pending_after:
            raise RuntimeError(
                f"Post-run audit found {len(pending_after)} non-terminal tasks: "
                f"{pending_after[:20]}"
            )

        if args.skip_postprocess:
            state["status"] = "chunks_complete"
        else:
            aggregate = SCRIPTS / "aggregate_results.py"
            build_notebooks = SCRIPTS / "build_notebooks.py"
            postprocessors: list[str] = []
            if aggregate.is_file():
                state["status"] = "aggregating"
                update_progress(state)
                run_logged([str(PYTHON), str(aggregate)], environment, "aggregate.log")
                postprocessors.append(aggregate.name)
            if build_notebooks.is_file():
                state["status"] = "building_notebooks"
                update_progress(state)
                run_logged(
                    [str(PYTHON), str(build_notebooks)],
                    environment,
                    "build_notebooks.log",
                )
                postprocessors.append(build_notebooks.name)
            state["postprocessors_run"] = postprocessors
            state["status"] = "complete" if postprocessors else "chunks_complete"
        state["completed_at_utc"] = utc_now()
        state["estimated_remaining_seconds"] = 0.0
        update_progress(state)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["active_tasks"] = []
        state["controller_error"] = f"{type(exc).__name__}: {exc}"
        state["controller_traceback"] = traceback.format_exc()
        update_progress(state)
        raise
    finally:
        try:
            if PID_PATH.is_file() and PID_PATH.read_text(encoding="utf-8").strip() == str(
                os.getpid()
            ):
                PID_PATH.unlink()
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
