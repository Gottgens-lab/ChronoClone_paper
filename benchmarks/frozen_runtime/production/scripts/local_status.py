#!/usr/bin/env python3
"""Print live JSON progress for the corrected eight-worker local benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHUNKS = RESULTS / "chunks"
STATE_PATH = RESULTS / "controller_state.json"
PID_PATH = RESULTS / "controller.pid"
EXPECTED_TASKS = 600
EXPECTED_ROWS = 1_300_500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError):
        return False
    return True


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def valid_local_marker(marker: Path) -> tuple[bool, int, str | None]:
    payload = read_json(marker)
    chunk_path = payload.get("chunk_path")
    if not isinstance(chunk_path, str):
        return False, 0, None
    chunk = Path(chunk_path)
    if not chunk.is_file():
        return False, 0, None
    valid = (
        payload.get("schema") == "chronoclone_day0_locked_chunk_completion_v1"
        and payload.get("status") == "complete"
        and chunk.parent.resolve() == CHUNKS.resolve()
        and marker.name
        == chunk.name.removesuffix(".csv.gz") + ".complete.json"
        and int(payload.get("row_count", -1))
        == int(payload.get("terminal_row_count", -2))
        and payload.get("chunk_sha256") == sha256_file(chunk)
    )
    rows = int(payload.get("terminal_row_count", 0)) if valid else 0
    return valid, rows, chunk.name if valid else None


def materialized_rows(path: Path) -> int:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except (EOFError, OSError, UnicodeError):
        return 0


def status_snapshot() -> dict[str, Any]:
    state = read_json(STATE_PATH)
    pid = None
    if PID_PATH.is_file():
        try:
            pid = int(PID_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None

    chunks = sorted(CHUNKS.glob("repeat_*__gblock_*.csv.gz"))
    markers = sorted(CHUNKS.glob("repeat_*__gblock_*.complete.json"))
    verified_names: set[str] = set()
    terminal_rows = 0
    for marker in markers:
        valid, rows, chunk_name = valid_local_marker(marker)
        if valid:
            assert chunk_name is not None
            verified_names.add(chunk_name)
            terminal_rows += rows
    unverified = [path for path in chunks if path.name not in verified_names]
    checkpoint_rows = sum(materialized_rows(path) for path in unverified)
    rows_materialized = terminal_rows + checkpoint_rows

    complete_tasks = len(verified_names)
    controller_status = str(state.get("status", "not_started"))
    if not process_alive(pid) and controller_status in {
        "starting",
        "scanning",
        "running",
        "post_run_audit",
        "aggregating",
        "building_notebooks",
    }:
        controller_status = "controller_not_alive_state_stale"
    return {
        "schema": "chronoclone_day0_locked_local_status_v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller_status": controller_status,
        "controller_pid": pid,
        "controller_alive": process_alive(pid),
        "workers": state.get("workers", 8),
        "threads_per_worker": state.get("threads_per_worker", 1),
        "active_tasks": state.get("active_tasks", []),
        "failed_task_ids": [
            row.get("task_id") for row in state.get("failed_tasks", [])
        ],
        "complete_tasks": complete_tasks,
        "expected_tasks": EXPECTED_TASKS,
        "progress_percent": round(100.0 * complete_tasks / EXPECTED_TASKS, 3),
        "chunk_files_present": len(chunks),
        "validated_completion_sidecars": len(verified_names),
        "unverified_or_partial_chunk_files": len(unverified),
        "terminal_rows_from_sidecars": terminal_rows,
        "checkpoint_rows_in_unverified_chunks": checkpoint_rows,
        "rows_materialized": rows_materialized,
        "expected_rows": EXPECTED_ROWS,
        "row_progress_percent": round(100.0 * rows_materialized / EXPECTED_ROWS, 3),
        "mean_task_seconds_this_run": state.get("mean_task_seconds_this_run"),
        "estimated_remaining_seconds": state.get("estimated_remaining_seconds"),
        "updated_at_utc": state.get("updated_at_utc"),
        "controller_log": str(ROOT / "logs" / "controller.out"),
        "state_json": str(STATE_PATH),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Refresh continuously; zero prints one JSON snapshot.",
    )
    args = parser.parse_args()
    if args.watch < 0:
        raise ValueError("--watch must be non-negative")
    while True:
        print(json.dumps(status_snapshot(), indent=2, sort_keys=True), flush=True)
        if args.watch == 0:
            return 0
        time.sleep(max(1.0, args.watch))


if __name__ == "__main__":
    raise SystemExit(main())
