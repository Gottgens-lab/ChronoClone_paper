#!/usr/bin/env python3
"""Execute the truth-blind structural gate for the corrected noise pipeline."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_runtime import activate_source_imports, sha256_file  # noqa: E402

SOURCE = activate_source_imports()

import benchmark_core as legacy  # noqa: E402
import corrected_core as corrected  # noqa: E402


LEVELS = [0, 2, 4, 6, 8, 10, 12, 16, 20]
REPEATS = [0, 1, 2]
OUTPUT = ROOT / "results" / "preflight"


def main() -> int:
    config_path = ROOT / "config" / "benchmark_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reference = legacy.load_reference_programs(ROOT / "reference")
    clean_counts = corrected.rounded_clean_counts(reference.clean_means)
    clean_mask = corrected.clean_fixed_eligibility_mask(
        clean_counts,
        start_states=config["allowed_start_states"],
        min_persistence=int(config["min_persistence"]),
        min_total_count=int(config["min_total_count"]),
    )
    if int(clean_mask.sum()) != 517:
        raise AssertionError("Clean eligibility must retain all 517 clones")

    rows: list[dict] = []
    gaussian_seeds: set[int] = set()
    uniform_seeds: set[int] = set()
    for repeat in REPEATS:
        for gaussian_pct in LEVELS:
            gaussian, gaussian_seed = corrected.gaussian_stage_day0_locked(
                reference.clean_means,
                gaussian_pct,
                repeat,
                int(config["master_seed"]),
            )
            gaussian_seeds.add(int(gaussian_seed))
            gaussian_mask = corrected.lineage_eligibility_mask(
                gaussian,
                start_states=config["allowed_start_states"],
                min_persistence=int(config["min_persistence"]),
                min_total_count=int(config["min_total_count"]),
            )
            for uniform_pct in LEVELS:
                counts, uniform_seed = (
                    corrected.uniform_redistribution_stage_day0_locked(
                        gaussian,
                        gaussian_pct,
                        uniform_pct,
                        repeat,
                        int(config["master_seed"]),
                        start_states=config["allowed_start_states"],
                        min_persistence=int(config["min_persistence"]),
                        min_total_count=int(config["min_total_count"]),
                    )
                )
                uniform_seeds.add(int(uniform_seed))
                current_mask = corrected.lineage_eligibility_mask(
                    counts,
                    start_states=config["allowed_start_states"],
                    min_persistence=int(config["min_persistence"]),
                    min_total_count=int(config["min_total_count"]),
                )
                corrected.assert_day0_frozen(clean_counts, counts)
                if not np.array_equal(current_mask, gaussian_mask):
                    raise AssertionError("Uniform noise changed lineage eligibility")
                clone_time_before = gaussian.sum(axis=2, dtype=np.int64)
                clone_time_after = counts.sum(axis=2, dtype=np.int64)
                if not np.array_equal(clone_time_before, clone_time_after):
                    raise AssertionError("Uniform noise changed clone-time totals")
                rows.append(
                    {
                        "repeat": repeat,
                        "gaussian_pct": gaussian_pct,
                        "uniform_pct": uniform_pct,
                        "gaussian_seed": str(gaussian_seed),
                        "uniform_seed": str(uniform_seed),
                        "day0_changed_entries": int(
                            np.count_nonzero(counts[:, 0, :] != clean_counts[:, 0, :])
                        ),
                        "day0_forbidden_state_count": int(counts[:, 0, 2:].sum()),
                        "n_pipeline_eligible": int(current_mask.sum()),
                        "n_clean_fixed_eligible": int(clean_mask.sum()),
                        "uniform_changed_eligibility": int(
                            np.count_nonzero(current_mask != gaussian_mask)
                        ),
                        "uniform_changed_clone_time_totals": int(
                            np.count_nonzero(clone_time_before != clone_time_after)
                        ),
                    }
                )

    table = pd.DataFrame(rows)
    expected_rows = len(REPEATS) * len(LEVELS) ** 2
    if len(table) != expected_rows:
        raise AssertionError("Structural grid is incomplete")
    if table.duplicated(["repeat", "gaussian_pct", "uniform_pct"]).any():
        raise AssertionError("Structural grid contains duplicate coordinates")
    if int(table["day0_changed_entries"].sum()) != 0:
        raise AssertionError("Day0 changed in structural preflight")
    if int(table["day0_forbidden_state_count"].sum()) != 0:
        raise AssertionError("Forbidden Day0 states appeared")
    if int(table["uniform_changed_eligibility"].sum()) != 0:
        raise AssertionError("Uniform eligibility invariance failed")
    if int(table["uniform_changed_clone_time_totals"].sum()) != 0:
        raise AssertionError("Uniform total preservation failed")
    if len(gaussian_seeds) != len(REPEATS) * len(LEVELS):
        raise AssertionError("Gaussian seed collision in sentinel design")
    if len(uniform_seeds) != expected_rows:
        raise AssertionError("Uniform seed collision in sentinel design")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "structural_conditions.csv"
    table.to_csv(csv_path, index=False)
    report = {
        "status": "pass",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_name": "chronoclone",
        "environment_path": "/path/to/ChronoClone/.runtime/chronoclone",
        "kernel": "chronoclone",
        "python_version": platform.python_version(),
        "execution_command": (
            "/path/to/ChronoClone/.runtime/chronoclone/bin/python "
            "scripts/run_structural_preflight.py"
        ),
        "sentinel_repeats": REPEATS,
        "sentinel_levels": LEVELS,
        "n_conditions": int(len(table)),
        "clean_eligible_clones": int(clean_mask.sum()),
        "day0_changed_entries_total": int(table["day0_changed_entries"].sum()),
        "day0_forbidden_state_count_total": int(
            table["day0_forbidden_state_count"].sum()
        ),
        "uniform_eligibility_changes_total": int(
            table["uniform_changed_eligibility"].sum()
        ),
        "uniform_clone_time_total_changes_total": int(
            table["uniform_changed_clone_time_totals"].sum()
        ),
        "config_sha256": sha256_file(config_path),
        "corrected_core_sha256": sha256_file(ROOT / "corrected_core.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "structural_conditions_sha256": sha256_file(csv_path),
        **SOURCE,
    }
    report_path = OUTPUT / "structural_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
