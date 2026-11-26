"""One condition of the four manuscript estimators from frozen source.

This adapter relocates exact files into temporary sibling directories expected
by the historical corrected core. It neither edits that source nor calls its
path-bound production controller. Its output is a separate smoke-run record;
it cannot resume, replace, or append to the historical benchmark shards.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

BENCHMARKS = Path(__file__).resolve().parents[2]
MODULE = BENCHMARKS / "ode_noise"
FROZEN = BENCHMARKS / "frozen_runtime"
METHODS = ["ChronoClone", "CLADES preprocessing", "K-means", "Ward"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian", type=int, default=0, choices=range(0, 101, 2))
    parser.add_argument("--uniform", type=int, default=0, choices=range(0, 101, 2))
    parser.add_argument("--repeat", type=int, default=0, choices=range(100))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(MODULE / "results/aggregate"):
        raise ValueError("Smoke output must not replace historical aggregate evidence")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    aggregate = json.loads((MODULE / "results/aggregate/aggregation_metadata.json").read_text())
    hashes = aggregate["hash_contract"]
    production = FROZEN / "production"
    native = FROZEN / "native_20260803"
    numerical_runtime = FROZEN / "normalization_runtime"
    sources = {
        "corrected_core": (production / "corrected_core.py", hashes["corrected_core_sha256"]),
        "low_level_core": (native / "benchmark_package/scripts/benchmark_core.py", hashes["source_core_sha256"]),
        "configuration": (production / "config/benchmark_config.json", hashes["config_sha256"]),
        "reference_profiles": (production / "reference/deterministic_profiles.csv", hashes["profile_sha256"]),
        "reference_manifest": (production / "reference/program_manifest.csv", hashes["manifest_sha256"]),
        "normalization": (numerical_runtime / "chronoclone/normalization.py", "ea9626c444f3527505bd1d594e0fff2896eb34609769de108b86748f962ce9cb"),
        "runtime_initializer": (numerical_runtime / "chronoclone/__init__.py", "1199332fe6c10e81af55eec91d583ae6da245e39bab44d86cc4112705f8c7480"),
    }
    source_records = {}
    for name, (path, expected) in sources.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(f"Frozen source mismatch: {path}")
        source_records[name] = {"path": str(path.relative_to(BENCHMARKS.parent)), "sha256": observed}
    historical = json.loads((native / "results/provenance/environment_manifest.json").read_text())
    versions = {name: (platform.python_version() if name == "python" else importlib.metadata.version(name))
                for name in historical["versions"]}
    mismatches = {name: {"historical": expected, "current": versions[name]}
                  for name, expected in historical["versions"].items() if expected != versions[name]}
    if mismatches:
        raise RuntimeError(f"Use the recorded benchmark dependency versions: {mismatches}")
    if "chronoclone" in sys.modules:
        raise RuntimeError("Start a fresh Python process so another ChronoClone cannot enter the smoke run")
    with tempfile.TemporaryDirectory(prefix="chronoclone_paper_benchmark_") as directory:
        temporary = Path(directory)
        os.environ["NUMBA_CACHE_DIR"] = str(temporary / "numba")
        os.environ["MPLCONFIGDIR"] = str(temporary / "matplotlib")
        core_dir = temporary / "corrected"
        legacy_dir = temporary / "native_method_noise_benchmark_local_20260803/benchmark_package/scripts"
        core_dir.mkdir()
        legacy_dir.mkdir(parents=True)
        shutil.copy2(sources["corrected_core"][0], core_dir / "corrected_core.py")
        shutil.copy2(sources["low_level_core"][0], legacy_dir / "benchmark_core.py")
        sys.path[:0] = [str(numerical_runtime), str(core_dir)]
        import numpy as np
        import pandas as pd
        import chronoclone as cc
        import corrected_core as corrected
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        if str(numerical_runtime) not in cc.__file__:
            raise AssertionError("ChronoClone did not load from the included exact numerical module")
        legacy = corrected.LEGACY_CORE
        config = json.loads(sources["configuration"][0].read_text())
        reference = legacy.load_reference_programs(production / "reference")
        clean = corrected.rounded_clean_counts(reference.clean_means)
        gaussian, gaussian_seed = corrected.gaussian_stage_day0_locked(
            reference.clean_means, args.gaussian, args.repeat, config["master_seed"])
        counts, uniform_seed = corrected.uniform_redistribution_stage_day0_locked(
            gaussian, args.gaussian, args.uniform, args.repeat, config["master_seed"],
            start_states=config["allowed_start_states"], min_persistence=config["min_persistence"],
            min_total_count=config["min_total_count"])
        corrected.assert_day0_frozen(clean, counts)
        assert np.array_equal(gaussian[:, 1:].sum(axis=2), counts[:, 1:].sum(axis=2))
        mask = corrected.condition_eligibility_masks(clean, counts,
            start_states=config["allowed_start_states"], min_persistence=config["min_persistence"],
            min_total_count=config["min_total_count"]).current_lineage_mask
        eligible_counts, eligible_ids = counts[mask], reference.clone_ids[mask]
        # Every estimator is fitted before truth or metric evaluation is accessed.
        fits = {method: legacy.fit_named_method(method, eligible_counts, eligible_ids, config)
                for method in METHODS}
        raw = pd.DataFrame(eligible_counts.reshape(len(eligible_counts), -1),
                           index=eligible_ids, columns=legacy.WIDE_COLUMNS)
        features = cc.normalize_clone_table(raw,
            times_order=[f"Day{t}" for t in legacy.TIMES], features_order=list(legacy.STATES))
        selection = corrected.chronoclone_silhouette_postfilter(features.to_numpy(), fits["ChronoClone"].labels)
        truth = reference.truth[mask]
        rows = []
        for method, fit in fits.items():
            rows.append({"method": method, "ari": float(adjusted_rand_score(truth, fit.labels)),
                         "nmi": float(normalized_mutual_info_score(truth, fit.labels)),
                         "recovered_k": int(fit.recovered_k),
                         "selected_parameter": float(fit.selection_parameter),
                         "label_sha256": hashlib.sha256(np.asarray(fit.labels, dtype="<i8").tobytes()).hexdigest()})
        if args.gaussian == args.uniform == 0:
            assert all(row["ari"] == 1.0 and row["recovered_k"] == 12 for row in rows)
            assert int(mask.sum()) == len(clean) == int(selection.n_retained)
        payload = {"schema": "chronoclone_paper_bounded_smoke_v1", "status": "pass",
                   "scope": "One condition, four manuscript estimators; no production shards created or replaced.",
                   "historical_aggregate_recomputed": False,
                   "repeat": args.repeat, "gaussian_pct": args.gaussian, "uniform_pct": args.uniform,
                   "gaussian_seed": int(gaussian_seed), "uniform_seed": int(uniform_seed),
                   "input_clones": len(clean), "eligible_clones": int(mask.sum()),
                   "t1_changed_entries": int(np.count_nonzero(clean[:, 0] != counts[:, 0])),
                   "postfit_retained_clones": int(selection.n_retained),
                   "postfit_retention_fraction": float(selection.retained_fraction_of_fitted),
                   "postfit_rule": "sample silhouette >= -0.1; no size filter, no refit",
                   "runtime": {"python": sys.executable, "prefix": sys.prefix, "platform": platform.platform(),
                               "versions": versions, "historical_versions_match": not mismatches},
                   "source_files": source_records, "results": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
