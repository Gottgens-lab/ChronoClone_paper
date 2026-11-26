"""Reconstruct one configured synthetic count tensor without fitting estimators.

Uses byte-verified historical count-generation cores. Output is independent of
the historical benchmark aggregate and includes the ordered eligible clone IDs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile


SOURCE_LOCKS = {
    "corrected_core": ("production/corrected_core.py", "52c6925233945d0b450d123453141ab6583cb1d789576e6d7e81687eb2f237a4"),
    "low_level_core": ("native_20260803/benchmark_package/scripts/benchmark_core.py", "4641e28b317308473a65472fbf2ebf533db41e37f2954d9702991c110bdb5ca1"),
    "configuration": ("production/config/benchmark_config.json", "cb1d87ff5e0f0e6a280335449623a0808987185342ab1df05efaf65d0a2a0f96"),
    "reference_profiles": ("production/reference/deterministic_profiles.csv", "e4275d94c1513026dc684bcca44703135bba4af4694ec55cd714d3c0fa21bf6f"),
    "reference_manifest": ("production/reference/program_manifest.csv", "ba890ce14f052113d203904aa158cab79c36794c4edaf638ee0afd5b85063d11"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_benchmark_root(explicit=None):
    """Find either the paper benchmarks root or the package in_silico root."""
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    elif os.environ.get("CHRONOCLONE_BENCHMARK_ROOT"):
        candidates = [Path(os.environ["CHRONOCLONE_BENCHMARK_ROOT"]).expanduser().resolve()]
    else:
        candidates = [Path(__file__).resolve().parent, *Path(__file__).resolve().parents,
                      Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "frozen_runtime/production/corrected_core.py").is_file():
            return candidate
    raise FileNotFoundError("Cannot find frozen_runtime/production; provide --benchmark-root")


@contextmanager
def frozen_generator(benchmark_root=None):
    """Load unchanged generation cores in their expected temporary topology."""
    import numpy as np
    root = locate_benchmark_root(benchmark_root)
    frozen = root / "frozen_runtime"
    sources = {}
    for role, (relative, expected) in SOURCE_LOCKS.items():
        path = frozen / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen {role} does not match its published SHA-256")
        sources[role] = {"path": "frozen_runtime/" + relative, "sha256": expected,
                         "bytes": path.stat().st_size}
    manifest = json.loads((frozen / "native_20260803/results/provenance/environment_manifest.json").read_text())
    versions = {name: platform.python_version() if name == "python" else importlib.metadata.version(name)
                for name in manifest["versions"]}
    mismatches = {name: {"expected": expected, "observed": versions[name]}
                  for name, expected in manifest["versions"].items() if versions[name] != expected}
    if mismatches:
        raise RuntimeError(f"Use the recorded benchmark dependency versions: {mismatches}")
    names = ("_chronoclone_noise_corrected", "_chronoclone_locked_legacy_benchmark_core")
    if any(name in sys.modules for name in names):
        raise RuntimeError("A frozen generation context is already active")
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with tempfile.TemporaryDirectory(prefix="chronoclone_counts_") as directory:
            temporary = Path(directory)
            corrected_dir = temporary / "corrected"
            legacy_dir = temporary / "native_method_noise_benchmark_local_20260803/benchmark_package/scripts"
            corrected_dir.mkdir()
            legacy_dir.mkdir(parents=True)
            shutil.copy2(frozen / SOURCE_LOCKS["corrected_core"][0], corrected_dir / "corrected_core.py")
            shutil.copy2(frozen / SOURCE_LOCKS["low_level_core"][0], legacy_dir / "benchmark_core.py")
            spec = importlib.util.spec_from_file_location(names[0], corrected_dir / "corrected_core.py")
            corrected = importlib.util.module_from_spec(spec)
            sys.modules[names[0]] = corrected
            spec.loader.exec_module(corrected)
            reference = corrected.LEGACY_CORE.load_reference_programs(frozen / "production/reference")
            configuration = json.loads((frozen / SOURCE_LOCKS["configuration"][0]).read_text())
            clean = corrected.rounded_clean_counts(reference.clean_means)
            clean.flags.writeable = False
            reference.clean_means.flags.writeable = False
            yield {"root": root, "core": corrected, "reference": reference, "configuration": configuration,
                   "clean_counts": clean, "source_files": sources, "versions": versions}
        for role, (relative, expected) in SOURCE_LOCKS.items():
            if sha256(frozen / relative) != expected:
                raise AssertionError(f"Frozen {role} changed while generating counts")
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.dont_write_bytecode = previous_bytecode


def generate_condition(context, gaussian, uniform, repeat):
    """Generate, independently check, and describe one complete count tensor."""
    import numpy as np
    config = context["configuration"]
    for name, value in (("gaussian", gaussian), ("uniform", uniform)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value not in config["noise_levels_pct"]:
            raise ValueError(f"{name} must be one of the configured integer percentages")
    if isinstance(repeat, bool) or not isinstance(repeat, (int, np.integer)) or not 0 <= repeat < config["repeats"]:
        raise ValueError("repeat must be a configured integer repeat index")
    core, reference, clean = context["core"], context["reference"], context["clean_counts"]
    eligibility_args = {key: config[key] for key in ("min_persistence", "min_total_count")}
    eligibility_args["start_states"] = config["allowed_start_states"]
    gaussian_counts, gaussian_seed = core.gaussian_stage_day0_locked(
        reference.clean_means, gaussian, repeat, config["master_seed"])
    counts, uniform_seed = core.uniform_redistribution_stage_day0_locked(
        gaussian_counts, gaussian, uniform, repeat, config["master_seed"], **eligibility_args)
    mask = core.condition_eligibility_masks(clean, counts, **eligibility_args).current_lineage_mask
    before_mask = core.lineage_eligibility_mask(gaussian_counts, **eligibility_args)
    checks = {
        "nonnegative_integer_counts": bool(np.issubdtype(counts.dtype, np.integer) and np.all(counts >= 0)),
        "fixed_t1_after_gaussian": bool(np.array_equal(clean[:, 0], gaussian_counts[:, 0])),
        "fixed_t1_after_uniform": bool(np.array_equal(clean[:, 0], counts[:, 0])),
        "gaussian_preserves_structural_zeros": bool(np.all(gaussian_counts[reference.clean_means == 0] == 0)),
        "uniform_preserves_clone_time_totals": bool(np.array_equal(gaussian_counts.sum(axis=2), counts.sum(axis=2))),
        "uniform_preserves_eligibility": bool(np.array_equal(mask, before_mask)),
    }
    if gaussian == uniform == 0:
        checks["clean_condition_equals_rounded_reference"] = bool(np.array_equal(clean, counts))
    if not all(checks.values()):
        raise AssertionError(checks)
    clone_ids = np.asarray(reference.clone_ids, dtype=str)
    eligible_ids = clone_ids[mask]
    ordered_id_sha = hashlib.sha256("\n".join(eligible_ids.tolist()).encode()).hexdigest()
    arrays = {"counts": counts, "gaussian_counts": gaussian_counts, "clean_counts": clean,
              "clean_means": reference.clean_means, "clone_ids": clone_ids,
              "truth_program": np.asarray(reference.truth, dtype=str),
              "timepoints": np.asarray(core.TIMES, dtype=np.int64), "states": np.asarray(core.STATES, dtype=str),
              "eligible_mask": mask, "eligible_clone_ids": eligible_ids}
    metadata = {"schema": "chronoclone_reproduced_count_tensor_v1", "gaussian_pct": int(gaussian),
                "uniform_pct": int(uniform), "repeat": int(repeat), "master_seed": config["master_seed"],
                "gaussian_seed": int(gaussian_seed), "uniform_seed": int(uniform_seed),
                "shape": list(counts.shape), "counts_dtype": str(counts.dtype),
                "counts_array_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
                "input_clones": len(counts), "eligible_clones": int(mask.sum()),
                "ordered_eligible_clone_ids_sha256": ordered_id_sha,
                "method_input_clone_ids_sha256": {method: ordered_id_sha for method in config["method_order"]},
                "checks": checks, "source_files": context["source_files"], "versions": context["versions"],
                "first_timepoint_fixed": True, "noise_order": "Gaussian then uniform redistribution at t2-t4",
                "estimator_fits_performed": False, "historical_aggregate_modified": False,
                "truth_scope": "Simulation programs for inspection and evaluation only; never encoded in clone IDs.",
                "runtime_scope": "Count generation uses only the unchanged frozen generation cores."}
    return arrays, metadata


def save_condition(arrays, metadata, output, benchmark_root):
    """Write a fresh NPZ and same-stem JSON without changing benchmark evidence."""
    import numpy as np
    output = Path(output).expanduser().resolve()
    root = Path(benchmark_root).resolve()
    if output.suffix != ".npz":
        raise ValueError("output must end in .npz")
    protected = [root / "frozen_runtime", root / "reference", root / "config", root / "results/aggregate",
                 root / "ode_noise/reference", root / "ode_noise/config", root / "ode_noise/results/aggregate"]
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError("Output cannot replace a frozen source, reference, configuration, or historical aggregate")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new output filename; existing tensor evidence is never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    receipt = {**metadata, "file": output.name, "file_bytes": output.stat().st_size, "file_sha256": sha256(output)}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--gaussian", type=int, default=0)
    parser.add_argument("--uniform", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with frozen_generator(args.benchmark_root) as context:
        arrays, metadata = generate_condition(context, args.gaussian, args.uniform, args.repeat)
        receipt = save_condition(arrays, metadata, args.output, context["root"])
    print(json.dumps({key: receipt[key] for key in ("file", "shape", "eligible_clones", "file_sha256", "checks")}, indent=2))


if __name__ == "__main__":
    main()
