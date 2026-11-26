#!/usr/bin/env python3
"""Load the locked method-adapter runtime used by the corrected benchmark."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/path/to/ChronoClone")
SOURCE = PROJECT_ROOT / "analysis_workspace/analyses/native_method_noise_benchmark_local_20260803"
SOURCE_SCRIPTS = SOURCE / "benchmark_package/scripts"
SOURCE_SITE_PACKAGES = SOURCE / "runtime/site-packages"
SOURCE_MANIFEST = SOURCE / "results/provenance/environment_manifest.json"
SOURCE_PROFILE = SOURCE / "benchmark_package/reference/deterministic_profiles.csv"
SOURCE_PROGRAM_MANIFEST = SOURCE / "benchmark_package/reference/program_manifest.csv"

EXPECTED = {
    "source_core_sha256": "4641e28b317308473a65472fbf2ebf533db41e37f2954d9702991c110bdb5ca1",
    "source_reference_sha256": "e4275d94c1513026dc684bcca44703135bba4af4694ec55cd714d3c0fa21bf6f",
    "source_program_manifest_sha256": "ba890ce14f052113d203904aa158cab79c36794c4edaf638ee0afd5b85063d11",
    "chronoclone_commit": "0b09e41a381e4e2219160a3ea513aae008cae14d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_runtime() -> dict[str, str]:
    paths = {
        "source_core_sha256": SOURCE_SCRIPTS / "benchmark_core.py",
        "source_reference_sha256": SOURCE_PROFILE,
        "source_program_manifest_sha256": SOURCE_PROGRAM_MANIFEST,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED[name], "observed": value}
        for name, value in observed.items()
        if value != EXPECTED[name]
    }
    if mismatches:
        raise RuntimeError(f"Locked benchmark source changed: {mismatches}")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if str(manifest["chronoclone_commit"]) != EXPECTED["chronoclone_commit"]:
        raise RuntimeError("Locked ChronoClone commit changed")
    if not (SOURCE_SITE_PACKAGES / "chronoclone/__init__.py").is_file():
        raise FileNotFoundError("Locked ChronoClone runtime is missing")
    return {
        **observed,
        "source_environment_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "chronoclone_commit": str(manifest["chronoclone_commit"]),
        "chronoclone_module_path": str(manifest["chronoclone_module_path"]),
    }


def activate_source_imports() -> dict[str, str]:
    provenance = validate_source_runtime()
    for path in (SOURCE_SITE_PACKAGES, SOURCE_SCRIPTS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return provenance
