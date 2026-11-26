#!/usr/bin/env python3
"""Standalone mechanical validation of aggregate, notebook, and SVG outputs."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "aggregate"
NOTEBOOK = ROOT / "notebooks" / "0003_day0_locked_method_noise_results.ipynb"
FIGURES = ROOT / "figures"
METHODS = {
    "ChronoClone",
    "CLADES preprocessing",
    "Cyclone abundance-only",
    "K-means",
    "Ward",
}
EXPECTED_FIGURES = {
    "gaussian_only_mean_ari_by_method.svg",
    "uniform_only_mean_ari_by_method.svg",
    "chronoclone_mean_ari_heatmap.svg",
    "clades_preprocessing_mean_ari_heatmap.svg",
    "cyclone_abundance_only_mean_ari_heatmap.svg",
    "kmeans_mean_ari_heatmap.svg",
    "ward_mean_ari_heatmap.svg",
    "gaussian_only_method_success_fraction.svg",
    "uniform_only_method_success_fraction.svg",
    "isolated_axes_eligible_clone_fraction.svg",
    "gaussian_only_recovered_k.svg",
    "uniform_only_recovered_k.svg",
    "chronoclone_isolated_axes_postfit_coverage.svg",
    "gaussian_only_same_retained_ari_sensitivity.svg",
    "uniform_only_same_retained_ari_sensitivity.svg",
}
POSTPROCESSOR_SOURCES = (
    ROOT / "scripts" / "aggregate_results.py",
    ROOT / "scripts" / "build_notebooks.py",
    ROOT / "scripts" / "execute_notebook.py",
    ROOT / "scripts" / "validate_postprocessing.py",
)
POSTPROCESSOR_MANIFEST = (
    ROOT / "results" / "provenance" / "postprocessing_source_manifest.sha256"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def main() -> int:
    metadata_path = AGGREGATE / "aggregation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass":
        raise ValueError("Aggregation commit marker is not pass")
    if metadata.get("observed_chunks") != 600 or metadata.get("observed_rows") != 1_300_500:
        raise ValueError("Aggregation cardinality differs from the production contract")
    for field in (
        "day0_lock_validated",
        "five_method_common_cohort_validated",
        "uniform_total_and_eligibility_invariance_validated",
        "cyclone_uniform_invariance_validated",
        "deterministic_clean_baseline_validated",
        "postfilter_payload_validated",
    ):
        if metadata.get(field) is not True:
            raise ValueError(f"Aggregation invariant did not pass: {field}")
    expected_postprocessor_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in POSTPROCESSOR_SOURCES
    }
    if not POSTPROCESSOR_MANIFEST.is_file():
        raise FileNotFoundError(POSTPROCESSOR_MANIFEST)
    locked_postprocessor_hashes = {}
    for line in POSTPROCESSOR_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        locked_postprocessor_hashes[name] = digest
    if locked_postprocessor_hashes != expected_postprocessor_hashes:
        raise ValueError("Postprocessing sources differ from their predeclared manifest")
    if metadata.get("postprocessor_sha256") != expected_postprocessor_hashes:
        raise ValueError("Postprocessing source hashes differ from aggregation provenance")
    if metadata.get("postprocessor_manifest_sha256") != sha256(POSTPROCESSOR_MANIFEST):
        raise ValueError("Postprocessing manifest hash differs from aggregation provenance")
    for name, expected in metadata["artifact_sha256"].items():
        path = AGGREGATE / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Aggregate artifact hash mismatch: {name}")

    summary = pd.read_csv(AGGREGATE / "native_method_noise_summary.csv")
    completeness = pd.read_csv(AGGREGATE / "result_completeness.csv")
    eligibility = pd.read_csv(AGGREGATE / "eligibility_summary.csv")
    selective = pd.read_csv(AGGREGATE / "chronoclone_selective_summary.csv")
    same_retained = pd.read_csv(AGGREGATE / "same_retained_subset_summary.csv")
    failure_counts = pd.read_csv(AGGREGATE / "method_failure_counts.csv")
    chunk_manifest = pd.read_csv(AGGREGATE / "chunk_manifest.csv")
    if len(summary) != 13_005 or len(completeness) != 13_005 or len(same_retained) != 13_005:
        raise ValueError("A method-coordinate summary is incomplete")
    if len(eligibility) != 2_601 or len(selective) != 2_601:
        raise ValueError("A condition-coordinate summary is incomplete")
    if len(chunk_manifest) != 600 or set(chunk_manifest["task_id"]) != set(range(600)):
        raise ValueError("Chunk manifest coverage is incomplete")
    expected_method_coordinates = {
        (method, gaussian, uniform)
        for method in METHODS
        for gaussian in range(0, 101, 2)
        for uniform in range(0, 101, 2)
    }
    for table in (summary, completeness, same_retained):
        observed = set(
            table[["method", "gaussian_pct", "uniform_pct"]].itertuples(
                index=False, name=None
            )
        )
        if observed != expected_method_coordinates:
            raise ValueError("Method-coordinate key set differs")
    expected_conditions = {
        (gaussian, uniform)
        for gaussian in range(0, 101, 2)
        for uniform in range(0, 101, 2)
    }
    for table in (eligibility, selective):
        observed = set(
            table[["gaussian_pct", "uniform_pct"]].itertuples(index=False, name=None)
        )
        if observed != expected_conditions:
            raise ValueError("Condition-coordinate key set differs")
    if not (completeness["n_expected"] == 100).all():
        raise ValueError("Completeness denominator differs from 100")
    if not (
        completeness["n_success"] + completeness["n_method_failure"] == 100
    ).all():
        raise ValueError("Terminal statuses do not sum to 100")
    if not (summary["ari_n_finite"] == summary["n_success"]).all():
        raise ValueError("Successful full-cohort ARI count differs from status count")
    bounded_columns = {
        "ari": (-1, 1),
        "nmi": (0, 1),
        "hungarian_macro_f1": (0, 1),
    }
    for prefix, (low, high) in bounded_columns.items():
        if not (summary[f"{prefix}_n_finite"] == summary["n_success"]).all():
            raise ValueError(f"Successful {prefix} count differs from status count")
        for suffix in ("mean", "median", "q025", "q975"):
            values = summary.loc[summary[f"{prefix}_n_finite"].gt(0), f"{prefix}_{suffix}"]
            if not values.between(low - 1e-12, high + 1e-12).all():
                raise ValueError(f"{prefix}_{suffix} is outside its metric domain")
    if not (eligibility["eligible_fraction_n_finite"] == 100).all():
        raise ValueError("Eligibility summaries do not contain all repeats")
    for suffix in ("mean", "median", "q025", "q975"):
        if not eligibility[f"eligible_fraction_{suffix}"].between(0, 1).all():
            raise ValueError("Eligibility fraction summary is outside [0, 1]")
    postfilter_counts = selective[
        [
            "n_postfilter_ok",
            "n_postfilter_all_abstained",
            "n_postfilter_insufficient",
            "n_postfilter_other",
        ]
    ].sum(axis=1)
    if not postfilter_counts.eq(100).all():
        raise ValueError("ChronoClone postfilter statuses do not sum to 100")
    for prefix in (
        "postfit_retained_fraction_of_eligible",
        "postfit_retained_fraction_of_all_input",
        "shared_retained_subset_nmi",
        "shared_retained_subset_hungarian_macro_f1",
        "shared_retained_subset_abstention_aware_hungarian_macro_f1",
    ):
        for suffix in ("mean", "median", "q025", "q975"):
            values = selective.loc[selective[f"{prefix}_n_finite"].gt(0), f"{prefix}_{suffix}"]
            if not values.between(0, 1).all():
                raise ValueError(f"Selective summary {prefix}_{suffix} is outside [0, 1]")
    for suffix in ("mean", "median", "q025", "q975"):
        values = selective.loc[
            selective["shared_retained_subset_ari_n_finite"].gt(0),
            f"shared_retained_subset_ari_{suffix}",
        ]
        if not values.between(-1, 1).all():
            raise ValueError("Selective ARI summary is outside [-1, 1]")
    for prefix, (low, high) in (
        ("shared_retained_subset_ari", (-1, 1)),
        ("shared_retained_subset_nmi", (0, 1)),
        ("shared_retained_subset_hungarian_macro_f1", (0, 1)),
        ("shared_retained_subset_abstention_aware_hungarian_macro_f1", (0, 1)),
    ):
        for suffix in ("mean", "median", "q025", "q975"):
            values = same_retained.loc[
                same_retained[f"{prefix}_n_finite"].gt(0), f"{prefix}_{suffix}"
            ]
            if not values.between(low, high).all():
                raise ValueError(f"Same-retained summary {prefix}_{suffix} is out of range")
    observed_failure_count = int(failure_counts["count"].sum()) if len(failure_counts) else 0
    if observed_failure_count != int(metadata["method_failure_rows"]):
        raise ValueError("Failure-count table does not reconcile")

    if not NOTEBOOK.is_file():
        raise FileNotFoundError(NOTEBOOK)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if notebook.metadata.get("execution", {}).get("status") != "ok":
        raise ValueError("Results notebook is not marked executed")
    if [cell.execution_count for cell in code_cells] != list(
        range(1, len(code_cells) + 1)
    ):
        raise ValueError("Results notebook execution counts are not sequential")
    if any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    ):
        raise ValueError("Results notebook contains an error output")
    plot_cells = [cell for cell in code_cells if "plt.subplots(" in cell.source]
    if len(plot_cells) != len(EXPECTED_FIGURES):
        raise ValueError("Unexpected number of plot-output cells")
    for cell in plot_cells:
        if cell.source.count("plt.subplots(") != 1:
            raise ValueError("A plot cell creates more than one figure")
        if cell.source.count("fig.savefig(") != 1 or cell.source.count("plt.show()") != 1:
            raise ValueError("A plot cell does not save/show exactly one figure")
        rich_outputs = [
            output
            for output in cell.get("outputs", [])
            if output.output_type in {"display_data", "execute_result"}
            and any(key.startswith("image/") for key in output.get("data", {}))
        ]
        if len(rich_outputs) != 1:
            raise ValueError("A plot cell does not retain exactly one visible figure output")
    for name in EXPECTED_FIGURES:
        source_hits = sum(name in cell.source for cell in plot_cells)
        if source_hits != 1:
            raise ValueError(f"Figure stem does not map to one plot cell: {name}")

    figure_hashes = {}
    observed_svg_names = {path.name for path in FIGURES.glob("*.svg")}
    if observed_svg_names != EXPECTED_FIGURES:
        raise ValueError(
            "Figure SVG allowlist differs: "
            f"{sorted(observed_svg_names ^ EXPECTED_FIGURES)}"
        )
    for name in sorted(EXPECTED_FIGURES):
        path = FIGURES / name
        if not path.is_file():
            raise FileNotFoundError(path)
        root = ET.parse(path).getroot()
        tags = [element.tag.rsplit("}", 1)[-1] for element in root.iter()]
        if "text" not in tags:
            raise ValueError(f"SVG lacks live text: {name}")
        if "image" in tags:
            raise ValueError(f"SVG embeds a raster image: {name}")
        figure_hashes[name] = sha256(path)

    report = {
        "schema": "chronoclone_day0_locked_postprocessing_validation_v1",
        "status": "pass",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_rows": int(metadata["observed_rows"]),
        "aggregate_chunks": int(metadata["observed_chunks"]),
        "summary_rows": len(summary),
        "eligibility_rows": len(eligibility),
        "selective_rows": len(selective),
        "notebook_code_cells": len(code_cells),
        "notebook_plot_cells": len(plot_cells),
        "notebook_sha256": sha256(NOTEBOOK),
        "figure_sha256": figure_hashes,
        "postprocessor_sha256": expected_postprocessor_hashes,
        "postprocessor_manifest_sha256": sha256(POSTPROCESSOR_MANIFEST),
    }
    atomic_json(report, AGGREGATE / "postprocessing_validation.json")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
