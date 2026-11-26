#!/usr/bin/env python3
"""Build the reader-facing Cyclone applicability diagnostic notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "0004_cyclone_applicability_diagnostic.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def main() -> int:
    cells = [
        markdown(
            """
# Cyclone abundance-only applicability diagnostic

This notebook audits why the Cyclone abundance-only panel in the fixed-t1
technical-noise benchmark is mostly undefined and whether it supports a
reader-facing accuracy comparison. Computed findings are emitted by executed
code cells rather than written into notebook Markdown.
"""
        ),
        code(
            """
from pathlib import Path
import hashlib
import json
import sys
import warnings

import nbclient
import nbformat
import numpy as np
import pandas as pd
import scipy
import sklearn
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
assert ROOT.name == "chronoclone_method_noise_benchmark_day0_locked_20260812"

ENVIRONMENT = {
    "manager": "micromamba (existing prefix reused without mutation)",
    "name": "chronoclone",
    "path": "/path/to/ChronoClone/.runtime/chronoclone",
    "kernel": "chronoclone",
    "python": sys.version.split()[0],
    "execution_command": (
        "/path/to/ChronoClone/.runtime/chronoclone/bin/python "
        "scripts/execute_notebook.py notebooks/0004_cyclone_applicability_diagnostic.ipynb"
    ),
}
assert ENVIRONMENT["python"] == "3.11.15"
assert Path(ENVIRONMENT["path"]).is_dir()

versions = pd.DataFrame(
    {
        "component": [
            "Python", "NumPy", "pandas", "SciPy", "scikit-learn",
            "nbformat", "nbclient",
        ],
        "version": [
            ENVIRONMENT["python"], np.__version__, pd.__version__,
            scipy.__version__, sklearn.__version__, nbformat.__version__,
            nbclient.__version__,
        ],
    }
)
display(pd.DataFrame([ENVIRONMENT]))
display(versions)
"""
        ),
        markdown(
            """
## Context & Methods

### Diagnostic question

Does the Cyclone abundance-only output represent a fair accuracy comparison,
a software defect, or a method-applicability mismatch?

### Key assumptions

- The completed, hash-validated aggregate is the controlling quantitative
  source.
- `Cyclone abundance-only` denotes the published abundance/correlation/Ward
  core with fixed default K=6 and its dynamic clone filter disabled; it is not
  the complete Seurat workflow.
- Uniform state redistribution preserves every clone-time total by contract.
- A finite-fit conditional ARI is not interpreted as performance over failed
  repeats.
"""
        ),
        code(
            """
AGGREGATE = ROOT / "results" / "aggregate"
SUMMARY_PATH = AGGREGATE / "native_method_noise_summary.csv"
COMPLETENESS_PATH = AGGREGATE / "result_completeness.csv"
FAILURE_PATH = AGGREGATE / "method_failure_counts.csv"
RUNS_PATH = AGGREGATE / "native_method_noise_runs.csv.gz"
METADATA_PATH = AGGREGATE / "aggregation_metadata.json"
CONFIG_PATH = ROOT / "config" / "benchmark_config.json"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
summary = pd.read_csv(SUMMARY_PATH)
completeness = pd.read_csv(COMPLETENESS_PATH)
failure_counts = pd.read_csv(FAILURE_PATH)

for filename, expected_hash in metadata["artifact_sha256"].items():
    assert sha256_file(AGGREGATE / filename) == expected_hash
assert metadata["status"] == "pass"
assert metadata["observed_rows"] == metadata["expected_rows"] == 1_300_500
assert metadata["cyclone_uniform_invariance_validated"] is True

display(
    pd.DataFrame(
        {
            "source": [
                SUMMARY_PATH.relative_to(ROOT),
                COMPLETENESS_PATH.relative_to(ROOT),
                FAILURE_PATH.relative_to(ROOT),
                RUNS_PATH.relative_to(ROOT),
            ],
            "grain": [
                "method x Gaussian x uniform (100-repeat summary)",
                "method x Gaussian x uniform fit coverage",
                "method x Gaussian x uniform x failure type",
                "repeat x Gaussian x uniform x method",
            ],
        }
    )
)
display(
    pd.DataFrame(
        {
            "validation": [
                "aggregate status", "terminal rows", "uniform-invariance gate",
                "Cyclone dynamic filter enabled",
            ],
            "value": [
                metadata["status"], metadata["observed_rows"],
                metadata["cyclone_uniform_invariance_validated"],
                config["cyclone"]["dynamic_clone_filter"],
            ],
        }
    )
)
"""
        ),
        markdown("## Data"),
        code(
            """
method_coverage = (
    summary.groupby("method", sort=False)
    .agg(
        requested_fits=("n_expected", "sum"),
        finite_partitions=("n_success", "sum"),
        method_failures=("n_method_failure", "sum"),
        coordinates_with_any_fit=("n_success", lambda values: int((values > 0).sum())),
        total_coordinates=("n_success", "size"),
    )
    .reset_index()
)
method_coverage["finite_fit_fraction"] = (
    method_coverage["finite_partitions"] / method_coverage["requested_fits"]
)
assert method_coverage["requested_fits"].eq(260_100).all()
display(method_coverage)

cyclone_summary = summary.loc[summary["method"].eq("Cyclone abundance-only")].copy()
assert len(cyclone_summary) == 2_601
assert int(cyclone_summary["n_success"].sum()) == 11_934
assert int(cyclone_summary["n_method_failure"].sum()) == 248_166
assert failure_counts["method"].eq("Cyclone abundance-only").all()
assert failure_counts["error_type"].eq("NativeSelectionFailure").all()
assert failure_counts["error_message"].nunique() == 1
display(failure_counts[["error_type", "error_message", "count"]].groupby(
    ["error_type", "error_message"], as_index=False
).agg(failed_fits=("count", "sum")))
"""
        ),
        markdown("## Results"),
        code(
            """
cyclone_run_parts = []
run_columns = [
    "method", "repeat", "gaussian_pct", "uniform_pct", "status", "ari",
    "recovered_k", "error_type", "error_message", "method_diagnostic_json",
]
for chunk in pd.read_csv(RUNS_PATH, compression="gzip", usecols=run_columns, chunksize=100_000):
    cyclone_part = chunk.loc[chunk["method"].eq("Cyclone abundance-only")]
    if len(cyclone_part):
        cyclone_run_parts.append(cyclone_part)
cyclone_runs = pd.concat(cyclone_run_parts, ignore_index=True)
assert len(cyclone_runs) == 260_100

pair_group = cyclone_runs.groupby(["repeat", "gaussian_pct"], sort=True)
pair_size = pair_group.size()
assert pair_size.eq(51).all()

uniform_invariance_checks = pd.DataFrame(
    {
        "field": ["status", "ARI", "recovered K", "error type", "error message"],
        "maximum distinct values across 51 uniform levels": [
            int(pair_group["status"].nunique(dropna=False).max()),
            int(pair_group["ari"].nunique(dropna=False).max()),
            int(pair_group["recovered_k"].nunique(dropna=False).max()),
            int(pair_group["error_type"].nunique(dropna=False).max()),
            int(pair_group["error_message"].nunique(dropna=False).max()),
        ],
    }
)
assert uniform_invariance_checks.iloc[:, 1].eq(1).all()
display(uniform_invariance_checks)

pair_first = pair_group.first().reset_index()
successful_pairs = pair_first.loc[pair_first["status"].eq("ok")].copy()
assert len(successful_pairs) == 234
assert len(successful_pairs) * 51 == int(cyclone_summary["n_success"].sum())

by_gaussian = (
    pair_first.groupby("gaussian_pct")
    .agg(
        successful_repeats=("status", lambda values: int((values == "ok").sum())),
        requested_repeats=("status", "size"),
    )
    .reset_index()
)
by_gaussian["fit_fraction"] = (
    by_gaussian["successful_repeats"] / by_gaussian["requested_repeats"]
)

coordinate_summary = pd.DataFrame(
    {
        "quantity": [
            "requested Cyclone rows",
            "finite partitions",
            "finite-fit fraction",
            "uniform-only finite partitions",
            "independent successful repeat/Gaussian pairs",
            "Gaussian levels with at least one finite fit",
            "first Gaussian level with a finite fit",
            "maximum repeat fit fraction at any Gaussian level",
            "heatmap coordinates with any finite fit",
            "heatmap coordinates with no finite fit",
        ],
        "value": [
            len(cyclone_runs),
            int(cyclone_runs["status"].eq("ok").sum()),
            float(cyclone_runs["status"].eq("ok").mean()),
            int(cyclone_runs.loc[cyclone_runs["gaussian_pct"].eq(0), "status"].eq("ok").sum()),
            len(successful_pairs),
            int((by_gaussian["successful_repeats"] > 0).sum()),
            int(by_gaussian.loc[by_gaussian["successful_repeats"] > 0, "gaussian_pct"].min()),
            float(by_gaussian["fit_fraction"].max()),
            int(cyclone_summary["n_success"].gt(0).sum()),
            int(cyclone_summary["n_success"].eq(0).sum()),
        ],
    }
)
display(coordinate_summary)
display(by_gaussian.loc[by_gaussian["successful_repeats"].gt(0)].reset_index(drop=True))
"""
        ),
        code(
            """
successful_pair_ari = successful_pairs["ari"].dropna()
finite_coordinate_ari = cyclone_summary.loc[
    cyclone_summary["ari_n_finite"].gt(0), "ari_mean"
]
assert len(successful_pair_ari) == len(successful_pairs)
assert successful_pairs["recovered_k"].eq(6).all()

conditional_accuracy = pd.DataFrame(
    {
        "quantity": [
            "successful independent repeat/Gaussian pairs",
            "conditional ARI mean",
            "conditional ARI median",
            "conditional ARI 2.5th percentile",
            "conditional ARI 97.5th percentile",
            "minimum coordinate-wise conditional mean ARI",
            "maximum coordinate-wise conditional mean ARI",
            "recovered K for every finite fit",
        ],
        "value": [
            len(successful_pair_ari),
            float(successful_pair_ari.mean()),
            float(successful_pair_ari.median()),
            float(successful_pair_ari.quantile(0.025)),
            float(successful_pair_ari.quantile(0.975)),
            float(finite_coordinate_ari.min()),
            float(finite_coordinate_ari.max()),
            int(successful_pairs["recovered_k"].iloc[0]),
        ],
    }
)
display(conditional_accuracy)
"""
        ),
        code(
            """
REFERENCE = (
    ROOT.parent / "native_method_noise_benchmark_local_20260803" /
    "benchmark_package" / "reference"
)
profiles = np.loadtxt(REFERENCE / "deterministic_profiles.csv", delimiter=",").reshape(12, 4, 13)
program_manifest = pd.read_csv(REFERENCE / "program_manifest.csv")
assert len(program_manifest) == len(profiles) == 12
assert int(program_manifest["n_clones"].sum()) == 517

intended_abundance = profiles.sum(axis=2)
rounded_profiles = np.rint(profiles).astype(np.int64)
rounded_abundance = rounded_profiles.sum(axis=2)
expanded_clean_counts = np.repeat(
    rounded_profiles, program_manifest["n_clones"].to_numpy(int), axis=0
)

information_audit = pd.DataFrame(
    {
        "quantity": [
            "truth programs",
            "distinct 52-dimensional state profiles",
            "distinct intended four-time abundance trajectories",
            "distinct rounded four-time abundance trajectories",
            "eligible clone count represented by the clean tensor",
            "distinct clean t1 clone totals",
        ],
        "value": [
            len(profiles),
            len(np.unique(np.round(profiles.reshape(12, -1), 10), axis=0)),
            len(np.unique(np.round(intended_abundance, 10), axis=0)),
            len(np.unique(rounded_abundance, axis=0)),
            len(expanded_clean_counts),
            len(np.unique(expanded_clean_counts[:, 0, :].sum(axis=1))),
        ],
    }
)
display(information_audit)

program_abundance_table = program_manifest[["truth_metaclone", "n_clones"]].copy()
for time_index, time_label in enumerate(["t1", "t2", "t3", "t4"]):
    program_abundance_table[f"{time_label}_intended_total"] = intended_abundance[:, time_index]
    program_abundance_table[f"{time_label}_rounded_total"] = rounded_abundance[:, time_index]
display(program_abundance_table)
"""
        ),
        code(
            """
sys.path.insert(0, str(ROOT / "scripts"))
import cyclone_adapter

clean_abundance = expanded_clean_counts.sum(axis=2, dtype=np.int64).astype(float)
time_by_clone = (clean_abundance / clean_abundance.sum(axis=0, keepdims=True) * 100.0).T
row_means = time_by_clone.mean(axis=1, keepdims=True)
row_sd = time_by_clone.std(axis=1, ddof=1, keepdims=True)
constant_time_rows = np.ptp(clean_abundance, axis=0) == 0
row_sd[constant_time_rows] = 0.0
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    scaled = (time_by_clone - row_means) / row_sd
clean_correlation = cyclone_adapter.r_pairwise_complete_correlation(scaled)

try:
    cyclone_adapter.cyclone_correlation_core(expanded_clean_counts, n_clusters=6)
except Exception as error:
    clean_failure_type = type(error).__name__
    clean_failure_message = str(error)
else:
    raise AssertionError("The locked clean Cyclone fixture unexpectedly returned a fit")

clean_numerical_audit = pd.DataFrame(
    {
        "quantity": [
            "constant time rows after state summation",
            "finite clean clone-correlation entries",
            "all-nonfinite clean correlation profiles",
            "clean fit outcome",
            "clean failure message",
        ],
        "value": [
            int(constant_time_rows.sum()),
            int(np.isfinite(clean_correlation).sum()),
            int((~np.isfinite(clean_correlation)).all(axis=1).sum()),
            clean_failure_type,
            clean_failure_message,
        ],
    }
)
display(clean_numerical_audit)
"""
        ),
        markdown("## Takeaways"),
        code(
            '''
cyclone_coverage = method_coverage.loc[
    method_coverage["method"].eq("Cyclone abundance-only")
].iloc[0]
fit_percent = 100.0 * float(cyclone_coverage["finite_fit_fraction"])
ari_mean = float(successful_pair_ari.mean())
max_success_percent = 100.0 * float(by_gaussian["fit_fraction"].max())

display(Markdown(f"""
**Observed evidence.** The abundance-only adapter returned a finite partition
for **{int(cyclone_coverage['finite_partitions']):,} of
{int(cyclone_coverage['requested_fits']):,} requested rows ({fit_percent:.2f}%)**.
Those rows reduce to only **{len(successful_pairs)} independent
repeat/Gaussian outcomes**, each copied unchanged across all 51 uniform levels.
No uniform-only condition fitted. Even where the core returned K=6, its
conditional mean ARI was **{ari_mean:.3f}**, and no Gaussian level fitted more
than **{max_success_percent:.0f}%** of repeats.

**Mechanistic diagnosis.** This is not evidence that full Cyclone is simply a
poor clustering algorithm. The adapter discards all 13-state composition and
uses only four clone-abundance totals. The 12 truth programs collapse to
{int(information_audit.loc[information_audit['quantity'].eq('distinct intended four-time abundance trajectories'), 'value'].iloc[0])}
intended abundance trajectories. Uniform redistribution is therefore exactly
invisible. The fixed clean t1 abundance row is constant, and the clean
correlation profiles are non-finite. Large Gaussian perturbations occasionally
break numerical ties by injecting random abundance variation, but that
variation is not the 12-program state-composition truth.

**Presentation decision.** The current Cyclone mean-ARI heatmap is not a fair
primary accuracy panel. It mixes gray undefined cells with conditional means
based on 1--22 successful repeats and duplicates every result down the uniform
axis. A reviewer could reasonably mistake this for a performance surface or a
full published Cyclone comparison. Cyclone should be removed from the primary
ARI figures and reported only as a clearly labelled applicability audit. A
valid Cyclone accuracy experiment would require a separate abundance-trajectory
benchmark (or suitable longitudinal biological data), the published dynamic
filter/full workflow, and a truth target defined in terms Cyclone can observe.
"""))

presentation_scope = pd.DataFrame(
    {
        "location": [
            "Primary accuracy figures",
            "Supplementary applicability audit",
            "Future Cyclone-specific benchmark",
        ],
        "Cyclone treatment": [
            "Exclude from ARI curves and heatmaps",
            "Report finite-fit fraction, exact uniform invariance, and core-only scope; do not rank by ARI",
            "Use abundance-informative truth/data and the published filter/full workflow",
        ],
    }
)
display(presentation_scope)
'''
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "chronoclone",
                "language": "python",
                "name": "chronoclone",
            },
            "language_info": {"name": "python", "version": "3.11.15"},
            "execution": {"status": "not_executed"},
        },
    )
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
