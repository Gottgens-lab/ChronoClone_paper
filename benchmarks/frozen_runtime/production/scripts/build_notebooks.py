#!/usr/bin/env python3
"""Build, execute, and validate the corrected production-results notebook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "0003_day0_locked_method_noise_results.ipynb"
EXECUTOR = ROOT / "scripts" / "execute_notebook.py"
VALIDATOR = ROOT / "scripts" / "validate_postprocessing.py"
METHODS = (
    "ChronoClone",
    "CLADES preprocessing",
    "Cyclone abundance-only",
    "K-means",
    "Ward",
)
SLUGS = {
    "ChronoClone": "chronoclone",
    "CLADES preprocessing": "clades_preprocessing",
    "Cyclone abundance-only": "cyclone_abundance_only",
    "K-means": "kmeans",
    "Ward": "ward",
}


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def axis_ari_cell(axis: str) -> str:
    if axis == "gaussian":
        subset = 'summary.loc[summary["uniform_pct"].eq(0)].copy()'
        x_column = "gaussian_pct"
        xlabel = "Gaussian noise (% count CV; t2/t3/t4 only)"
        title = "Gaussian noise only"
        stem = "gaussian_only_mean_ari_by_method"
    elif axis == "uniform":
        subset = 'summary.loc[summary["gaussian_pct"].eq(0)].copy()'
        x_column = "uniform_pct"
        xlabel = "Uniform state redistribution (%; t2/t3/t4 only)"
        title = "Uniform redistribution only"
        stem = "uniform_only_mean_ari_by_method"
    else:
        raise ValueError(axis)
    return f'''# Plot-specific shaping and full construction: {title} mean ARI
axis_data = {subset}
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for method in METHODS:
    data = axis_data.loc[axis_data["method"].eq(method)].sort_values({x_column!r})
    x = data[{x_column!r}].to_numpy(float)
    mean = data["ari_mean"].to_numpy(float)
    lower = data["ari_q025"].to_numpy(float)
    upper = data["ari_q975"].to_numpy(float)
    finite = np.isfinite(mean)
    if finite.any():
        ax.plot(x[finite], mean[finite], color=COLORS[method], linewidth=2.0, label=method)
        band = finite & np.isfinite(lower) & np.isfinite(upper)
        if band.any():
            ax.fill_between(x[band], lower[band], upper[band], color=COLORS[method], alpha=0.14, linewidth=0)
ax.set_xlim(0, 100)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel({xlabel!r})
ax.set_ylabel("Adjusted Rand Index")
ax.set_title({title!r})
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, labels, frameon=False, fontsize=8)
fig.tight_layout()
output_path = FIGURES / {stem + '.svg'!r}
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)'''


def heatmap_cell(method: str) -> str:
    slug = SLUGS[method]
    return f'''# Plot-specific shaping and full construction: {method} mean-ARI heatmap
heat_source = summary.loc[summary["method"].eq({method!r}), ["gaussian_pct", "uniform_pct", "ari_mean"]].copy()
heat = heat_source.pivot(index="uniform_pct", columns="gaussian_pct", values="ari_mean").sort_index().sort_index(axis=1)
assert heat.shape == (51, 51)
x = heat.columns.to_numpy(float)
y = heat.index.to_numpy(float)
x_edges = np.r_[x - 1.0, x[-1] + 1.0]
y_edges = np.r_[y - 1.0, y[-1] + 1.0]
values = np.ma.masked_invalid(heat.to_numpy(float))
fig, ax = plt.subplots(figsize=(7.1, 5.8))
ax.set_facecolor("#EEEEEE")
mesh = ax.pcolormesh(x_edges, y_edges, values, shading="flat", vmin=0, vmax=1, cmap="viridis")
if values.count() == 0:
    ax.text(50, 50, "No successful fits", ha="center", va="center", fontsize=12, color="#555555")
ax.set_xlim(-1, 101)
ax.set_ylim(-1, 101)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_yticks(np.arange(0, 101, 20))
ax.set_xlabel("Gaussian noise (%)")
ax.set_ylabel("Uniform redistribution applied after Gaussian noise (%)")
ax.set_title({method!r} + " — mean ARI")
colorbar = fig.colorbar(mesh, ax=ax)
colorbar.solids.set_rasterized(False)
colorbar.set_label("Mean ARI across successful repeats")
fig.tight_layout()
output_path = FIGURES / {slug + '_mean_ari_heatmap.svg'!r}
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)'''


def axis_success_cell(axis: str) -> str:
    if axis == "gaussian":
        filter_text = 'completeness.loc[completeness["uniform_pct"].eq(0)].copy()'
        x_column = "gaussian_pct"
        xlabel = "Gaussian noise (%)"
        title = "Gaussian-only native-fit success"
        stem = "gaussian_only_method_success_fraction"
    else:
        filter_text = 'completeness.loc[completeness["gaussian_pct"].eq(0)].copy()'
        x_column = "uniform_pct"
        xlabel = "Uniform redistribution (%)"
        title = "Uniform-only native-fit success"
        stem = "uniform_only_method_success_fraction"
    return f'''# Plot-specific shaping and full construction: {title}
success_axis = {filter_text}
fig, ax = plt.subplots(figsize=(7.6, 4.9))
for method in METHODS:
    data = success_axis.loc[success_axis["method"].eq(method)].sort_values({x_column!r})
    x = data[{x_column!r}].to_numpy(float)
    y = data["success_fraction"].to_numpy(float)
    ax.plot(x, y, color=COLORS[method], linewidth=2.0, label=method)
ax.set_xlim(0, 100)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel({xlabel!r})
ax.set_ylabel("Successful native fit fraction (of 100 repeats)")
ax.set_title({title!r})
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
output_path = FIGURES / {stem + '.svg'!r}
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)'''


def axis_k_cell(axis: str) -> str:
    if axis == "gaussian":
        filter_text = 'summary.loc[summary["uniform_pct"].eq(0)].copy()'
        x_column = "gaussian_pct"
        xlabel = "Gaussian noise (%)"
        title = "Gaussian-only recovered cluster count"
        stem = "gaussian_only_recovered_k"
    else:
        filter_text = 'summary.loc[summary["gaussian_pct"].eq(0)].copy()'
        x_column = "uniform_pct"
        xlabel = "Uniform redistribution (%)"
        title = "Uniform-only recovered cluster count"
        stem = "uniform_only_recovered_k"
    return f'''# Plot-specific shaping and full construction: {title}
k_axis = {filter_text}
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for method in METHODS:
    data = k_axis.loc[k_axis["method"].eq(method)].sort_values({x_column!r})
    x = data[{x_column!r}].to_numpy(float)
    mean = data["recovered_k_mean"].to_numpy(float)
    lower = data["recovered_k_q025"].to_numpy(float)
    upper = data["recovered_k_q975"].to_numpy(float)
    finite = np.isfinite(mean)
    if finite.any():
        ax.plot(x[finite], mean[finite], color=COLORS[method], linewidth=2.0, label=method)
        band = finite & np.isfinite(lower) & np.isfinite(upper)
        if band.any():
            ax.fill_between(x[band], lower[band], upper[band], color=COLORS[method], alpha=0.14, linewidth=0)
ax.axhline(12, color="#555555", linewidth=1, linestyle="--", label="Reference programs (12)")
ax.set_xlim(0, 100)
ax.set_xlabel({xlabel!r})
ax.set_ylabel("Recovered cluster count")
ax.set_title({title!r})
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
output_path = FIGURES / {stem + '.svg'!r}
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)'''


def axis_sensitivity_cell(axis: str) -> str:
    if axis == "gaussian":
        filter_text = 'same_retained.loc[same_retained["uniform_pct"].eq(0)].copy()'
        x_column = "gaussian_pct"
        xlabel = "Gaussian noise (%)"
        title = "Same retained-set ARI sensitivity — Gaussian only"
        stem = "gaussian_only_same_retained_ari_sensitivity"
    else:
        filter_text = 'same_retained.loc[same_retained["gaussian_pct"].eq(0)].copy()'
        x_column = "uniform_pct"
        xlabel = "Uniform redistribution (%)"
        title = "Same retained-set ARI sensitivity — Uniform only"
        stem = "uniform_only_same_retained_ari_sensitivity"
    return f'''# Plot-specific shaping and full construction: {title}
sensitivity_axis = {filter_text}
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for method in METHODS:
    data = sensitivity_axis.loc[sensitivity_axis["method"].eq(method)].sort_values({x_column!r})
    x = data[{x_column!r}].to_numpy(float)
    mean = data["shared_retained_minus_full_ari_mean"].to_numpy(float)
    lower = data["shared_retained_minus_full_ari_q025"].to_numpy(float)
    upper = data["shared_retained_minus_full_ari_q975"].to_numpy(float)
    finite = np.isfinite(mean)
    if finite.any():
        ax.plot(x[finite], mean[finite], color=COLORS[method], linewidth=2.0, label=method)
        band = finite & np.isfinite(lower) & np.isfinite(upper)
        if band.any():
            ax.fill_between(x[band], lower[band], upper[band], color=COLORS[method], alpha=0.14, linewidth=0)
ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
ax.set_xlim(0, 100)
ax.set_xlabel({xlabel!r})
ax.set_ylabel("ARI on ChronoClone-retained clones minus full-cohort ARI")
ax.set_title({title!r})
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(handles, labels, frameon=False, fontsize=8)
fig.tight_layout()
output_path = FIGURES / {stem + '.svg'!r}
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)'''


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "chronoclone",
            "language": "python",
            "name": "chronoclone",
        },
        "language_info": {"name": "python", "version": "3.11.15"},
        "execution": {
            "environment_manager": "micromamba-compatible existing prefix",
            "environment_name": "chronoclone",
            "environment_path": "/path/to/ChronoClone/.runtime/chronoclone",
            "kernel": "chronoclone",
            "command": (
                "/path/to/ChronoClone/.runtime/chronoclone/bin/python "
                "scripts/execute_notebook.py "
                "notebooks/0003_day0_locked_method_noise_results.ipynb"
            ),
            "status": "not_executed",
        },
    }
    cells = [
        markdown(
            """
# Fixed-t1 five-method noise benchmark

This notebook reads only the strict 100-repeat aggregate produced after all
600 current-hash shards pass validation. Accuracy summaries exclude native
method failures rather than treating failure as accuracy zero. Ribbons are
empirical 2.5th--97.5th percentiles across repeat indices, not confidence
intervals for biological generalization.

The corrected estimand holds t1 fixed, reapplies realized-tensor lineage
eligibility before every fit, passes the same ordered clone IDs to all methods,
and uses the sole inclusive ChronoClone post-fit cutoff `silhouette >= -0.1`.
"""
        ),
        code(
            """
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import xml.etree.ElementTree as ET
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
AGGREGATE = ROOT / "results" / "aggregate"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
ENVIRONMENT = {
    "manager": "micromamba-compatible existing prefix",
    "name": "chronoclone",
    "path": "/path/to/ChronoClone/.runtime/chronoclone",
    "kernel": "chronoclone",
    "python": platform.python_version(),
    "versions": {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "scanpy", "matplotlib", "nbformat", "nbclient"]
    },
    "execution_command": "/path/to/ChronoClone/.runtime/chronoclone/bin/python scripts/execute_notebook.py notebooks/0003_day0_locked_method_noise_results.ipynb",
}
assert ENVIRONMENT["python"] == "3.11.15"
assert Path(ENVIRONMENT["path"]).is_dir()
mpl.rcParams["svg.fonttype"] = "none"
METHODS = ["ChronoClone", "CLADES preprocessing", "Cyclone abundance-only", "K-means", "Ward"]
COLORS = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))
ENVIRONMENT
"""
        ),
        code(
            """
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

metadata = json.loads((AGGREGATE / "aggregation_metadata.json").read_text())
assert metadata["status"] == "pass"
for name, expected in metadata["artifact_sha256"].items():
    assert sha256(AGGREGATE / name) == expected
summary = pd.read_csv(AGGREGATE / "native_method_noise_summary.csv")
completeness = pd.read_csv(AGGREGATE / "result_completeness.csv")
eligibility = pd.read_csv(AGGREGATE / "eligibility_summary.csv")
selective = pd.read_csv(AGGREGATE / "chronoclone_selective_summary.csv")
same_retained = pd.read_csv(AGGREGATE / "same_retained_subset_summary.csv")
failures = pd.read_csv(AGGREGATE / "method_failure_counts.csv")
assert len(summary) == len(completeness) == len(same_retained) == 5 * 51 * 51
assert len(eligibility) == len(selective) == 51 * 51
assert not summary.duplicated(["method", "gaussian_pct", "uniform_pct"]).any()
assert not eligibility.duplicated(["gaussian_pct", "uniform_pct"]).any()
assert (completeness["n_expected"] == 100).all()
assert (completeness["n_success"] + completeness["n_method_failure"] == 100).all()
metadata_display = dict(metadata)
metadata_display["t1_lock_validated"] = metadata_display.pop("day0_lock_validated")
metadata_display.pop("schema", None)
display(pd.DataFrame([metadata_display]).T.rename(columns={0: "value"}))
display(completeness.groupby("method", sort=False)[["n_expected", "n_success", "n_method_failure"]].sum())
"""
        ),
        markdown(
            """
## Full-cohort accuracy

The two axis plots isolate one noise mechanism. Each method heatmap then shows
the complete Gaussian-then-uniform grid on a common 0--1 ARI scale. Gray cells
have no successful native fit and therefore no accuracy estimate.
"""
        ),
        code(axis_ari_cell("gaussian")),
        code(axis_ari_cell("uniform")),
    ]
    for method in METHODS:
        cells.append(code(heatmap_cell(method)))
    cells.extend(
        [
            markdown(
                """
## Applicability, eligibility, and native selection

Success fractions expose method applicability separately from accuracy.
Recovered K is descriptive of each truth-blind native selector. Selection
parameters are not numerically comparable across methods (for example,
ChronoClone consensus threshold versus CLADES resolution versus requested K),
so their axis values are tabulated rather than placed on one scale.
"""
            ),
            code(axis_success_cell("gaussian")),
            code(axis_success_cell("uniform")),
            code(
                """# Plot-specific shaping and full construction: common eligibility on both isolated axes
gaussian_eligibility = eligibility.loc[eligibility["uniform_pct"].eq(0)].sort_values("gaussian_pct").copy()
uniform_eligibility = eligibility.loc[eligibility["gaussian_pct"].eq(0)].sort_values("uniform_pct").copy()
fig, ax = plt.subplots(figsize=(7.6, 4.9))
for label, data, x_column, color in [
    ("Gaussian only", gaussian_eligibility, "gaussian_pct", "#0072B2"),
    ("Uniform only", uniform_eligibility, "uniform_pct", "#D55E00"),
]:
    x = data[x_column].to_numpy(float)
    mean = data["eligible_fraction_mean"].to_numpy(float)
    lower = data["eligible_fraction_q025"].to_numpy(float)
    upper = data["eligible_fraction_q975"].to_numpy(float)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
ax.set_xlim(0, 100)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Noise level (%)")
ax.set_ylabel("Lineage-eligible fraction of input clones")
ax.set_title("Common method cohort after realized-tensor lineage eligibility")
ax.legend(frameon=False)
fig.tight_layout()
output_path = FIGURES / "isolated_axes_eligible_clone_fraction.svg"
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)"""
            ),
            code(axis_k_cell("gaussian")),
            code(axis_k_cell("uniform")),
            code(
                """
selector_axes = pd.concat(
    [
        summary.loc[summary["uniform_pct"].eq(0), ["method", "gaussian_pct", "selection_parameter_mean", "selected_silhouette_mean", "recovered_k_mean"]].assign(axis="Gaussian only", noise_pct=lambda x: x["gaussian_pct"]),
        summary.loc[summary["gaussian_pct"].eq(0), ["method", "uniform_pct", "selection_parameter_mean", "selected_silhouette_mean", "recovered_k_mean"]].assign(axis="Uniform only", noise_pct=lambda x: x["uniform_pct"]),
    ],
    ignore_index=True,
)
selector_endpoints = selector_axes.loc[selector_axes["noise_pct"].isin([0, 10, 20, 50, 100]), ["axis", "noise_pct", "method", "selection_parameter_mean", "selected_silhouette_mean", "recovered_k_mean"]]
display(selector_endpoints.sort_values(["axis", "noise_pct", "method"]))
"""
            ),
            markdown(
                """
## ChronoClone selection and common retained-set sensitivity

ChronoClone conditional accuracy is paired with retained-clone coverage. For
the sensitivity curves, every already-fitted comparator is rescored on the
same ChronoClone-retained IDs without refitting. The plotted difference is a
within-method denominator sensitivity, not a filtered-ChronoClone superiority
contrast.
"""
            ),
            code(
                """# Plot-specific shaping and full construction: ChronoClone post-fit coverage
gaussian_coverage = selective.loc[selective["uniform_pct"].eq(0)].sort_values("gaussian_pct").copy()
uniform_coverage = selective.loc[selective["gaussian_pct"].eq(0)].sort_values("uniform_pct").copy()
fig, ax = plt.subplots(figsize=(7.6, 4.9))
for label, data, x_column, color in [
    ("Gaussian only", gaussian_coverage, "gaussian_pct", "#0072B2"),
    ("Uniform only", uniform_coverage, "uniform_pct", "#D55E00"),
]:
    x = data[x_column].to_numpy(float)
    mean = data["postfit_retained_fraction_of_eligible_mean"].to_numpy(float)
    lower = data["postfit_retained_fraction_of_eligible_q025"].to_numpy(float)
    upper = data["postfit_retained_fraction_of_eligible_q975"].to_numpy(float)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
ax.set_xlim(0, 100)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Noise level (%)")
ax.set_ylabel("Fraction retained after silhouette >= -0.1")
ax.set_title("ChronoClone post-fit clone coverage")
ax.legend(frameon=False)
fig.tight_layout()
output_path = FIGURES / "chronoclone_isolated_axes_postfit_coverage.svg"
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)"""
            ),
            code(axis_sensitivity_cell("gaussian")),
            code(axis_sensitivity_cell("uniform")),
            code(
                """
selective_endpoints = pd.concat(
    [
        selective.loc[selective["uniform_pct"].eq(0)].assign(axis="Gaussian only", noise_pct=lambda x: x["gaussian_pct"]),
        selective.loc[selective["gaussian_pct"].eq(0)].assign(axis="Uniform only", noise_pct=lambda x: x["uniform_pct"]),
    ],
    ignore_index=True,
)
selective_columns = [
    "axis", "noise_pct", "postfit_retained_fraction_of_eligible_mean",
    "shared_retained_subset_ari_mean", "postfit_retained_k_mean",
    "postfit_clusters_lost_mean", "n_postfilter_ok", "n_postfilter_all_abstained",
    "n_postfilter_insufficient", "n_postfilter_other",
]
display(selective_endpoints.loc[selective_endpoints["noise_pct"].isin([0, 10, 20, 50, 100]), selective_columns].sort_values(["axis", "noise_pct"]))
failure_totals = completeness.groupby("method", sort=False)[["n_success", "n_method_failure"]].sum().reset_index()
display(failure_totals)
if len(failures):
    display(failures.groupby(["method", "error_type"], as_index=False)["count"].sum())
else:
    display(pd.DataFrame({"status": ["No native method failures"]}))
"""
            ),
            markdown(
                """
## Scope limits

- CLADES denotes its published mouse preprocessing partition, not its
  NeuralODE/Gillespie stages.
- Cyclone is abundance-only here; its failures are applicability outcomes.
- K-means and Ward use a predeclared truth-blind silhouette K grid.
- The five methods retain their native feature/selection rules, so the result
  is not an isolated comparison of clustering machinery.
- The deterministic 12-program generator is a technical-noise stress test,
  not evidence of real-data biological generality.
"""
            ),
            code(
                """
expected_figures = {
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
svg_rows = []
for name in sorted(expected_figures):
    path = FIGURES / name
    root = ET.parse(path).getroot()
    tags = [element.tag.rsplit("}", 1)[-1] for element in root.iter()]
    svg_rows.append({
        "figure": name,
        "exists": path.is_file(),
        "has_live_text": "text" in tags,
        "has_embedded_raster": "image" in tags,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    })
svg_validation = pd.DataFrame(svg_rows)
assert len(svg_validation) == 15
assert svg_validation["exists"].all()
assert svg_validation["has_live_text"].all()
assert not svg_validation["has_embedded_raster"].any()
display(svg_validation)
"""
            ),
        ]
    )
    notebook.cells = cells
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


def main() -> int:
    metadata_path = ROOT / "results" / "aggregate" / "aggregation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Run scripts/aggregate_results.py first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "pass":
        raise RuntimeError("Strict aggregation did not pass")
    build_notebook()
    subprocess.run([sys.executable, str(EXECUTOR), str(NOTEBOOK)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(VALIDATOR)], cwd=ROOT, check=True)
    print(NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
