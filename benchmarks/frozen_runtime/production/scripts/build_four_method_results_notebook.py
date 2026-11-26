#!/usr/bin/env python3
"""Create a four-method reader-facing copy of executed notebook 0003."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "0003_day0_locked_method_noise_results.ipynb"
TARGET = ROOT / "notebooks" / "0005_day0_locked_four_method_noise_results.ipynb"

PLOTTED_METHODS = (
    "ChronoClone",
    "CLADES preprocessing",
    "K-means",
    "Ward",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def success_plot_source(axis: str) -> str:
    if axis == "gaussian":
        filter_expression = 'completeness_plot.loc[completeness_plot["uniform_pct"].eq(0)].copy()'
        x_column = "gaussian_pct"
        xlabel = "Gaussian noise (%)"
        title = "Gaussian-only native-fit success (four methods)"
        output_name = "gaussian_only_method_success_fraction.svg"
    elif axis == "uniform":
        filter_expression = 'completeness_plot.loc[completeness_plot["gaussian_pct"].eq(0)].copy()'
        x_column = "uniform_pct"
        xlabel = "Uniform redistribution (%)"
        title = "Uniform-only native-fit success (four methods)"
        output_name = "uniform_only_method_success_fraction.svg"
    else:  # pragma: no cover - builder-internal contract
        raise ValueError(axis)

    return f'''# Plot-specific shaping and full construction: {title}
success_axis = {filter_expression}
assert set(success_axis["method"]) == set(METHODS)
assert success_axis["success_fraction"].eq(1.0).all()
success_by_noise = (
    success_axis.groupby({x_column!r}, as_index=False)
    .agg(
        minimum_success_fraction=("success_fraction", "min"),
        maximum_success_fraction=("success_fraction", "max"),
    )
    .sort_values({x_column!r})
)
assert np.allclose(
    success_by_noise["minimum_success_fraction"],
    success_by_noise["maximum_success_fraction"],
)
x = success_by_noise[{x_column!r}].to_numpy(float)
y = success_by_noise["minimum_success_fraction"].to_numpy(float)
fig, ax = plt.subplots(figsize=(7.6, 4.9))
ax.plot(x, y, color="#4D4D4D", linewidth=2.0, label="All four plotted methods")
ax.set_xlim(0, 100)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel({xlabel!r})
ax.set_ylabel("Successful native fit fraction (of 100 repeats)")
ax.set_title({title!r})
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
output_path = FIGURES / {output_name!r}
fig.savefig(output_path, format="svg", bbox_inches="tight")
plt.show()
print(output_path)'''


def data_loading_source() -> str:
    return '''def sha256(path):
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

# The aggregate remains the complete five-method source. Reader-facing tables
# and plots below use only the four estimator-compatible methods requested here.
summary_plot = summary.loc[summary["method"].isin(METHODS)].copy()
completeness_plot = completeness.loc[completeness["method"].isin(METHODS)].copy()
same_retained_plot = same_retained.loc[same_retained["method"].isin(METHODS)].copy()
failures_plot = failures.loc[failures["method"].isin(METHODS)].copy()
assert set(summary_plot["method"]) == set(METHODS)
assert len(summary_plot) == len(completeness_plot) == len(same_retained_plot) == 4 * 51 * 51
assert completeness_plot["n_success"].eq(100).all()
assert completeness_plot["n_method_failure"].eq(0).all()
assert failures_plot.empty

metadata_display = dict(metadata)
metadata_display["t1_lock_validated"] = metadata_display.pop("day0_lock_validated")
metadata_display.pop("schema", None)
metadata_display.pop("method_failure_rows", None)
metadata_display["plotted_methods"] = ", ".join(METHODS)
metadata_display["plotted_method_failure_rows"] = int(completeness_plot["n_method_failure"].sum())
display(pd.DataFrame([metadata_display]).T.rename(columns={0: "value"}))
display(completeness_plot.groupby("method", sort=False)[["n_expected", "n_success", "n_method_failure"]].sum())'''


def scope_markdown() -> str:
    return '''## Scope limits

- CLADES denotes its published mouse preprocessing partition, not its
  NeuralODE/Gillespie stages.
- K-means and Ward use a predeclared truth-blind silhouette K grid.
- The four plotted methods retain their native feature/selection rules, so the
  result is not an isolated comparison of clustering machinery.
- The deterministic 12-program generator is a technical-noise stress test,
  not evidence of real-data biological generality.
- The abundance-only Cyclone core is deliberately excluded from every plot in
  this copy because the separate executed applicability audit shows that the
  state-composition benchmark does not provide it with an identifiable target.
  Its completed rows remain preserved in the source aggregate and are not
  recoded as accuracy zero.'''


def final_validation_source() -> str:
    return '''expected_figures = {
    "gaussian_only_mean_ari_by_method.svg",
    "uniform_only_mean_ari_by_method.svg",
    "chronoclone_mean_ari_heatmap.svg",
    "clades_preprocessing_mean_ari_heatmap.svg",
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
    svg_text = path.read_text(encoding="utf-8")
    svg_rows.append({
        "figure": name,
        "exists": path.is_file(),
        "has_live_text": "text" in tags,
        "has_embedded_raster": "image" in tags,
        "contains_excluded_method_label": "Cyclone" in svg_text,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    })
svg_validation = pd.DataFrame(svg_rows)
assert len(svg_validation) == 14
assert svg_validation["exists"].all()
assert svg_validation["has_live_text"].all()
assert not svg_validation["has_embedded_raster"].any()
assert not svg_validation["contains_excluded_method_label"].any()
display(svg_validation)'''


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_hash = sha256_file(SOURCE)
    notebook = nbformat.read(SOURCE, as_version=4)
    transformed_cells = []

    for source_index, original_cell in enumerate(notebook.cells):
        if (
            original_cell.cell_type == "code"
            and "Cyclone abundance-only mean-ARI heatmap" in original_cell.source
        ):
            continue

        cell = copy.deepcopy(original_cell)
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

        if source_index == 0:
            cell.source = '''# Fixed-t1 four-method noise benchmark

This reader-facing copy of notebook 0003 reads the same strict 100-repeat
aggregate but excludes the abundance-only Cyclone core from every plot. The
separate executed notebook `0004_cyclone_applicability_diagnostic.ipynb`
documents why that core is not an estimator-compatible accuracy comparator for
this state-composition simulation.

Accuracy ribbons are empirical 2.5th--97.5th percentiles across repeat indices,
not confidence intervals for biological generalization. The corrected estimand
holds t1 fixed, reapplies realized-tensor lineage eligibility before every fit,
passes the same ordered clone IDs to all included methods, and uses the sole
inclusive ChronoClone post-fit cutoff `silhouette >= -0.1`.'''
        elif source_index == 1:
            cell.source = cell.source.replace(
                'FIGURES = ROOT / "figures"',
                'FIGURES = ROOT / "figures_four_method"',
            )
            cell.source = cell.source.replace(
                '"manager": "micromamba-compatible existing prefix"',
                '"manager": "micromamba (existing prefix reused without mutation)"',
            )
            cell.source = cell.source.replace(
                "notebooks/0003_day0_locked_method_noise_results.ipynb",
                "notebooks/0005_day0_locked_four_method_noise_results.ipynb",
            )
            cell.source = cell.source.replace(
                'METHODS = ["ChronoClone", "CLADES preprocessing", "Cyclone abundance-only", "K-means", "Ward"]',
                'METHODS = ["ChronoClone", "CLADES preprocessing", "K-means", "Ward"]',
            )
            cell.source = cell.source.replace(
                'COLORS = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))',
                'COLORS = {"ChronoClone": "#0072B2", "CLADES preprocessing": "#D55E00", "K-means": "#CC79A7", "Ward": "#E69F00"}',
            )
        elif source_index == 2:
            cell.source = data_loading_source()
        elif source_index == 3:
            cell.source = '''## Full-cohort accuracy

The two axis plots isolate one noise mechanism. Each of the four included
method heatmaps then shows the complete Gaussian-then-uniform grid on a common
0--1 ARI scale.'''
        elif source_index in {4, 5, 6, 7, 9, 10, 15, 16}:
            cell.source = cell.source.replace("summary.loc[", "summary_plot.loc[")
        elif source_index == 11:
            cell.source = '''## Applicability, eligibility, and native selection

All four plotted methods return a finite native partition in every requested
condition. The success plots therefore show their shared 100% fit fraction as
one neutral curve rather than four perfectly overlapping method lines.
Recovered K remains descriptive of each truth-blind native selector. Selection
parameters are not numerically comparable across methods, so their values are
tabulated rather than placed on one scale.'''
        elif source_index == 12:
            cell.source = success_plot_source("gaussian")
        elif source_index == 13:
            cell.source = success_plot_source("uniform")
        elif source_index == 17:
            cell.source = cell.source.replace("summary.loc[", "summary_plot.loc[")
        elif source_index in {20, 21}:
            cell.source = cell.source.replace(
                "same_retained.loc[", "same_retained_plot.loc["
            )
        elif source_index == 22:
            cell.source = cell.source.replace(
                'failure_totals = completeness.groupby("method", sort=False)',
                'failure_totals = completeness_plot.groupby("method", sort=False)',
            )
            cell.source = cell.source.replace("if len(failures):", "if len(failures_plot):")
            cell.source = cell.source.replace(
                'display(failures.groupby(["method", "error_type"], as_index=False)["count"].sum())',
                'display(failures_plot.groupby(["method", "error_type"], as_index=False)["count"].sum())',
            )
            cell.source = cell.source.replace(
                'display(pd.DataFrame({"status": ["No native method failures"]}))',
                'display(pd.DataFrame({"status": ["No native method failures among the four plotted methods"]}))',
            )
        elif source_index == 23:
            cell.source = scope_markdown()
        elif source_index == 24:
            cell.source = final_validation_source()

        # Publication styling is applied only to this reader-facing display
        # notebook.  The quantitative 0003 aggregate and all estimands remain
        # unchanged.  Panels are built close to their 6.5-inch manuscript
        # width so labels are not shrunk after placement.
        if cell.cell_type == "code":
            cell.source = cell.source.replace(
                'mpl.rcParams["svg.fonttype"] = "none"',
                'mpl.rcParams.update({"svg.fonttype": "none", "font.size": 9.5, '
                '"axes.labelsize": 9.5, "xtick.labelsize": 8.5, '
                '"ytick.labelsize": 8.5, "legend.fontsize": 8.0})',
            )
            for old, new in {
                "figsize=(7.6, 5.2)": "figsize=(6.8, 4.45)",
                "figsize=(7.1, 5.8)": "figsize=(6.8, 5.15)",
                "figsize=(7.6, 4.9)": "figsize=(6.8, 4.25)",
                "figsize=(7.6, 5.0)": "figsize=(6.8, 4.35)",
            }.items():
                cell.source = cell.source.replace(old, new)

            # Captions already identify the isolated-axis and diagnostic
            # panels, so their internal titles only consumed plotting area.
            for title_line in (
                "ax.set_title('Gaussian noise only')\n",
                "ax.set_title('Uniform redistribution only')\n",
                'ax.set_title("Common method cohort after realized-tensor lineage eligibility")\n',
                'ax.set_title("ChronoClone post-fit clone coverage")\n',
            ):
                cell.source = cell.source.replace(title_line, "")

            if source_index == 14:
                cell.source = cell.source.replace(
                    "ax.set_ylim(-0.03, 1.03)",
                    "ax.set_ylim(0.95, 1.002)\n"
                    "ax.set_yticks(np.arange(0.95, 1.001, 0.01))\n"
                    "ax.text(0.99, 0.06, 'expanded y-axis', transform=ax.transAxes, "
                    "ha='right', va='bottom', fontsize=7.5, color='#5B6472')",
                )
            elif source_index == 19:
                cell.source = cell.source.replace(
                    "ax.set_ylim(-0.03, 1.03)",
                    "ax.set_ylim(0.90, 1.002)\n"
                    "ax.set_yticks(np.arange(0.90, 1.001, 0.02))\n"
                    "ax.text(0.99, 0.06, 'expanded y-axis', transform=ax.transAxes, "
                    "ha='right', va='bottom', fontsize=7.5, color='#5B6472')",
                )

            # Heatmaps retain only a compact method tag for standalone panel
            # identification, rather than a repeated descriptive title.
            heatmap_labels = {
                "ax.set_title('ChronoClone' + \" — mean ARI\")": "ChronoClone",
                "ax.set_title('CLADES preprocessing' + \" — mean ARI\")": "CLADES preprocessing",
                "ax.set_title('K-means' + \" — mean ARI\")": "K-means",
                "ax.set_title('Ward' + \" — mean ARI\")": "Ward",
            }
            for old, label in heatmap_labels.items():
                cell.source = cell.source.replace(
                    old,
                    f'ax.text(0.02, 0.98, {label!r}, transform=ax.transAxes, '
                    'ha="left", va="top", fontsize=9.5, fontweight="bold", '
                    'color="#26235F", bbox=dict(facecolor="white", '
                    'edgecolor="none", alpha=0.82, pad=2.0))',
                )

            # Keep a high-resolution companion for document assembly while
            # preserving the editable SVG as the primary artifact.
            cell.source = cell.source.replace(
                'fig.savefig(output_path, format="svg", bbox_inches="tight")',
                'fig.savefig(output_path, format="svg", bbox_inches="tight")\n'
                'fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")',
            )

        transformed_cells.append(cell)

    notebook.cells = transformed_cells
    notebook.metadata["execution"] = {"status": "not_executed"}
    notebook.metadata["derivation"] = {
        "source_notebook": str(SOURCE.relative_to(ROOT)),
        "source_notebook_sha256": source_hash,
        "change": "Cyclone abundance-only excluded from every plot; outputs isolated",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, TARGET)
    print(TARGET)
    print(f"source_sha256={source_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
