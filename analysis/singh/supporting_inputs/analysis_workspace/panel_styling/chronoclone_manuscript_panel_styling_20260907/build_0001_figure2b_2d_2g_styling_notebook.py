#!/usr/bin/env python3
"""Build the intentionally unexecuted Figure 2B/2D-2G styling notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path("/path/to/ChronoClone")
ANALYSIS_DIR = ROOT / "analysis_workspace/analyses/chronoclone_manuscript_panel_styling_20260907"
NOTEBOOK_PATH = ANALYSIS_DIR / "0001_figure2b_2d_2g_publication_styling.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str, *, tags=None, output_stem=None):
    metadata = {}
    if tags:
        metadata["tags"] = tags
    if output_stem:
        metadata["output_stem"] = output_stem
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n", metadata=metadata)


cells = [
    markdown(
        """
        # Publication styling rebuild for Figure 2B and Figure 2D–2G

        This notebook rebuilds only the manuscript panels that compare the
        neutrophil- and monocyte-output Weinreb metaclones. It changes layout,
        typography, and metaclone-label colours while preserving the exported
        coordinates, local-density values, differentiated-output fractions,
        pseudotime observations, medians, and Kruskal–Wallis tests.

        The source tables under `reference_workspace/` are read-only. No scientific
        analysis is re-fit, and no protected source file is modified.

        **Build status:** constructed on 2026-09-07 and intentionally left
        unexecuted at the authors' request. The exact execution command is recorded
        below for the next validation step.
        """
    ),
    markdown(
        """
        ## Environment and execution

        - Environment manager: existing conda-compatible prefix; do not mutate
        - Environment name: `chronoclone`
        - Absolute MacBook path: `/path/to/ChronoClone/.runtime/chronoclone`
        - Jupyter kernel: `chronoclone`
        - Python: 3.11.15
        - Observed package versions at build time: NumPy 2.4.6, pandas 2.3.3,
          Matplotlib 3.11.0, SciPy 1.17.1, Pillow 12.2.0, nbformat 5.10.4
        - Exact execution command from the workspace root:

          `NUMBA_CACHE_DIR=/private/tmp/chronoclone_numba_cache MPLCONFIGDIR=/private/tmp/chronoclone_mpl /path/to/ChronoClone/.runtime/chronoclone/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=chronoclone --ExecutePreprocessor.timeout=7200 analysis_workspace/analyses/chronoclone_manuscript_panel_styling_20260907/0001_figure2b_2d_2g_publication_styling.ipynb`
        """
    ),
    markdown(
        """
        ## Context and chart contract

        - **Figure 2B:** one integrated, stacked Neu/Mono density map at a
          manuscript-width canvas. The 27 small maps retain the exported
          same-metaclone/same-day count among 100 nearest assigned cells, with
          the viridis scale normalized independently within each metaclone.
        - **Figure 2D and 2F:** horizontal 100%-composition bars; cell-output
          category colours are unchanged, while MC tick labels use the central
          manuscript palette.
        - **Figure 2E and 2G:** grouped cell-level Palantir-pseudotime violins;
          MC order, cell values, medians, and within-day Kruskal–Wallis tests
          are unchanged. Each panel has its own separate MC-key stem.
        - All panels use live SVG text. Dense UMAP marks in Figure 2B are
          intentionally rasterized inside the SVG; the remaining panels are
          fully vector.
        - Each plot is constructed, saved as one SVG stem plus a same-stem PNG,
          and displayed in one dedicated code cell.
        """
    ),
    code(
        """
        # Setup: paths, imports, environment identity, and global rendering policy.
        from pathlib import Path
        import hashlib
        import platform
        import sys
        import xml.etree.ElementTree as ET

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import MultipleLocator
        import nbformat
        import numpy as np
        import pandas as pd
        import PIL
        from IPython.display import display
        from scipy import __version__ as scipy_version
        from scipy.stats import kruskal

        ROOT = Path("/path/to/ChronoClone")
        ANALYSIS_DIR = ROOT / "analysis_workspace/analyses/chronoclone_manuscript_panel_styling_20260907"
        NOTEBOOK_PATH = ANALYSIS_DIR / "0001_figure2b_2d_2g_publication_styling.ipynb"
        FIGURE_DIR = ANALYSIS_DIR / "figures"
        TABLE_DIR = ANALYSIS_DIR / "tables"
        REPORT_DIR = ANALYSIS_DIR / "reports"
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        ENV_NAME = "chronoclone"
        ENV_PATH = Path("/path/to/ChronoClone/.runtime/chronoclone")
        EXPECTED_PYTHON = ENV_PATH / "bin/python"
        assert EXPECTED_PYTHON.exists(), EXPECTED_PYTHON
        assert Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), sys.executable

        matplotlib.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "Arial",
            "font.size": 9,
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#172033",
            "axes.linewidth": 0.8,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "text.color": "#172033",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        })

        print(f"Environment: {ENV_NAME} ({ENV_PATH})")
        print(f"Kernel: chronoclone; executable: {sys.executable}")
        print(f"Python: {platform.python_version()}")
        print(
            "Versions: "
            f"numpy={np.__version__}; pandas={pd.__version__}; "
            f"matplotlib={matplotlib.__version__}; scipy={scipy_version}; "
            f"Pillow={PIL.__version__}; nbformat={nbformat.__version__}"
        )
        """
    ),
    markdown("## Data and immutable source provenance"),
    code(
        """
        # Load the immutable exported plotting tables and verify every locked hash.
        INPUT_PATHS = {
            "manuscript_metaclone_palette": TABLE_DIR / "manuscript_metaclone_palette.csv",
            "figure_2b_focal_density": ROOT / "analysis_workspace/analyses/weinreb_clone_trajectory_umap_20260810/tables/weinreb_cell_state_umap_neutrophil_monocyte_metaclone_density_by_day_source.csv",
            "figure_2b_background": ROOT / "analysis_workspace/analyses/weinreb_metaclone_cohesion_power_20260819/tables/iseh2026_registered_cell_state_umap_source.csv",
            "figure_2d_neutrophil_composition": ROOT / "reference_workspace/Figure/fig2/data/split/2e_neu_output_composition.csv",
            "figure_2e_neutrophil_pseudotime": ROOT / "reference_workspace/Figure/fig2/data/fig2d_pseudotime_by_mc_day.csv",
            "figure_2e_neutrophil_kruskal": ROOT / "reference_workspace/Figure/fig2/data/fig2d_perday_kruskal.csv",
            "figure_2f_monocyte_composition": ROOT / "reference_workspace/Figure/fig2/data/split/2f_mono_output_composition.csv",
            "figure_2g_monocyte_pseudotime": ROOT / "reference_workspace/Figure/fig2/data/fig2e_pseudotime_by_mc_day.csv",
            "figure_2g_monocyte_kruskal": ROOT / "reference_workspace/Figure/fig2/data/fig2e_perday_kruskal.csv",
        }
        EXPECTED_SHA256 = {
            "manuscript_metaclone_palette": "8606561063d2d1c0326d9084881c91f24700799211067238822676e9cff84433",
            "figure_2b_focal_density": "7854f3fbacb67c7e9296de8713f0b2c10799e21af57dc1e9c5bbe6b6a0a1e438",
            "figure_2b_background": "4c5c532303ced39451a1de3bae67e6d75eb46358ee9450c798ff337526e86c49",
            "figure_2d_neutrophil_composition": "4fd701e86733409e9f471bbb42e18bb165f0220e4f022b72fe254821a198af3c",
            "figure_2e_neutrophil_pseudotime": "b308be25f765be25cf1bb969964689bdb02dd3a00d7af1e84a4893c0277dc74c",
            "figure_2e_neutrophil_kruskal": "2e2b84c1a61e0df2ff153ffd62e67b2c71eda9c85a7d7290b875edd91ac677b5",
            "figure_2f_monocyte_composition": "2d00df36ddd3983a29dd08c2829148b4bbdd24e4cad3ba122d845b088d2c124d",
            "figure_2g_monocyte_pseudotime": "1767fb6b86317938ce1b03f025c50b0ff356a37a5a636afbc2c93f8b91048129",
            "figure_2g_monocyte_kruskal": "23bcbb366405621439e8547a14b0fc639c0fe4b8619457444561f5656a5b6992",
        }
        SOURCE_NOTEBOOKS = {
            "figure_2b_executed_source": ROOT / "analysis_workspace/analyses/weinreb_clone_trajectory_umap_20260810/0001_weinreb_clone_trajectory_umap_comparison.ipynb",
            "figure_2d_to_2g_executed_source": ROOT / "reference_workspace/notebooks/Figure2_klein_pace.ipynb",
        }
        EXPECTED_SOURCE_NOTEBOOK_SHA256 = {
            "figure_2b_executed_source": "4c63cb763ecb5ed6280ce822db43442e353efe95b6f727d562d5f99fd1214112",
            "figure_2d_to_2g_executed_source": "08833f0796dd972a544e19bf82ca722ac964a48aad748470784468b1fa4bd4ec",
        }

        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        source_hash_rows = []
        for source_name, source_path in {**INPUT_PATHS, **SOURCE_NOTEBOOKS}.items():
            observed_hash = sha256_file(source_path)
            expected_hash = (
                EXPECTED_SHA256.get(source_name)
                or EXPECTED_SOURCE_NOTEBOOK_SHA256[source_name]
            )
            assert observed_hash == expected_hash, (source_name, observed_hash, expected_hash)
            source_hash_rows.append({
                "source": source_name,
                "path": str(source_path.relative_to(ROOT)),
                "bytes": source_path.stat().st_size,
                "sha256": observed_hash,
            })
        source_hashes = pd.DataFrame(source_hash_rows)
        source_hashes.to_csv(TABLE_DIR / "input_source_hashes.csv", index=False)

        palette_table = pd.read_csv(INPUT_PATHS["manuscript_metaclone_palette"])
        density_focal = pd.read_csv(INPUT_PATHS["figure_2b_focal_density"])
        density_background = pd.read_csv(INPUT_PATHS["figure_2b_background"])
        neu_composition = pd.read_csv(INPUT_PATHS["figure_2d_neutrophil_composition"])
        neu_pseudotime = pd.read_csv(INPUT_PATHS["figure_2e_neutrophil_pseudotime"])
        neu_kruskal = pd.read_csv(INPUT_PATHS["figure_2e_neutrophil_kruskal"])
        mono_composition = pd.read_csv(INPUT_PATHS["figure_2f_monocyte_composition"])
        mono_pseudotime = pd.read_csv(INPUT_PATHS["figure_2g_monocyte_pseudotime"])
        mono_kruskal = pd.read_csv(INPUT_PATHS["figure_2g_monocyte_kruskal"])

        palette_table["metaclone"] = palette_table["metaclone"].astype(int)
        MC_COLOR = dict(zip(palette_table["metaclone"], palette_table["color"]))
        NEU_MCS = [1, 22, 9, 16, 14]
        MONO_MCS = [6, 8, 7, 12]
        DAYS = [2, 4, 6]
        OUTPUT_ORDER = ["Neu", "Mono", "Ery/Mk", "Baso", "Mast", "Other"]
        OUTPUT_COLOR = {
            "Neu": "#E0913C",
            "Mono": "#B23A48",
            "Ery/Mk": "#2E7DBF",
            "Baso": "#3F9E4D",
            "Mast": "#8E6FB8",
            "Other": "#5B6472",
        }
        INK = "#172033"
        MID_GREY = "#5B6472"
        LIGHT_GREY = "#D8DDE3"
        GRID_GREY = "#E8EBEF"

        assert palette_table["metaclone"].is_unique
        assert set(NEU_MCS + MONO_MCS).issubset(MC_COLOR)
        assert not any(str(path).startswith(str(ROOT / "data/quarantine")) for path in INPUT_PATHS.values())
        display(source_hashes)
        """
    ),
    markdown("## Results"),
    markdown("### Figure 2B — time-resolved metaclone occupancy"),
    code(
        """
        # Figure 2B: plot-specific shaping, stacked Neu/Mono layout, export, and display.
        panel_2b_stem = FIGURE_DIR / "figure_2b_neu_mono_cell_state_density_stacked"
        panel_2b_focal = density_focal.loc[
            density_focal["metaclone"].astype(int).isin(NEU_MCS + MONO_MCS)
            & density_focal["day"].astype(int).isin(DAYS)
        ].copy()
        panel_2b_focal["metaclone"] = panel_2b_focal["metaclone"].astype(int)
        panel_2b_focal["day"] = panel_2b_focal["day"].astype(int)
        panel_2b_background = density_background[["cell_id", "UMAP1", "UMAP2"]].copy()
        panel_2b_background = panel_2b_background.drop_duplicates("cell_id", keep="first")
        assert len(panel_2b_background) == 9714
        assert len(panel_2b_focal) == 2805
        assert panel_2b_focal["cell_id"].is_unique

        x_limits = (
            float(panel_2b_background["UMAP1"].min() - 0.8),
            float(panel_2b_background["UMAP1"].max() + 0.8),
        )
        y_limits = (
            float(panel_2b_background["UMAP2"].min() - 0.8),
            float(panel_2b_background["UMAP2"].max() + 0.8),
        )
        clone_counts = (
            panel_2b_focal.groupby("metaclone", observed=True)["n_mc_clones_total"]
            .first().astype(int).to_dict()
        )
        density_vmax = (
            panel_2b_focal.groupby("metaclone", observed=True)["density_vmax_within_mc"]
            .first().astype(float).to_dict()
        )
        family_blocks = [
            ("Neutrophil-output metaclones", NEU_MCS),
            ("Monocyte-output metaclones", MONO_MCS),
        ]

        figure_2b = plt.figure(figsize=(7.2, 7.8), facecolor="white")
        outer_grid = figure_2b.add_gridspec(
            2, 1, height_ratios=[1.0, 1.0], hspace=0.33,
            left=0.10, right=0.995, top=0.94, bottom=0.025,
        )
        block_axes = []
        background_x = panel_2b_background["UMAP1"].to_numpy(float)
        background_y = panel_2b_background["UMAP2"].to_numpy(float)

        for block_index, (family_label, metaclones) in enumerate(family_blocks):
            inner_grid = outer_grid[block_index].subgridspec(
                3, len(metaclones), wspace=0.02, hspace=-0.04
            )
            axes_for_block = np.empty((3, len(metaclones)), dtype=object)
            for row_index, day in enumerate(DAYS):
                for column_index, metaclone in enumerate(metaclones):
                    axis = figure_2b.add_subplot(inner_grid[row_index, column_index])
                    axes_for_block[row_index, column_index] = axis
                    axis.scatter(
                        background_x, background_y,
                        s=2.2, c=LIGHT_GREY, linewidths=0,
                        rasterized=True, zorder=0,
                    )
                    panel_points = panel_2b_focal.loc[
                        panel_2b_focal["metaclone"].eq(metaclone)
                        & panel_2b_focal["day"].eq(day)
                    ].sort_values("local_same_mc_day_neighbors", kind="stable")
                    axis.scatter(
                        panel_points["UMAP1"].to_numpy(float),
                        panel_points["UMAP2"].to_numpy(float),
                        c=panel_points["local_same_mc_day_neighbors"].to_numpy(float),
                        cmap="viridis", vmin=0, vmax=density_vmax[metaclone],
                        s=15, alpha=0.82, edgecolors="#111827", linewidths=0.12,
                        rasterized=True, zorder=2,
                    )
                    axis.set_aspect("equal")
                    axis.set_xlim(*x_limits)
                    axis.set_ylim(*y_limits)
                    axis.axis("off")
                    if column_index == 0:
                        axis.text(
                            -0.08, 0.50, f"Day {day}", transform=axis.transAxes,
                            ha="right", va="center", fontsize=9.5,
                            fontweight="bold", color=INK,
                        )
                    if row_index == 0:
                        axis.text(
                            0.50, 1.22, f"MC{metaclone}", transform=axis.transAxes,
                            ha="center", va="bottom", fontsize=10.5,
                            fontweight="bold", color=MC_COLOR[metaclone],
                        )
                        axis.text(
                            0.50, 1.06, f"n={clone_counts[metaclone]} clones",
                            transform=axis.transAxes, ha="center", va="bottom",
                            fontsize=7.4, color=MID_GREY,
                        )
            block_axes.append((family_label, axes_for_block))

        figure_2b.canvas.draw()
        for family_label, axes_for_block in block_axes:
            left_edge = axes_for_block[0, 0].get_position().x0
            right_edge = axes_for_block[0, -1].get_position().x1
            top_edge = axes_for_block[0, 0].get_position().y1
            figure_2b.text(
                (left_edge + right_edge) / 2, top_edge + 0.065, family_label,
                ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=INK,
            )

        figure_2b.savefig(
            panel_2b_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04
        )
        figure_2b.savefig(
            panel_2b_stem.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.04
        )
        display(figure_2b)
        plt.close(figure_2b)
        print(panel_2b_stem.with_suffix(".svg"))
        print(panel_2b_stem.with_suffix(".png"))
        """,
        tags=["plot-cell", "figure-2b"],
        output_stem="figure_2b_neu_mono_cell_state_density_stacked",
    ),
    markdown("### Figure 2D — differentiated output of neutrophil-output metaclones"),
    code(
        """
        # Figure 2D: exact source fractions, publication composition bars, export, and display.
        panel_2d_stem = FIGURE_DIR / "figure_2d_neutrophil_output_composition"
        panel_2d_data = (
            neu_composition.assign(metaclone=lambda frame: frame["metaclone"].astype(int))
            .set_index("metaclone").loc[NEU_MCS, OUTPUT_ORDER].copy()
        )
        assert np.allclose(panel_2d_data.sum(axis=1).to_numpy(float), 1.0)
        y_positions = np.arange(len(NEU_MCS))[::-1]
        left_edge = np.zeros(len(NEU_MCS), dtype=float)

        figure_2d, axis_2d = plt.subplots(figsize=(5.2, 3.0), layout="constrained")
        shown_categories = []
        for output_category in OUTPUT_ORDER:
            values = panel_2d_data[output_category].to_numpy(float)
            axis_2d.barh(
                y_positions, values, left=left_edge,
                color=OUTPUT_COLOR[output_category], edgecolor="white",
                linewidth=0.55, height=0.68,
            )
            left_edge = left_edge + values
            if np.any(values > 0.02):
                shown_categories.append(output_category)

        axis_2d.set_yticks(y_positions)
        axis_2d.set_yticklabels([f"MC{metaclone}" for metaclone in NEU_MCS])
        for tick_label, metaclone in zip(axis_2d.get_yticklabels(), NEU_MCS):
            tick_label.set_color(MC_COLOR[metaclone])
            tick_label.set_fontsize(10.5)
            tick_label.set_fontweight("bold")
        axis_2d.set_xlim(0, 1)
        axis_2d.xaxis.set_major_locator(MultipleLocator(0.25))
        axis_2d.set_xlabel("Differentiated output fraction", fontsize=9.5)
        axis_2d.tick_params(axis="x", labelsize=8.5)
        axis_2d.tick_params(axis="y", length=0, pad=5)
        axis_2d.spines[["top", "right", "left"]].set_visible(False)
        axis_2d.grid(axis="x", color=GRID_GREY, linewidth=0.6, zorder=0)
        axis_2d.set_axisbelow(True)
        composition_handles = [
            Patch(facecolor=OUTPUT_COLOR[category], edgecolor="none", label=category)
            for category in shown_categories
        ]
        axis_2d.legend(
            handles=composition_handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.22), ncol=len(composition_handles),
            frameon=False, fontsize=8.2, handlelength=1.2,
            columnspacing=1.0,
        )

        figure_2d.savefig(
            panel_2d_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04
        )
        figure_2d.savefig(
            panel_2d_stem.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.04
        )
        display(figure_2d)
        plt.close(figure_2d)
        print(panel_2d_stem.with_suffix(".svg"))
        print(panel_2d_stem.with_suffix(".png"))
        """,
        tags=["plot-cell", "figure-2d"],
        output_stem="figure_2d_neutrophil_output_composition",
    ),
    markdown("### Figure 2E — neutrophil-branch developmental position"),
    code(
        """
        # Figure 2E: exact cell values and tests, grouped violins, export, and display.
        panel_2e_stem = FIGURE_DIR / "figure_2e_neutrophil_branch_pseudotime"
        panel_2e_data = neu_pseudotime.loc[
            neu_pseudotime["metaclone"].astype(int).isin(NEU_MCS)
            & neu_pseudotime["day"].astype(int).isin(DAYS)
        ].copy()
        panel_2e_data["metaclone"] = panel_2e_data["metaclone"].astype(int)
        panel_2e_data["day"] = panel_2e_data["day"].astype(int)
        panel_2e_stats = neu_kruskal.set_index(neu_kruskal["day"].astype(int)).loc[DAYS].copy()
        violin_width = 0.80 / len(NEU_MCS)

        figure_2e, axis_2e = plt.subplots(figsize=(7.0, 3.7), layout="constrained")
        for day_index, day in enumerate(DAYS):
            for mc_index, metaclone in enumerate(NEU_MCS):
                values = panel_2e_data.loc[
                    panel_2e_data["day"].eq(day)
                    & panel_2e_data["metaclone"].eq(metaclone), "pt"
                ].dropna().to_numpy(float)
                assert len(values) >= 3
                x_position = day_index + (mc_index - (len(NEU_MCS) - 1) / 2) * violin_width
                violin = axis_2e.violinplot(
                    [values], positions=[x_position], widths=violin_width * 0.92,
                    showmedians=True, showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(MC_COLOR[metaclone])
                    body.set_edgecolor(MC_COLOR[metaclone])
                    body.set_linewidth(0.65)
                    body.set_alpha(0.86)
                violin["cmedians"].set_color("#111827")
                violin["cmedians"].set_linewidth(1.1)

            p_value = float(panel_2e_stats.loc[day, "p"])
            exponent = int(np.floor(np.log10(p_value)))
            mantissa = p_value / (10 ** exponent)
            axis_2e.text(
                day_index, 1.075, rf"KW $P={mantissa:.1f}\\times10^{{{exponent}}}$",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=INK,
            )

        axis_2e.set_xticks(range(len(DAYS)))
        axis_2e.set_xticklabels([f"Day {day}" for day in DAYS], fontsize=9.5)
        axis_2e.set_ylabel("Palantir pseudotime", fontsize=9.5)
        axis_2e.set_ylim(0, 1.17)
        axis_2e.set_xlim(-0.50, len(DAYS) - 0.50)
        axis_2e.yaxis.set_major_locator(MultipleLocator(0.2))
        axis_2e.tick_params(axis="y", labelsize=8.5)
        axis_2e.spines[["top", "right"]].set_visible(False)
        axis_2e.grid(axis="y", color=GRID_GREY, linewidth=0.6)
        axis_2e.set_axisbelow(True)

        figure_2e.savefig(
            panel_2e_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04
        )
        figure_2e.savefig(
            panel_2e_stem.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.04
        )
        display(figure_2e)
        plt.close(figure_2e)
        print(panel_2e_stem.with_suffix(".svg"))
        print(panel_2e_stem.with_suffix(".png"))
        """,
        tags=["plot-cell", "figure-2e"],
        output_stem="figure_2e_neutrophil_branch_pseudotime",
    ),
    code(
        """
        # Figure 2E key: exact MC order and central colors, separate export and display.
        panel_2e_legend_stem = FIGURE_DIR / "figure_2e_neutrophil_branch_pseudotime_legend"
        panel_2e_legend_rows = (
            palette_table.set_index("metaclone").loc[NEU_MCS, ["label", "color"]].copy()
        )
        panel_2e_legend_handles = [
            Line2D(
                [0], [0], marker="s", linestyle="none", markersize=8,
                markerfacecolor=row["color"], markeredgecolor=row["color"],
                label=row["label"],
            )
            for _, row in panel_2e_legend_rows.iterrows()
        ]

        figure_2e_legend = plt.figure(figsize=(7.0, 0.62), facecolor="white")
        figure_2e_legend.legend(
            handles=panel_2e_legend_handles, loc="center", ncol=len(NEU_MCS),
            frameon=False, fontsize=9.5, handletextpad=0.45,
            columnspacing=1.35,
        )
        figure_2e_legend.savefig(
            panel_2e_legend_stem.with_suffix(".svg"),
            bbox_inches="tight", pad_inches=0.03
        )
        figure_2e_legend.savefig(
            panel_2e_legend_stem.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.03
        )
        display(figure_2e_legend)
        plt.close(figure_2e_legend)
        print(panel_2e_legend_stem.with_suffix(".svg"))
        print(panel_2e_legend_stem.with_suffix(".png"))
        """,
        tags=["plot-cell", "figure-2e", "legend"],
        output_stem="figure_2e_neutrophil_branch_pseudotime_legend",
    ),
    markdown("### Figure 2F — differentiated output of monocyte-output metaclones"),
    code(
        """
        # Figure 2F: exact source fractions, publication composition bars, export, and display.
        panel_2f_stem = FIGURE_DIR / "figure_2f_monocyte_output_composition"
        panel_2f_data = (
            mono_composition.assign(metaclone=lambda frame: frame["metaclone"].astype(int))
            .set_index("metaclone").loc[MONO_MCS, OUTPUT_ORDER].copy()
        )
        assert np.allclose(panel_2f_data.sum(axis=1).to_numpy(float), 1.0)
        y_positions = np.arange(len(MONO_MCS))[::-1]
        left_edge = np.zeros(len(MONO_MCS), dtype=float)

        figure_2f, axis_2f = plt.subplots(figsize=(5.2, 2.7), layout="constrained")
        shown_categories = []
        for output_category in OUTPUT_ORDER:
            values = panel_2f_data[output_category].to_numpy(float)
            axis_2f.barh(
                y_positions, values, left=left_edge,
                color=OUTPUT_COLOR[output_category], edgecolor="white",
                linewidth=0.55, height=0.68,
            )
            left_edge = left_edge + values
            if np.any(values > 0.02):
                shown_categories.append(output_category)

        axis_2f.set_yticks(y_positions)
        axis_2f.set_yticklabels([f"MC{metaclone}" for metaclone in MONO_MCS])
        for tick_label, metaclone in zip(axis_2f.get_yticklabels(), MONO_MCS):
            tick_label.set_color(MC_COLOR[metaclone])
            tick_label.set_fontsize(10.5)
            tick_label.set_fontweight("bold")
        axis_2f.set_xlim(0, 1)
        axis_2f.xaxis.set_major_locator(MultipleLocator(0.25))
        axis_2f.set_xlabel("Differentiated output fraction", fontsize=9.5)
        axis_2f.tick_params(axis="x", labelsize=8.5)
        axis_2f.tick_params(axis="y", length=0, pad=5)
        axis_2f.spines[["top", "right", "left"]].set_visible(False)
        axis_2f.grid(axis="x", color=GRID_GREY, linewidth=0.6, zorder=0)
        axis_2f.set_axisbelow(True)
        composition_handles = [
            Patch(facecolor=OUTPUT_COLOR[category], edgecolor="none", label=category)
            for category in shown_categories
        ]
        axis_2f.legend(
            handles=composition_handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.25), ncol=len(composition_handles),
            frameon=False, fontsize=8.2, handlelength=1.2,
            columnspacing=1.0,
        )

        figure_2f.savefig(
            panel_2f_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04
        )
        figure_2f.savefig(
            panel_2f_stem.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.04
        )
        display(figure_2f)
        plt.close(figure_2f)
        print(panel_2f_stem.with_suffix(".svg"))
        print(panel_2f_stem.with_suffix(".png"))
        """,
        tags=["plot-cell", "figure-2f"],
        output_stem="figure_2f_monocyte_output_composition",
    ),
    markdown("### Figure 2G — monocyte-branch developmental position"),
    code(
        """
        # Figure 2G: exact cell values and tests, grouped violins, export, and display.
        panel_2g_stem = FIGURE_DIR / "figure_2g_monocyte_branch_pseudotime"
        panel_2g_data = mono_pseudotime.loc[
            mono_pseudotime["metaclone"].astype(int).isin(MONO_MCS)
            & mono_pseudotime["day"].astype(int).isin(DAYS)
        ].copy()
        panel_2g_data["metaclone"] = panel_2g_data["metaclone"].astype(int)
        panel_2g_data["day"] = panel_2g_data["day"].astype(int)
        panel_2g_stats = mono_kruskal.set_index(mono_kruskal["day"].astype(int)).loc[DAYS].copy()
        violin_width = 0.80 / len(MONO_MCS)

        figure_2g, axis_2g = plt.subplots(figsize=(7.0, 3.7), layout="constrained")
        for day_index, day in enumerate(DAYS):
            for mc_index, metaclone in enumerate(MONO_MCS):
                values = panel_2g_data.loc[
                    panel_2g_data["day"].eq(day)
                    & panel_2g_data["metaclone"].eq(metaclone), "pt"
                ].dropna().to_numpy(float)
                assert len(values) >= 3
                x_position = day_index + (mc_index - (len(MONO_MCS) - 1) / 2) * violin_width
                violin = axis_2g.violinplot(
                    [values], positions=[x_position], widths=violin_width * 0.92,
                    showmedians=True, showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(MC_COLOR[metaclone])
                    body.set_edgecolor(MC_COLOR[metaclone])
                    body.set_linewidth(0.65)
                    body.set_alpha(0.86)
                violin["cmedians"].set_color("#111827")
                violin["cmedians"].set_linewidth(1.1)

            p_value = float(panel_2g_stats.loc[day, "p"])
            exponent = int(np.floor(np.log10(p_value)))
            mantissa = p_value / (10 ** exponent)
            axis_2g.text(
                day_index, 1.075, rf"KW $P={mantissa:.1f}\\times10^{{{exponent}}}$",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=INK,
            )

        axis_2g.set_xticks(range(len(DAYS)))
        axis_2g.set_xticklabels([f"Day {day}" for day in DAYS], fontsize=9.5)
        axis_2g.set_ylabel("Palantir pseudotime", fontsize=9.5)
        axis_2g.set_ylim(0, 1.17)
        axis_2g.set_xlim(-0.50, len(DAYS) - 0.50)
        axis_2g.yaxis.set_major_locator(MultipleLocator(0.2))
        axis_2g.tick_params(axis="y", labelsize=8.5)
        axis_2g.spines[["top", "right"]].set_visible(False)
        axis_2g.grid(axis="y", color=GRID_GREY, linewidth=0.6)
        axis_2g.set_axisbelow(True)

        figure_2g.savefig(
            panel_2g_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04
        )
        figure_2g.savefig(
            panel_2g_stem.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.04
        )
        display(figure_2g)
        plt.close(figure_2g)
        print(panel_2g_stem.with_suffix(".svg"))
        print(panel_2g_stem.with_suffix(".png"))
        """,
        tags=["plot-cell", "figure-2g"],
        output_stem="figure_2g_monocyte_branch_pseudotime",
    ),
    code(
        """
        # Figure 2G key: exact MC order and central colors, separate export and display.
        panel_2g_legend_stem = FIGURE_DIR / "figure_2g_monocyte_branch_pseudotime_legend"
        panel_2g_legend_rows = (
            palette_table.set_index("metaclone").loc[MONO_MCS, ["label", "color"]].copy()
        )
        panel_2g_legend_handles = [
            Line2D(
                [0], [0], marker="s", linestyle="none", markersize=8,
                markerfacecolor=row["color"], markeredgecolor=row["color"],
                label=row["label"],
            )
            for _, row in panel_2g_legend_rows.iterrows()
        ]

        figure_2g_legend = plt.figure(figsize=(7.0, 0.62), facecolor="white")
        figure_2g_legend.legend(
            handles=panel_2g_legend_handles, loc="center", ncol=len(MONO_MCS),
            frameon=False, fontsize=9.5, handletextpad=0.45,
            columnspacing=1.55,
        )
        figure_2g_legend.savefig(
            panel_2g_legend_stem.with_suffix(".svg"),
            bbox_inches="tight", pad_inches=0.03
        )
        figure_2g_legend.savefig(
            panel_2g_legend_stem.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.03
        )
        display(figure_2g_legend)
        plt.close(figure_2g_legend)
        print(panel_2g_legend_stem.with_suffix(".svg"))
        print(panel_2g_legend_stem.with_suffix(".png"))
        """,
        tags=["plot-cell", "figure-2g", "legend"],
        output_stem="figure_2g_monocyte_branch_pseudotime_legend",
    ),
    markdown("## Validation"),
    code(
        """
        # Validate source integrity, quantitative preservation, plot-cell structure, and exports.
        import ast
        from PIL import Image

        validation_rows = []

        def record(check, passed, detail):
            validation_rows.append({"check": check, "passed": bool(passed), "detail": str(detail)})

        # Immutable inputs and source notebooks.
        observed_hashes = {
            name: sha256_file(path) for name, path in {**INPUT_PATHS, **SOURCE_NOTEBOOKS}.items()
        }
        expected_hashes = {**EXPECTED_SHA256, **EXPECTED_SOURCE_NOTEBOOK_SHA256}
        record(
            "all source hashes locked",
            observed_hashes == expected_hashes,
            f"{len(observed_hashes)} sources",
        )

        # Figure 2B source-table invariants.
        record(
            "Figure 2B exact focal universe",
            len(panel_2b_focal) == 2805
            and panel_2b_focal["cell_id"].is_unique
            and panel_2b_focal["LARRY_ID"].nunique() == 244
            and set(panel_2b_focal["metaclone"]) == set(NEU_MCS + MONO_MCS)
            and set(panel_2b_focal["day"]) == set(DAYS),
            "2,805 focal cells; 244 clones; 9 metaclones; 3 days",
        )
        record(
            "Figure 2B density metric unchanged",
            panel_2b_focal["n_neighbors"].eq(100).all()
            and panel_2b_focal["local_same_mc_day_neighbors"].between(1, 100).all()
            and np.allclose(
                panel_2b_focal["local_same_mc_day_fraction"].to_numpy(float),
                panel_2b_focal["local_same_mc_day_neighbors"].to_numpy(float) / 100.0,
            )
            and panel_2b_focal.groupby("metaclone", observed=True)["density_vmax_within_mc"].nunique().eq(1).all(),
            "same-MC/day count among 100 neighbours; one scale per MC",
        )
        record(
            "Figure 2B exact background universe",
            len(panel_2b_background) == 9714
            and panel_2b_background["cell_id"].is_unique
            and np.isfinite(panel_2b_background[["UMAP1", "UMAP2"]].to_numpy(float)).all(),
            "9,714 unique background cells",
        )

        # Composition values and order.
        record(
            "Figure 2D composition preserved",
            panel_2d_data.index.tolist() == NEU_MCS
            and np.allclose(panel_2d_data.sum(axis=1).to_numpy(float), 1.0),
            f"order={NEU_MCS}; rows sum to 1",
        )
        record(
            "Figure 2F composition preserved",
            panel_2f_data.index.tolist() == MONO_MCS
            and np.allclose(panel_2f_data.sum(axis=1).to_numpy(float), 1.0),
            f"order={MONO_MCS}; rows sum to 1",
        )

        # Recompute the source Kruskal-Wallis statistics from the exact plotted observations.
        for panel_label, plotted_data, plotted_stats, metaclones in [
            ("Figure 2E", panel_2e_data, panel_2e_stats, NEU_MCS),
            ("Figure 2G", panel_2g_data, panel_2g_stats, MONO_MCS),
        ]:
            statistic_matches = []
            for day in DAYS:
                samples = [
                    plotted_data.loc[
                        plotted_data["day"].eq(day)
                        & plotted_data["metaclone"].eq(metaclone), "pt"
                    ].dropna().to_numpy(float)
                    for metaclone in metaclones
                ]
                statistic, p_value = kruskal(*samples)
                statistic_matches.append(
                    np.isclose(round(float(statistic), 2), float(plotted_stats.loc[day, "H"]), atol=0.005)
                    and np.isclose(float(p_value), float(plotted_stats.loc[day, "p"]), rtol=1e-12, atol=0.0)
                )
            record(
                f"{panel_label} tests reproduced",
                all(statistic_matches),
                "three within-day Kruskal-Wallis H and P values",
            )

        # One dedicated code cell and one output stem for every plot.
        notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        plot_cells = [
            cell for cell in notebook.cells
            if cell.cell_type == "code" and "plot-cell" in cell.metadata.get("tags", [])
        ]
        expected_output_stems = [
            "figure_2b_neu_mono_cell_state_density_stacked",
            "figure_2d_neutrophil_output_composition",
            "figure_2e_neutrophil_branch_pseudotime",
            "figure_2e_neutrophil_branch_pseudotime_legend",
            "figure_2f_monocyte_output_composition",
            "figure_2g_monocyte_branch_pseudotime",
            "figure_2g_monocyte_branch_pseudotime_legend",
        ]
        plot_cell_checks = []
        for plot_cell in plot_cells:
            syntax_tree = ast.parse(plot_cell.source)
            figure_calls = [
                node for node in ast.walk(syntax_tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "plt"
                and node.func.attr in {"figure", "subplots"}
            ]
            save_calls = [
                node for node in ast.walk(syntax_tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "savefig"
            ]
            function_definitions = [
                node for node in ast.walk(syntax_tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            plot_cell_checks.append(
                len(figure_calls) == 1
                and len(save_calls) == 2
                and '.with_suffix(".svg")' in plot_cell.source
                and '.with_suffix(".png")' in plot_cell.source
                and "display(" in plot_cell.source
                and not function_definitions
            )
        record(
            "one plot cell and one stem per panel",
            len(plot_cells) == 7
            and [cell.metadata.get("output_stem") for cell in plot_cells] == expected_output_stems
            and all(plot_cell_checks),
            "7 plot cells; one figure, one SVG+PNG stem, inline construction and display",
        )

        # Export, text, palette, raster/vector, and resolution checks.
        stem_to_mcs = {
            "figure_2b_neu_mono_cell_state_density_stacked": NEU_MCS + MONO_MCS,
            "figure_2d_neutrophil_output_composition": NEU_MCS,
            "figure_2e_neutrophil_branch_pseudotime": NEU_MCS,
            "figure_2e_neutrophil_branch_pseudotime_legend": NEU_MCS,
            "figure_2f_monocyte_output_composition": MONO_MCS,
            "figure_2g_monocyte_branch_pseudotime": MONO_MCS,
            "figure_2g_monocyte_branch_pseudotime_legend": MONO_MCS,
        }
        export_checks = []
        for stem_name, metaclones in stem_to_mcs.items():
            svg_path = FIGURE_DIR / f"{stem_name}.svg"
            png_path = FIGURE_DIR / f"{stem_name}.png"
            svg_text = svg_path.read_text(encoding="utf-8").lower()
            svg_root = ET.parse(svg_path).getroot()
            png_width, png_height = Image.open(png_path).size
            expected_colors_present = all(MC_COLOR[metaclone].lower() in svg_text for metaclone in metaclones)
            is_density_panel = stem_name.startswith("figure_2b_")
            is_compact_legend = stem_name.endswith("_legend")
            raster_policy_ok = ("<image" in svg_text) if is_density_panel else ("<image" not in svg_text)
            minimum_png_height = 150 if is_compact_legend else 300
            export_checks.append(
                svg_root.tag.endswith("svg")
                and "<text" in svg_text
                and expected_colors_present
                and raster_policy_ok
                and png_width >= 1000
                and png_height >= minimum_png_height
            )
        record(
            "publication exports",
            len(export_checks) == 7 and all(export_checks),
            "7 live-text SVGs + 600-dpi PNGs; only Figure 2B contains rasterized dense marks",
        )

        validation = pd.DataFrame(validation_rows)
        validation.to_csv(REPORT_DIR / "validation_checks.csv", index=False)
        assert validation["passed"].all(), validation.loc[~validation["passed"]]

        artifact_paths = sorted(FIGURE_DIR.glob("*")) + [
            TABLE_DIR / "input_source_hashes.csv",
            REPORT_DIR / "validation_checks.csv",
        ]
        artifact_manifest = pd.DataFrame({
            "path": [str(path.relative_to(ROOT)) for path in artifact_paths],
            "bytes": [path.stat().st_size for path in artifact_paths],
            "sha256": [sha256_file(path) for path in artifact_paths],
        })
        artifact_manifest.to_csv(REPORT_DIR / "artifact_manifest.csv", index=False)

        display(validation)
        display(artifact_manifest)
        print(f"Validation checks passed: {int(validation['passed'].sum())}/{len(validation)}")
        print(REPORT_DIR / "validation_checks.csv")
        print(REPORT_DIR / "artifact_manifest.csv")
        """
    ),
    markdown(
        """
        ## Takeaways

        Populate this section only after the notebook has been executed and
        every standalone SVG/PNG has been inspected at its manuscript size.
        Until then, treat the notebook as a reproducible build specification,
        not a validated result release.
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (chronoclone)",
            "language": "python",
            "name": "chronoclone",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.15",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "chronoclone": {
            "analysis_scope": "Figure 2B and Figure 2D-2G publication styling only",
            "build_date": "2026-09-07",
            "execution_status": "constructed_unexecuted_by_explicit_instruction",
            "environment_name": "chronoclone",
            "environment_path": "/path/to/ChronoClone/.runtime/chronoclone",
        },
    },
)

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
