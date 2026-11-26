#!/usr/bin/env python3
"""Build the Weinreb Day-2 HSPC metaclone gene-program notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

from cnmf_append_cells import build_cnmf_cells


ROOT = Path("/path/to/ChronoClone")
ANALYSIS = ROOT / "analysis_workspace/analyses/weinreb_day2_hspc_metaclone_gene_programs_20260902"
OUT = ANALYSIS / "0001_weinreb_day2_hspc_metaclone_gene_programs.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    md(
        r"""
        # Weinreb Day-2 HSPC gene programs and targeted metaclonal signatures

        This analysis uses every Day-2 cell assigned to a current Weinreb
        metaclone whose source label is `prog_1` or `prog_2`. Those labels are
        collapsed into one HSPC compartment before eligibility, aggregation,
        modeling, program scoring, or plotting. Each LARRY clone contributes one
        recovered-count pseudobulk, so nested cells are not treated as independent
        replicates.

        The original three comparison tiers are retained as broad context:

        1. every metaclone versus the equal-metaclone mean of the other 19;
        2. pooled lineage-bias families, including all pairwise family contrasts;
        3. each metaclone versus peers in the same lineage-bias family, including
           every within-family pair.

        The manuscript-facing primary analysis tests four locked, non-exclusive
        Day-6 behavioral axes within the seven Multi-lineage metaclones: Baso-rich,
        Neu-rich, Mono-rich, and persistent-undifferentiated output. Each axis is
        tested as an equal-metaclone focal pair versus the equal-metaclone mean of
        the other five, with both member effects, max-P conjunction, comparator-drop,
        and cell-removal/library-refit sensitivities. The broad 1,450-estimable
        correlation screen is retained only as a supplementary sensitivity.

        The fixed program universe is MSigDB Hallmark 2020. Program scores are
        unweighted means of gene-wise standardized log-normalized expression in
        clone pseudobulks. Gene-level negative-binomial contrasts and preranked
        Hallmark enrichment are retained as supporting evidence. All results are
        retrospective associations in the realized current partition, not causal
        or prospective predictors of later fate.
        """
    ),
    md(
        r"""
        ## tl;dr

        Computed findings are printed from live objects in the computed-findings section rather
        than hard-coded into notebook Markdown. The principal tables and figures are
        created before that summary and validated at the end of the run.
        """
    ),
    md(
        r"""
        ## Context & Methods

        ### Reproducible environment

        - Environment manager/name: existing Conda-compatible environment,
          `chronoclone` (reused without mutation)
        - Absolute MacBook-local path:
          `/path/to/ChronoClone/.runtime/chronoclone`
        - Jupyter kernel: `chronoclone`
        - Python: 3.11.15
        - Exact execution command:

          `/path/to/ChronoClone/.runtime/chronoclone/bin/jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=chronoclone --ExecutePreprocessor.timeout=7200 /path/to/ChronoClone/analysis_workspace/analyses/weinreb_day2_hspc_metaclone_gene_programs_20260902/0001_weinreb_day2_hspc_metaclone_gene_programs.ipynb`

        The next cell verifies the executable, kernel registration, and package
        versions. No package is installed or changed by this notebook.
        """
    ),
    code(
        r"""
        import os
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/research_weinreb_programs_mpl")
        os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/research_weinreb_programs_numba")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        import hashlib
        import importlib.metadata as im
        import itertools
        import json
        import shutil
        import sys
        import warnings
        from pathlib import Path
        from textwrap import fill

        import anndata as ad
        import gseapy as gp
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import nbformat
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp
        import statsmodels.api as sm
        from IPython.display import display
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Patch

        PROJECT_ROOT = Path("/path/to/ChronoClone")
        ANALYSIS_DIR = PROJECT_ROOT / "analysis_workspace/analyses/weinreb_day2_hspc_metaclone_gene_programs_20260902"
        FIGDIR = ANALYSIS_DIR / "figures"
        TABLEDIR = ANALYSIS_DIR / "tables"
        REPORTDIR = ANALYSIS_DIR / "reports"
        RESOURCEDIR = ANALYSIS_DIR / "resources"
        for directory in (FIGDIR, TABLEDIR, REPORTDIR, RESOURCEDIR):
            directory.mkdir(parents=True, exist_ok=True)

        NOTEBOOK_PATH = ANALYSIS_DIR / "0001_weinreb_day2_hspc_metaclone_gene_programs.ipynb"
        BUILDER_PATH = ANALYSIS_DIR / "scripts/build_notebook.py"
        INPUT_H5AD = PROJECT_ROOT / "reference_workspace/data/klein_metaclone_palantir_trajectories.h5ad"
        LINEAGE_MAP_PATH = PROJECT_ROOT / "reference_workspace/Figure/fig2/data/fig2h_lineage_bias_map.csv"
        HALLMARK_SOURCE = PROJECT_ROOT / "reference_workspace/data/gene_sets/MSigDB_Hallmark_2020.gmt"
        HALLMARK_LOCAL = RESOURCEDIR / "MSigDB_Hallmark_2020.gmt"

        ENV_PATH = Path("/path/to/ChronoClone/.runtime/chronoclone")
        PYTHON_PATH = ENV_PATH / "bin/python"
        JUPYTER_PATH = ENV_PATH / "bin/jupyter"
        NBCONVERT_PATH = ENV_PATH / "bin/jupyter-nbconvert"
        KERNEL_JSON = Path("~/Library/Jupyter/kernels/chronoclone/kernel.json")
        EXECUTION_COMMAND = (
            f"{NBCONVERT_PATH} --to notebook --execute --inplace "
            f"--ExecutePreprocessor.kernel_name=chronoclone --ExecutePreprocessor.timeout=7200 {NOTEBOOK_PATH}"
        )
        assert (
            PYTHON_PATH.exists() and JUPYTER_PATH.exists()
            and NBCONVERT_PATH.exists() and KERNEL_JSON.exists()
        )
        assert Path(os.path.realpath(sys.executable)) == Path(os.path.realpath(PYTHON_PATH))

        packages = [
            "adjustText", "anndata", "gseapy", "matplotlib", "nbclient", "nbformat", "numpy",
            "pandas", "pydeseq2", "scipy", "statsmodels"
        ]
        versions = {package: im.version(package) for package in packages}
        environment = pd.DataFrame({
            "field": [
                "manager/name", "environment_path", "python", "kernel",
                "kernel_json", "python_version", "versions", "execution_command"
            ],
            "value": [
                "existing Conda-compatible / chronoclone (reused; not mutated)",
                str(ENV_PATH), sys.executable, "chronoclone", str(KERNEL_JSON),
                sys.version.split()[0], json.dumps(versions, sort_keys=True), EXECUTION_COMMAND
            ]
        })
        environment.to_csv(REPORTDIR / "environment.csv", index=False)
        display(environment)

        EXPECTED_INPUT_SHA256 = "4cb08c12dc70f5c3789a75a09a6399db06b3f6efe18147b9f2cca0908ba93911"
        EXPECTED_LINEAGE_MAP_SHA256 = "b4d1af7b481a9b1a26d781feb449fb88d300d770c20633054e152f213fa18eb6"
        EXPECTED_HALLMARK_SHA256 = "4275592957a1587652092bb398cf77216fde5b8daa2aedaa0e016f7d10bbdb81"
        HSPC_SOURCE_LABELS = ["prog_1", "prog_2"]
        FAMILY_GROUPS = {
            "Neutrophil": [9, 14, 16, 22, 1],
            "Monocyte": [12, 7, 8, 6],
            "Erythroid / Mk": [10, 17, 23],
            "Basophil": [13],
            "Multi-lineage": [0, 2, 5, 3, 18, 4, 11],
        }
        FAMILY_KEY = {
            "Neutrophil": "Neutrophil", "Monocyte": "Monocyte",
            "Erythroid / Mk": "Erythroid_Mk", "Basophil": "Basophil",
            "Multi-lineage": "Multi_lineage",
        }
        FAMILY_DISPLAY = {value: key for key, value in FAMILY_KEY.items()}
        MC_ORDER = [mc for members in FAMILY_GROUPS.values() for mc in members]
        FAMILY_ORDER = list(FAMILY_GROUPS)
        FAMILY_KEY_ORDER = [FAMILY_KEY[name] for name in FAMILY_ORDER]
        MC_TO_FAMILY = {mc: family for family, members in FAMILY_GROUPS.items() for mc in members}
        MC_TO_FAMILY_KEY = {mc: FAMILY_KEY[family] for mc, family in MC_TO_FAMILY.items()}

        FAMILY_COLORS = {
            "Neutrophil": "#E0913C", "Monocyte": "#B23A48",
            "Erythroid / Mk": "#2E7DBF", "Basophil": "#3F9E4D",
            "Multi-lineage": "#7A8190",
        }
        INK = "#17324D"
        ORANGE = "#E0913C"
        GREY = "#68717D"
        LIGHT_GREY = "#D8DDE3"
        PALE = "#F4F1EA"
        WHITE = "#FFFFFF"
        PROGRAM_CMAP = LinearSegmentedColormap.from_list(
            "program_navy_neutral_orange", [INK, PALE, ORANGE]
        )
        mpl.rcParams.update({
            "figure.facecolor": WHITE, "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE, "font.family": "DejaVu Sans",
            "svg.fonttype": "none", "axes.edgecolor": GREY,
            "axes.labelcolor": INK, "text.color": INK,
            "xtick.color": GREY, "ytick.color": GREY,
        })

        def sha256(path, chunk=1024 * 1024):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(chunk), b""):
                    digest.update(block)
            return digest.hexdigest()

        def bh_adjust(values):
            p = np.asarray(values, dtype=float)
            q = np.full(len(p), np.nan, dtype=float)
            ok = np.isfinite(p)
            if not ok.any():
                return q
            pv = p[ok]
            order = np.argsort(pv, kind="mergesort")
            adjusted = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
            adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
            restored = np.empty(len(pv), dtype=float)
            restored[order] = np.clip(adjusted, 0, 1)
            q[ok] = restored
            return q

        def safe_name(value):
            return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")

        def clean_program_name(value):
            return str(value).replace("HALLMARK_", "").replace("_", " ").title()
        """
    ),
    md(
        r"""
        ### Key assumptions and interpretation boundaries

        - The current executed Figure-2 lineage map is controlling. Its manual
          five-family grouping classifies MC0 and MC5 as Multi-lineage, unlike an
          older automatic dominant-output threshold. The discrepancy is audited,
          not silently reconciled.
        - Metaclone and lineage-family labels are outcome-informed summaries of the
          same longitudinal clones. Day-2 differences are retrospective correlates.
        - Family-level clone models pool member metaclones and therefore estimate a
          clone-weighted contrast within this realized partition; metaclones are not
          independent biological replicates of a universal family.
        - The Basophil family has one metaclone and has no within-family contrast.
        - Stored expression is log-normalized and has no independent integer layer.
          Recovered counts use the already audited integer-lattice reconstruction;
          its assumption is recorded explicitly.
        """
    ),
    md(
        r"""
        ## Data

        The active corrected 9,714-cell trajectory object and current Figure-2
        lineage map are read-only. `obs["population"]` is removed immediately and
        is never used for filtering, adjustment, selection, summarization, export,
        or interpretation.
        """
    ),
    code(
        r"""
        input_hash = sha256(INPUT_H5AD)
        lineage_map_hash = sha256(LINEAGE_MAP_PATH)
        assert input_hash == EXPECTED_INPUT_SHA256
        assert lineage_map_hash == EXPECTED_LINEAGE_MAP_SHA256

        A = ad.read_h5ad(INPUT_H5AD)
        A.obs = A.obs.drop(columns=["population"], errors="ignore")
        assert "population" not in A.obs.columns
        obs = A.obs.copy()
        obs["metaclone"] = pd.to_numeric(obs["metaclone"], errors="coerce")
        obs["cell_type"] = obs["cell_type"].astype(str)
        obs["Library"] = obs["Library"].astype(str)
        obs["LARRY_ID"] = obs["LARRY_ID"].astype(str)
        obs["day"] = pd.to_numeric(obs["day"], errors="coerce")
        POS = pd.Series(np.arange(A.n_obs), index=A.obs_names)

        lineage_map = pd.read_csv(LINEAGE_MAP_PATH)
        lineage_map["metaclone"] = lineage_map["metaclone"].astype(int)
        asserted_map = lineage_map.set_index("metaclone")["family"].to_dict()
        assert asserted_map == MC_TO_FAMILY
        assert lineage_map.metaclone.tolist() == MC_ORDER

        old_auto = []
        top_to_family = {
            "Neu": "Neutrophil", "Mono": "Monocyte", "Ery/Mk": "Erythroid / Mk",
            "Baso": "Basophil", "Mast": "Multi-lineage", "Other": "Multi-lineage"
        }
        for row in lineage_map.itertuples(index=False):
            legacy_family = top_to_family.get(str(row.top_output), "Multi-lineage")
            if float(row.top_frac) < 0.60:
                legacy_family = "Multi-lineage"
            old_auto.append({
                "metaclone": int(row.metaclone), "current_family": str(row.family),
                "legacy_threshold_family": legacy_family,
                "top_output": str(row.top_output), "top_fraction": float(row.top_frac),
                "classification_differs": legacy_family != str(row.family),
            })
        lineage_rule_sensitivity = pd.DataFrame(old_auto)
        lineage_rule_sensitivity.to_csv(TABLEDIR / "lineage_bias_rule_sensitivity.csv", index=False)

        input_audit = pd.DataFrame([
            ["input_path", str(INPUT_H5AD)],
            ["input_sha256", input_hash],
            ["lineage_map_path", str(LINEAGE_MAP_PATH)],
            ["lineage_map_sha256", lineage_map_hash],
            ["shape", f"{A.n_obs} x {A.n_vars}"],
            ["population_removed_before_analysis", True],
            ["n_current_metaclones", len(MC_ORDER)],
            ["n_current_lineage_families", len(FAMILY_ORDER)],
            ["legacy_threshold_disagreements", int(lineage_rule_sensitivity.classification_differs.sum())],
        ], columns=["check", "value"])
        input_audit.to_csv(TABLEDIR / "input_provenance_and_scope_audit.csv", index=False)
        display(input_audit)
        display(lineage_rule_sensitivity[lineage_rule_sensitivity.classification_differs])
        """
    ),
    code(
        r"""
        expression = A.X.tocsr().copy() if sp.issparse(A.X) else sp.csr_matrix(A.X)
        expression.data = np.expm1(expression.data)
        row_nnz = np.diff(expression.indptr)
        assert (row_nnz > 0).all()
        row_min = np.minimum.reduceat(expression.data, expression.indptr[:-1])
        primitive = expression.data / np.repeat(row_min, row_nnz)
        COUNTS = expression.copy()
        COUNTS.data = np.rint(primitive).astype(np.int64)
        reconstruction_error = float(np.max(np.abs(primitive - COUNTS.data)))
        expm1_row_sums = np.asarray(expression.sum(axis=1)).ravel()
        recovered_row_sums = np.asarray(COUNTS.sum(axis=1)).ravel()
        assert reconstruction_error < 1e-6
        assert np.all(COUNTS.data >= 0)
        assert np.ptp(expm1_row_sums) < 1e-6

        count_audit = pd.DataFrame([
            ["native_integer_layer", False],
            ["integer_recovery_assumption", "smallest positive value per cell represents one UMI"],
            ["max_integer_lattice_error", reconstruction_error],
            ["expm1_row_sum_min", float(expm1_row_sums.min())],
            ["expm1_row_sum_max", float(expm1_row_sums.max())],
            ["recovered_library_min", int(recovered_row_sums.min())],
            ["recovered_library_median", float(np.median(recovered_row_sums))],
            ["recovered_library_max", int(recovered_row_sums.max())],
        ], columns=["check", "value"])
        count_audit.to_csv(TABLEDIR / "count_recovery_audit.csv", index=False)
        display(count_audit)
        """
    ),
    code(
        r"""
        all_day2_hspc = obs[
            (obs.day == 2) & obs.cell_type.isin(HSPC_SOURCE_LABELS)
        ].copy()
        hspc_cells = all_day2_hspc[
            all_day2_hspc.metaclone.isin(MC_ORDER)
        ].copy()
        hspc_cells["metaclone"] = hspc_cells.metaclone.astype(int)
        hspc_cells["lineage_bias"] = hspc_cells.metaclone.map(MC_TO_FAMILY)
        hspc_cells["lineage_bias_key"] = hspc_cells.metaclone.map(MC_TO_FAMILY_KEY)

        assert len(all_day2_hspc) == 1041
        assert len(hspc_cells) == 775
        assert hspc_cells.LARRY_ID.nunique() == 569
        assert set(hspc_cells.metaclone.unique()) == set(MC_ORDER)
        assert hspc_cells.groupby("LARRY_ID", observed=True).metaclone.nunique().max() == 1
        assert hspc_cells.groupby("LARRY_ID", observed=True).lineage_bias.nunique().max() == 1

        hspc_support = (hspc_cells.groupby(["lineage_bias", "metaclone"], observed=True)
                        .agg(n_cells=("LARRY_ID", "size"),
                             n_clones=("LARRY_ID", "nunique"),
                             n_libraries=("Library", "nunique"))
                        .reset_index())
        hspc_support["metaclone"] = pd.Categorical(
            hspc_support.metaclone, categories=MC_ORDER, ordered=True
        )
        hspc_support = hspc_support.sort_values("metaclone").reset_index(drop=True)
        hspc_support["metaclone"] = hspc_support.metaclone.astype(int)
        hspc_support.to_csv(TABLEDIR / "day2_hspc_support_by_metaclone.csv", index=False)

        family_support = (hspc_cells.groupby("lineage_bias", observed=True)
                          .agg(n_cells=("LARRY_ID", "size"),
                               n_clones=("LARRY_ID", "nunique"),
                               n_metaclones=("metaclone", "nunique"),
                               n_libraries=("Library", "nunique"))
                          .reindex(FAMILY_ORDER).reset_index())
        family_support.to_csv(TABLEDIR / "day2_hspc_support_by_lineage_bias.csv", index=False)

        source_sublabel = (hspc_cells.groupby(["lineage_bias", "metaclone", "cell_type"], observed=True)
                           .agg(n_cells=("LARRY_ID", "size"), n_clones=("LARRY_ID", "nunique"))
                           .reset_index())
        source_sublabel.to_csv(TABLEDIR / "day2_hspc_source_sublabel_diagnostic.csv", index=False)
        mixed_source_clones = int(
            (hspc_cells.groupby("LARRY_ID", observed=True).cell_type.nunique() > 1).sum()
        )

        cell_export = hspc_cells[[
            "LARRY_ID", "day", "cell_type", "Library", "metaclone",
            "lineage_bias", "lineage_bias_key"
        ]].copy()
        cell_export.index.name = "cell"
        assert "population" not in cell_export.columns
        cell_export.to_csv(TABLEDIR / "day2_hspc_analysis_cells.csv")

        cohort_audit = pd.DataFrame([
            ["all_Day2_prog1_plus_prog2_cells", len(all_day2_hspc)],
            ["assigned_Day2_HSPC_cells_analyzed", len(hspc_cells)],
            ["unassigned_Day2_HSPC_cells_excluded_no_metaclone", len(all_day2_hspc) - len(hspc_cells)],
            ["analyzed_LARRY_clones", hspc_cells.LARRY_ID.nunique()],
            ["analyzed_metaclones", hspc_cells.metaclone.nunique()],
            ["lineage_bias_families", hspc_cells.lineage_bias.nunique()],
            ["clones_spanning_both_source_sublabels", mixed_source_clones],
            ["minimum_clones_in_any_metaclone", int(hspc_support.n_clones.min())],
        ], columns=["check", "value"])
        cohort_audit.to_csv(TABLEDIR / "cohort_audit.csv", index=False)
        display(cohort_audit)
        display(family_support)
        display(hspc_support)
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_01_metaclone_support"
        plot_data = hspc_support.set_index("metaclone").reindex(MC_ORDER).reset_index()
        plot_data["bar_color"] = plot_data.lineage_bias.map(FAMILY_COLORS)

        fig, ax = plt.subplots(figsize=(8.2, 8.5))
        y = np.arange(len(plot_data))
        ax.barh(
            y, plot_data.n_clones.to_numpy(), height=0.68,
            color=plot_data.bar_color.to_list(), edgecolor=WHITE, linewidth=0.7
        )
        for position, row in plot_data.iterrows():
            ax.text(
                float(row.n_clones) + 0.7, position,
                f"{int(row.n_clones)} clones · {int(row.n_cells)} cells",
                va="center", ha="left", fontsize=8.1, color=GREY
            )
        boundaries = np.cumsum([len(FAMILY_GROUPS[name]) for name in FAMILY_ORDER])[:-1]
        for boundary in boundaries:
            ax.axhline(boundary - 0.5, color=LIGHT_GREY, linewidth=0.9)
        ax.set_yticks(y, [f"MC{mc}" for mc in plot_data.metaclone])
        ax.invert_yaxis()
        ax.set_xlim(0, float(plot_data.n_clones.max()) * 1.45)
        ax.set_xlabel("Day-2 HSPC clone pseudobulks")
        ax.set_title("Day-2 HSPC support across the current 20 metaclones", loc="left", pad=12)
        ax.text(
            0, 1.012, "All assigned prog_1 + prog_2 cells; one inferential unit per LARRY clone",
            transform=ax.transAxes, fontsize=8.6, color=GREY
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.6)
        ax.set_axisbelow(True)
        legend_handles = [Patch(facecolor=FAMILY_COLORS[name], label=name) for name in FAMILY_ORDER]
        ax.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=2, loc="lower right")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    md(
        r"""
        ### Clone pseudobulks and library adjustment

        Recovered cell counts are summed once per LARRY clone across both HSPC
        source labels. Genes are filtered without group labels: at least 10 total
        recovered counts and detection in at least five clone pseudobulks. Day-2
        library fractions are represented by the largest full-rank set of centered,
        standardized fraction covariates; the same covariates are used in the
        20-metaclone and five-family models. Source `prog_1`/`prog_2` fractions are
        diagnostic only and never enter a model.
        """
    ),
    code(
        r"""
        pseudobulk_rows = {}
        metadata_rows = []
        day2_libraries = sorted(hspc_cells.Library.unique())
        for clone, clone_cells in hspc_cells.groupby("LARRY_ID", observed=True):
            assert clone_cells.metaclone.nunique() == 1
            mc = int(clone_cells.metaclone.iloc[0])
            indices = POS.loc[clone_cells.index].to_numpy()
            pseudobulk_rows[str(clone)] = np.asarray(COUNTS[indices].sum(axis=0)).ravel().astype(np.int64)
            library_counts = clone_cells.Library.value_counts()
            label_counts = clone_cells.cell_type.value_counts()
            record = {
                "sample": str(clone), "clone": str(clone), "metaclone": mc,
                "mc_group": f"MC{mc}", "lineage_bias": MC_TO_FAMILY[mc],
                "lineage_bias_key": MC_TO_FAMILY_KEY[mc], "n_cells": int(len(clone_cells)),
                "n_prog_1_cells": int(label_counts.get("prog_1", 0)),
                "n_prog_2_cells": int(label_counts.get("prog_2", 0)),
            }
            for library in day2_libraries:
                record[f"libfrac__{safe_name(library)}"] = float(
                    library_counts.get(library, 0) / len(clone_cells)
                )
            metadata_rows.append(record)

        pseudobulk_counts = pd.DataFrame(pseudobulk_rows).T
        pseudobulk_counts.columns = A.var_names.astype(str)
        pseudobulk_counts.index.name = "sample"
        pseudobulk_metadata = pd.DataFrame(metadata_rows).set_index("sample").loc[pseudobulk_counts.index]
        pseudobulk_metadata["mc_group"] = pd.Categorical(
            pseudobulk_metadata.mc_group,
            categories=[f"MC{mc}" for mc in MC_ORDER], ordered=True
        )
        pseudobulk_metadata["lineage_bias_key"] = pd.Categorical(
            pseudobulk_metadata.lineage_bias_key,
            categories=FAMILY_KEY_ORDER, ordered=True
        )
        assert len(pseudobulk_counts) == len(pseudobulk_metadata) == 569
        assert (pseudobulk_counts.sum(axis=1) > 0).all()
        assert not pseudobulk_counts.index.duplicated().any()

        libfrac_columns = [column for column in pseudobulk_metadata if column.startswith("libfrac__")]
        reference_library_fraction = max(
            libfrac_columns,
            key=lambda column: (
                (pseudobulk_metadata.groupby("mc_group", observed=True)[column].mean() > 0).sum(),
                pseudobulk_metadata[column].mean(),
            )
        )
        candidate_library_columns = [
            column for column in libfrac_columns
            if column != reference_library_fraction and pseudobulk_metadata[column].std(ddof=0) > 1e-12
        ]
        mc_dummies = pd.get_dummies(pseudobulk_metadata.mc_group, drop_first=True, dtype=float)
        rank_matrix = np.column_stack([np.ones(len(pseudobulk_metadata)), mc_dummies.to_numpy(float)])
        library_covariates = []
        rank_dropped_library_columns = []
        for column in candidate_library_columns:
            z = (
                (pseudobulk_metadata[column] - pseudobulk_metadata[column].mean())
                / pseudobulk_metadata[column].std(ddof=0)
            )
            trial = np.column_stack([rank_matrix, z.to_numpy(float)])
            if np.linalg.matrix_rank(trial) > np.linalg.matrix_rank(rank_matrix):
                name = column + "__z"
                pseudobulk_metadata[name] = z
                library_covariates.append(name)
                rank_matrix = trial
            else:
                rank_dropped_library_columns.append(column)

        detected_clones = (pseudobulk_counts > 0).sum(axis=0)
        total_counts = pseudobulk_counts.sum(axis=0)
        keep_gene = (detected_clones >= 5) & (total_counts >= 10)
        filtered_counts = pseudobulk_counts.loc[:, keep_gene].copy()
        gene_filter = pd.DataFrame({
            "gene": pseudobulk_counts.columns, "detected_clones": detected_clones.to_numpy(),
            "total_recovered_counts": total_counts.to_numpy(), "retained": keep_gene.to_numpy()
        })
        gene_filter.to_csv(TABLEDIR / "group_blind_gene_filter.csv", index=False)

        metadata_export = pseudobulk_metadata.copy()
        metadata_export.index.name = "sample"
        assert "population" not in metadata_export.columns
        metadata_export.reset_index().to_csv(TABLEDIR / "day2_hspc_clone_pseudobulk_metadata.csv", index=False)
        library_design_audit = pd.DataFrame([
            ["day2_libraries", "|".join(day2_libraries)],
            ["reference_fraction", reference_library_fraction],
            ["library_covariates", "|".join(library_covariates)],
            ["rank_dropped_fractions", "|".join(rank_dropped_library_columns)],
            ["n_library_covariates", len(library_covariates)],
            ["genes_before_filter", pseudobulk_counts.shape[1]],
            ["genes_after_group_blind_filter", filtered_counts.shape[1]],
        ], columns=["check", "value"])
        library_design_audit.to_csv(TABLEDIR / "library_design_and_gene_filter_audit.csv", index=False)
        display(library_design_audit)
        """
    ),
    md(
        r"""
        ## Results

        ### Gene-level negative-binomial models

        Two full-rank PyDESeq2 models use identical clone pseudobulks and the same
        library covariates: one has a 20-level metaclone coefficient and one has a
        five-level lineage-bias coefficient. One-versus-rest contrasts use an
        equal-level comparator rather than weighting the comparator by clone count.
        Pairwise within-family contrasts are all returned, with a separate
        shared-library support flag for interpretation.
        """
    ),
    code(
        r"""
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from pydeseq2.default_inference import DefaultInference

        inference = DefaultInference(n_cpus=1)

        def fit_deseq_model(factor, label):
            metadata = pseudobulk_metadata.copy()
            design = "~ " + " + ".join(library_covariates + [factor])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dds = DeseqDataSet(
                    counts=filtered_counts.astype(int), metadata=metadata,
                    design=design, inference=inference, quiet=True, refit_cooks=False
                )
                design_matrix = np.asarray(dds.obsm["design_matrix"], dtype=float)
                design_rank = int(np.linalg.matrix_rank(design_matrix))
                assert design_rank == design_matrix.shape[1]
                dds.deseq2()
            warning_text = sorted({str(item.message) for item in caught})
            summary = {
                "model": label, "factor": factor, "design": design,
                "n_clone_units": len(metadata), "n_genes": filtered_counts.shape[1],
                "design_rank": design_rank, "design_columns": design_matrix.shape[1],
                "condition_number": float(np.linalg.cond(design_matrix)),
                "warning_count": len(warning_text), "warnings": " | ".join(warning_text),
            }
            return dds, summary

        dds_mc, mc_fit_summary = fit_deseq_model("mc_group", "20_metaclones")
        dds_family, family_fit_summary = fit_deseq_model("lineage_bias_key", "5_lineage_bias_families")
        fit_summary = pd.DataFrame([mc_fit_summary, family_fit_summary])
        fit_summary.to_csv(TABLEDIR / "deseq_model_fit_summary.csv", index=False)
        display(fit_summary)
        """
    ),
    code(
        r"""
        mc_library_support = (hspc_cells.groupby(["metaclone", "Library"], observed=True)
                              .LARRY_ID.nunique().unstack(fill_value=0)
                              .reindex(index=MC_ORDER, columns=day2_libraries, fill_value=0))
        within_pair_rows = []
        for family, members in FAMILY_GROUPS.items():
            if len(members) < 2:
                continue
            for target_mc, reference_mc in itertools.combinations(members, 2):
                shared = [
                    library for library in day2_libraries
                    if int(mc_library_support.loc[target_mc, library]) >= 3
                    and int(mc_library_support.loc[reference_mc, library]) >= 3
                ]
                within_pair_rows.append({
                    "lineage_bias": family,
                    "target_mc": target_mc,
                    "reference_mc": reference_mc,
                    "target_clones": int(hspc_support.set_index("metaclone").loc[target_mc, "n_clones"]),
                    "reference_clones": int(hspc_support.set_index("metaclone").loc[reference_mc, "n_clones"]),
                    "qualifying_shared_libraries_ge3_per_arm": "|".join(shared),
                    "n_qualifying_shared_libraries": len(shared),
                    "support_gate": bool(shared),
                })
        within_pair_support = pd.DataFrame(within_pair_rows)
        assert len(within_pair_support) == 40
        within_pair_support.to_csv(TABLEDIR / "within_lineage_pair_support.csv", index=False)

        def condition_mean_vector(dds, factor, levels):
            # Mean adjusted design row for explicitly named categorical levels.
            rows = [
                np.asarray(dds.cond(**{factor: level}), dtype=float)
                for level in levels
            ]
            return np.mean(np.vstack(rows), axis=0)

        def target_minus_reference_vector(dds, factor, targets, references):
            # Explicit target-minus-reference contrast; never rely on argument order.
            vector = (
                condition_mean_vector(dds, factor, targets)
                - condition_mean_vector(dds, factor, references)
            )
            if len(targets) == 1 and len(references) == 1:
                # FormulaicContrasts.contrast(column, baseline, group_to_compare)
                # returns group_to_compare - baseline.  This independent equality
                # assertion prevents the previously observed sign inversion.
                formulaic_vector = np.asarray(
                    dds.contrast(factor, references[0], targets[0]), dtype=float
                )
                assert np.allclose(vector, formulaic_vector, atol=1e-12)
            return vector

        contrast_specs = []
        for mc in MC_ORDER:
            target = f"MC{mc}"
            comparators = [f"MC{other}" for other in MC_ORDER if other != mc]
            contrast_specs.append({
                "model": "metaclone", "tier": "all_metaclones_ovr",
                "contrast_id": f"MC{mc}_vs_equal_mean_other_19",
                "contrast_label": f"MC{mc} vs equal-MC mean of other 19",
                "target": target, "reference": "equal_mean_other_19",
                "lineage_bias": MC_TO_FAMILY[mc], "target_mc": mc,
                "reference_mc": np.nan, "support_gate": True,
                "target_levels": [target], "reference_levels": comparators,
                "vector": target_minus_reference_vector(
                    dds_mc, "mc_group", [target], comparators
                ),
            })

        for family_key in FAMILY_KEY_ORDER:
            comparators = [key for key in FAMILY_KEY_ORDER if key != family_key]
            contrast_specs.append({
                "model": "family", "tier": "lineage_bias_ovr",
                "contrast_id": f"{family_key}_vs_equal_mean_other_4",
                "contrast_label": f"{FAMILY_DISPLAY[family_key]} vs equal-family mean of other 4",
                "target": family_key, "reference": "equal_mean_other_4",
                "lineage_bias": FAMILY_DISPLAY[family_key], "target_mc": np.nan,
                "reference_mc": np.nan, "support_gate": True,
                "target_levels": [family_key], "reference_levels": comparators,
                "vector": target_minus_reference_vector(
                    dds_family, "lineage_bias_key", [family_key], comparators
                ),
            })

        for target_family, reference_family in itertools.combinations(FAMILY_KEY_ORDER, 2):
            contrast_specs.append({
                "model": "family", "tier": "lineage_bias_pairwise",
                "contrast_id": f"{target_family}_vs_{reference_family}",
                "contrast_label": f"{FAMILY_DISPLAY[target_family]} vs {FAMILY_DISPLAY[reference_family]}",
                "target": target_family, "reference": reference_family,
                "lineage_bias": "between_families", "target_mc": np.nan,
                "reference_mc": np.nan, "support_gate": True,
                "target_levels": [target_family], "reference_levels": [reference_family],
                "vector": target_minus_reference_vector(
                    dds_family, "lineage_bias_key", [target_family], [reference_family]
                ),
            })

        for family, members in FAMILY_GROUPS.items():
            if len(members) < 2:
                continue
            for mc in members:
                target = f"MC{mc}"
                comparators = [f"MC{other}" for other in members if other != mc]
                contrast_specs.append({
                    "model": "metaclone", "tier": "within_lineage_ovr",
                    "contrast_id": f"MC{mc}_vs_equal_mean_{safe_name(family)}_peers",
                    "contrast_label": f"MC{mc} vs equal-MC mean of {family} peers",
                    "target": target, "reference": f"equal_mean_{safe_name(family)}_peers",
                    "lineage_bias": family, "target_mc": mc, "reference_mc": np.nan,
                    "support_gate": True,
                    "target_levels": [target], "reference_levels": comparators,
                    "vector": target_minus_reference_vector(
                        dds_mc, "mc_group", [target], comparators
                    ),
                })

            family_support_pairs = within_pair_support[
                within_pair_support.lineage_bias == family
            ].set_index(["target_mc", "reference_mc"])
            for target_mc, reference_mc in itertools.combinations(members, 2):
                support_row = family_support_pairs.loc[(target_mc, reference_mc)]
                contrast_specs.append({
                    "model": "metaclone", "tier": "within_lineage_pairwise",
                    "contrast_id": f"MC{target_mc}_vs_MC{reference_mc}",
                    "contrast_label": f"MC{target_mc} vs MC{reference_mc} within {family}",
                    "target": f"MC{target_mc}", "reference": f"MC{reference_mc}",
                    "lineage_bias": family, "target_mc": target_mc,
                    "reference_mc": reference_mc,
                    "support_gate": bool(support_row.support_gate),
                    "target_levels": [f"MC{target_mc}"],
                    "reference_levels": [f"MC{reference_mc}"],
                    "vector": target_minus_reference_vector(
                        dds_mc, "mc_group", [f"MC{target_mc}"], [f"MC{reference_mc}"]
                    ),
                })

        assert len(contrast_specs) == 94
        orientation_rows = []
        for spec in contrast_specs:
            dds = dds_mc if spec["model"] == "metaclone" else dds_family
            factor = "mc_group" if spec["model"] == "metaclone" else "lineage_bias_key"
            expected = target_minus_reference_vector(
                dds, factor, spec["target_levels"], spec["reference_levels"]
            )
            max_error = float(np.max(np.abs(np.asarray(spec["vector"]) - expected)))
            orientation_rows.append({
                "contrast_id": spec["contrast_id"],
                "model": spec["model"],
                "target_levels": "|".join(spec["target_levels"]),
                "reference_levels": "|".join(spec["reference_levels"]),
                "max_vector_error": max_error,
                "target_minus_reference_verified": max_error < 1e-12,
            })
        orientation_audit = pd.DataFrame(orientation_rows)
        assert orientation_audit.target_minus_reference_verified.all()
        orientation_audit.to_csv(
            TABLEDIR / "contrast_orientation_audit.csv", index=False
        )

        spec_table = pd.DataFrame([
            {
                key: ("|".join(value) if key in {"target_levels", "reference_levels"} else value)
                for key, value in spec.items() if key != "vector"
            }
            for spec in contrast_specs
        ])
        spec_table.to_csv(TABLEDIR / "contrast_registry.csv", index=False)
        display(spec_table.groupby("tier", observed=True).agg(
            n_contrasts=("contrast_id", "nunique"),
            n_supported=("support_gate", "sum")
        ).reset_index())
        """
    ),
    code(
        r"""
        def extract_deseq_result(dds, vector, spec):
            stats = DeseqStats(
                dds, contrast=np.asarray(vector, dtype=float), inference=inference,
                quiet=True, cooks_filter=False, independent_filter=False
            )
            stats.summary()
            result = stats.results_df.reset_index().rename(columns={"index": "gene"})
            if "gene" not in result.columns:
                result = result.rename(columns={result.columns[0]: "gene"})
            result["pvalue"] = result.pvalue.fillna(1.0)
            result["padj"] = result.padj.fillna(1.0)
            for key in [
                "model", "tier", "contrast_id", "contrast_label", "target",
                "reference", "lineage_bias", "target_mc", "reference_mc", "support_gate"
            ]:
                result[key] = spec[key]
            return result

        deseq_parts = []
        for number, spec in enumerate(contrast_specs, start=1):
            dds = dds_mc if spec["model"] == "metaclone" else dds_family
            deseq_parts.append(extract_deseq_result(dds, spec["vector"], spec))
            if number % 10 == 0 or number == len(contrast_specs):
                print(f"completed {number}/{len(contrast_specs)} DESeq2 contrasts")

        deseq_results = pd.concat(deseq_parts, ignore_index=True)
        deseq_results["contrast_fdr"] = deseq_results.padj
        deseq_results["tier_fdr"] = (
            deseq_results.groupby("tier", observed=True, group_keys=False).pvalue
            .transform(bh_adjust)
        )
        deseq_results["effect_ge_0_5"] = deseq_results.log2FoldChange.abs() >= 0.5
        deseq_results.to_csv(
            TABLEDIR / "gene_level_deseq2_contrasts.csv.gz", index=False, compression="gzip"
        )

        deseq_summary = (deseq_results.groupby("tier", observed=True)
                         .agg(n_contrasts=("contrast_id", "nunique"),
                              n_gene_tests=("gene", "size"),
                              n_contrast_fdr_lt_0_05=("contrast_fdr", lambda x: int((x < 0.05).sum())),
                              n_tier_fdr_lt_0_05=("tier_fdr", lambda x: int((x < 0.05).sum())),
                              n_tier_fdr_and_abs_lfc_ge_0_5=(
                                  "tier_fdr", lambda x: 0
                              ))
                         .reset_index())
        effect_counts = (deseq_results.assign(
            pass_both=(deseq_results.tier_fdr < 0.05) & deseq_results.effect_ge_0_5
        ).groupby("tier", observed=True).pass_both.sum())
        deseq_summary["n_tier_fdr_and_abs_lfc_ge_0_5"] = (
            deseq_summary.tier.map(effect_counts).fillna(0).astype(int)
        )
        deseq_summary.to_csv(TABLEDIR / "gene_level_deseq2_tier_summary.csv", index=False)

        within_pair_gene_summary = (deseq_results[
            deseq_results.tier == "within_lineage_pairwise"
        ].groupby([
            "contrast_id", "contrast_label", "lineage_bias", "target_mc",
            "reference_mc", "support_gate"
        ], observed=True)
         .agg(n_genes=("gene", "size"),
              n_contrast_fdr_lt_0_05=("contrast_fdr", lambda x: int((x < 0.05).sum())),
              n_tier_fdr_lt_0_05=("tier_fdr", lambda x: int((x < 0.05).sum())),
              min_tier_fdr=("tier_fdr", "min"))
         .reset_index())
        within_pair_gene_summary.to_csv(
            TABLEDIR / "within_lineage_pairwise_gene_summary.csv", index=False
        )
        display(deseq_summary)
        display(within_pair_gene_summary)
        """
    ),
    md(
        r"""
        ### Fixed Hallmark program scores

        The complete 50-program Hallmark 2020 universe is copied byte-for-byte into
        this research analysis as a reproducible checkpoint. Mouse symbols and Hallmark
        symbols are matched case-insensitively by uppercase symbol; mapping coverage
        is audited. No program is selected from the observed Day-2 contrasts before
        testing.

        For each clone, DESeq2-normalized counts are log2 transformed, each gene is
        z-scored across clones, and each program is the unweighted mean of its mapped
        gene z-scores. Program scores are then standardized across clones. HC3-robust
        linear models use the same design matrices and contrasts as the gene-level
        models.
        """
    ),
    code(
        r"""
        hallmark_source_hash = sha256(HALLMARK_SOURCE)
        assert hallmark_source_hash == EXPECTED_HALLMARK_SHA256
        if (not HALLMARK_LOCAL.exists()) or sha256(HALLMARK_LOCAL) != hallmark_source_hash:
            shutil.copy2(HALLMARK_SOURCE, HALLMARK_LOCAL)
        assert sha256(HALLMARK_LOCAL) == hallmark_source_hash

        hallmark_sets = {}
        with open(HALLMARK_LOCAL) as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                term = fields[0]
                genes = [gene.upper() for gene in fields[2:] if gene]
                hallmark_sets[term] = sorted(set(genes))
        assert len(hallmark_sets) == 50

        normalized_counts = np.asarray(dds_mc.layers["normed_counts"], dtype=float)
        assert normalized_counts.shape == filtered_counts.shape
        log_normalized = np.log2(normalized_counts + 1.0)
        gene_mean = log_normalized.mean(axis=0)
        gene_sd = log_normalized.std(axis=0, ddof=1)
        valid_sd = gene_sd > 1e-12
        gene_z = np.zeros_like(log_normalized, dtype=float)
        gene_z[:, valid_sd] = (
            (log_normalized[:, valid_sd] - gene_mean[valid_sd]) / gene_sd[valid_sd]
        )

        gene_names = np.asarray(filtered_counts.columns.astype(str))
        upper_to_indices = {}
        for index, gene in enumerate(gene_names):
            upper_to_indices.setdefault(gene.upper(), []).append(index)

        program_scores_raw = {}
        mapping_rows = []
        for term, requested_genes in hallmark_sets.items():
            mapped_indices = sorted({
                index for gene in requested_genes for index in upper_to_indices.get(gene, [])
            })
            mapped_names = gene_names[mapped_indices].tolist()
            mapping_rows.append({
                "program": term, "n_library_genes": len(requested_genes),
                "n_mapped_filtered_genes": len(mapped_indices),
                "mapped_fraction": len(mapped_indices) / len(requested_genes),
                "mapped_genes": "|".join(mapped_names),
            })
            assert len(mapped_indices) >= 10
            program_scores_raw[term] = gene_z[:, mapped_indices].mean(axis=1)

        program_mapping = pd.DataFrame(mapping_rows).sort_values("program")
        program_mapping.to_csv(TABLEDIR / "hallmark_gene_mapping_audit.csv", index=False)
        program_scores = pd.DataFrame(program_scores_raw, index=pseudobulk_metadata.index)
        program_scores = (
            (program_scores - program_scores.mean(axis=0))
            / program_scores.std(axis=0, ddof=1)
        )
        assert np.isfinite(program_scores.to_numpy()).all()
        program_score_export = pseudobulk_metadata[[
            "clone", "metaclone", "mc_group", "lineage_bias", "lineage_bias_key", "n_cells"
        ]].join(program_scores)
        assert "population" not in program_score_export.columns
        program_score_export.to_csv(
            TABLEDIR / "clone_hallmark_program_scores.csv.gz", compression="gzip"
        )

        resource_audit = pd.DataFrame([
            ["source_path", str(HALLMARK_SOURCE)],
            ["local_checkpoint", str(HALLMARK_LOCAL)],
            ["sha256", hallmark_source_hash],
            ["programs", len(hallmark_sets)],
            ["minimum_mapped_genes", int(program_mapping.n_mapped_filtered_genes.min())],
            ["median_mapped_genes", float(program_mapping.n_mapped_filtered_genes.median())],
            ["maximum_mapped_genes", int(program_mapping.n_mapped_filtered_genes.max())],
        ], columns=["check", "value"])
        resource_audit.to_csv(REPORTDIR / "hallmark_resource_audit.csv", index=False)
        display(resource_audit)
        display(program_mapping[[
            "program", "n_library_genes", "n_mapped_filtered_genes", "mapped_fraction"
        ]])
        """
    ),
    code(
        r"""
        mc_design = dds_mc.obsm["design_matrix"].copy()
        family_design = dds_family.obsm["design_matrix"].copy()
        assert mc_design.index.tolist() == program_scores.index.tolist()
        assert family_design.index.tolist() == program_scores.index.tolist()
        assert all(column in mc_design.columns for column in library_covariates)
        assert all(column in family_design.columns for column in library_covariates)

        program_fits_mc = {
            program: sm.OLS(program_scores[program].to_numpy(), mc_design.to_numpy(float)).fit(cov_type="HC3")
            for program in program_scores.columns
        }
        program_fits_family = {
            program: sm.OLS(program_scores[program].to_numpy(), family_design.to_numpy(float)).fit(cov_type="HC3")
            for program in program_scores.columns
        }

        program_test_rows = []
        for spec in contrast_specs:
            fits = program_fits_mc if spec["model"] == "metaclone" else program_fits_family
            for program, fit_result in fits.items():
                test = fit_result.t_test(np.asarray(spec["vector"], dtype=float))
                program_test_rows.append({
                    "program": program,
                    "program_label": clean_program_name(program),
                    "model": spec["model"], "tier": spec["tier"],
                    "contrast_id": spec["contrast_id"],
                    "contrast_label": spec["contrast_label"],
                    "target": spec["target"], "reference": spec["reference"],
                    "lineage_bias": spec["lineage_bias"],
                    "target_mc": spec["target_mc"], "reference_mc": spec["reference_mc"],
                    "support_gate": spec["support_gate"],
                    "effect_sd": float(np.asarray(test.effect).squeeze()),
                    "standard_error": float(np.asarray(test.sd).squeeze()),
                    "t_stat": float(np.asarray(test.tvalue).squeeze()),
                    "pvalue": float(np.asarray(test.pvalue).squeeze()),
                })
        program_tests = pd.DataFrame(program_test_rows)
        program_tests["contrast_fdr"] = (
            program_tests.groupby("contrast_id", observed=True, group_keys=False).pvalue
            .transform(bh_adjust)
        )
        program_tests["tier_fdr"] = (
            program_tests.groupby("tier", observed=True, group_keys=False).pvalue
            .transform(bh_adjust)
        )

        program_orientation_rows = []
        for spec in contrast_specs:
            dds = dds_mc if spec["model"] == "metaclone" else dds_family
            factor = "mc_group" if spec["model"] == "metaclone" else "lineage_bias_key"
            fits = program_fits_mc if spec["model"] == "metaclone" else program_fits_family
            target_row = condition_mean_vector(dds, factor, spec["target_levels"])
            reference_row = condition_mean_vector(dds, factor, spec["reference_levels"])
            observed = program_tests.loc[
                program_tests.contrast_id == spec["contrast_id"],
                ["program", "effect_sd"],
            ].set_index("program").effect_sd
            expected = pd.Series({
                program: float(
                    target_row @ np.asarray(fit_result.params, dtype=float)
                    - reference_row @ np.asarray(fit_result.params, dtype=float)
                )
                for program, fit_result in fits.items()
            })
            max_error = float((observed.reindex(expected.index) - expected).abs().max())
            program_orientation_rows.append({
                "contrast_id": spec["contrast_id"],
                "target_minus_reference_max_abs_error": max_error,
                "passed": max_error < 1e-10,
            })
        program_orientation_audit = pd.DataFrame(program_orientation_rows)
        assert program_orientation_audit.passed.all()
        program_orientation_audit.to_csv(
            TABLEDIR / "program_effect_orientation_audit.csv", index=False
        )
        program_tests.to_csv(TABLEDIR / "hallmark_program_contrasts.csv", index=False)

        program_tier_summary = (program_tests.groupby("tier", observed=True)
                                .agg(n_contrasts=("contrast_id", "nunique"),
                                     n_program_tests=("program", "size"),
                                     n_contrast_fdr_lt_0_05=(
                                         "contrast_fdr", lambda x: int((x < 0.05).sum())
                                     ),
                                     n_tier_fdr_lt_0_05=(
                                         "tier_fdr", lambda x: int((x < 0.05).sum())
                                     ))
                                .reset_index())
        program_tier_summary.to_csv(TABLEDIR / "hallmark_program_tier_summary.csv", index=False)

        def adjusted_program_means(design, fits, factor_values, factor_name):
            coefficient_matrix = np.column_stack([fits[program].params for program in program_scores.columns])
            rows = []
            for value in factor_values:
                sample_index = pseudobulk_metadata.index[
                    pseudobulk_metadata[factor_name].astype(str) == str(value)
                ][0]
                design_row = design.loc[sample_index].copy()
                design_row.loc[library_covariates] = 0.0
                estimates = design_row.to_numpy(float) @ coefficient_matrix
                record = {"group": str(value)}
                record.update(dict(zip(program_scores.columns, estimates)))
                rows.append(record)
            return pd.DataFrame(rows)

        adjusted_mc_program_means = adjusted_program_means(
            mc_design, program_fits_mc, [f"MC{mc}" for mc in MC_ORDER], "mc_group"
        )
        adjusted_mc_program_means["metaclone"] = (
            adjusted_mc_program_means.group.str.replace("MC", "", regex=False).astype(int)
        )
        adjusted_mc_program_means["lineage_bias"] = (
            adjusted_mc_program_means.metaclone.map(MC_TO_FAMILY)
        )
        adjusted_family_program_means = adjusted_program_means(
            family_design, program_fits_family, FAMILY_KEY_ORDER, "lineage_bias_key"
        )
        adjusted_family_program_means["lineage_bias"] = (
            adjusted_family_program_means.group.map(FAMILY_DISPLAY)
        )
        adjusted_mc_program_means.to_csv(TABLEDIR / "adjusted_program_means_by_metaclone.csv", index=False)
        adjusted_family_program_means.to_csv(
            TABLEDIR / "adjusted_program_means_by_lineage_bias.csv", index=False
        )
        display(program_tier_summary)
        """
    ),
    md(
        r"""
        ### Preranked Hallmark enrichment sensitivity

        The DESeq2 Wald statistic supplies a signed full-gene ranking for each
        one-versus-rest contrast. GSEA uses the same fixed Hallmark checkpoint,
        1,000 permutations, and a deterministic seed. This is a directional
        enrichment sensitivity, not an independent validation dataset. GSEA NES is
        interpreted within each ranked contrast; cross-contrast visual comparisons
        use the directly comparable adjusted clone program scores instead.
        """
    ),
    code(
        r"""
        gsea_tiers = {"all_metaclones_ovr", "lineage_bias_ovr", "within_lineage_ovr"}
        gsea_parts = []
        gsea_contrasts = deseq_results[deseq_results.tier.isin(gsea_tiers)].contrast_id.unique()
        for number, contrast_id in enumerate(gsea_contrasts, start=1):
            ranked = deseq_results[
                deseq_results.contrast_id == contrast_id
            ][["gene", "stat"]].copy()
            ranked["gene_upper"] = ranked.gene.astype(str).str.upper()
            ranked["abs_stat"] = ranked.stat.abs().fillna(-np.inf)
            ranked = (ranked.sort_values(["abs_stat", "gene_upper"], ascending=[False, True])
                      .drop_duplicates("gene_upper", keep="first"))
            ranked["rank_stat"] = ranked.stat.fillna(0.0)
            ranked = ranked.sort_values(
                ["rank_stat", "gene_upper"], ascending=[False, True]
            ).reset_index(drop=True)
            # Resolve exact ties deterministically at machine precision so GSEA
            # never inherits an arbitrary gene order or emits tie warnings.
            rank_values = ranked.rank_stat.to_numpy(float)
            for position in range(1, len(rank_values)):
                if rank_values[position] >= rank_values[position - 1]:
                    rank_values[position] = np.nextafter(
                        rank_values[position - 1], -np.inf
                    )
            assert pd.Series(rank_values).is_unique
            assert np.all(np.diff(rank_values) < 0)
            rank_series = pd.Series(
                rank_values, index=ranked.gene_upper, name="wald_stat"
            )
            prerank = gp.prerank(
                rnk=rank_series, gene_sets=str(HALLMARK_LOCAL), min_size=10,
                max_size=500, permutation_num=1000, seed=20260902, threads=1,
                outdir=None, no_plot=True, verbose=False
            )
            terms = prerank.res2d.copy()
            source_row = deseq_results[deseq_results.contrast_id == contrast_id].iloc[0]
            for column in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
                terms[column] = pd.to_numeric(terms[column], errors="coerce")
            terms["contrast_id"] = contrast_id
            terms["contrast_label"] = source_row.contrast_label
            terms["tier"] = source_row.tier
            terms["target"] = source_row.target
            terms["reference"] = source_row.reference
            terms["lineage_bias"] = source_row.lineage_bias
            terms["target_mc"] = source_row.target_mc
            terms["reference_mc"] = source_row.reference_mc
            terms["support_gate"] = source_row.support_gate
            terms["gmt_sha256"] = hallmark_source_hash
            gsea_parts.append(terms)
            if number % 10 == 0 or number == len(gsea_contrasts):
                print(f"completed {number}/{len(gsea_contrasts)} preranked GSEA contrasts")

        gsea_results = pd.concat(gsea_parts, ignore_index=True)
        gsea_results["tier_fdr_from_nominal_p"] = (
            gsea_results.groupby("tier", observed=True, group_keys=False)["NOM p-val"]
            .transform(bh_adjust)
        )
        gsea_results.to_csv(
            TABLEDIR / "hallmark_preranked_gsea.csv.gz", index=False, compression="gzip"
        )
        gsea_summary = (gsea_results.groupby("tier", observed=True)
                        .agg(n_contrasts=("contrast_id", "nunique"),
                             n_program_tests=("Term", "size"),
                             n_contrast_fdr_lt_0_25=(
                                 "FDR q-val", lambda x: int((x < 0.25).sum())
                             ),
                             n_tier_fdr_lt_0_05=(
                                 "tier_fdr_from_nominal_p", lambda x: int((x < 0.05).sum())
                             ))
                        .reset_index())
        gsea_summary.to_csv(TABLEDIR / "hallmark_preranked_gsea_summary.csv", index=False)
        display(gsea_summary)
        """
    ),
    md(
        r"""
        ### Program landscape across all metaclones

        Heatmaps show library-adjusted Hallmark score means at the centered
        library composition. Programs are ordered by their observed range only to
        make the fixed 50-program universe legible; all program tests remain in
        the accompanying tables. A dot marks a one-versus-rest program contrast
        passing the prespecified tier-wide Benjamini-Hochberg FDR of 0.05.
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_02_all_metaclone_program_landscape"
        plot_matrix_full = (adjusted_mc_program_means.set_index("metaclone")
                            .reindex(MC_ORDER)[list(program_scores.columns)])
        plot_programs = (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
        plot_programs = plot_programs.sort_values(ascending=False).head(20).index.tolist()
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_long = (plot_matrix.rename_axis("program").reset_index()
                     .melt(id_vars="program", var_name="metaclone", value_name="adjusted_score_sd"))
        plot_long["lineage_bias"] = plot_long.metaclone.map(MC_TO_FAMILY)
        plot_long.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant = program_tests[
            (program_tests.tier == "all_metaclones_ovr") &
            (program_tests.tier_fdr < 0.05)
        ][["program", "target_mc"]].copy()
        significant["target_mc"] = significant.target_mc.astype(int)
        significant_pairs = set(map(tuple, significant.to_numpy()))
        limit = max(0.5, float(np.nanmax(np.abs(plot_matrix.to_numpy()))))

        fig, ax = plt.subplots(figsize=(12.2, 8.4))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, metaclone in enumerate(plot_matrix.columns):
                if (program, int(metaclone)) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "•", ha="center", va="center",
                            color=INK, fontsize=10, fontweight="bold")
        boundaries = np.cumsum([len(FAMILY_GROUPS[name]) for name in FAMILY_ORDER])[:-1]
        for boundary in boundaries:
            ax.axvline(boundary, color=WHITE, linewidth=2.0)
        ax.set_xticks(np.arange(len(MC_ORDER)) + 0.5, [f"MC{mc}" for mc in MC_ORDER], rotation=90)
        for tick, metaclone in zip(ax.get_xticklabels(), MC_ORDER):
            tick.set_color(FAMILY_COLORS[MC_TO_FAMILY[metaclone]])
            tick.set_fontweight("bold")
        ax.set_yticks(
            np.arange(len(plot_programs)) + 0.5,
            [clean_program_name(program) for program in plot_programs], fontsize=8.2
        )
        ax.set_xlim(0, len(MC_ORDER))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_xlabel("Metaclone (tick color denotes current lineage-bias family)")
        ax.set_title("Adjusted Hallmark program landscape across all metaclones", loc="left", pad=12)
        ax.text(
            0, 1.012, "Top 20 of 50 by adjusted-mean range · dot: tier-wide FDR < 0.05",
            transform=ax.transAxes, fontsize=8.6, color=GREY
        )
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.025, pad=0.02)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    md(
        r"""
        ### Programs across pooled lineage-bias families

        These adjusted means pool clones from metaclones assigned to the same
        current lineage-bias family. They are clone-weighted summaries of this
        realized partition, not independent family replicates.
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_03_lineage_bias_program_landscape"
        plot_matrix_full = (adjusted_family_program_means.set_index("lineage_bias")
                            .reindex(FAMILY_ORDER)[list(program_scores.columns)])
        plot_programs = (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
        plot_programs = plot_programs.sort_values(ascending=False).head(20).index.tolist()
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_long = (plot_matrix.rename_axis("program").reset_index()
                     .melt(id_vars="program", var_name="lineage_bias", value_name="adjusted_score_sd"))
        plot_long.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant = program_tests[
            (program_tests.tier == "lineage_bias_ovr") &
            (program_tests.tier_fdr < 0.05)
        ][["program", "lineage_bias"]]
        significant_pairs = set(map(tuple, significant.to_numpy()))
        limit = max(0.5, float(np.nanmax(np.abs(plot_matrix.to_numpy()))))

        fig, ax = plt.subplots(figsize=(8.1, 8.4))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, family in enumerate(plot_matrix.columns):
                if (program, family) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "•", ha="center", va="center",
                            color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(FAMILY_ORDER)) + 0.5, FAMILY_ORDER, rotation=28, ha="right")
        for tick, family in zip(ax.get_xticklabels(), FAMILY_ORDER):
            tick.set_color(FAMILY_COLORS[family])
            tick.set_fontweight("bold")
        ax.set_yticks(
            np.arange(len(plot_programs)) + 0.5,
            [clean_program_name(program) for program in plot_programs], fontsize=8.2
        )
        ax.set_xlim(0, len(FAMILY_ORDER))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_xlabel("Current lineage-bias family")
        ax.set_title("Adjusted Hallmark programs across pooled lineage-bias families", loc="left", pad=12)
        ax.text(
            0, 1.012, "Top 20 of 50 by adjusted-mean range · dot: tier-wide FDR < 0.05",
            transform=ax.transAxes, fontsize=8.6, color=GREY
        )
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.038, pad=0.03)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_04_lineage_bias_pairwise_program_counts"
        pair_tests = program_tests[program_tests.tier == "lineage_bias_pairwise"].copy()
        plot_matrix = pd.DataFrame(
            np.nan, index=FAMILY_ORDER, columns=FAMILY_ORDER, dtype=float
        )
        for family in FAMILY_ORDER:
            plot_matrix.loc[family, family] = 0
        directional_rows = []
        for (target_key, reference_key), frame in pair_tests.groupby(["target", "reference"], observed=True):
            target_family = FAMILY_DISPLAY[target_key]
            reference_family = FAMILY_DISPLAY[reference_key]
            forward = int(((frame.tier_fdr < 0.05) & (frame.effect_sd > 0)).sum())
            reverse = int(((frame.tier_fdr < 0.05) & (frame.effect_sd < 0)).sum())
            plot_matrix.loc[target_family, reference_family] = forward
            plot_matrix.loc[reference_family, target_family] = reverse
            directional_rows.extend([
                {"enriched_in": target_family, "relative_to": reference_family,
                 "n_programs_tier_fdr_lt_0_05": forward},
                {"enriched_in": reference_family, "relative_to": target_family,
                 "n_programs_tier_fdr_lt_0_05": reverse},
            ])
        pd.DataFrame(directional_rows).to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        display_matrix = np.ma.masked_invalid(plot_matrix.to_numpy())
        vmax = max(1.0, float(np.nanmax(plot_matrix.to_numpy())))

        fig, ax = plt.subplots(figsize=(6.8, 5.9))
        image_artist = ax.pcolormesh(
            np.arange(display_matrix.shape[1] + 1),
            np.arange(display_matrix.shape[0] + 1),
            display_matrix, cmap="Blues", vmin=0, vmax=vmax, shading="flat",
        )
        for row_index, row_family in enumerate(FAMILY_ORDER):
            for column_index, column_family in enumerate(FAMILY_ORDER):
                value = plot_matrix.iloc[row_index, column_index]
                if np.isfinite(value) and row_index != column_index:
                    ax.text(column_index + 0.5, row_index + 0.5, str(int(value)), ha="center", va="center",
                            color=WHITE if value > vmax * 0.55 else INK, fontweight="bold")
        ax.set_xticks(np.arange(len(FAMILY_ORDER)) + 0.5, FAMILY_ORDER, rotation=32, ha="right")
        ax.set_yticks(np.arange(len(FAMILY_ORDER)) + 0.5, FAMILY_ORDER)
        ax.set_xlim(0, len(FAMILY_ORDER))
        ax.set_ylim(0, len(FAMILY_ORDER))
        ax.invert_yaxis()
        ax.set_xlabel("Relative to family")
        ax.set_ylabel("Programs higher in family")
        ax.set_title("Directional Hallmark differences between lineage-bias families", loc="left", pad=12)
        ax.text(
            0, 1.012, "Cell value: programs with tier-wide FDR < 0.05; diagonal not tested",
            transform=ax.transAxes, fontsize=8.4, color=GREY
        )
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Significant programs")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    md(
        r"""
        ### Metaclone-resolved programs within each lineage-bias family

        Four multi-metaclone families are estimable. Basophil contains only MC13,
        so no within-Basophil metaclone contrast exists. Each panel selects the 12
        programs with the largest adjusted range within that family for display;
        all 50 program tests remain tabulated. Dots use the tier-wide FDR across
        every within-family one-versus-peer program test.
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_05_neutrophil_within_family_programs"
        plot_family = "Neutrophil"
        plot_metaclones = FAMILY_GROUPS[plot_family]
        plot_matrix_full = (adjusted_mc_program_means.set_index("metaclone")
                            .reindex(plot_metaclones)[list(program_scores.columns)])
        plot_programs = (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
        plot_programs = plot_programs.sort_values(ascending=False).head(12).index.tolist()
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_long = (plot_matrix.rename_axis("program").reset_index()
                     .melt(id_vars="program", var_name="metaclone", value_name="adjusted_score_sd"))
        plot_long["lineage_bias"] = plot_family
        plot_long.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant = program_tests[
            (program_tests.tier == "within_lineage_ovr") &
            (program_tests.lineage_bias == plot_family) &
            (program_tests.tier_fdr < 0.05)
        ][["program", "target_mc"]].copy()
        significant["target_mc"] = significant.target_mc.astype(int)
        significant_pairs = set(map(tuple, significant.to_numpy()))
        limit = max(0.5, float(np.nanmax(np.abs(plot_matrix.to_numpy()))))

        fig, ax = plt.subplots(figsize=(6.9, 6.0))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, metaclone in enumerate(plot_matrix.columns):
                if (program, int(metaclone)) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "•", ha="center", va="center",
                            color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(plot_metaclones)) + 0.5, [f"MC{mc}" for mc in plot_metaclones])
        ax.set_yticks(np.arange(len(plot_programs)) + 0.5,
                      [clean_program_name(program) for program in plot_programs], fontsize=8.2)
        ax.set_xlim(0, len(plot_metaclones))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_xlabel("Neutrophil-biased metaclone")
        ax.set_title("Within-family Hallmark landscape · Neutrophil", loc="left", pad=12)
        ax.text(0, 1.012, "Top 12 by within-family range · dot: tier-wide FDR < 0.05",
                transform=ax.transAxes, fontsize=8.4, color=GREY)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.038, pad=0.03)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_06_monocyte_within_family_programs"
        plot_family = "Monocyte"
        plot_metaclones = FAMILY_GROUPS[plot_family]
        plot_matrix_full = (adjusted_mc_program_means.set_index("metaclone")
                            .reindex(plot_metaclones)[list(program_scores.columns)])
        plot_programs = (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
        plot_programs = plot_programs.sort_values(ascending=False).head(12).index.tolist()
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_long = (plot_matrix.rename_axis("program").reset_index()
                     .melt(id_vars="program", var_name="metaclone", value_name="adjusted_score_sd"))
        plot_long["lineage_bias"] = plot_family
        plot_long.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant = program_tests[
            (program_tests.tier == "within_lineage_ovr") &
            (program_tests.lineage_bias == plot_family) &
            (program_tests.tier_fdr < 0.05)
        ][["program", "target_mc"]].copy()
        significant["target_mc"] = significant.target_mc.astype(int)
        significant_pairs = set(map(tuple, significant.to_numpy()))
        limit = max(0.5, float(np.nanmax(np.abs(plot_matrix.to_numpy()))))

        fig, ax = plt.subplots(figsize=(6.6, 6.0))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, metaclone in enumerate(plot_matrix.columns):
                if (program, int(metaclone)) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "•", ha="center", va="center",
                            color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(plot_metaclones)) + 0.5, [f"MC{mc}" for mc in plot_metaclones])
        ax.set_yticks(np.arange(len(plot_programs)) + 0.5,
                      [clean_program_name(program) for program in plot_programs], fontsize=8.2)
        ax.set_xlim(0, len(plot_metaclones))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_xlabel("Monocyte-biased metaclone")
        ax.set_title("Within-family Hallmark landscape · Monocyte", loc="left", pad=12)
        ax.text(0, 1.012, "Top 12 by within-family range · dot: tier-wide FDR < 0.05",
                transform=ax.transAxes, fontsize=8.4, color=GREY)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.038, pad=0.03)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_07_erythroid_mk_within_family_programs"
        plot_family = "Erythroid / Mk"
        plot_metaclones = FAMILY_GROUPS[plot_family]
        plot_matrix_full = (adjusted_mc_program_means.set_index("metaclone")
                            .reindex(plot_metaclones)[list(program_scores.columns)])
        plot_programs = (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
        plot_programs = plot_programs.sort_values(ascending=False).head(12).index.tolist()
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_long = (plot_matrix.rename_axis("program").reset_index()
                     .melt(id_vars="program", var_name="metaclone", value_name="adjusted_score_sd"))
        plot_long["lineage_bias"] = plot_family
        plot_long.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant = program_tests[
            (program_tests.tier == "within_lineage_ovr") &
            (program_tests.lineage_bias == plot_family) &
            (program_tests.tier_fdr < 0.05)
        ][["program", "target_mc"]].copy()
        significant["target_mc"] = significant.target_mc.astype(int)
        significant_pairs = set(map(tuple, significant.to_numpy()))
        limit = max(0.5, float(np.nanmax(np.abs(plot_matrix.to_numpy()))))

        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, metaclone in enumerate(plot_matrix.columns):
                if (program, int(metaclone)) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "•", ha="center", va="center",
                            color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(plot_metaclones)) + 0.5, [f"MC{mc}" for mc in plot_metaclones])
        ax.set_yticks(np.arange(len(plot_programs)) + 0.5,
                      [clean_program_name(program) for program in plot_programs], fontsize=8.2)
        ax.set_xlim(0, len(plot_metaclones))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_xlabel("Erythroid / Mk-biased metaclone")
        ax.set_title("Within-family Hallmark landscape · Erythroid / Mk", loc="left", pad=12)
        ax.text(0, 1.012, "Top 12 by within-family range · dot: tier-wide FDR < 0.05",
                transform=ax.transAxes, fontsize=8.4, color=GREY)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.038, pad=0.03)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_08_multilineage_within_family_programs"
        plot_family = "Multi-lineage"
        plot_metaclones = FAMILY_GROUPS[plot_family]
        plot_matrix_full = (adjusted_mc_program_means.set_index("metaclone")
                            .reindex(plot_metaclones)[list(program_scores.columns)])
        plot_programs = (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
        plot_programs = plot_programs.sort_values(ascending=False).head(12).index.tolist()
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_long = (plot_matrix.rename_axis("program").reset_index()
                     .melt(id_vars="program", var_name="metaclone", value_name="adjusted_score_sd"))
        plot_long["lineage_bias"] = plot_family
        plot_long.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant = program_tests[
            (program_tests.tier == "within_lineage_ovr") &
            (program_tests.lineage_bias == plot_family) &
            (program_tests.tier_fdr < 0.05)
        ][["program", "target_mc"]].copy()
        significant["target_mc"] = significant.target_mc.astype(int)
        significant_pairs = set(map(tuple, significant.to_numpy()))
        limit = max(0.5, float(np.nanmax(np.abs(plot_matrix.to_numpy()))))

        fig, ax = plt.subplots(figsize=(7.8, 6.0))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, metaclone in enumerate(plot_matrix.columns):
                if (program, int(metaclone)) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "•", ha="center", va="center",
                            color=INK, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(plot_metaclones)) + 0.5, [f"MC{mc}" for mc in plot_metaclones])
        ax.set_yticks(np.arange(len(plot_programs)) + 0.5,
                      [clean_program_name(program) for program in plot_programs], fontsize=8.2)
        ax.set_xlim(0, len(plot_metaclones))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_xlabel("Multi-lineage metaclone")
        ax.set_title("Within-family Hallmark landscape · Multi-lineage", loc="left", pad=12)
        ax.text(0, 1.012, "Top 12 by within-family range · dot: tier-wide FDR < 0.05",
                transform=ax.transAxes, fontsize=8.4, color=GREY)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.032, pad=0.025)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # PLOT_CELL_INLINE_V1
        plot_stem = "weinreb_day2_hspc_09_within_family_pairwise_program_counts"
        pair_tests = program_tests[program_tests.tier == "within_lineage_pairwise"].copy()
        plot_matrix = pd.DataFrame(np.nan, index=MC_ORDER, columns=MC_ORDER, dtype=float)
        support_matrix = pd.DataFrame(False, index=MC_ORDER, columns=MC_ORDER, dtype=bool)
        for metaclone in MC_ORDER:
            plot_matrix.loc[metaclone, metaclone] = 0
            support_matrix.loc[metaclone, metaclone] = True
        directional_rows = []
        for (target_mc, reference_mc), frame in pair_tests.groupby(
            ["target_mc", "reference_mc"], observed=True
        ):
            target_mc = int(target_mc)
            reference_mc = int(reference_mc)
            is_supported = bool(frame.support_gate.iloc[0])
            forward = int(((frame.tier_fdr < 0.05) & (frame.effect_sd > 0)).sum())
            reverse = int(((frame.tier_fdr < 0.05) & (frame.effect_sd < 0)).sum())
            plot_matrix.loc[target_mc, reference_mc] = forward
            plot_matrix.loc[reference_mc, target_mc] = reverse
            support_matrix.loc[target_mc, reference_mc] = is_supported
            support_matrix.loc[reference_mc, target_mc] = is_supported
            family = MC_TO_FAMILY[target_mc]
            directional_rows.extend([
                {"lineage_bias": family, "enriched_in_metaclone": target_mc,
                 "relative_to_metaclone": reference_mc,
                 "n_programs_tier_fdr_lt_0_05": forward, "support_gate": is_supported},
                {"lineage_bias": family, "enriched_in_metaclone": reference_mc,
                 "relative_to_metaclone": target_mc,
                 "n_programs_tier_fdr_lt_0_05": reverse, "support_gate": is_supported},
            ])
        pd.DataFrame(directional_rows).to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        display_matrix = np.ma.masked_invalid(plot_matrix.to_numpy())
        vmax = max(1.0, float(np.nanmax(plot_matrix.to_numpy())))

        fig, ax = plt.subplots(figsize=(10.2, 9.1))
        image_artist = ax.pcolormesh(
            np.arange(display_matrix.shape[1] + 1),
            np.arange(display_matrix.shape[0] + 1),
            display_matrix, cmap="Blues", vmin=0, vmax=vmax, shading="flat",
        )
        for row_index, target_mc in enumerate(MC_ORDER):
            for column_index, reference_mc in enumerate(MC_ORDER):
                value = plot_matrix.iloc[row_index, column_index]
                if np.isfinite(value) and row_index != column_index:
                    marker = "" if support_matrix.iloc[row_index, column_index] else "†"
                    ax.text(column_index + 0.5, row_index + 0.5, f"{int(value)}{marker}",
                            ha="center", va="center", fontsize=6.5,
                            color=WHITE if value > vmax * 0.55 else INK)
        boundaries = np.cumsum([len(FAMILY_GROUPS[name]) for name in FAMILY_ORDER])[:-1]
        for boundary in boundaries:
            ax.axvline(boundary, color=WHITE, linewidth=2.2)
            ax.axhline(boundary, color=WHITE, linewidth=2.2)
        ax.set_xticks(np.arange(len(MC_ORDER)) + 0.5, [f"MC{mc}" for mc in MC_ORDER], rotation=90)
        ax.set_yticks(np.arange(len(MC_ORDER)) + 0.5, [f"MC{mc}" for mc in MC_ORDER])
        ax.set_xlim(0, len(MC_ORDER))
        ax.set_ylim(0, len(MC_ORDER))
        ax.invert_yaxis()
        for tick, metaclone in zip(ax.get_xticklabels(), MC_ORDER):
            tick.set_color(FAMILY_COLORS[MC_TO_FAMILY[metaclone]])
        for tick, metaclone in zip(ax.get_yticklabels(), MC_ORDER):
            tick.set_color(FAMILY_COLORS[MC_TO_FAMILY[metaclone]])
        ax.set_xlabel("Relative to metaclone")
        ax.set_ylabel("Programs higher in metaclone")
        ax.set_title("Directional Hallmark differences within lineage-bias families", loc="left", pad=32)
        ax.text(
            0, 1.004,
            "Tier-wide FDR < 0.05 · blank: different family · †: shared-library support gate not met",
            transform=ax.transAxes, fontsize=8.4, color=GREY
        )
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.035, pad=0.025)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Significant programs")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    md(
        r"""
        ## Computed findings

        The following narrative is assembled from the executed result objects.
        It deliberately separates observed evidence from interpretation and avoids
        causal or prospective claims.
        """
    ),
    code(
        r"""
        summary_lines = [
            "OBSERVED EVIDENCE",
            (
                f"- Day-2 prog_1 + prog_2 contained {len(all_day2_hspc):,} cells; "
                f"{len(hspc_cells):,} cells from {hspc_cells.LARRY_ID.nunique():,} assigned "
                f"LARRY clones entered the 20-metaclone analysis."
            ),
            (
                f"- Metaclone support ranged from {int(hspc_support.n_clones.min())} to "
                f"{int(hspc_support.n_clones.max())} clone pseudobulks."
            ),
        ]
        for row in program_tier_summary.itertuples(index=False):
            summary_lines.append(
                f"- {row.tier}: {int(row.n_contrasts)} contrasts, "
                f"{int(row.n_tier_fdr_lt_0_05)} of {int(row.n_program_tests)} Hallmark tests "
                "passed tier-wide FDR < 0.05."
            )

        for tier, label in [
            ("all_metaclones_ovr", "all-metaclone"),
            ("lineage_bias_ovr", "lineage-family"),
            ("within_lineage_ovr", "within-family"),
        ]:
            candidates = program_tests[
                (program_tests.tier == tier) & (program_tests.tier_fdr < 0.05)
            ].copy()
            if candidates.empty:
                summary_lines.append(f"- No {label} program test passed tier-wide FDR < 0.05.")
            else:
                candidates = candidates.assign(abs_effect=candidates.effect_sd.abs())
                top = candidates.sort_values(
                    ["abs_effect", "tier_fdr", "contrast_id", "program"],
                    ascending=[False, True, True, True]
                ).head(3)
                rendered = "; ".join(
                    f"{row.contrast_label}: {row.program_label} ({row.effect_sd:+.2f} SD, q={row.tier_fdr:.2g})"
                    for row in top.itertuples(index=False)
                )
                summary_lines.append(f"- Largest significant {label} program effects: {rendered}.")

        supported_pairs = within_pair_support[within_pair_support.support_gate]
        summary_lines.extend([
            "",
            "INTERPRETATION BOUNDARIES",
            (
                f"- {len(supported_pairs)} of {len(within_pair_support)} within-family metaclone pairs "
                "met the shared-library support gate; unsupported estimates remain visible and flagged."
            ),
            "- Program associations may reflect early maturation state or technical library composition despite adjustment.",
            "- Lineage-bias families are outcome-informed, nested partitions; pooled-family effects are clone-weighted in this dataset.",
            "- MC13 is the only Basophil metaclone, so no within-Basophil comparison is estimable.",
            "- MC0 and MC5 follow the current manual Figure-2 map (Multi-lineage); an older threshold rule would differ.",
        ])
        result_summary_text = "\n".join(summary_lines)
        (REPORTDIR / "computed_findings_and_boundaries.txt").write_text(
            result_summary_text + "\n", encoding="utf-8"
        )
        print(result_summary_text)
        """
    ),
    md(
        r"""
        ## Validation & provenance

        Validation covers cohort identity, protected-column exclusion, count
        reconstruction, model rank, contrast completeness, multiple-testing
        outputs, plot-cell structure, file existence, and reproducible resource
        hashes. A machine-readable artifact manifest follows the checks.
        """
    ),
    code(
        r"""
        expected_tier_counts = {
            "all_metaclones_ovr": 20,
            "lineage_bias_ovr": 5,
            "lineage_bias_pairwise": 10,
            "within_lineage_ovr": 19,
            "within_lineage_pairwise": 40,
        }
        observed_tier_counts = spec_table.groupby("tier", observed=True).contrast_id.nunique().to_dict()
        plot_stems = [
            "weinreb_day2_hspc_01_metaclone_support",
            "weinreb_day2_hspc_02_all_metaclone_program_landscape",
            "weinreb_day2_hspc_03_lineage_bias_program_landscape",
            "weinreb_day2_hspc_04_lineage_bias_pairwise_program_counts",
            "weinreb_day2_hspc_05_neutrophil_within_family_programs",
            "weinreb_day2_hspc_06_monocyte_within_family_programs",
            "weinreb_day2_hspc_07_erythroid_mk_within_family_programs",
            "weinreb_day2_hspc_08_multilineage_within_family_programs",
            "weinreb_day2_hspc_09_within_family_pairwise_program_counts",
        ]
        notebook_source = nbformat.read(NOTEBOOK_PATH, as_version=4)
        plot_marker = "# PLOT_CELL_" + "INLINE_V1"
        plot_sources = [
            cell.source for cell in notebook_source.cells
            if cell.cell_type == "code" and plot_marker in cell.source
        ]
        plot_source_contract = all(
            source.count("plt.subplots") == 1
            and source.count("fig.savefig(svg") == 1
            and source.count("fig.savefig(png") == 1
            and source.count("display(fig)") == 1
            and source.count("plt.close(fig)") == 1
            for source in plot_sources
        )
        output_headers_are_population_free = True
        for table_path in sorted(TABLEDIR.glob("*.csv*")):
            header = pd.read_csv(table_path, nrows=0).columns.astype(str).str.lower().tolist()
            output_headers_are_population_free &= "population" not in header

        validation_rows = [
            ("input_object_sha256", input_hash == EXPECTED_INPUT_SHA256,
             f"observed={input_hash}"),
            ("lineage_map_sha256", lineage_map_hash == EXPECTED_LINEAGE_MAP_SHA256,
             f"observed={lineage_map_hash}"),
            ("population_removed", "population" not in A.obs.columns,
             "removed immediately after object read"),
            ("output_headers_population_free", output_headers_are_population_free,
             f"checked={len(list(TABLEDIR.glob('*.csv*')))} tables"),
            ("all_day2_prog1_plus_prog2_cells", len(all_day2_hspc) == 1041,
             f"observed={len(all_day2_hspc)}"),
            ("assigned_analysis_cells", len(hspc_cells) == 775,
             f"observed={len(hspc_cells)}"),
            ("clone_pseudobulks", len(pseudobulk_counts) == 569,
             f"observed={len(pseudobulk_counts)}"),
            ("current_metaclones", hspc_cells.metaclone.nunique() == 20,
             f"observed={hspc_cells.metaclone.nunique()}"),
            ("current_lineage_families", hspc_cells.lineage_bias.nunique() == 5,
             f"observed={hspc_cells.lineage_bias.nunique()}"),
            ("minimum_metaclone_clone_support", int(hspc_support.n_clones.min()) >= 10,
             f"observed_min={int(hspc_support.n_clones.min())}"),
            ("count_integer_lattice_error", reconstruction_error < 1e-6,
             f"observed={reconstruction_error:.3g}"),
            ("full_rank_metaclone_model",
             mc_fit_summary["design_rank"] == mc_fit_summary["design_columns"],
             f"rank={mc_fit_summary['design_rank']}/{mc_fit_summary['design_columns']}"),
            ("full_rank_lineage_model",
             family_fit_summary["design_rank"] == family_fit_summary["design_columns"],
             f"rank={family_fit_summary['design_rank']}/{family_fit_summary['design_columns']}"),
            ("contrast_registry_complete", observed_tier_counts == expected_tier_counts,
             json.dumps(observed_tier_counts, sort_keys=True)),
            ("program_test_grid_complete", len(program_tests) == 94 * 50,
             f"observed={len(program_tests)} expected={94 * 50}"),
            ("gsea_test_grid_complete",
             gsea_results.contrast_id.nunique() == 44 and len(gsea_results) == 44 * 50,
             f"contrasts={gsea_results.contrast_id.nunique()} tests={len(gsea_results)}"),
            ("finite_program_statistics",
             np.isfinite(program_tests[["effect_sd", "standard_error", "t_stat", "pvalue"]].to_numpy()).all(),
             "effect, SE, t, and p checked"),
            ("basophil_within_family_not_estimated",
             not ((program_tests.tier.str.startswith("within_lineage")) &
                  (program_tests.lineage_bias == "Basophil")).any(),
             "one Basophil metaclone"),
            ("legacy_rule_disagreement_audited",
             set(lineage_rule_sensitivity.loc[
                 lineage_rule_sensitivity.classification_differs, "metaclone"
             ]) == {0, 5},
             "expected current-vs-legacy disagreements: MC0, MC5"),
            ("plot_cell_count", len(plot_sources) == len(plot_stems),
             f"observed={len(plot_sources)} expected={len(plot_stems)}"),
            ("plot_cell_inline_contract", plot_source_contract,
             "one inline figure construction/save/display/close per plot cell"),
            ("svg_outputs", all((FIGDIR / f"{stem}.svg").stat().st_size > 0 for stem in plot_stems),
             f"expected={len(plot_stems)}"),
            ("png_outputs", all((FIGDIR / f"{stem}.png").stat().st_size > 0 for stem in plot_stems),
             f"expected={len(plot_stems)}"),
            ("hallmark_resource_checkpoint", sha256(HALLMARK_LOCAL) == hallmark_source_hash,
             f"sha256={hallmark_source_hash}"),
        ]
        validation = pd.DataFrame(validation_rows, columns=["check", "passed", "detail"])
        validation.to_csv(REPORTDIR / "validation.csv", index=False)
        assert validation.passed.all(), validation.loc[~validation.passed]

        artifact_paths = []
        for folder in [ANALYSIS_DIR / "scripts", FIGDIR, TABLEDIR, REPORTDIR, RESOURCEDIR]:
            for path in sorted(folder.rglob("*")):
                if (path.is_file() and path.name != "artifact_manifest.csv"
                        and path.suffix != ".pyc" and "__pycache__" not in path.parts):
                    artifact_paths.append(path)
        artifact_manifest = pd.DataFrame([
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ])
        artifact_manifest.to_csv(REPORTDIR / "artifact_manifest.csv", index=False)
        display(validation)
        display(artifact_manifest)
        print("All validation checks passed.")
        print("Notebook:", NOTEBOOK_PATH)
        print("Environment: chronoclone |", ENV_PATH)
        print("Execution command:", EXECUTION_COMMAND)
        """
    ),
    md(
        r"""
        ## Version 2 — Multi-lineage program–output correlations

        This section asks whether the
        library-adjusted, clone-pseudobulk Day-2 Hallmark score of each current
        Multi-lineage metaclone is associated with the cell types produced by
        that metaclone at Days 4 and 6.

        The seven current Multi-lineage metaclones are the independent
        observations. Primary outcomes are the author `cell_type_weinreb`
        fractions within each observed clone and day, averaged equally across
        clones. An unweighted pooled-cell result and a mature-only clone-balanced
        result are retained as sensitivities. Entirely unobserved clone-days are
        missing rather than zero; within an observed clone-day, an absent cell
        type contributes a structural zero.

        Each program–output association uses an exhaustive two-sided Spearman
        label-permutation test over all `7! = 5,040` metaclone permutations.
        Multiplicity is reported both within each cell-type/time endpoint (50
        programs) and across the full primary screen. Leave-one-metaclone-out
        directions are retained because seven observations cannot support a
        stable high-dimensional predictive claim. Output fractions are
        compositional, and the Multi-lineage family itself was defined from later
        output; all associations are retrospective and exploratory.
        """
    ),
    code(
        r"""
        # Build clone-balanced D4/D6 author cell-type outputs for all seven Multi-lineage MCs
        from scipy.spatial.distance import jensenshannon, pdist, squareform
        from scipy.stats import rankdata, spearmanr

        MULTI_MCS = list(FAMILY_GROUPS["Multi-lineage"])
        OUTPUT_DAYS = [4, 6]
        AUTHOR_OUTPUT_TYPES = [
            "Neutrophil", "Monocyte", "Meg", "Erythroid", "Baso", "Mast",
            "Eos", "Ccr7_DC", "Lymphoid", "Undifferentiated",
        ]
        MATURE_OUTPUT_TYPES = [
            output_type for output_type in AUTHOR_OUTPUT_TYPES
            if output_type != "Undifferentiated"
        ]
        required_output_columns = [
            "day", "cell_type_weinreb", "LARRY_ID", "metaclone", "Library"
        ]
        assert all(column in obs.columns for column in required_output_columns)

        output_cells = obs.loc[
            obs.metaclone.isin(MULTI_MCS) & obs.day.isin(OUTPUT_DAYS),
            required_output_columns,
        ].copy()
        output_cells["metaclone"] = output_cells.metaclone.astype(int)
        output_cells["day"] = output_cells.day.astype(int)
        output_cells["cell_type_weinreb"] = output_cells.cell_type_weinreb.astype(str)
        observed_output_types = set(output_cells.cell_type_weinreb)
        assert observed_output_types <= set(AUTHOR_OUTPUT_TYPES), observed_output_types

        output_support_rows = []
        for (metaclone, day), group in output_cells.groupby(
            ["metaclone", "day"], observed=True
        ):
            mature = group[group.cell_type_weinreb != "Undifferentiated"]
            output_support_rows.append({
                "metaclone": int(metaclone),
                "day": int(day),
                "n_cells": int(len(group)),
                "n_clones": int(group.LARRY_ID.nunique()),
                "n_mature_cells": int(len(mature)),
                "n_mature_informative_clones": int(mature.LARRY_ID.nunique()),
                "fraction_clones_with_mature_output": float(
                    mature.LARRY_ID.nunique() / group.LARRY_ID.nunique()
                ),
            })
        multilineage_output_support = (
            pd.DataFrame(output_support_rows)
            .set_index(["metaclone", "day"])
            .reindex(pd.MultiIndex.from_product(
                [MULTI_MCS, OUTPUT_DAYS], names=["metaclone", "day"]
            ))
            .reset_index()
        )
        assert multilineage_output_support[
            ["n_cells", "n_clones"]
        ].notna().all().all()

        clone_keys = output_cells[
            ["metaclone", "day", "LARRY_ID"]
        ].drop_duplicates()
        clone_type_grid = (
            clone_keys.assign(_join_key=1)
            .merge(
                pd.DataFrame({
                    "cell_type_weinreb": AUTHOR_OUTPUT_TYPES,
                    "_join_key": 1,
                }),
                on="_join_key",
            )
            .drop(columns="_join_key")
        )
        clone_type_counts = (
            output_cells.groupby(
                ["metaclone", "day", "LARRY_ID", "cell_type_weinreb"],
                observed=True,
            )
            .size()
            .rename("n_cells")
            .reset_index()
        )
        clone_type_complete = clone_type_grid.merge(
            clone_type_counts,
            on=["metaclone", "day", "LARRY_ID", "cell_type_weinreb"],
            how="left",
            validate="one_to_one",
        )
        clone_type_complete["n_cells"] = clone_type_complete.n_cells.fillna(0).astype(int)
        clone_type_complete["clone_day_total"] = clone_type_complete.groupby(
            ["metaclone", "day", "LARRY_ID"], observed=True
        ).n_cells.transform("sum")
        assert (clone_type_complete.clone_day_total > 0).all()
        clone_type_complete["fraction"] = (
            clone_type_complete.n_cells / clone_type_complete.clone_day_total
        )
        multilineage_output_clone_balanced = (
            clone_type_complete.groupby(
                ["metaclone", "day", "cell_type_weinreb"], observed=True
            ).fraction.mean().reset_index()
        )

        pooled_type_counts = (
            output_cells.groupby(
                ["metaclone", "day", "cell_type_weinreb"], observed=True
            ).size().rename("n_cells").reset_index()
        )
        mc_day_type_grid = (
            output_cells[["metaclone", "day"]].drop_duplicates().assign(_join_key=1)
            .merge(
                pd.DataFrame({
                    "cell_type_weinreb": AUTHOR_OUTPUT_TYPES,
                    "_join_key": 1,
                }),
                on="_join_key",
            )
            .drop(columns="_join_key")
        )
        multilineage_output_pooled_cells = mc_day_type_grid.merge(
            pooled_type_counts,
            on=["metaclone", "day", "cell_type_weinreb"],
            how="left",
            validate="one_to_one",
        )
        multilineage_output_pooled_cells["n_cells"] = (
            multilineage_output_pooled_cells.n_cells.fillna(0).astype(int)
        )
        multilineage_output_pooled_cells["fraction"] = (
            multilineage_output_pooled_cells.n_cells
            / multilineage_output_pooled_cells.groupby(
                ["metaclone", "day"], observed=True
            ).n_cells.transform("sum")
        )

        mature_cells = output_cells[
            output_cells.cell_type_weinreb != "Undifferentiated"
        ].copy()
        mature_clone_keys = mature_cells[
            ["metaclone", "day", "LARRY_ID"]
        ].drop_duplicates()
        mature_clone_type_grid = (
            mature_clone_keys.assign(_join_key=1)
            .merge(
                pd.DataFrame({
                    "cell_type_weinreb": MATURE_OUTPUT_TYPES,
                    "_join_key": 1,
                }),
                on="_join_key",
            )
            .drop(columns="_join_key")
        )
        mature_clone_counts = (
            mature_cells.groupby(
                ["metaclone", "day", "LARRY_ID", "cell_type_weinreb"],
                observed=True,
            ).size().rename("n_cells").reset_index()
        )
        mature_clone_complete = mature_clone_type_grid.merge(
            mature_clone_counts,
            on=["metaclone", "day", "LARRY_ID", "cell_type_weinreb"],
            how="left",
            validate="one_to_one",
        )
        mature_clone_complete["n_cells"] = (
            mature_clone_complete.n_cells.fillna(0).astype(int)
        )
        mature_clone_complete["mature_clone_day_total"] = mature_clone_complete.groupby(
            ["metaclone", "day", "LARRY_ID"], observed=True
        ).n_cells.transform("sum")
        assert (mature_clone_complete.mature_clone_day_total > 0).all()
        mature_clone_complete["fraction"] = (
            mature_clone_complete.n_cells / mature_clone_complete.mature_clone_day_total
        )
        multilineage_output_mature_clone_balanced = (
            mature_clone_complete.groupby(
                ["metaclone", "day", "cell_type_weinreb"], observed=True
            ).fraction.mean().reset_index()
        )

        program_columns = list(program_scores.columns)
        multilineage_day2_program_means = (
            adjusted_mc_program_means.set_index("metaclone")
            .loc[MULTI_MCS, program_columns]
            .rename_axis("metaclone")
            .reset_index()
        )
        assert multilineage_day2_program_means.shape == (7, 51)
        assert np.isfinite(
            multilineage_day2_program_means[program_columns].to_numpy(float)
        ).all()

        multilineage_output_support.to_csv(
            TABLEDIR / "multilineage_output_support_by_day.csv", index=False
        )
        multilineage_output_clone_balanced.to_csv(
            TABLEDIR / "multilineage_clone_balanced_author_output_by_day.csv", index=False
        )
        multilineage_output_pooled_cells.to_csv(
            TABLEDIR / "multilineage_pooled_cell_author_output_by_day.csv", index=False
        )
        multilineage_output_mature_clone_balanced.to_csv(
            TABLEDIR / "multilineage_clone_balanced_mature_output_by_day.csv", index=False
        )
        multilineage_day2_program_means.to_csv(
            TABLEDIR / "multilineage_day2_adjusted_program_means.csv", index=False
        )

        primary_fraction_sums = multilineage_output_clone_balanced.groupby(
            ["metaclone", "day"], observed=True
        ).fraction.sum()
        mature_fraction_sums = multilineage_output_mature_clone_balanced.groupby(
            ["metaclone", "day"], observed=True
        ).fraction.sum()
        assert np.allclose(primary_fraction_sums, 1.0)
        assert np.allclose(mature_fraction_sums, 1.0)
        display(multilineage_output_support)
        """
    ),
    md(
        r"""
        ### Exact metaclone-level correlations

        Day 4 and Day 6 are tested separately. A Day-6-minus-Day-4 snapshot
        difference is retained as a sensitivity; it can reflect both biological
        change and incomplete recovery of different clones at the two destructive
        snapshots. Endpoint-specific FDR asks which of the 50 programs relates to
        one declared cell-type/time outcome. Analysis-wide FDR additionally
        protects across all cell types and temporal scopes.
        """
    ),
    code(
        r"""
        # Exhaustive 7! Spearman correlations, endpoint/global FDR, and LOMO stability
        PERMUTATION_INDEX = np.asarray(
            list(itertools.permutations(np.arange(len(MULTI_MCS)))), dtype=int
        )
        assert PERMUTATION_INDEX.shape == (5040, 7)

        def exact_spearman_permutation(x_values, y_values):
            x_values = np.asarray(x_values, dtype=float)
            y_values = np.asarray(y_values, dtype=float)
            complete = np.isfinite(x_values) & np.isfinite(y_values)
            if complete.sum() != len(MULTI_MCS):
                return np.nan, np.nan
            x_rank = rankdata(x_values, method="average")
            y_rank = rankdata(y_values, method="average")
            x_centered = x_rank - x_rank.mean()
            y_centered = y_rank - y_rank.mean()
            denominator = np.sqrt(
                np.sum(x_centered ** 2) * np.sum(y_centered ** 2)
            )
            if denominator <= 0:
                return np.nan, np.nan
            observed_rho = float(np.dot(x_centered, y_centered) / denominator)
            permuted_centered = y_centered[PERMUTATION_INDEX]
            permutation_rho = permuted_centered @ x_centered / denominator
            exact_pvalue = float(
                np.mean(np.abs(permutation_rho) >= abs(observed_rho) - 1e-12)
            )
            return observed_rho, exact_pvalue

        def lomo_spearman_summary(x_values, y_values, observed_rho):
            lomo_rho = []
            for omitted in range(len(MULTI_MCS)):
                keep = np.arange(len(MULTI_MCS)) != omitted
                x_subset = np.asarray(x_values, dtype=float)[keep]
                y_subset = np.asarray(y_values, dtype=float)[keep]
                complete = np.isfinite(x_subset) & np.isfinite(y_subset)
                if (
                    complete.sum() < 2
                    or np.unique(x_subset[complete]).size < 2
                    or np.unique(y_subset[complete]).size < 2
                ):
                    rho = np.nan
                else:
                    rho = spearmanr(
                        x_subset[complete], y_subset[complete]
                    ).statistic
                lomo_rho.append(float(rho) if np.isfinite(rho) else np.nan)
            finite = np.asarray(lomo_rho)[np.isfinite(lomo_rho)]
            sign_stable = bool(
                len(finite) == len(MULTI_MCS)
                and np.isfinite(observed_rho)
                and abs(observed_rho) > 0
                and np.all(np.sign(finite) == np.sign(observed_rho))
            )
            return {
                "n_finite_lomo": int(len(finite)),
                "lomo_sign_stable": sign_stable,
                "lomo_min_rho": float(finite.min()) if len(finite) else np.nan,
                "lomo_max_rho": float(finite.max()) if len(finite) else np.nan,
                "lomo_min_abs_rho": float(np.abs(finite).min()) if len(finite) else np.nan,
            }

        def build_program_output_correlations(
            output_table, output_types, estimand, analysis_role
        ):
            wide = (
                output_table.pivot_table(
                    index="metaclone", columns=["day", "cell_type_weinreb"],
                    values="fraction", observed=True
                )
                .reindex(index=MULTI_MCS)
            )
            program_matrix = (
                multilineage_day2_program_means.set_index("metaclone")
                .loc[MULTI_MCS, program_columns]
            )
            rows = []
            for temporal_scope in ["Day 4", "Day 6", "Day 6 minus Day 4"]:
                for output_type in output_types:
                    day4 = wide[(4, output_type)].to_numpy(float)
                    day6 = wide[(6, output_type)].to_numpy(float)
                    if temporal_scope == "Day 4":
                        outcome = day4
                    elif temporal_scope == "Day 6":
                        outcome = day6
                    else:
                        outcome = day6 - day4
                    for program in program_columns:
                        predictor = program_matrix[program].to_numpy(float)
                        rho, exact_pvalue = exact_spearman_permutation(
                            predictor, outcome
                        )
                        lomo = lomo_spearman_summary(predictor, outcome, rho)
                        rows.append({
                            "estimand": estimand,
                            "analysis_role": analysis_role,
                            "temporal_scope": temporal_scope,
                            "output_type": output_type,
                            "program": program,
                            "program_label": clean_program_name(program),
                            "n_metaclones": int(
                                np.sum(np.isfinite(predictor) & np.isfinite(outcome))
                            ),
                            "n_nonzero_output_metaclones": int(
                                np.sum(np.abs(outcome) > 1e-12)
                            ),
                            "n_unique_output_values": int(
                                len(np.unique(np.round(outcome[np.isfinite(outcome)], 12)))
                            ),
                            "spearman_rho": rho,
                            "exact_permutation_pvalue": exact_pvalue,
                            "n_label_permutations": len(PERMUTATION_INDEX),
                            **lomo,
                        })
            result = pd.DataFrame(rows)
            result["endpoint_fdr"] = result.groupby(
                ["temporal_scope", "output_type"],
                observed=True,
                group_keys=False,
            ).exact_permutation_pvalue.transform(bh_adjust)
            result["temporal_scope_fdr"] = result.groupby(
                "temporal_scope", observed=True, group_keys=False
            ).exact_permutation_pvalue.transform(bh_adjust)
            result["analysis_wide_fdr"] = bh_adjust(
                result.exact_permutation_pvalue.to_numpy(float)
            )
            return result

        primary_correlations = build_program_output_correlations(
            multilineage_output_clone_balanced,
            AUTHOR_OUTPUT_TYPES,
            "clone_balanced_all_observed_cells",
            "primary",
        )
        pooled_correlations = build_program_output_correlations(
            multilineage_output_pooled_cells,
            AUTHOR_OUTPUT_TYPES,
            "pooled_cells_all_observed_cells",
            "sensitivity",
        )
        mature_correlations = build_program_output_correlations(
            multilineage_output_mature_clone_balanced,
            MATURE_OUTPUT_TYPES,
            "clone_balanced_mature_only",
            "sensitivity",
        )

        correlation_key = ["temporal_scope", "output_type", "program"]
        primary_correlations = primary_correlations.merge(
            pooled_correlations[correlation_key + ["spearman_rho"]].rename(
                columns={"spearman_rho": "pooled_cell_sensitivity_rho"}
            ),
            on=correlation_key,
            how="left",
            validate="one_to_one",
        )
        primary_correlations = primary_correlations.merge(
            mature_correlations[correlation_key + ["spearman_rho"]].rename(
                columns={"spearman_rho": "mature_only_sensitivity_rho"}
            ),
            on=correlation_key,
            how="left",
            validate="one_to_one",
        )
        primary_correlations["pooled_direction_agrees"] = (
            np.sign(primary_correlations.spearman_rho)
            == np.sign(primary_correlations.pooled_cell_sensitivity_rho)
        )
        primary_correlations.loc[
            primary_correlations.pooled_cell_sensitivity_rho.isna(),
            "pooled_direction_agrees",
        ] = False
        primary_correlations["mature_direction_agrees"] = (
            np.sign(primary_correlations.spearman_rho)
            == np.sign(primary_correlations.mature_only_sensitivity_rho)
        )
        primary_correlations.loc[
            primary_correlations.mature_only_sensitivity_rho.isna(),
            "mature_direction_agrees",
        ] = False
        primary_correlations["stable_endpoint_association"] = (
            (primary_correlations.endpoint_fdr < 0.05)
            & primary_correlations.lomo_sign_stable
            & primary_correlations.pooled_direction_agrees
        )

        all_program_output_correlations = pd.concat(
            [primary_correlations, pooled_correlations, mature_correlations],
            ignore_index=True,
            sort=False,
        )
        primary_correlations.to_csv(
            TABLEDIR / "multilineage_program_output_correlations.csv", index=False
        )
        all_program_output_correlations.to_csv(
            TABLEDIR / "multilineage_program_output_correlation_sensitivities.csv",
            index=False,
        )

        correlation_summary = (
            all_program_output_correlations.groupby(
                ["estimand", "analysis_role", "temporal_scope"], observed=True
            )
            .agg(
                n_tests=("program", "size"),
                n_estimable=("exact_permutation_pvalue", lambda x: int(x.notna().sum())),
                n_nominal_p_lt_0_05=(
                    "exact_permutation_pvalue", lambda x: int((x < 0.05).sum())
                ),
                n_endpoint_fdr_lt_0_05=(
                    "endpoint_fdr", lambda x: int((x < 0.05).sum())
                ),
                n_analysis_wide_fdr_lt_0_05=(
                    "analysis_wide_fdr", lambda x: int((x < 0.05).sum())
                ),
                n_lomo_sign_stable=(
                    "lomo_sign_stable", lambda x: int(x.fillna(False).sum())
                ),
            )
            .reset_index()
        )
        correlation_summary.to_csv(
            TABLEDIR / "multilineage_program_output_correlation_summary.csv",
            index=False,
        )
        display(correlation_summary)
        """
    ),
    code(
        r"""
        # Global distance association between the complete Day-2 program profile and output composition
        program_profile = (
            multilineage_day2_program_means.set_index("metaclone")
            .loc[MULTI_MCS, program_columns]
            .to_numpy(float)
        )
        program_profile_sd = program_profile.std(axis=0, ddof=1)
        variable_programs = program_profile_sd > 1e-12
        program_profile_z = (
            (program_profile[:, variable_programs]
             - program_profile[:, variable_programs].mean(axis=0))
            / program_profile_sd[variable_programs]
        )
        program_distance_matrix = squareform(pdist(program_profile_z, metric="euclidean"))
        upper_triangle = np.triu_indices(len(MULTI_MCS), k=1)
        program_distances = program_distance_matrix[upper_triangle]

        def exact_distance_association(output_table, output_types, estimand):
            rows = []
            for day in OUTPUT_DAYS:
                output_matrix = (
                    output_table[output_table.day == day]
                    .pivot_table(
                        index="metaclone", columns="cell_type_weinreb",
                        values="fraction", observed=True
                    )
                    .reindex(index=MULTI_MCS, columns=output_types)
                    .to_numpy(float)
                )
                if not np.isfinite(output_matrix).all():
                    rows.append({
                        "estimand": estimand, "day": day,
                        "spearman_distance_rho": np.nan,
                        "exact_permutation_pvalue": np.nan,
                        "n_label_permutations": len(PERMUTATION_INDEX),
                    })
                    continue
                output_distance_matrix = squareform(
                    pdist(
                        output_matrix,
                        metric=lambda left, right: jensenshannon(
                            left, right, base=2
                        ),
                    )
                )
                output_distances = output_distance_matrix[upper_triangle]
                observed = float(
                    spearmanr(program_distances, output_distances).statistic
                )
                permuted_rho = []
                for permutation in PERMUTATION_INDEX:
                    permuted_matrix = output_distance_matrix[
                        np.ix_(permutation, permutation)
                    ]
                    permuted_rho.append(
                        spearmanr(
                            program_distances, permuted_matrix[upper_triangle]
                        ).statistic
                    )
                permuted_rho = np.asarray(permuted_rho, dtype=float)
                exact_pvalue = float(
                    np.mean(np.abs(permuted_rho) >= abs(observed) - 1e-12)
                )
                rows.append({
                    "estimand": estimand,
                    "day": day,
                    "spearman_distance_rho": observed,
                    "exact_permutation_pvalue": exact_pvalue,
                    "n_label_permutations": len(PERMUTATION_INDEX),
                    "n_programs_in_distance": int(variable_programs.sum()),
                    "n_output_types_in_distance": len(output_types),
                })
            result = pd.DataFrame(rows)
            result["fdr_within_estimand"] = bh_adjust(
                result.exact_permutation_pvalue.to_numpy(float)
            )
            return result

        global_program_output_association = pd.concat([
            exact_distance_association(
                multilineage_output_clone_balanced,
                AUTHOR_OUTPUT_TYPES,
                "clone_balanced_all_observed_cells",
            ),
            exact_distance_association(
                multilineage_output_mature_clone_balanced,
                MATURE_OUTPUT_TYPES,
                "clone_balanced_mature_only",
            ),
        ], ignore_index=True)
        global_program_output_association.to_csv(
            TABLEDIR / "multilineage_program_output_global_association.csv",
            index=False,
        )
        display(global_program_output_association)
        """
    ),
    md(
        r"""
        ### Output composition and correlation views

        Each figure is generated in its own cell. Filled bars show the primary
        equal-clone output estimand. The heatmap and labelled scatter retain all
        seven metaclones; symbols distinguish nominal exact permutation evidence
        from endpoint-level FDR without implying analysis-wide discovery.
        """
    ),
    code(
        r"""
        # PROGRAM_OUTPUT_PLOT_INLINE_V1 — clone-balanced output composition
        plot_stem = "weinreb_day2_hspc_19_multilineage_clone_balanced_outputs"
        plot_output_map = {
            "Neutrophil": "Neutrophil",
            "Monocyte": "Monocyte",
            "Meg": "Erythroid / Mk",
            "Erythroid": "Erythroid / Mk",
            "Baso": "Basophil",
            "Mast": "Mast",
            "Eos": "Other mature",
            "Ccr7_DC": "Other mature",
            "Lymphoid": "Other mature",
            "Undifferentiated": "Undifferentiated",
        }
        plot_output_order = [
            "Neutrophil", "Monocyte", "Erythroid / Mk", "Basophil", "Mast",
            "Other mature", "Undifferentiated",
        ]
        plot_output_colors = {
            "Neutrophil": "#D4B441",
            "Monocyte": "#B23A48",
            "Erythroid / Mk": "#2E7DBF",
            "Basophil": "#3F9E4D",
            "Mast": "#8E6FB8",
            "Other mature": "#8693A3",
            "Undifferentiated": "#D8DDE3",
        }
        plot_data = multilineage_output_clone_balanced.copy()
        plot_data["output_class"] = plot_data.cell_type_weinreb.map(plot_output_map)
        plot_data = (
            plot_data.groupby(
                ["metaclone", "day", "output_class"], observed=True
            ).fraction.sum().reset_index()
        )
        plot_data.to_csv(
            TABLEDIR / f"{plot_stem}_plot_data.csv", index=False
        )
        plot_row_order = [(mc, day) for mc in MULTI_MCS for day in OUTPUT_DAYS]
        plot_matrix = (
            plot_data.pivot_table(
                index=["metaclone", "day"], columns="output_class",
                values="fraction", fill_value=0.0, observed=True
            )
            .reindex(
                index=pd.MultiIndex.from_tuples(
                    plot_row_order, names=["metaclone", "day"]
                ),
                columns=plot_output_order,
                fill_value=0.0,
            )
        )
        plot_support = multilineage_output_support.set_index(
            ["metaclone", "day"]
        )
        y_positions = []
        for mc_index in range(len(MULTI_MCS)):
            base = (len(MULTI_MCS) - 1 - mc_index) * 2.7
            y_positions.extend([base + 0.55, base - 0.55])
        y_positions = np.asarray(y_positions, dtype=float)

        fig, ax = plt.subplots(figsize=(10.8, 7.8))
        left = np.zeros(len(plot_matrix), dtype=float)
        for output_class in plot_output_order:
            values = plot_matrix[output_class].to_numpy(float)
            ax.barh(
                y_positions, values, left=left, height=0.82,
                color=plot_output_colors[output_class], edgecolor=WHITE,
                linewidth=0.45, label=output_class,
            )
            left += values
        row_labels = [
            f"MC{mc} · D{day}  n={int(plot_support.loc[(mc, day), 'n_clones'])}"
            for mc, day in plot_row_order
        ]
        ax.set_yticks(y_positions, row_labels, fontsize=8.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean within-clone fraction of observed cells")
        ax.set_title(
            "Clone-balanced cell-type output · Multi-lineage metaclones",
            loc="left", pad=14,
        )
        ax.text(
            0, 1.012,
            "Author D4/D6 annotations · each observed clone contributes equally",
            transform=ax.transAxes, fontsize=8.6, color=GREY,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.legend(
            title="Output class", loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=False, fontsize=8.5, title_fontsize=9,
        )
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # PROGRAM_OUTPUT_PLOT_INLINE_V1 — top exact program/output correlations
        plot_stem = "weinreb_day2_hspc_20_multilineage_program_output_correlations"
        plot_correlations = primary_correlations[
            primary_correlations.temporal_scope.isin(["Day 4", "Day 6"])
            & primary_correlations.exact_permutation_pvalue.notna()
        ].copy()
        program_priority = (
            plot_correlations.groupby("program", observed=True)
            .agg(
                minimum_endpoint_fdr=("endpoint_fdr", "min"),
                minimum_exact_p=("exact_permutation_pvalue", "min"),
                maximum_abs_rho=("spearman_rho", lambda x: float(np.abs(x).max())),
            )
            .sort_values(
                ["minimum_endpoint_fdr", "minimum_exact_p", "maximum_abs_rho"],
                ascending=[True, True, False],
            )
        )
        plot_programs = program_priority.head(12).index.tolist()
        plot_correlations = plot_correlations[
            plot_correlations.program.isin(plot_programs)
        ].copy()
        plot_correlations["output_time"] = (
            plot_correlations.temporal_scope.str.replace("Day ", "D", regex=False)
            + " · " + plot_correlations.output_type
        )
        plot_column_order = [
            f"D{day} · {output_type}"
            for day in OUTPUT_DAYS for output_type in AUTHOR_OUTPUT_TYPES
        ]
        plot_matrix = (
            plot_correlations.pivot_table(
                index="program", columns="output_time", values="spearman_rho",
                observed=True
            )
            .reindex(index=plot_programs, columns=plot_column_order)
        )
        plot_correlations.to_csv(
            TABLEDIR / f"{plot_stem}_plot_data.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(14.8, 7.2))
        mesh = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(float),
            cmap=PROGRAM_CMAP, vmin=-1, vmax=1,
            edgecolors=WHITE, linewidth=0.35, shading="flat",
        )
        endpoint_hit_lookup = plot_correlations.set_index(
            ["program", "output_time"]
        ).endpoint_fdr
        nominal_lookup = plot_correlations.set_index(
            ["program", "output_time"]
        ).exact_permutation_pvalue
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, output_time in enumerate(plot_matrix.columns):
                endpoint_q = endpoint_hit_lookup.get((program, output_time), np.nan)
                exact_p = nominal_lookup.get((program, output_time), np.nan)
                symbol = (
                    "★"
                    if np.isfinite(endpoint_q) and endpoint_q < 0.05
                    else (
                        "•"
                        if np.isfinite(exact_p) and exact_p < 0.05
                        else ""
                    )
                )
                if symbol:
                    ax.text(
                        column_index + 0.5, row_index + 0.5, symbol,
                        ha="center", va="center", fontsize=8.5, color=INK,
                    )
        ax.set_xticks(
            np.arange(len(plot_column_order)) + 0.5,
            plot_column_order, rotation=58, ha="right", fontsize=7.5,
        )
        ax.set_yticks(
            np.arange(len(plot_programs)) + 0.5,
            [clean_program_name(program) for program in plot_programs],
            fontsize=8.4,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Author cell type and output day")
        ax.set_ylabel("Day-2 Hallmark program")
        ax.set_title(
            "Metaclone-level Day-2 program versus later cell-type output",
            loc="left", pad=14,
        )
        ax.text(
            0, 1.012,
            "Exact 7! Spearman tests · • nominal P < 0.05 · ★ endpoint FDR < 0.05",
            transform=ax.transAxes, fontsize=8.6, color=GREY,
        )
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Spearman rho across seven metaclones")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # PROGRAM_OUTPUT_PLOT_INLINE_V1 — strongest LOMO-stable endpoint association
        plot_stem = "weinreb_day2_hspc_21_multilineage_strongest_program_output_scatter"
        plot_candidates = primary_correlations[
            primary_correlations.temporal_scope.isin(["Day 4", "Day 6"])
            & primary_correlations.exact_permutation_pvalue.notna()
            & primary_correlations.lomo_sign_stable
        ].copy()
        if plot_candidates.empty:
            plot_candidates = primary_correlations[
                primary_correlations.temporal_scope.isin(["Day 4", "Day 6"])
                & primary_correlations.exact_permutation_pvalue.notna()
            ].copy()
        plot_candidates["abs_rho"] = plot_candidates.spearman_rho.abs()
        strongest = plot_candidates.sort_values(
            ["analysis_wide_fdr", "endpoint_fdr", "exact_permutation_pvalue", "abs_rho"],
            ascending=[True, True, True, False],
        ).iloc[0]
        plot_day = int(str(strongest.temporal_scope).replace("Day ", ""))
        plot_program = str(strongest.program)
        plot_output_type = str(strongest.output_type)
        plot_x = (
            multilineage_day2_program_means.set_index("metaclone")
            .loc[MULTI_MCS, plot_program]
        )
        plot_y = (
            multilineage_output_clone_balanced[
                (multilineage_output_clone_balanced.day == plot_day)
                & (multilineage_output_clone_balanced.cell_type_weinreb == plot_output_type)
            ]
            .set_index("metaclone")
            .reindex(MULTI_MCS).fraction
        )
        plot_data = pd.DataFrame({
            "metaclone": MULTI_MCS,
            "day2_program_score_sd": plot_x.to_numpy(float),
            "clone_balanced_output_fraction": plot_y.to_numpy(float),
            "program": plot_program,
            "output_type": plot_output_type,
            "day": plot_day,
        })
        plot_data.to_csv(
            TABLEDIR / f"{plot_stem}_plot_data.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        ax.scatter(
            plot_data.day2_program_score_sd,
            plot_data.clone_balanced_output_fraction,
            s=74, color=ORANGE, edgecolor=INK, linewidth=0.75, zorder=3,
        )
        if plot_data.day2_program_score_sd.nunique() > 1:
            slope, intercept = np.polyfit(
                plot_data.day2_program_score_sd,
                plot_data.clone_balanced_output_fraction,
                1,
            )
            x_line = np.linspace(
                plot_data.day2_program_score_sd.min(),
                plot_data.day2_program_score_sd.max(), 100
            )
            ax.plot(
                x_line, intercept + slope * x_line,
                color=INK, linewidth=1.15, linestyle="--", zorder=2,
            )
        for row in plot_data.itertuples(index=False):
            ax.annotate(
                f"MC{int(row.metaclone)}",
                (row.day2_program_score_sd, row.clone_balanced_output_fraction),
                xytext=(5, 5), textcoords="offset points", fontsize=8.3,
                color=INK,
            )
        ax.set_xlabel(f"Day-2 {clean_program_name(plot_program)} score (adjusted SD)")
        ax.set_ylabel(f"D{plot_day} clone-balanced {plot_output_type} fraction")
        ax.set_title(
            f"{clean_program_name(plot_program)} versus {plot_output_type} output · D{plot_day}",
            loc="left", pad=14,
        )
        ax.text(
            0, 1.012,
            (
                f"Spearman rho={strongest.spearman_rho:+.2f} · "
                f"exact P={strongest.exact_permutation_pvalue:.3g} · "
                f"endpoint q={strongest.endpoint_fdr:.3g} · "
                f"analysis-wide q={strongest.analysis_wide_fdr:.3g}"
            ),
            transform=ax.transAxes, fontsize=8.4, color=GREY,
        )
        ax.grid(color=LIGHT_GREY, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    md(
        r"""
        ## Multi-lineage correlation findings and boundaries

        Findings below are generated from the executed result tables. The
        distinction between analysis-wide evidence, endpoint-level exploratory
        evidence, and nominal rankings is kept explicit.
        """
    ),
    code(
        r"""
        # Print and save computed findings without hard-coding results in Markdown
        valid_primary = primary_correlations[
            primary_correlations.exact_permutation_pvalue.notna()
        ].copy()
        analysis_wide_hits = valid_primary[
            valid_primary.analysis_wide_fdr < 0.05
        ].copy()
        endpoint_hits = valid_primary[
            valid_primary.endpoint_fdr < 0.05
        ].copy()
        stable_endpoint_hits = valid_primary[
            valid_primary.stable_endpoint_association
        ].copy()
        top_correlations = valid_primary.sort_values(
            ["analysis_wide_fdr", "endpoint_fdr", "exact_permutation_pvalue"],
            ascending=True,
        ).head(12)
        output_support_min = int(multilineage_output_support.n_clones.min())
        output_support_max = int(multilineage_output_support.n_clones.max())
        mature_support_min = int(
            multilineage_output_support.n_mature_informative_clones.min()
        )
        mature_support_max = int(
            multilineage_output_support.n_mature_informative_clones.max()
        )

        finding_lines = [
            "OBSERVED METACLONE-LEVEL EVIDENCE",
            (
                f"- Seven current Multi-lineage metaclones were tested with 50 "
                f"library-adjusted Day-2 clone-pseudobulk Hallmark scores."
            ),
            (
                f"- Primary D4/D6 and D6-minus-D4 output correlations: "
                f"{len(primary_correlations)} rows; {len(valid_primary)} estimable; "
                f"{len(analysis_wide_hits)} passed analysis-wide FDR < 0.05; "
                f"{len(endpoint_hits)} passed cell-type/time endpoint FDR < 0.05; "
                f"{len(stable_endpoint_hits)} also retained LOMO sign and pooled-output direction."
            ),
            (
                f"- Observed-clone support per metaclone/day ranged from "
                f"{output_support_min} to {output_support_max}; mature-informative "
                f"clone support ranged from {mature_support_min} to {mature_support_max}."
            ),
        ]
        for row in global_program_output_association.itertuples(index=False):
            finding_lines.append(
                f"- Global {row.estimand}, Day {int(row.day)}: "
                f"distance rho={row.spearman_distance_rho:+.2f}, "
                f"exact P={row.exact_permutation_pvalue:.3g}, "
                f"within-estimand q={row.fdr_within_estimand:.3g}."
            )
        finding_lines.append("- Highest-ranked individual correlations:")
        for row in top_correlations.itertuples(index=False):
            finding_lines.append(
                f"  - {row.program_label} vs {row.output_type} ({row.temporal_scope}): "
                f"rho={row.spearman_rho:+.2f}, exact P={row.exact_permutation_pvalue:.3g}, "
                f"endpoint q={row.endpoint_fdr:.3g}, analysis-wide q={row.analysis_wide_fdr:.3g}, "
                f"LOMO-sign-stable={bool(row.lomo_sign_stable)}."
            )
        finding_lines.extend([
            "",
            "INTERPRETATION BOUNDARIES",
            "- The independent sample size is seven metaclones; exact permutation P-values are discrete and effect sizes are more informative than asymptotic precision.",
            "- Output fractions are compositional, so enrichment of one cell type necessarily reduces at least one other fraction.",
            "- The Multi-lineage family was defined retrospectively from later output; correlations are descriptive associations, not prospective validation.",
            "- Mature-only composition omits clones without a mature sister at that day and is a sensitivity, not the primary estimand.",
            "- Day-6-minus-Day-4 differences compare destructive snapshots with partially different observed-clone cohorts and cannot establish within-clone temporal change.",
        ])
        correlation_findings_text = "\n".join(finding_lines)
        (REPORTDIR / "multilineage_program_output_findings_and_boundaries.txt").write_text(
            correlation_findings_text + "\n"
        )
        print(correlation_findings_text)
        display(top_correlations[[
            "temporal_scope", "output_type", "program_label", "spearman_rho",
            "exact_permutation_pvalue", "endpoint_fdr", "analysis_wide_fdr",
            "lomo_sign_stable", "pooled_direction_agrees", "mature_direction_agrees",
        ]])
        """
    ),
    md(
        r"""
        ## Multi-lineage correlation validation and provenance

        This final cell validates the complete metaclone/output grids, exhaustive
        permutation implementation, multiplicity fields, physical absence of the
        removed analysis, plot-cell contract, editable SVG exports, and artifact
        hashes.
        """
    ),
    code(
        r"""
        # Validate the appended correlation analysis and its reader-facing visibility
        built_notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        notebook_sources = ["".join(cell.get("source", "")) for cell in built_notebook.cells]
        removed_tag = "cell-" + "level-not-for-use"
        removed_markers = [
            "## Version 2 — " + "cell-level sensitivity",
            "CELL_" + "LEVEL_PLOT_INLINE_V1",
            "cell_" + "level_program_tests",
            "cell_" + "level_gene_tests",
        ]
        removed_marker_hits = [
            marker for marker in removed_markers
            if any(marker in source for source in notebook_sources)
        ]
        removed_tagged_cells = sum(
            removed_tag in cell.metadata.get("tags", [])
            for cell in built_notebook.cells
        )
        removed_active_artifacts = sorted(
            str(path.relative_to(ANALYSIS_DIR))
            for folder in [FIGDIR, TABLEDIR, REPORTDIR]
            for path in folder.glob("*cell_level*")
        )
        cell_level_analysis_absent = (
            not removed_marker_hits
            and removed_tagged_cells == 0
            and not removed_active_artifacts
        )

        correlation_plot_stems = [
            "weinreb_day2_hspc_19_multilineage_clone_balanced_outputs",
            "weinreb_day2_hspc_20_multilineage_program_output_correlations",
            "weinreb_day2_hspc_21_multilineage_strongest_program_output_scatter",
        ]
        correlation_plot_sources = [
            source for source in notebook_sources
            if source.lstrip().startswith("# PROGRAM_OUTPUT_PLOT_INLINE_V1")
        ]
        correlation_plot_contract = all(
            source.count("plt.subplots") == 1
            and source.count("fig.savefig(svg") == 1
            and source.count("fig.savefig(png") == 1
            and source.count("display(fig)") == 1
            and source.count("plt.close(fig)") == 1
            for source in correlation_plot_sources
        )
        correlation_svg_editable = all(
            "<svg" in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
            and "<image" not in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
            for stem in correlation_plot_stems
        )
        finite_exact_p = primary_correlations.exact_permutation_pvalue.dropna()
        exact_p_grid_ok = np.allclose(
            finite_exact_p.to_numpy() * len(PERMUTATION_INDEX),
            np.rint(finite_exact_p.to_numpy() * len(PERMUTATION_INDEX)),
            atol=1e-10,
        )
        correlation_numeric_probability_columns = [
            "exact_permutation_pvalue", "endpoint_fdr", "temporal_scope_fdr",
            "analysis_wide_fdr",
        ]
        probability_ranges_ok = all(
            primary_correlations[column].dropna().between(0, 1).all()
            for column in correlation_numeric_probability_columns
        )
        new_table_paths = [
            TABLEDIR / "multilineage_output_support_by_day.csv",
            TABLEDIR / "multilineage_clone_balanced_author_output_by_day.csv",
            TABLEDIR / "multilineage_pooled_cell_author_output_by_day.csv",
            TABLEDIR / "multilineage_clone_balanced_mature_output_by_day.csv",
            TABLEDIR / "multilineage_day2_adjusted_program_means.csv",
            TABLEDIR / "multilineage_program_output_correlations.csv",
            TABLEDIR / "multilineage_program_output_correlation_sensitivities.csv",
            TABLEDIR / "multilineage_program_output_correlation_summary.csv",
            TABLEDIR / "multilineage_program_output_global_association.csv",
        ] + [TABLEDIR / f"{stem}_plot_data.csv" for stem in correlation_plot_stems]
        new_table_headers_exclude_population = all(
            "population" not in pd.read_csv(path, nrows=0).columns
            for path in new_table_paths
        )

        multilineage_correlation_validation_rows = [
            ("seven_multilineage_metaclones", MULTI_MCS == [0, 2, 5, 3, 18, 4, 11], str(MULTI_MCS)),
            ("output_day_grid_complete", len(multilineage_output_support) == 14, str(multilineage_output_support.shape)),
            ("primary_output_grid_complete", len(multilineage_output_clone_balanced) == 7 * 2 * 10, str(multilineage_output_clone_balanced.shape)),
            ("mature_output_grid_complete", len(multilineage_output_mature_clone_balanced) == 7 * 2 * 9, str(multilineage_output_mature_clone_balanced.shape)),
            ("clone_balanced_fractions_sum_to_one", np.allclose(primary_fraction_sums, 1.0), f"max_error={float(np.max(np.abs(primary_fraction_sums - 1))):.3g}"),
            ("mature_fractions_sum_to_one", np.allclose(mature_fraction_sums, 1.0), f"max_error={float(np.max(np.abs(mature_fraction_sums - 1))):.3g}"),
            ("day2_program_grid_complete", multilineage_day2_program_means.shape == (7, 51), str(multilineage_day2_program_means.shape)),
            ("primary_correlation_grid_complete", len(primary_correlations) == 50 * 10 * 3, f"observed={len(primary_correlations)} expected=1500"),
            ("exhaustive_permutation_grid", PERMUTATION_INDEX.shape == (5040, 7), str(PERMUTATION_INDEX.shape)),
            ("exact_pvalue_grid", exact_p_grid_ok, "all finite exact P values are multiples of 1/5040"),
            ("probability_ranges", probability_ranges_ok, "all finite P/q values within [0,1]"),
            ("global_association_grid", len(global_program_output_association) == 4, str(global_program_output_association.shape)),
            (
                "cell_level_analysis_absent",
                cell_level_analysis_absent,
                f"marker_hits={removed_marker_hits}; tagged_cells={removed_tagged_cells}; active_artifacts={removed_active_artifacts}",
            ),
            ("correlation_plot_cell_count", len(correlation_plot_sources) == len(correlation_plot_stems), f"observed={len(correlation_plot_sources)} expected={len(correlation_plot_stems)}"),
            ("correlation_plot_inline_contract", correlation_plot_contract, "one inline build/save/display/close per plot cell"),
            ("correlation_svg_outputs", all((FIGDIR / f"{stem}.svg").stat().st_size > 0 for stem in correlation_plot_stems), f"expected={len(correlation_plot_stems)}"),
            ("correlation_png_outputs", all((FIGDIR / f"{stem}.png").stat().st_size > 0 for stem in correlation_plot_stems), f"expected={len(correlation_plot_stems)}"),
            ("correlation_svg_editable", correlation_svg_editable, "all new SVGs are vector-only with no embedded image"),
            ("new_tables_exclude_population", new_table_headers_exclude_population, f"checked={len(new_table_paths)}"),
            ("input_landing_zone_empty", not any((PROJECT_ROOT / "Input").iterdir()), "Input/ contains no registered-intake residue"),
        ]
        multilineage_correlation_validation = pd.DataFrame(
            multilineage_correlation_validation_rows,
            columns=["check", "passed", "detail"],
        )
        multilineage_correlation_validation.to_csv(
            REPORTDIR / "multilineage_program_output_validation.csv", index=False
        )
        assert multilineage_correlation_validation.passed.all(), (
            multilineage_correlation_validation.loc[
                ~multilineage_correlation_validation.passed
            ]
        )

        correlation_artifact_paths = [BUILDER_PATH] + new_table_paths
        for stem in correlation_plot_stems:
            correlation_artifact_paths.extend([
                FIGDIR / f"{stem}.svg", FIGDIR / f"{stem}.png"
            ])
        correlation_artifact_paths.extend([
            REPORTDIR / "multilineage_program_output_findings_and_boundaries.txt",
            REPORTDIR / "multilineage_program_output_validation.csv",
        ])
        multilineage_correlation_manifest = pd.DataFrame([
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(correlation_artifact_paths))
        ])
        multilineage_correlation_manifest.to_csv(
            REPORTDIR / "multilineage_program_output_artifact_manifest.csv",
            index=False,
        )

        # Refresh the broad manifest only after the correlation outputs and its
        # dedicated manifest have reached their delivered state.
        active_artifact_paths = []
        for folder in [ANALYSIS_DIR / "scripts", FIGDIR, TABLEDIR, REPORTDIR, RESOURCEDIR]:
            for path in sorted(folder.rglob("*")):
                if (path.is_file() and path.name != "artifact_manifest.csv"
                        and path.suffix != ".pyc" and "__pycache__" not in path.parts):
                    active_artifact_paths.append(path)
        artifact_manifest = pd.DataFrame([
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in active_artifact_paths
        ])
        artifact_manifest.to_csv(REPORTDIR / "artifact_manifest.csv", index=False)

        display(multilineage_correlation_validation)
        display(multilineage_correlation_manifest)
        print("All Multi-lineage program-output validation checks passed.")
        print("Notebook:", NOTEBOOK_PATH)
        print("Environment: chronoclone |", ENV_PATH)
        print("Execution command:", EXECUTION_COMMAND)
        """
    ),
]

# The targeted behavior-signature analysis is appended after the retained broad
# correlation sensitivity so it can reuse the exact clone-balanced output tables.
# It is the manuscript-facing primary analysis despite its later notebook position.
cells.extend([
    md(
        r"""
        ## Targeted metaclonal behavior signatures

        Four non-exclusive behavioral axes are locked from clone-balanced Day-6
        author annotations before their targeted expression contrasts are computed
        or inspected. These retrospective, analysis-defined axes are not prospectively
        preregistered. The independent expression
        unit is one Day-2 pooled-HSPC LARRY clone pseudobulk. The seven Multi-lineage
        metaclones contain 217 such units. The gene universe remains the original
        group-blind 9,777-gene filter and the program universe remains all 50 fixed
        Hallmarks.

        A *shared behavioral signature* requires a globally significant equal-MC
        focal-pair contrast, same-direction effects in both focal metaclones, a
        globally significant max-P conjunction test, same-direction combined and
        member effects after removal of every comparator MC, and same-direction
        combined and member effects in every estimable leave-one-library-out refit.
        Combined-only results are candidates, not shared signatures.
        """
    ),
    code(
        r"""
        # Lock the seven-MC cohort and behavior-axis registry before targeted tests
        TARGET_MCS = [0, 2, 5, 3, 18, 4, 11]
        assert TARGET_MCS == MULTI_MCS
        BEHAVIOR_AXES = [
            {
                "axis_id": "baso_rich", "axis_label": "Baso-rich",
                "focal_mcs": [2, 4], "output_type": "Baso",
                "threshold": 0.10, "rule": "Day-6 Baso fraction >= 10%",
            },
            {
                "axis_id": "neu_rich", "axis_label": "Neu-rich",
                "focal_mcs": [0, 5], "output_type": "Neutrophil",
                "threshold": 0.60, "rule": "Day-6 Neutrophil fraction >= 60%",
            },
            {
                "axis_id": "mono_rich", "axis_label": "Mono-rich",
                "focal_mcs": [0, 3], "output_type": "Monocyte",
                "threshold": 0.20, "rule": "Day-6 Monocyte fraction >= 20%",
            },
            {
                "axis_id": "persistent_undifferentiated",
                "axis_label": "Persistent-undifferentiated",
                "focal_mcs": [11, 18], "output_type": "Undifferentiated",
                "threshold": 0.75,
                "rule": "Day-6 Undifferentiated fraction >= 75%",
            },
        ]

        day6_output_lookup = (
            multilineage_output_clone_balanced[
                multilineage_output_clone_balanced.day == 6
            ]
            .set_index(["cell_type_weinreb", "metaclone"]).fraction
        )
        axis_definition_rows = []
        axis_registry_rows = []
        for axis in BEHAVIOR_AXES:
            fractions = pd.Series({
                mc: float(day6_output_lookup.loc[(axis["output_type"], mc)])
                for mc in TARGET_MCS
            })
            focal = list(axis["focal_mcs"])
            comparators = [mc for mc in TARGET_MCS if mc not in focal]
            assert set(fractions[fractions >= axis["threshold"]].index) == set(focal)
            assert fractions[focal].min() >= axis["threshold"]
            assert fractions[comparators].max() < axis["threshold"]
            axis_definition_rows.append({
                "axis_id": axis["axis_id"], "axis_label": axis["axis_label"],
                "output_type": axis["output_type"], "day": 6,
                "focal_mcs": "|".join(map(str, focal)),
                "comparator_mcs": "|".join(map(str, comparators)),
                "threshold": axis["threshold"], "rule": axis["rule"],
                "focal_min_fraction": float(fractions[focal].min()),
                "focal_max_fraction": float(fractions[focal].max()),
                "next_nonfocal_fraction": float(fractions[comparators].max()),
            })
            for mc in TARGET_MCS:
                axis_registry_rows.append({
                    "axis_id": axis["axis_id"], "axis_label": axis["axis_label"],
                    "output_type": axis["output_type"], "metaclone": mc,
                    "day6_clone_balanced_fraction": float(fractions[mc]),
                    "is_focal": mc in focal, "threshold": axis["threshold"],
                })
        behavior_axis_definitions = pd.DataFrame(axis_definition_rows)
        behavior_axis_registry = pd.DataFrame(axis_registry_rows)
        behavior_axis_definitions.to_csv(
            TABLEDIR / "targeted_behavior_axis_definitions.csv", index=False
        )
        behavior_axis_registry.to_csv(
            TABLEDIR / "targeted_behavior_axis_registry.csv", index=False
        )

        targeted_ids = pseudobulk_metadata.index[
            pseudobulk_metadata.metaclone.isin(TARGET_MCS)
        ]
        targeted_counts = filtered_counts.loc[targeted_ids].copy()
        targeted_metadata = pseudobulk_metadata.loc[targeted_ids].copy()
        targeted_metadata["multi_mc_group"] = pd.Categorical(
            targeted_metadata.metaclone.map(lambda value: f"MC{int(value)}"),
            categories=[f"MC{mc}" for mc in TARGET_MCS], ordered=True,
        )
        assert len(targeted_counts) == len(targeted_metadata) == 217
        assert targeted_counts.shape[1] == 9777
        assert not targeted_counts.index.duplicated().any()
        assert "population" not in targeted_metadata.columns
        assert set(targeted_metadata.metaclone.unique()) == set(TARGET_MCS)
        targeted_support = (
            targeted_metadata.groupby("metaclone", observed=True)
            .agg(n_clone_pseudobulks=("clone", "nunique"), n_day2_hspc_cells=("n_cells", "sum"))
            .reindex(TARGET_MCS).reset_index()
        )
        targeted_support.to_csv(
            TABLEDIR / "targeted_multilineage_day2_clone_support.csv", index=False
        )
        display(behavior_axis_definitions)
        display(targeted_support)
        """
    ),
    code(
        r"""
        # Build a full-rank seven-MC design with library fractions only
        def select_library_covariates(metadata, factor, suffix):
            metadata = metadata.copy()
            raw_columns = [
                column for column in metadata.columns
                if column.startswith("libfrac__") and not column.endswith("__z")
                and "__loo_" not in column
            ]
            reference = max(
                raw_columns,
                key=lambda column: (
                    (metadata.groupby(factor, observed=True)[column].mean() > 0).sum(),
                    metadata[column].mean(),
                ),
            )
            candidates = [
                column for column in raw_columns
                if column != reference and metadata[column].std(ddof=0) > 1e-12
            ]
            factor_dummies = pd.get_dummies(
                metadata[factor], drop_first=True, dtype=float
            )
            rank_matrix = np.column_stack([
                np.ones(len(metadata)), factor_dummies.to_numpy(float)
            ])
            selected, dropped = [], []
            for column in candidates:
                z = (
                    (metadata[column] - metadata[column].mean())
                    / metadata[column].std(ddof=0)
                )
                trial = np.column_stack([rank_matrix, z.to_numpy(float)])
                if np.linalg.matrix_rank(trial) > np.linalg.matrix_rank(rank_matrix):
                    name = f"{column}__{suffix}_z"
                    metadata[name] = z
                    selected.append(name)
                    rank_matrix = trial
                else:
                    dropped.append(column)
            return metadata, selected, reference, dropped

        targeted_metadata, targeted_library_covariates, targeted_reference_library, targeted_dropped = (
            select_library_covariates(targeted_metadata, "multi_mc_group", "multi")
        )
        targeted_design_formula = "~ " + " + ".join(
            targeted_library_covariates + ["multi_mc_group"]
        )
        with warnings.catch_warnings(record=True) as targeted_warnings:
            warnings.simplefilter("always")
            dds_targeted = DeseqDataSet(
                counts=targeted_counts.astype(int), metadata=targeted_metadata,
                design=targeted_design_formula, inference=inference,
                quiet=True, refit_cooks=False,
            )
            targeted_design = np.asarray(
                dds_targeted.obsm["design_matrix"], dtype=float
            )
            assert np.linalg.matrix_rank(targeted_design) == targeted_design.shape[1]
            dds_targeted.deseq2()
        targeted_fit_audit = pd.DataFrame([{
            "model": "seven_multilineage_metaclones",
            "design": targeted_design_formula,
            "n_clone_pseudobulks": len(targeted_metadata),
            "n_genes": targeted_counts.shape[1],
            "design_rank": int(np.linalg.matrix_rank(targeted_design)),
            "design_columns": int(targeted_design.shape[1]),
            "condition_number": float(np.linalg.cond(targeted_design)),
            "reference_library_fraction": targeted_reference_library,
            "library_covariates": "|".join(targeted_library_covariates),
            "rank_dropped": "|".join(targeted_dropped),
            "warning_count": len(targeted_warnings),
            "warnings": " | ".join(sorted({str(item.message) for item in targeted_warnings})),
        }])
        targeted_fit_audit.to_csv(
            TABLEDIR / "targeted_seven_mc_model_fit_audit.csv", index=False
        )
        display(targeted_fit_audit)
        """
    ),
    code(
        r"""
        # Register explicit target-minus-reference vectors for the primary axes
        def explicit_group_mean_vector(dds, factor, levels):
            return np.mean(np.vstack([
                np.asarray(dds.cond(**{factor: level}), dtype=float)
                for level in levels
            ]), axis=0)

        def explicit_target_minus_reference(dds, factor, targets, references):
            vector = (
                explicit_group_mean_vector(dds, factor, targets)
                - explicit_group_mean_vector(dds, factor, references)
            )
            design_columns = [
                str(column) for column in dds.obsm["design_matrix"].columns
            ]
            nonfactor_indices = [
                index for index, column in enumerate(design_columns)
                if factor not in column
            ]
            # Equal target/reference means cancel the intercept and every
            # library covariate exactly; only factor coefficients may remain.
            assert np.allclose(vector[nonfactor_indices], 0.0, atol=1e-12)
            if len(targets) == 1 and len(references) == 1:
                assert np.allclose(
                    vector,
                    np.asarray(dds.contrast(factor, references[0], targets[0]), dtype=float),
                    atol=1e-12,
                )
            return vector

        targeted_primary_specs = []
        for axis in BEHAVIOR_AXES:
            focal_levels = [f"MC{mc}" for mc in axis["focal_mcs"]]
            comparator_mcs = [mc for mc in TARGET_MCS if mc not in axis["focal_mcs"]]
            comparator_levels = [f"MC{mc}" for mc in comparator_mcs]
            targeted_primary_specs.append({
                "analysis": "primary_behavior_axis", "axis_id": axis["axis_id"],
                "axis_label": axis["axis_label"], "role": "combined",
                "member_mc": np.nan, "target_levels": focal_levels,
                "reference_levels": comparator_levels,
                "contrast_id": f"{axis['axis_id']}__focal_pair_vs_other_five",
                "contrast_label": f"{axis['axis_label']}: equal focal pair vs equal other five",
                "vector": explicit_target_minus_reference(
                    dds_targeted, "multi_mc_group", focal_levels, comparator_levels
                ),
            })
            for member_mc in axis["focal_mcs"]:
                targeted_primary_specs.append({
                    "analysis": "primary_behavior_axis", "axis_id": axis["axis_id"],
                    "axis_label": axis["axis_label"], "role": "member",
                    "member_mc": member_mc, "target_levels": [f"MC{member_mc}"],
                    "reference_levels": comparator_levels,
                    "contrast_id": f"{axis['axis_id']}__MC{member_mc}_vs_other_five",
                    "contrast_label": f"{axis['axis_label']}: MC{member_mc} vs equal other five",
                    "vector": explicit_target_minus_reference(
                        dds_targeted, "multi_mc_group", [f"MC{member_mc}"], comparator_levels
                    ),
                })

        targeted_secondary_specs = [
            {
                "analysis": "secondary_pair", "axis_id": "mc5_vs_mc3_divergence",
                "axis_label": "MC5-MC3 divergence", "role": "secondary",
                "member_mc": np.nan, "target_levels": ["MC5"],
                "reference_levels": ["MC3"], "contrast_id": "MC5_vs_MC3",
                "contrast_label": "MC5 vs MC3",
                "vector": explicit_target_minus_reference(
                    dds_targeted, "multi_mc_group", ["MC5"], ["MC3"]
                ),
            },
            {
                "analysis": "secondary_pair", "axis_id": "mc3_vs_mc18_convergence",
                "axis_label": "MC3-MC18 branch-compatibility convergence",
                "role": "secondary", "member_mc": np.nan,
                "target_levels": ["MC3"], "reference_levels": ["MC18"],
                "contrast_id": "MC3_vs_MC18", "contrast_label": "MC3 vs MC18",
                "vector": explicit_target_minus_reference(
                    dds_targeted, "multi_mc_group", ["MC3"], ["MC18"]
                ),
            },
        ]
        targeted_specs = targeted_primary_specs + targeted_secondary_specs
        targeted_contrast_registry = pd.DataFrame([{
            key: ("|".join(value) if key in {"target_levels", "reference_levels"} else value)
            for key, value in spec.items() if key != "vector"
        } for spec in targeted_specs])
        assert len(targeted_primary_specs) == 12
        assert len(targeted_secondary_specs) == 2
        targeted_contrast_registry.to_csv(
            TABLEDIR / "targeted_contrast_registry.csv", index=False
        )
        display(targeted_contrast_registry)
        """
    ),
    code(
        r"""
        # Gene-level DESeq2 effects for combined, member, and secondary contrasts
        def targeted_deseq_result(dds, spec):
            stats = DeseqStats(
                dds, contrast=np.asarray(spec["vector"], dtype=float),
                inference=inference, quiet=True, cooks_filter=False,
                independent_filter=False,
            )
            stats.summary()
            result = stats.results_df.reset_index().rename(columns={"index": "gene"})
            if "gene" not in result.columns:
                result = result.rename(columns={result.columns[0]: "gene"})
            result["pvalue"] = result.pvalue.fillna(1.0)
            result["padj"] = result.padj.fillna(1.0)
            for key in [
                "analysis", "axis_id", "axis_label", "role", "member_mc",
                "contrast_id", "contrast_label",
            ]:
                result[key] = spec[key]
            return result

        targeted_gene_parts = []
        for number, spec in enumerate(targeted_specs, start=1):
            targeted_gene_parts.append(targeted_deseq_result(dds_targeted, spec))
            print(f"targeted DESeq2 contrast {number}/{len(targeted_specs)}: {spec['contrast_id']}")
        targeted_gene_all = pd.concat(targeted_gene_parts, ignore_index=True)
        assert targeted_gene_all.gene.nunique() == 9777
        targeted_gene_all.to_csv(
            TABLEDIR / "targeted_gene_effects_all.csv.gz", index=False, compression="gzip"
        )

        primary_gene_combined = targeted_gene_all[
            (targeted_gene_all.analysis == "primary_behavior_axis")
            & (targeted_gene_all.role == "combined")
        ][[
            "axis_id", "axis_label", "gene", "baseMean", "log2FoldChange",
            "lfcSE", "stat", "pvalue", "padj",
        ]].rename(columns={
            "log2FoldChange": "combined_log2fc", "lfcSE": "combined_lfc_se",
            "stat": "combined_stat", "pvalue": "combined_pvalue",
            "padj": "combined_contrast_fdr",
        })
        primary_gene_members = targeted_gene_all[
            (targeted_gene_all.analysis == "primary_behavior_axis")
            & (targeted_gene_all.role == "member")
        ].copy()
        member_gene_parts = []
        for axis in BEHAVIOR_AXES:
            axis_members = primary_gene_members[
                primary_gene_members.axis_id == axis["axis_id"]
            ]
            left = axis_members[axis_members.member_mc == axis["focal_mcs"][0]][[
                "gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"
            ]].rename(columns={
                "log2FoldChange": "member1_log2fc", "lfcSE": "member1_lfc_se",
                "stat": "member1_stat", "pvalue": "member1_pvalue",
                "padj": "member1_contrast_fdr",
            })
            right = axis_members[axis_members.member_mc == axis["focal_mcs"][1]][[
                "gene", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"
            ]].rename(columns={
                "log2FoldChange": "member2_log2fc", "lfcSE": "member2_lfc_se",
                "stat": "member2_stat", "pvalue": "member2_pvalue",
                "padj": "member2_contrast_fdr",
            })
            merged = left.merge(right, on="gene", validate="one_to_one")
            merged["axis_id"] = axis["axis_id"]
            merged["member1_mc"] = axis["focal_mcs"][0]
            merged["member2_mc"] = axis["focal_mcs"][1]
            member_gene_parts.append(merged)
        member_gene_wide = pd.concat(member_gene_parts, ignore_index=True)
        targeted_gene_signatures = primary_gene_combined.merge(
            member_gene_wide, on=["axis_id", "gene"], validate="one_to_one"
        )
        targeted_gene_signatures["combined_global_fdr"] = bh_adjust(
            targeted_gene_signatures.combined_pvalue.to_numpy(float)
        )
        targeted_gene_signatures["conjunction_pvalue"] = targeted_gene_signatures[[
            "member1_pvalue", "member2_pvalue"
        ]].max(axis=1)
        targeted_gene_signatures["conjunction_global_fdr"] = bh_adjust(
            targeted_gene_signatures.conjunction_pvalue.to_numpy(float)
        )
        targeted_gene_signatures["members_same_direction_as_combined"] = (
            (np.sign(targeted_gene_signatures.member1_log2fc)
             == np.sign(targeted_gene_signatures.combined_log2fc))
            & (np.sign(targeted_gene_signatures.member2_log2fc)
               == np.sign(targeted_gene_signatures.combined_log2fc))
            & (targeted_gene_signatures.combined_log2fc != 0)
        )
        assert len(targeted_gene_signatures) == 4 * 9777
        assert targeted_gene_signatures.combined_global_fdr.notna().all()
        assert targeted_gene_signatures.conjunction_global_fdr.notna().all()

        targeted_secondary_genes = targeted_gene_all[
            targeted_gene_all.analysis == "secondary_pair"
        ].copy()
        targeted_secondary_genes["secondary_global_fdr"] = bh_adjust(
            targeted_secondary_genes.pvalue.to_numpy(float)
        )
        assert len(targeted_secondary_genes) == 2 * 9777
        targeted_secondary_genes.to_csv(
            TABLEDIR / "targeted_secondary_gene_contrasts.csv.gz",
            index=False, compression="gzip",
        )
        """
    ),
    code(
        r"""
        # Hallmark effects from the same seven-MC design and explicit vectors
        targeted_normed = np.asarray(
            dds_targeted.layers["normed_counts"], dtype=float
        )
        targeted_log = np.log2(targeted_normed + 1.0)
        targeted_gene_mean = targeted_log.mean(axis=0)
        targeted_gene_sd = targeted_log.std(axis=0, ddof=1)
        targeted_valid_sd = targeted_gene_sd > 1e-12
        targeted_gene_z = np.zeros_like(targeted_log, dtype=float)
        targeted_gene_z[:, targeted_valid_sd] = (
            (targeted_log[:, targeted_valid_sd] - targeted_gene_mean[targeted_valid_sd])
            / targeted_gene_sd[targeted_valid_sd]
        )
        fixed_program_indices = {}
        for program, requested_genes in hallmark_sets.items():
            fixed_program_indices[program] = sorted({
                index for gene in requested_genes
                for index in upper_to_indices.get(gene, [])
            })
            assert len(fixed_program_indices[program]) >= 10
        targeted_program_scores = pd.DataFrame({
            program: targeted_gene_z[:, indices].mean(axis=1)
            for program, indices in fixed_program_indices.items()
        }, index=targeted_metadata.index)
        targeted_program_scores = (
            (targeted_program_scores - targeted_program_scores.mean(axis=0))
            / targeted_program_scores.std(axis=0, ddof=1)
        )
        assert np.isfinite(targeted_program_scores.to_numpy()).all()
        targeted_program_export = targeted_metadata[[
            "clone", "metaclone", "multi_mc_group", "n_cells"
        ]].join(targeted_program_scores)
        targeted_program_export.to_csv(
            TABLEDIR / "targeted_multilineage_clone_hallmark_program_scores.csv.gz",
            index=False, compression="gzip",
        )
        targeted_program_design = dds_targeted.obsm["design_matrix"].copy()
        assert targeted_program_scores.index.tolist() == targeted_program_design.index.tolist()
        targeted_program_fits = {
            program: sm.OLS(
                targeted_program_scores[program].to_numpy(float),
                targeted_program_design.to_numpy(float),
            ).fit(cov_type="HC3")
            for program in program_columns
        }
        targeted_program_rows = []
        for spec in targeted_specs:
            for program, fit_result in targeted_program_fits.items():
                test = fit_result.t_test(np.asarray(spec["vector"], dtype=float))
                targeted_program_rows.append({
                    "analysis": spec["analysis"], "axis_id": spec["axis_id"],
                    "axis_label": spec["axis_label"], "role": spec["role"],
                    "member_mc": spec["member_mc"], "contrast_id": spec["contrast_id"],
                    "contrast_label": spec["contrast_label"], "program": program,
                    "program_label": clean_program_name(program),
                    "effect_sd": float(np.asarray(test.effect).squeeze()),
                    "standard_error": float(np.asarray(test.sd).squeeze()),
                    "t_stat": float(np.asarray(test.tvalue).squeeze()),
                    "pvalue": float(np.asarray(test.pvalue).squeeze()),
                })
        targeted_program_all = pd.DataFrame(targeted_program_rows)
        assert len(targeted_program_all) == 14 * 50
        targeted_program_all.to_csv(
            TABLEDIR / "targeted_hallmark_effects_all.csv", index=False
        )

        primary_program_combined = targeted_program_all[
            (targeted_program_all.analysis == "primary_behavior_axis")
            & (targeted_program_all.role == "combined")
        ][[
            "axis_id", "axis_label", "program", "program_label", "effect_sd",
            "standard_error", "t_stat", "pvalue",
        ]].rename(columns={
            "effect_sd": "combined_effect_sd", "standard_error": "combined_se",
            "t_stat": "combined_t", "pvalue": "combined_pvalue",
        })
        primary_program_members = targeted_program_all[
            (targeted_program_all.analysis == "primary_behavior_axis")
            & (targeted_program_all.role == "member")
        ].copy()
        member_program_parts = []
        for axis in BEHAVIOR_AXES:
            axis_members = primary_program_members[
                primary_program_members.axis_id == axis["axis_id"]
            ]
            left = axis_members[axis_members.member_mc == axis["focal_mcs"][0]][[
                "program", "effect_sd", "standard_error", "t_stat", "pvalue"
            ]].rename(columns={
                "effect_sd": "member1_effect_sd", "standard_error": "member1_se",
                "t_stat": "member1_t", "pvalue": "member1_pvalue",
            })
            right = axis_members[axis_members.member_mc == axis["focal_mcs"][1]][[
                "program", "effect_sd", "standard_error", "t_stat", "pvalue"
            ]].rename(columns={
                "effect_sd": "member2_effect_sd", "standard_error": "member2_se",
                "t_stat": "member2_t", "pvalue": "member2_pvalue",
            })
            merged = left.merge(right, on="program", validate="one_to_one")
            merged["axis_id"] = axis["axis_id"]
            merged["member1_mc"] = axis["focal_mcs"][0]
            merged["member2_mc"] = axis["focal_mcs"][1]
            member_program_parts.append(merged)
        member_program_wide = pd.concat(member_program_parts, ignore_index=True)
        targeted_program_signatures = primary_program_combined.merge(
            member_program_wide, on=["axis_id", "program"], validate="one_to_one"
        )
        targeted_program_signatures["combined_global_fdr"] = bh_adjust(
            targeted_program_signatures.combined_pvalue.to_numpy(float)
        )
        targeted_program_signatures["conjunction_pvalue"] = targeted_program_signatures[[
            "member1_pvalue", "member2_pvalue"
        ]].max(axis=1)
        targeted_program_signatures["conjunction_global_fdr"] = bh_adjust(
            targeted_program_signatures.conjunction_pvalue.to_numpy(float)
        )
        targeted_program_signatures["members_same_direction_as_combined"] = (
            (np.sign(targeted_program_signatures.member1_effect_sd)
             == np.sign(targeted_program_signatures.combined_effect_sd))
            & (np.sign(targeted_program_signatures.member2_effect_sd)
               == np.sign(targeted_program_signatures.combined_effect_sd))
            & (targeted_program_signatures.combined_effect_sd != 0)
        )
        assert len(targeted_program_signatures) == 4 * 50

        targeted_secondary_programs = targeted_program_all[
            targeted_program_all.analysis == "secondary_pair"
        ].copy()
        targeted_secondary_programs["secondary_global_fdr"] = bh_adjust(
            targeted_secondary_programs.pvalue.to_numpy(float)
        )
        assert len(targeted_secondary_programs) == 2 * 50
        targeted_secondary_programs.to_csv(
            TABLEDIR / "targeted_secondary_hallmark_contrasts.csv", index=False
        )
        display(targeted_program_signatures.sort_values(
            ["combined_global_fdr", "conjunction_global_fdr"]
        ).head(20))
        """
    ),
])

targeted_e_f_cells = [
    code(
        r"""
        # TARGETED_MAIN_PLOT_INLINE_V1 — Figure 4E
        plot_stem = "targeted_figure_4e_monocyte_matched_output_program_landscape"
        plot_family = "Monocyte"
        plot_metaclones = FAMILY_GROUPS[plot_family]
        plot_matrix_full = (
            adjusted_mc_program_means.set_index("metaclone")
            .reindex(plot_metaclones)[list(program_scores.columns)]
        )
        plot_programs = (
            (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
            .sort_values(ascending=False).head(12).index.tolist()
        )
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_data = (
            plot_matrix.rename_axis("program").reset_index()
            .melt(id_vars="program", var_name="metaclone", value_name="adjusted_score_sd")
        )
        plot_data["lineage_bias"] = plot_family
        plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant_pairs = set(map(tuple, program_tests[
            (program_tests.tier == "within_lineage_ovr")
            & (program_tests.lineage_bias == plot_family)
            & (program_tests.tier_fdr < 0.05)
        ][["program", "target_mc"]].assign(target_mc=lambda frame: frame.target_mc.astype(int)).to_numpy()))
        # Use one reproducibly derived color scale across the monocyte and
        # neutrophil matched-output panels.
        shared_scale_values = []
        for shared_family in ["Monocyte", "Neutrophil"]:
            shared_matrix_full = (
                adjusted_mc_program_means.set_index("metaclone")
                .reindex(FAMILY_GROUPS[shared_family])[list(program_scores.columns)]
            )
            shared_programs = (
                (shared_matrix_full.max(axis=0) - shared_matrix_full.min(axis=0))
                .sort_values(ascending=False).head(12).index.tolist()
            )
            shared_scale_values.append(shared_matrix_full[shared_programs].to_numpy(float))
        limit = max(
            0.5,
            float(max(np.nanmax(np.abs(values)) for values in shared_scale_values)),
        )
        fig, ax = plt.subplots(figsize=(6.6, 6.0))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(float), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, metaclone in enumerate(plot_matrix.columns):
                if (program, int(metaclone)) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "●", ha="center", va="center",
                            fontsize=8.6, color=WHITE)
        ax.set_xticks(np.arange(len(plot_metaclones)) + 0.5, [f"MC{mc}" for mc in plot_metaclones])
        ax.set_yticks(np.arange(len(plot_programs)) + 0.5,
                      [clean_program_name(program) for program in plot_programs], fontsize=8.2)
        ax.set_xlim(0, len(plot_metaclones))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_title("Day-2 programs within monocyte-output metaclones", loc="left", pad=23)
        ax.text(0, 1.028, "One clone pseudobulk per LARRY barcode · dot: 950-test OVR BH q < 0.05",
                transform=ax.transAxes, fontsize=8.2, color=GREY)
        ax.text(0, 1.008, "Shared symmetric color scale across Figures 4E and 4F",
                transform=ax.transAxes, fontsize=8.1, color=GREY)
        ax.spines[:].set_visible(False)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.04, pad=0.04)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # TARGETED_MAIN_PLOT_INLINE_V1 — Figure 4F
        plot_stem = "targeted_figure_4f_neutrophil_matched_output_program_landscape"
        plot_family = "Neutrophil"
        plot_metaclones = FAMILY_GROUPS[plot_family]
        plot_matrix_full = (
            adjusted_mc_program_means.set_index("metaclone")
            .reindex(plot_metaclones)[list(program_scores.columns)]
        )
        plot_programs = (
            (plot_matrix_full.max(axis=0) - plot_matrix_full.min(axis=0))
            .sort_values(ascending=False).head(12).index.tolist()
        )
        plot_matrix = plot_matrix_full[plot_programs].T
        plot_data = (
            plot_matrix.rename_axis("program").reset_index()
            .melt(id_vars="program", var_name="metaclone", value_name="adjusted_score_sd")
        )
        plot_data["lineage_bias"] = plot_family
        plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        significant_pairs = set(map(tuple, program_tests[
            (program_tests.tier == "within_lineage_ovr")
            & (program_tests.lineage_bias == plot_family)
            & (program_tests.tier_fdr < 0.05)
        ][["program", "target_mc"]].assign(target_mc=lambda frame: frame.target_mc.astype(int)).to_numpy()))
        # Use one reproducibly derived color scale across the monocyte and
        # neutrophil matched-output panels.
        shared_scale_values = []
        for shared_family in ["Monocyte", "Neutrophil"]:
            shared_matrix_full = (
                adjusted_mc_program_means.set_index("metaclone")
                .reindex(FAMILY_GROUPS[shared_family])[list(program_scores.columns)]
            )
            shared_programs = (
                (shared_matrix_full.max(axis=0) - shared_matrix_full.min(axis=0))
                .sort_values(ascending=False).head(12).index.tolist()
            )
            shared_scale_values.append(shared_matrix_full[shared_programs].to_numpy(float))
        limit = max(
            0.5,
            float(max(np.nanmax(np.abs(values)) for values in shared_scale_values)),
        )
        fig, ax = plt.subplots(figsize=(6.9, 6.0))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(float), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, metaclone in enumerate(plot_matrix.columns):
                if (program, int(metaclone)) in significant_pairs:
                    ax.text(column_index + 0.5, row_index + 0.5, "●", ha="center", va="center",
                            fontsize=8.6, color=WHITE)
        ax.set_xticks(np.arange(len(plot_metaclones)) + 0.5, [f"MC{mc}" for mc in plot_metaclones])
        ax.set_yticks(np.arange(len(plot_programs)) + 0.5,
                      [clean_program_name(program) for program in plot_programs], fontsize=8.2)
        ax.set_xlim(0, len(plot_metaclones))
        ax.set_ylim(0, len(plot_programs))
        ax.invert_yaxis()
        ax.set_title("Day-2 programs within neutrophil-output metaclones", loc="left", pad=23)
        ax.text(0, 1.028, "One clone pseudobulk per LARRY barcode · dot: 950-test OVR BH q < 0.05",
                transform=ax.transAxes, fontsize=8.2, color=GREY)
        ax.text(0, 1.008, "Shared symmetric color scale across Figures 4E and 4F",
                transform=ax.transAxes, fontsize=8.1, color=GREY)
        ax.spines[:].set_visible(False)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.04, pad=0.04)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Adjusted program score (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
]

def targeted_volcano_cell(axis_id: str, axis_label: str):
    stem = f"targeted_supp_{axis_id}_gene_volcano"
    return code(
        rf"""
        # TARGETED_SUPPLEMENT_PLOT_INLINE_V1 — {axis_label} gene volcano
        plot_stem = "{stem}"
        plot_axis_id = "{axis_id}"
        plot_axis_label = "{axis_label}"
        plot_data = targeted_gene_signatures[
            targeted_gene_signatures.axis_id == plot_axis_id
        ].copy()
        plot_data["minus_log10_p"] = -np.log10(
            plot_data.combined_pvalue.clip(lower=np.nextafter(0, 1))
        )
        plot_data["evidence_class"] = np.select(
            [plot_data.shared_behavioral_signature, plot_data.pooled_only_candidate],
            ["Full shared-signature gate", "Combined-only candidate"],
            default="Not globally significant",
        )
        plot_data.to_csv(TABLEDIR / f"{{plot_stem}}_plot_data.csv", index=False)
        plot_colors = {{
            "Not globally significant": "#C8CED6",
            "Combined-only candidate": ORANGE,
            "Full shared-signature gate": INK,
        }}
        fig, ax = plt.subplots(figsize=(6.8, 5.4))
        for evidence_class in [
            "Not globally significant", "Combined-only candidate",
            "Full shared-signature gate",
        ]:
            subset = plot_data[plot_data.evidence_class == evidence_class]
            ax.scatter(
                subset.combined_log2fc, subset.minus_log10_p,
                s=13 if evidence_class == "Not globally significant" else 30,
                color=plot_colors[evidence_class],
                edgecolor="none" if evidence_class == "Not globally significant" else INK,
                linewidth=0 if evidence_class == "Not globally significant" else 0.45,
                alpha=0.55 if evidence_class == "Not globally significant" else 0.95,
                label=evidence_class, rasterized=False,
            )
        label_rows = plot_data[plot_data.combined_global_fdr_pass].sort_values(
            ["combined_global_fdr", "combined_pvalue"]
        )
        for row in label_rows.itertuples(index=False):
            ax.annotate(
                str(row.gene), (row.combined_log2fc, row.minus_log10_p),
                xytext=(4, 4), textcoords="offset points", fontsize=7.2, color=INK,
            )
        ax.axvline(0, color=GREY, linewidth=0.85, linestyle="--")
        ax.set_xlabel("Combined focal-pair vs other-five log2 fold change")
        ax.set_ylabel("-log10(two-sided Wald P)")
        ax.set_title(f"{{plot_axis_label}} · Day-2 clone-pseudobulk genes", loc="left", pad=14)
        ax.text(
            0, 1.012, "Global BH family: four axes x 9,777 genes",
            transform=ax.transAxes, fontsize=8.3, color=GREY,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=LIGHT_GREY, linewidth=0.65, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.8, loc="best")
        fig.tight_layout()
        svg = FIGDIR / f"{{plot_stem}}.svg"
        png = FIGDIR / f"{{plot_stem}}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    )


targeted_supplement_cells = [
    md(
        r"""
        ## Supplementary targeted-analysis figures

        The four axis-specific gene volcanoes, complete targeted GSEA matrix, and
        true cell-removal/library-refit effect diagnostic are kept separate from the
        manuscript panels.
        """
    ),
    targeted_volcano_cell("baso_rich", "Baso-rich"),
    targeted_volcano_cell("neu_rich", "Neu-rich"),
    targeted_volcano_cell("mono_rich", "Mono-rich"),
    targeted_volcano_cell("persistent_undifferentiated", "Persistent-undifferentiated"),
    code(
        r"""
        # TARGETED_SUPPLEMENT_PLOT_INLINE_V1 — complete targeted GSEA matrix
        plot_stem = "targeted_supp_hallmark_preranked_gsea_matrix"
        gsea_axis_order = [axis["axis_id"] for axis in BEHAVIOR_AXES] + [
            "mc5_vs_mc3_divergence", "mc3_vs_mc18_convergence"
        ]
        gsea_axis_labels = {
            **{axis["axis_id"]: axis["axis_label"] for axis in BEHAVIOR_AXES},
            "mc5_vs_mc3_divergence": "MC5 vs MC3",
            "mc3_vs_mc18_convergence": "MC3 vs MC18",
        }
        plot_data = targeted_gsea.copy()
        gsea_program_order = (
            plot_data.groupby("Term", observed=True).NES
            .apply(lambda values: float(np.nanmax(np.abs(values))))
            .sort_values(ascending=False).index.tolist()
        )
        gsea_matrix = (
            plot_data.pivot(index="Term", columns="axis_id", values="NES")
            .reindex(index=gsea_program_order, columns=gsea_axis_order)
        )
        plot_data["display_order"] = plot_data.Term.map(
            {program: index for index, program in enumerate(gsea_program_order)}
        )
        plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        gsea_significant = set(zip(
            plot_data.loc[plot_data.global_fdr_from_nominal_p < 0.05, "Term"],
            plot_data.loc[plot_data.global_fdr_from_nominal_p < 0.05, "axis_id"],
        ))
        gsea_limit = max(1.0, float(np.nanmax(np.abs(gsea_matrix.to_numpy(float)))))
        fig, ax = plt.subplots(figsize=(8.8, 8.9))
        image_artist = ax.pcolormesh(
            np.arange(gsea_matrix.shape[1] + 1),
            np.arange(gsea_matrix.shape[0] + 1),
            gsea_matrix.to_numpy(float), cmap=PROGRAM_CMAP,
            vmin=-gsea_limit, vmax=gsea_limit, shading="flat",
        )
        for row_index, program in enumerate(gsea_matrix.index):
            for column_index, axis_id in enumerate(gsea_matrix.columns):
                if (program, axis_id) in gsea_significant:
                    ax.text(column_index + 0.5, row_index + 0.5, "●", ha="center", va="center",
                            fontsize=7.8, color=WHITE)
        ax.set_xticks(
            np.arange(len(gsea_axis_order)) + 0.5,
            [gsea_axis_labels[value] for value in gsea_axis_order],
            rotation=28, ha="right", fontsize=8,
        )
        ax.set_yticks(
            np.arange(len(gsea_program_order)) + 0.5,
            [clean_program_name(value) for value in gsea_program_order], fontsize=6.1,
        )
        ax.set_xlim(0, len(gsea_axis_order))
        ax.set_ylim(0, len(gsea_program_order))
        ax.invert_yaxis()
        ax.set_title("Targeted preranked Hallmark enrichment", loc="left", pad=14)
        ax.text(
            0, 1.008, "DESeq2 Wald rankings · dot: global FDR < 0.05 within primary or secondary family",
            transform=ax.transAxes, fontsize=8.1, color=GREY,
        )
        ax.spines[:].set_visible(False)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.025, pad=0.03)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Normalized enrichment score")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # TARGETED_SUPPLEMENT_PLOT_INLINE_V1 — true library-LOO effects
        plot_stem = "targeted_supp_library_loo_hallmark_effect_diagnostic"
        loo_plot_data = targeted_library_loo_program_effects[
            targeted_library_loo_program_effects.role == "combined"
        ].merge(
            selected_pairs, on=["axis_id", "program"], how="inner", validate="many_to_one"
        )
        primary_plot_data = selected_signature_rows[[
            "axis_id", "axis_label", "program", "program_label", "combined_effect_sd"
        ]].rename(columns={"combined_effect_sd": "primary_effect_sd"})
        loo_plot_data = loo_plot_data.merge(
            primary_plot_data,
            on=["axis_id", "axis_label", "program", "program_label"],
            validate="many_to_one",
        )
        loo_plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        loo_pair_order = selected_signature_rows[[
            "axis_id", "axis_label", "program", "program_label"
        ]].copy()
        assert len(loo_pair_order) == 8
        loo_pair_labels = [
            f"{row.axis_label} · {row.program_label}"
            for row in loo_pair_order.itertuples(index=False)
        ]
        loo_y = {
            (row.axis_id, row.program): len(loo_pair_order) - 1 - index
            for index, row in enumerate(loo_pair_order.itertuples(index=False))
        }
        fig, ax = plt.subplots(figsize=(9.0, 7.6))
        for row in loo_pair_order.itertuples(index=False):
            subset = loo_plot_data[
                (loo_plot_data.axis_id == row.axis_id)
                & (loo_plot_data.program == row.program)
            ]
            y = loo_y[(row.axis_id, row.program)]
            ax.hlines(
                y, subset.effect_sd.min(), subset.effect_sd.max(),
                color="#9CB2C8", linewidth=1.2,
            )
            ax.scatter(
                subset.effect_sd, np.full(len(subset), y),
                s=22, color="#6F93B5", edgecolor=INK, linewidth=0.35,
                alpha=0.85,
            )
            ax.scatter(
                subset.primary_effect_sd.iloc[0], y, s=54, marker="D",
                color=ORANGE, edgecolor=INK, linewidth=0.55, zorder=4,
            )
        ax.axvline(0, color=GREY, linewidth=0.9, linestyle="--")
        ax.set_yticks(np.arange(len(loo_pair_order)), list(reversed(loo_pair_labels)), fontsize=8.2)
        ax.set_xlabel("Combined Hallmark effect (SD)")
        ax.set_title("Cell-removal leave-one-library-out effects", loc="left", pad=14)
        ax.text(
            0, 1.012, "Blue: seven refits after removing one library's cells · orange diamond: primary",
            transform=ax.transAxes, fontsize=8.3, color=GREY,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
]

deferred_post_sensitivity_cells = [
    md(
        r"""
        ### Direction and adjusted-mean reconciliation

        Every primary combined effect is independently reconstructed as the mean
        adjusted focal-MC profile minus the mean adjusted comparator-MC profile.
        This validates both the contrast vectors and the labels used downstream.
        """
    ),
    code(
        r"""
        # Reconcile target-minus-reference effects with adjusted MC means
        targeted_condition_rows = {
            mc: np.asarray(
                dds_targeted.cond(multi_mc_group=f"MC{mc}"), dtype=float
            )
            for mc in TARGET_MCS
        }
        targeted_program_adjusted_means = pd.DataFrame({
            mc: {
                program: float(
                    targeted_condition_rows[mc]
                    @ np.asarray(fit_result.params, dtype=float)
                )
                for program, fit_result in targeted_program_fits.items()
            }
            for mc in TARGET_MCS
        }).T
        targeted_program_adjusted_means.index.name = "metaclone"
        targeted_program_adjusted_means.reset_index().to_csv(
            TABLEDIR / "targeted_adjusted_hallmark_means_by_metaclone.csv", index=False
        )
        targeted_gene_adjusted_means = pd.DataFrame(
            {
                mc: (
                    targeted_gene_lfc_coefficients @ targeted_condition_rows[mc]
                    / np.log(2)
                )
                for mc in TARGET_MCS
            },
            index=targeted_gene_names,
        )
        targeted_gene_adjusted_means.index.name = "gene"

        adjusted_mean_audit_rows = []
        targeted_vector_audit_rows = []
        for spec in targeted_specs:
            target_mcs = [int(level.replace("MC", "")) for level in spec["target_levels"]]
            reference_mcs = [int(level.replace("MC", "")) for level in spec["reference_levels"]]
            vector = np.asarray(spec["vector"], dtype=float)
            design_columns = [str(column) for column in dds_targeted.obsm["design_matrix"].columns]
            nonfactor = [index for index, column in enumerate(design_columns) if "multi_mc_group" not in column]
            targeted_vector_audit_rows.append({
                "axis_id": spec["axis_id"],
                "role": spec["role"],
                "contrast_id": spec["contrast_id"],
                "intercept_and_library_max_abs_weight": float(
                    np.max(np.abs(vector[nonfactor]))
                ),
                "target_minus_reference_vector_verified": bool(
                    np.max(np.abs(vector[nonfactor])) < 1e-12
                ),
            })

            program_expected = (
                targeted_program_adjusted_means.loc[target_mcs].mean(axis=0)
                - targeted_program_adjusted_means.loc[reference_mcs].mean(axis=0)
            )
            program_observed = (
                targeted_program_all[
                    targeted_program_all.contrast_id == spec["contrast_id"]
                ].set_index("program").effect_sd
            )
            gene_expected = (
                targeted_gene_adjusted_means[target_mcs].mean(axis=1)
                - targeted_gene_adjusted_means[reference_mcs].mean(axis=1)
            )
            gene_observed = (
                targeted_gene_all[
                    targeted_gene_all.contrast_id == spec["contrast_id"]
                ].set_index("gene").log2FoldChange
            )
            adjusted_mean_audit_rows.append({
                "axis_id": spec["axis_id"],
                "role": spec["role"],
                "contrast_id": spec["contrast_id"],
                "n_target_mcs": len(target_mcs),
                "n_reference_mcs": len(reference_mcs),
                "n_programs": len(program_expected),
                "program_max_abs_error": float(
                    (program_expected - program_observed.reindex(program_expected.index)).abs().max()
                ),
                "n_genes": len(gene_expected),
                "gene_max_abs_error": float(
                    (gene_expected - gene_observed.reindex(gene_expected.index)).abs().max()
                ),
            })
        targeted_adjusted_mean_orientation_audit = pd.DataFrame(adjusted_mean_audit_rows)
        targeted_vector_neutrality_audit = pd.DataFrame(targeted_vector_audit_rows)
        assert len(targeted_adjusted_mean_orientation_audit) == len(targeted_specs) == 14
        assert len(targeted_vector_neutrality_audit) == len(targeted_specs)
        assert (targeted_adjusted_mean_orientation_audit.program_max_abs_error < 1e-10).all()
        assert (targeted_adjusted_mean_orientation_audit.gene_max_abs_error < 1e-10).all()
        assert targeted_vector_neutrality_audit.target_minus_reference_vector_verified.all()
        targeted_adjusted_mean_orientation_audit.to_csv(
            TABLEDIR / "targeted_adjusted_mean_orientation_audit.csv", index=False
        )
        targeted_vector_neutrality_audit.to_csv(
            TABLEDIR / "targeted_vector_neutrality_audit.csv", index=False
        )
        display(targeted_adjusted_mean_orientation_audit)
        """
    ),
    md(
        r"""
        ### Targeted preranked Hallmark sensitivity

        DESeq2 Wald statistics provide full-gene rankings for the four combined
        behavior-axis contrasts and two secondary MC contrasts. GSEA uses the fixed
        Hallmark resource and independent global BH families of 200 primary tests
        and 100 secondary tests. This is supporting pathway sensitivity, not an
        independent dataset.
        """
    ),
    code(
        r"""
        # Preranked GSEA for four primary combined axes and two secondary contrasts
        targeted_gsea_source = targeted_gene_all[
            ((targeted_gene_all.analysis == "primary_behavior_axis")
             & (targeted_gene_all.role == "combined"))
            | (targeted_gene_all.analysis == "secondary_pair")
        ].copy()
        targeted_gsea_parts = []
        for number, contrast_id in enumerate(
            targeted_gsea_source.contrast_id.unique(), start=1
        ):
            contrast_rows = targeted_gsea_source[
                targeted_gsea_source.contrast_id == contrast_id
            ].copy()
            ranked = contrast_rows[["gene", "stat"]].copy()
            ranked["gene_upper"] = ranked.gene.astype(str).str.upper()
            ranked["abs_stat"] = ranked.stat.abs().fillna(-np.inf)
            ranked = (
                ranked.sort_values(["abs_stat", "gene_upper"], ascending=[False, True])
                .drop_duplicates("gene_upper", keep="first")
            )
            ranked["rank_stat"] = ranked.stat.fillna(0.0)
            ranked = ranked.sort_values(
                ["rank_stat", "gene_upper"], ascending=[False, True]
            ).reset_index(drop=True)
            # Resolve exact ties deterministically at machine precision so GSEA
            # never inherits an arbitrary gene order or emits tie warnings.
            rank_values = ranked.rank_stat.to_numpy(float)
            for position in range(1, len(rank_values)):
                if rank_values[position] >= rank_values[position - 1]:
                    rank_values[position] = np.nextafter(
                        rank_values[position - 1], -np.inf
                    )
            assert pd.Series(rank_values).is_unique
            assert np.all(np.diff(rank_values) < 0)
            rank_series = pd.Series(
                rank_values, index=ranked.gene_upper, name="wald_stat",
            )
            prerank = gp.prerank(
                rnk=rank_series, gene_sets=str(HALLMARK_LOCAL), min_size=10,
                max_size=500, permutation_num=1000, seed=20260903,
                threads=1, outdir=None, no_plot=True, verbose=False,
            )
            terms = prerank.res2d.copy()
            for column in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
                terms[column] = pd.to_numeric(terms[column], errors="coerce")
            source = contrast_rows.iloc[0]
            terms["analysis"] = source.analysis
            terms["axis_id"] = source.axis_id
            terms["axis_label"] = source.axis_label
            terms["contrast_id"] = contrast_id
            terms["contrast_label"] = source.contrast_label
            terms["gmt_sha256"] = hallmark_source_hash
            targeted_gsea_parts.append(terms)
            print(f"targeted GSEA {number}/6: {contrast_id}")
        targeted_gsea = pd.concat(targeted_gsea_parts, ignore_index=True)
        targeted_gsea["global_family"] = np.where(
            targeted_gsea.analysis == "primary_behavior_axis",
            "four_primary_axes", "two_secondary_pairs",
        )
        targeted_gsea["global_fdr_from_nominal_p"] = (
            targeted_gsea.groupby("global_family", observed=True, group_keys=False)["NOM p-val"]
            .transform(bh_adjust)
        )
        assert len(targeted_gsea[targeted_gsea.global_family == "four_primary_axes"]) == 200
        assert len(targeted_gsea[targeted_gsea.global_family == "two_secondary_pairs"]) == 100
        targeted_gsea.to_csv(
            TABLEDIR / "targeted_hallmark_preranked_gsea.csv.gz",
            index=False, compression="gzip",
        )
        targeted_gsea_summary = (
            targeted_gsea.groupby(["global_family", "axis_id", "axis_label"], observed=True)
            .agg(
                n_programs=("Term", "size"),
                n_contrast_fdr_lt_0_25=("FDR q-val", lambda values: int((values < 0.25).sum())),
                n_global_fdr_lt_0_05=("global_fdr_from_nominal_p", lambda values: int((values < 0.05).sum())),
            ).reset_index()
        )
        targeted_gsea_summary.to_csv(
            TABLEDIR / "targeted_hallmark_preranked_gsea_summary.csv", index=False
        )
        display(targeted_gsea_summary)
        """
    ),
]

loo_cells_to_append = [
    md(
        r"""
        ### Leave-one-library-out refits

        Each Day-2 library is omitted at the cell level. Counts are re-summed per
        surviving LARRY clone, library fractions are recomputed, normalization and
        dispersion estimation are rerun, Hallmark scores are rebuilt, and a fresh
        full-rank seven-MC model is fitted. This is a genuine data-removal refit,
        not a covariate-deletion or existing-pseudobulk influence approximation.
        """
    ),
    code(
        r"""
        # True cell-removal leave-one-library-out refits
        multi_hspc_cells = hspc_cells[
            hspc_cells.metaclone.isin(TARGET_MCS)
        ].copy()
        assert multi_hspc_cells.LARRY_ID.nunique() == 217
        filtered_gene_indices = np.flatnonzero(keep_gene.to_numpy())
        assert len(filtered_gene_indices) == 9777
        fixed_program_indices = {}
        for program, requested_genes in hallmark_sets.items():
            fixed_program_indices[program] = sorted({
                index for gene in requested_genes
                for index in upper_to_indices.get(gene, [])
            })
            assert len(fixed_program_indices[program]) >= 10

        loo_gene_parts = []
        loo_program_rows = []
        loo_fit_rows = []
        for omitted_library in day2_libraries:
            loo_cells = multi_hspc_cells[
                multi_hspc_cells.Library != omitted_library
            ].copy()
            removed_cells = int((multi_hspc_cells.Library == omitted_library).sum())
            assert removed_cells > 0
            loo_pseudobulk_rows = {}
            loo_metadata_rows = []
            for clone, clone_cells in loo_cells.groupby("LARRY_ID", observed=True):
                assert clone_cells.metaclone.nunique() == 1
                mc = int(clone_cells.metaclone.iloc[0])
                indices = POS.loc[clone_cells.index].to_numpy()
                loo_pseudobulk_rows[str(clone)] = np.asarray(
                    COUNTS[indices][:, filtered_gene_indices].sum(axis=0)
                ).ravel().astype(np.int64)
                library_counts = clone_cells.Library.value_counts()
                record = {
                    "sample": str(clone), "clone": str(clone),
                    "metaclone": mc, "multi_mc_group": f"MC{mc}",
                    "n_cells": int(len(clone_cells)),
                }
                for library in day2_libraries:
                    record[f"libfrac__{safe_name(library)}"] = float(
                        library_counts.get(library, 0) / len(clone_cells)
                    )
                loo_metadata_rows.append(record)
            loo_counts = pd.DataFrame(loo_pseudobulk_rows).T
            loo_counts.columns = filtered_counts.columns
            loo_counts.index.name = "sample"
            loo_metadata = (
                pd.DataFrame(loo_metadata_rows).set_index("sample")
                .loc[loo_counts.index]
            )
            loo_metadata["multi_mc_group"] = pd.Categorical(
                loo_metadata.multi_mc_group,
                categories=[f"MC{mc}" for mc in TARGET_MCS], ordered=True,
            )
            assert omitted_library not in set(loo_cells.Library)
            assert set(loo_metadata.metaclone.unique()) == set(TARGET_MCS)
            assert not loo_counts.index.duplicated().any()
            assert loo_counts.shape[1] == 9777
            loo_metadata, loo_covariates, loo_reference, loo_dropped = (
                select_library_covariates(
                    loo_metadata, "multi_mc_group", f"loo_{safe_name(omitted_library)}"
                )
            )
            loo_formula = "~ " + " + ".join(
                loo_covariates + ["multi_mc_group"]
            )
            with warnings.catch_warnings(record=True) as loo_warnings:
                warnings.simplefilter("always")
                loo_dds = DeseqDataSet(
                    counts=loo_counts.astype(int), metadata=loo_metadata,
                    design=loo_formula, inference=inference, quiet=True,
                    refit_cooks=False,
                )
                loo_design = np.asarray(
                    loo_dds.obsm["design_matrix"], dtype=float
                )
                loo_rank = int(np.linalg.matrix_rank(loo_design))
                assert loo_rank == loo_design.shape[1]
                loo_dds.deseq2()

            loo_lfc_coefficients = np.asarray(loo_dds.varm["LFC"], dtype=float)
            assert loo_lfc_coefficients.shape == (
                9777, loo_design.shape[1]
            )
            loo_normed = np.asarray(loo_dds.layers["normed_counts"], dtype=float)
            loo_log = np.log2(loo_normed + 1.0)
            loo_gene_mean = loo_log.mean(axis=0)
            loo_gene_sd = loo_log.std(axis=0, ddof=1)
            loo_valid_sd = loo_gene_sd > 1e-12
            loo_gene_z = np.zeros_like(loo_log, dtype=float)
            loo_gene_z[:, loo_valid_sd] = (
                (loo_log[:, loo_valid_sd] - loo_gene_mean[loo_valid_sd])
                / loo_gene_sd[loo_valid_sd]
            )
            loo_program_scores = pd.DataFrame({
                program: loo_gene_z[:, indices].mean(axis=1)
                for program, indices in fixed_program_indices.items()
            }, index=loo_metadata.index)
            loo_program_sd = loo_program_scores.std(axis=0, ddof=1)
            loo_program_scores = (
                (loo_program_scores - loo_program_scores.mean(axis=0))
                / loo_program_sd
            )
            assert np.isfinite(loo_program_scores.to_numpy()).all()
            loo_design_frame = loo_dds.obsm["design_matrix"].copy()
            assert loo_design_frame.index.tolist() == loo_program_scores.index.tolist()
            loo_program_fits = {
                program: sm.OLS(
                    loo_program_scores[program].to_numpy(float),
                    loo_design_frame.to_numpy(float),
                ).fit(cov_type="HC3")
                for program in program_columns
            }

            for axis in BEHAVIOR_AXES:
                focal = list(axis["focal_mcs"])
                comparators = [mc for mc in TARGET_MCS if mc not in focal]
                role_targets = [
                    ("combined", np.nan, focal),
                    ("member1", focal[0], [focal[0]]),
                    ("member2", focal[1], [focal[1]]),
                ]
                for role, member_mc, target_mcs in role_targets:
                    vector = explicit_target_minus_reference(
                        loo_dds, "multi_mc_group",
                        [f"MC{mc}" for mc in target_mcs],
                        [f"MC{mc}" for mc in comparators],
                    )
                    gene_effects = loo_lfc_coefficients @ vector / np.log(2)
                    loo_gene_parts.append(pd.DataFrame({
                        "omitted_library": omitted_library,
                        "axis_id": axis["axis_id"],
                        "axis_label": axis["axis_label"],
                        "role": role, "member_mc": member_mc,
                        "gene": targeted_gene_names,
                        "log2fc": gene_effects,
                    }))
                    for program, fit_result in loo_program_fits.items():
                        loo_program_rows.append({
                            "omitted_library": omitted_library,
                            "axis_id": axis["axis_id"],
                            "axis_label": axis["axis_label"],
                            "role": role, "member_mc": member_mc,
                            "program": program,
                            "program_label": clean_program_name(program),
                            "effect_sd": float(
                                np.asarray(fit_result.params, dtype=float) @ vector
                            ),
                        })
            loo_fit_rows.append({
                "omitted_library": omitted_library,
                "removed_cells": removed_cells,
                "remaining_cells": int(len(loo_cells)),
                "remaining_clone_pseudobulks": int(len(loo_metadata)),
                "remaining_metaclones": int(loo_metadata.metaclone.nunique()),
                "design": loo_formula, "design_rank": loo_rank,
                "design_columns": int(loo_design.shape[1]),
                "condition_number": float(np.linalg.cond(loo_design)),
                "reference_library_fraction": loo_reference,
                "library_covariates": "|".join(loo_covariates),
                "rank_dropped": "|".join(loo_dropped),
                "warning_count": len(loo_warnings),
                "warnings": " | ".join(sorted({str(item.message) for item in loo_warnings})),
                "estimable": True,
            })
            print(
                f"completed cell-removal/refit LOO: {omitted_library} · "
                f"{len(loo_metadata)} surviving clone pseudobulks"
            )

        targeted_library_loo_fit_audit = pd.DataFrame(loo_fit_rows)
        targeted_library_loo_gene_effects = pd.concat(loo_gene_parts, ignore_index=True)
        targeted_library_loo_program_effects = pd.DataFrame(loo_program_rows)
        assert len(targeted_library_loo_fit_audit) == len(day2_libraries) == 7
        assert targeted_library_loo_fit_audit.estimable.all()
        assert len(targeted_library_loo_gene_effects) == 7 * 4 * 3 * 9777
        assert len(targeted_library_loo_program_effects) == 7 * 4 * 3 * 50
        targeted_library_loo_fit_audit.to_csv(
            TABLEDIR / "targeted_library_loo_model_fit_audit.csv", index=False
        )
        targeted_library_loo_gene_effects.to_csv(
            TABLEDIR / "targeted_library_loo_gene_effects.csv.gz",
            index=False, compression="gzip",
        )
        targeted_library_loo_program_effects.to_csv(
            TABLEDIR / "targeted_library_loo_hallmark_effects.csv", index=False
        )
        display(targeted_library_loo_fit_audit)
        """
    ),
    code(
        r"""
        # Reduce LOO effects to explicit all-estimable-direction Boolean components
        loo_gene_check = targeted_library_loo_gene_effects.merge(
            primary_gene_direction, on=["axis_id", "gene"], validate="many_to_one"
        )
        loo_gene_check["estimable_effect"] = np.isfinite(loo_gene_check.log2fc)
        loo_gene_check["direction_agrees"] = (
            loo_gene_check.estimable_effect
            & (np.sign(loo_gene_check.log2fc)
               == np.sign(loo_gene_check.combined_log2fc))
            & (loo_gene_check.combined_log2fc != 0)
        )
        loo_gene_stability = (
            loo_gene_check.groupby(["axis_id", "gene"], observed=True)
            .agg(
                library_loo_n_requested_effects=("direction_agrees", "size"),
                library_loo_n_estimable_effects=("estimable_effect", "sum"),
                library_loo_n_direction_agree=("direction_agrees", "sum"),
            ).reset_index()
        )
        loo_gene_stability["library_loo_all_estimable_same_direction"] = (
            (loo_gene_stability.library_loo_n_estimable_effects > 0)
            & (loo_gene_stability.library_loo_n_direction_agree
               == loo_gene_stability.library_loo_n_estimable_effects)
        )
        loo_gene_stability["library_loo_complete"] = (
            loo_gene_stability.library_loo_n_estimable_effects
            == loo_gene_stability.library_loo_n_requested_effects
        )
        assert (loo_gene_stability.library_loo_n_requested_effects == 21).all()

        loo_program_check = targeted_library_loo_program_effects.merge(
            primary_program_direction,
            on=["axis_id", "program"], validate="many_to_one"
        )
        loo_program_check["estimable_effect"] = np.isfinite(loo_program_check.effect_sd)
        loo_program_check["direction_agrees"] = (
            loo_program_check.estimable_effect
            & (np.sign(loo_program_check.effect_sd)
               == np.sign(loo_program_check.combined_effect_sd))
            & (loo_program_check.combined_effect_sd != 0)
        )
        loo_program_stability = (
            loo_program_check.groupby(["axis_id", "program"], observed=True)
            .agg(
                library_loo_n_requested_effects=("direction_agrees", "size"),
                library_loo_n_estimable_effects=("estimable_effect", "sum"),
                library_loo_n_direction_agree=("direction_agrees", "sum"),
            ).reset_index()
        )
        loo_program_stability["library_loo_all_estimable_same_direction"] = (
            (loo_program_stability.library_loo_n_estimable_effects > 0)
            & (loo_program_stability.library_loo_n_direction_agree
               == loo_program_stability.library_loo_n_estimable_effects)
        )
        loo_program_stability["library_loo_complete"] = (
            loo_program_stability.library_loo_n_estimable_effects
            == loo_program_stability.library_loo_n_requested_effects
        )
        assert (loo_program_stability.library_loo_n_requested_effects == 21).all()

        targeted_gene_signatures = (
            targeted_gene_signatures
            .merge(comparator_gene_stability, on=["axis_id", "gene"], validate="one_to_one")
            .merge(loo_gene_stability, on=["axis_id", "gene"], validate="one_to_one")
        )
        targeted_program_signatures = (
            targeted_program_signatures
            .merge(comparator_program_stability, on=["axis_id", "program"], validate="one_to_one")
            .merge(loo_program_stability, on=["axis_id", "program"], validate="one_to_one")
        )
        for result in [targeted_gene_signatures, targeted_program_signatures]:
            result["combined_global_fdr_pass"] = result.combined_global_fdr < 0.05
            result["conjunction_global_fdr_pass"] = result.conjunction_global_fdr < 0.05
            member1_effect = (
                result.member1_log2fc if "member1_log2fc" in result.columns
                else result.member1_effect_sd
            )
            member2_effect = (
                result.member2_log2fc if "member2_log2fc" in result.columns
                else result.member2_effect_sd
            )
            combined_effect = (
                result.combined_log2fc if "combined_log2fc" in result.columns
                else result.combined_effect_sd
            )
            result["member1_nominal_same_direction"] = (
                (result.member1_pvalue < 0.05)
                & (np.sign(member1_effect) == np.sign(combined_effect))
            )
            result["member2_nominal_same_direction"] = (
                (result.member2_pvalue < 0.05)
                & (np.sign(member2_effect) == np.sign(combined_effect))
            )
            result["member_conjunction_not_supported"] = (
                result.combined_global_fdr_pass
                & ~result.conjunction_global_fdr_pass
            )
            result["opposing_member_direction"] = (
                result.combined_global_fdr_pass
                & ~result.members_same_direction_as_combined
            )
            result["shared_behavioral_signature"] = (
                result.combined_global_fdr_pass
                & result.members_same_direction_as_combined
                & result.conjunction_global_fdr_pass
                & result.comparator_drop_all_same_direction
                & result.library_loo_all_estimable_same_direction
            )
            result["pooled_only_candidate"] = (
                result.combined_global_fdr_pass
                & ~result.shared_behavioral_signature
            )
            result["one_metaclone_driven"] = (
                result.combined_global_fdr_pass
                & result.members_same_direction_as_combined
                & (result.member1_nominal_same_direction
                   ^ result.member2_nominal_same_direction)
            )
            result["pooled_only_failure_reason"] = np.select(
                [
                    ~result.combined_global_fdr_pass,
                    result.opposing_member_direction,
                    result.one_metaclone_driven,
                    result.member_conjunction_not_supported,
                    ~result.comparator_drop_all_same_direction,
                    ~result.library_loo_all_estimable_same_direction,
                ],
                [
                    "combined_global_fdr_not_supported",
                    "member_direction_disagreement",
                    "exactly_one_member_nominal_same_direction",
                    "global_max_p_conjunction_not_supported",
                    "comparator_drop_direction_unstable",
                    "library_loo_direction_unstable",
                ],
                default="all_shared_signature_gates_pass",
            )

        targeted_gene_signatures.to_csv(
            TABLEDIR / "targeted_behavior_gene_signatures.csv.gz",
            index=False, compression="gzip",
        )
        targeted_program_signatures.to_csv(
            TABLEDIR / "targeted_behavior_hallmark_signatures.csv", index=False
        )
        targeted_signature_summary = pd.concat([
            targeted_gene_signatures.assign(feature_type="gene", feature=targeted_gene_signatures.gene),
            targeted_program_signatures.assign(feature_type="Hallmark", feature=targeted_program_signatures.program_label),
        ], ignore_index=True, sort=False).groupby(
            ["feature_type", "axis_id", "axis_label"], observed=True
        ).agg(
            n_tests=("feature", "size"),
            n_combined_global_fdr=("combined_global_fdr_pass", "sum"),
            n_conjunction_global_fdr=("conjunction_global_fdr_pass", "sum"),
            n_shared_signatures=("shared_behavioral_signature", "sum"),
            n_pooled_only_candidates=("pooled_only_candidate", "sum"),
            n_one_metaclone_driven=("one_metaclone_driven", "sum"),
            n_comparator_stable=("comparator_drop_all_same_direction", "sum"),
            n_library_loo_stable=("library_loo_all_estimable_same_direction", "sum"),
        ).reset_index()
        targeted_signature_summary.to_csv(
            TABLEDIR / "targeted_behavior_signature_summary.csv", index=False
        )
        display(targeted_signature_summary)
        """
    ),
]

cells.extend([
    md(
        r"""
        ### Comparator-removal stability

        For each behavior axis, each of the five nonfocal metaclones is removed
        from the equal-MC reference in turn. The fitted seven-MC coefficient model
        is unchanged; only the explicit target-minus-remaining-four estimand is
        recalculated. The combined focal-pair effect and both member effects must
        retain the primary combined direction for every removal.
        """
    ),
    code(
        r"""
        # Comparator-drop direction checks for all genes and all Hallmarks
        targeted_gene_lfc_coefficients = np.asarray(
            dds_targeted.varm["LFC"], dtype=float
        )
        assert targeted_gene_lfc_coefficients.shape == (
            targeted_counts.shape[1], targeted_design.shape[1]
        )
        targeted_gene_names = np.asarray(dds_targeted.var_names.astype(str))
        assert np.array_equal(targeted_gene_names, targeted_counts.columns.astype(str))

        # Independently reconcile coefficient projection with every primary and
        # secondary DESeqStats log2 fold change before using it for sensitivities.
        projection_errors = []
        for spec in targeted_specs:
            projected = (
                targeted_gene_lfc_coefficients @ np.asarray(spec["vector"], dtype=float)
                / np.log(2)
            )
            observed = (
                targeted_gene_all[targeted_gene_all.contrast_id == spec["contrast_id"]]
                .set_index("gene").reindex(targeted_gene_names).log2FoldChange.to_numpy(float)
            )
            projection_errors.append({
                "contrast_id": spec["contrast_id"],
                "max_abs_log2fc_error": float(np.nanmax(np.abs(projected - observed))),
            })
        targeted_projection_audit = pd.DataFrame(projection_errors)
        assert (targeted_projection_audit.max_abs_log2fc_error < 1e-10).all()
        targeted_projection_audit.to_csv(
            TABLEDIR / "targeted_gene_projection_orientation_audit.csv", index=False
        )

        comparator_gene_parts = []
        comparator_program_rows = []
        for axis in BEHAVIOR_AXES:
            focal = list(axis["focal_mcs"])
            comparators = [mc for mc in TARGET_MCS if mc not in focal]
            for omitted_mc in comparators:
                remaining = [mc for mc in comparators if mc != omitted_mc]
                role_targets = [
                    ("combined", np.nan, focal),
                    ("member1", focal[0], [focal[0]]),
                    ("member2", focal[1], [focal[1]]),
                ]
                for role, member_mc, target_mcs in role_targets:
                    vector = explicit_target_minus_reference(
                        dds_targeted, "multi_mc_group",
                        [f"MC{mc}" for mc in target_mcs],
                        [f"MC{mc}" for mc in remaining],
                    )
                    gene_effects = (
                        targeted_gene_lfc_coefficients @ vector / np.log(2)
                    )
                    comparator_gene_parts.append(pd.DataFrame({
                        "axis_id": axis["axis_id"], "axis_label": axis["axis_label"],
                        "omitted_comparator_mc": omitted_mc,
                        "remaining_comparator_mcs": "|".join(map(str, remaining)),
                        "role": role, "member_mc": member_mc,
                        "gene": targeted_gene_names, "log2fc": gene_effects,
                    }))
                    for program, fit_result in targeted_program_fits.items():
                        comparator_program_rows.append({
                            "axis_id": axis["axis_id"],
                            "axis_label": axis["axis_label"],
                            "omitted_comparator_mc": omitted_mc,
                            "remaining_comparator_mcs": "|".join(map(str, remaining)),
                            "role": role, "member_mc": member_mc,
                            "program": program,
                            "program_label": clean_program_name(program),
                            "effect_sd": float(
                                np.asarray(fit_result.params, dtype=float) @ vector
                            ),
                        })
        targeted_comparator_gene_effects = pd.concat(
            comparator_gene_parts, ignore_index=True
        )
        targeted_comparator_program_effects = pd.DataFrame(comparator_program_rows)
        assert len(targeted_comparator_gene_effects) == 4 * 5 * 3 * 9777
        assert len(targeted_comparator_program_effects) == 4 * 5 * 3 * 50
        targeted_comparator_gene_effects.to_csv(
            TABLEDIR / "targeted_comparator_drop_gene_effects.csv.gz",
            index=False, compression="gzip",
        )
        targeted_comparator_program_effects.to_csv(
            TABLEDIR / "targeted_comparator_drop_hallmark_effects.csv", index=False
        )

        primary_gene_direction = targeted_gene_signatures[[
            "axis_id", "gene", "combined_log2fc"
        ]].copy()
        comparator_gene_check = targeted_comparator_gene_effects.merge(
            primary_gene_direction, on=["axis_id", "gene"], validate="many_to_one"
        )
        comparator_gene_check["direction_agrees"] = (
            np.sign(comparator_gene_check.log2fc)
            == np.sign(comparator_gene_check.combined_log2fc)
        ) & (comparator_gene_check.combined_log2fc != 0)
        comparator_gene_stability = (
            comparator_gene_check.groupby(["axis_id", "gene"], observed=True)
            .agg(
                comparator_drop_n_effects=("direction_agrees", "size"),
                comparator_drop_n_direction_agree=("direction_agrees", "sum"),
                comparator_drop_all_same_direction=("direction_agrees", "all"),
            ).reset_index()
        )
        assert (comparator_gene_stability.comparator_drop_n_effects == 15).all()

        primary_program_direction = targeted_program_signatures[[
            "axis_id", "program", "combined_effect_sd"
        ]].copy()
        comparator_program_check = targeted_comparator_program_effects.merge(
            primary_program_direction, on=["axis_id", "program"], validate="many_to_one"
        )
        comparator_program_check["direction_agrees"] = (
            np.sign(comparator_program_check.effect_sd)
            == np.sign(comparator_program_check.combined_effect_sd)
        ) & (comparator_program_check.combined_effect_sd != 0)
        comparator_program_stability = (
            comparator_program_check.groupby(["axis_id", "program"], observed=True)
            .agg(
                comparator_drop_n_effects=("direction_agrees", "size"),
                comparator_drop_n_direction_agree=("direction_agrees", "sum"),
                comparator_drop_all_same_direction=("direction_agrees", "all"),
            ).reset_index()
        )
        assert (comparator_program_stability.comparator_drop_n_effects == 15).all()
        display(comparator_program_stability.groupby("axis_id", observed=True)
                .comparator_drop_all_same_direction.sum().reset_index())
        """
    ),
])

cells.extend(loo_cells_to_append)
cells.extend(deferred_post_sensitivity_cells)

cells.extend([
    md(
        r"""
        ## Manuscript Figure 4 panels

        Each panel is constructed, saved, and displayed in its own code cell.
        SVGs retain editable text and vector marks. Panels 4E and 4F re-express the
        corrected matched-output monocyte and neutrophil landscapes under explicit
        manuscript stems.
        """
    ),
    code(
        r"""
        # TARGETED_MAIN_PLOT_INLINE_V1 — Figure 4A
        plot_stem = "targeted_figure_4a_multilineage_day4_day6_output_composition"
        plot_output_map = {
            "Neutrophil": "Neutrophil", "Monocyte": "Monocyte",
            "Meg": "Erythroid / Mk", "Erythroid": "Erythroid / Mk",
            "Baso": "Basophil", "Mast": "Mast", "Eos": "Other mature",
            "Ccr7_DC": "Other mature", "Lymphoid": "Other mature",
            "Undifferentiated": "Undifferentiated",
        }
        plot_output_order = [
            "Neutrophil", "Monocyte", "Erythroid / Mk", "Basophil",
            "Mast", "Other mature", "Undifferentiated",
        ]
        plot_output_colors = {
            "Neutrophil": "#D4B441", "Monocyte": "#B23A48",
            "Erythroid / Mk": "#2E7DBF", "Basophil": "#3F9E4D",
            "Mast": "#8E6FB8", "Other mature": "#8693A3",
            "Undifferentiated": "#D8DDE3",
        }
        plot_data = multilineage_output_clone_balanced.copy()
        plot_data["output_class"] = plot_data.cell_type_weinreb.map(plot_output_map)
        plot_data = (
            plot_data.groupby(["metaclone", "day", "output_class"], observed=True)
            .fraction.sum().reset_index()
        )
        plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        plot_row_order = [(mc, day) for mc in TARGET_MCS for day in [4, 6]]
        plot_matrix = (
            plot_data.pivot_table(
                index=["metaclone", "day"], columns="output_class",
                values="fraction", fill_value=0.0, observed=True,
            ).reindex(
                index=pd.MultiIndex.from_tuples(
                    plot_row_order, names=["metaclone", "day"]
                ), columns=plot_output_order, fill_value=0.0,
            )
        )
        plot_support = multilineage_output_support.set_index(["metaclone", "day"])
        y_positions = []
        for mc_index in range(len(TARGET_MCS)):
            base = (len(TARGET_MCS) - 1 - mc_index) * 2.7
            y_positions.extend([base + 0.55, base - 0.55])
        y_positions = np.asarray(y_positions, dtype=float)
        fig, ax = plt.subplots(figsize=(10.8, 7.8))
        left = np.zeros(len(plot_matrix), dtype=float)
        for output_class in plot_output_order:
            values = plot_matrix[output_class].to_numpy(float)
            ax.barh(
                y_positions, values, left=left, height=0.82,
                color=plot_output_colors[output_class], edgecolor=WHITE,
                linewidth=0.45, label=output_class,
            )
            left += values
        row_labels = [
            f"MC{mc} · Day {day}  n={int(plot_support.loc[(mc, day), 'n_clones'])}"
            for mc, day in plot_row_order
        ]
        ax.set_yticks(y_positions, row_labels, fontsize=8.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean within-clone fraction of observed cells")
        ax.set_title("Clone-balanced output of Multi-lineage metaclones", loc="left", pad=14)
        ax.text(
            0, 1.012, "Author Day-4/Day-6 annotations · each observed clone contributes equally",
            transform=ax.transAxes, fontsize=8.6, color=GREY,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.legend(
            title="Output class", loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=False, fontsize=8.5, title_fontsize=9,
        )
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # TARGETED_MAIN_PLOT_INLINE_V1 — Figure 4B
        from matplotlib.patches import Rectangle
        plot_stem = "targeted_figure_4b_behavior_axis_registry"
        plot_axes = [axis["axis_id"] for axis in BEHAVIOR_AXES]
        plot_axis_labels = [
            f"{axis['axis_label']} ({axis['output_type']} >= {100 * axis['threshold']:.0f}%)"
            for axis in BEHAVIOR_AXES
        ]
        plot_data = behavior_axis_registry.copy()
        plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        plot_matrix = (
            plot_data.pivot(index="axis_id", columns="metaclone",
                            values="day6_clone_balanced_fraction")
            .reindex(index=plot_axes, columns=TARGET_MCS)
        )
        focal_lookup = set(
            zip(plot_data.loc[plot_data.is_focal, "axis_id"],
                plot_data.loc[plot_data.is_focal, "metaclone"])
        )
        sequential_cmap = LinearSegmentedColormap.from_list(
            "navy_fraction", [WHITE, "#DCE7F2", "#6F93B5", INK]
        )
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(float), cmap=sequential_cmap,
            vmin=0, vmax=1, shading="flat", edgecolors=WHITE, linewidth=0.8,
        )
        for row_index, axis_id in enumerate(plot_matrix.index):
            for column_index, mc in enumerate(plot_matrix.columns):
                value = float(plot_matrix.loc[axis_id, mc])
                ax.text(
                    column_index + 0.5, row_index + 0.5, f"{100 * value:.1f}%",
                    ha="center", va="center", fontsize=8.1,
                    color=WHITE if value > 0.55 else INK,
                )
                if (axis_id, mc) in focal_lookup:
                    ax.add_patch(Rectangle(
                        (column_index + 0.02, row_index + 0.02), 0.96, 0.96,
                        fill=False, edgecolor=ORANGE, linewidth=2.2,
                    ))
        ax.set_xticks(np.arange(len(TARGET_MCS)) + 0.5, [f"MC{mc}" for mc in TARGET_MCS])
        ax.set_yticks(np.arange(len(plot_axes)) + 0.5, plot_axis_labels)
        ax.set_xlim(0, len(TARGET_MCS))
        ax.set_ylim(0, len(plot_axes))
        ax.invert_yaxis()
        ax.set_xlabel("Multi-lineage metaclone")
        ax.set_title("Locked Day-6 behavioral-axis registry", loc="left", pad=14)
        ax.text(
            0, 1.02, "Orange outline: analysis-defined focal pair · values: clone-balanced author output",
            transform=ax.transAxes, fontsize=8.4, color=GREY,
        )
        ax.spines[:].set_visible(False)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.035, pad=0.03)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Day-6 output fraction")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # TARGETED_MAIN_PLOT_INLINE_V1 — Figure 4C
        plot_stem = "targeted_figure_4c_hallmark_signature_effect_matrix"
        axis_order = [axis["axis_id"] for axis in BEHAVIOR_AXES]
        axis_labels = {axis["axis_id"]: axis["axis_label"] for axis in BEHAVIOR_AXES}
        plot_data = targeted_program_signatures.copy()
        program_order = (
            plot_data.groupby(["program", "program_label"], observed=True)
            .combined_effect_sd.apply(lambda values: float(np.max(np.abs(values))))
            .sort_values(ascending=False).index.get_level_values("program").tolist()
        )
        plot_matrix = (
            plot_data.pivot(index="program", columns="axis_id", values="combined_effect_sd")
            .reindex(index=program_order, columns=axis_order)
        )
        plot_data["display_order"] = plot_data.program.map(
            {program: index for index, program in enumerate(program_order)}
        )
        plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        pooled_lookup = set(zip(
            plot_data.loc[plot_data.pooled_only_candidate, "program"],
            plot_data.loc[plot_data.pooled_only_candidate, "axis_id"],
        ))
        shared_lookup = set(zip(
            plot_data.loc[plot_data.shared_behavioral_signature, "program"],
            plot_data.loc[plot_data.shared_behavioral_signature, "axis_id"],
        ))
        limit = max(0.5, float(np.nanmax(np.abs(plot_matrix.to_numpy(float)))))
        fig, ax = plt.subplots(figsize=(8.2, 8.8))
        image_artist = ax.pcolormesh(
            np.arange(plot_matrix.shape[1] + 1),
            np.arange(plot_matrix.shape[0] + 1),
            plot_matrix.to_numpy(float), cmap=PROGRAM_CMAP,
            vmin=-limit, vmax=limit, shading="flat",
        )
        for row_index, program in enumerate(plot_matrix.index):
            for column_index, axis_id in enumerate(plot_matrix.columns):
                if (program, axis_id) in shared_lookup:
                    ax.text(column_index + 0.5, row_index + 0.5, "●", ha="center", va="center",
                            fontsize=9.2, color=WHITE)
                elif (program, axis_id) in pooled_lookup:
                    ax.text(column_index + 0.5, row_index + 0.5, "○", ha="center", va="center",
                            fontsize=9.2, color=INK)
        program_label_map = plot_data.drop_duplicates("program").set_index("program").program_label
        ax.set_xticks(np.arange(len(axis_order)) + 0.5, [axis_labels[value] for value in axis_order],
                      rotation=25, ha="right")
        ax.set_yticks(np.arange(len(program_order)) + 0.5,
                      [program_label_map.loc[value] for value in program_order], fontsize=6.1)
        ax.set_xlim(0, len(axis_order))
        ax.set_ylim(0, len(program_order))
        ax.invert_yaxis()
        ax.set_title("Day-2 Hallmark effects across locked behavioral axes", loc="left", pad=14)
        ax.text(
            0, 1.008, "Target minus equal-MC reference · ○ combined-only candidate · ● full shared-signature gate",
            transform=ax.transAxes, fontsize=8.2, color=GREY,
        )
        ax.spines[:].set_visible(False)
        colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.025, pad=0.03)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Combined effect (SD)")
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
    code(
        r"""
        # TARGETED_MAIN_PLOT_INLINE_V1 — Figure 4D
        plot_stem = "targeted_figure_4d_member_agreement_forest"
        # Compact representative display: two lowest combined-q Hallmarks per axis.
        # Complete results remain in Figure 4C and the exported 200-row table.
        top_per_axis = (
            targeted_program_signatures.sort_values(
                ["axis_id", "combined_global_fdr", "conjunction_global_fdr"]
            ).groupby("axis_id", observed=True).head(2)[["axis_id", "program"]]
        )
        selected_pairs = top_per_axis.drop_duplicates().reset_index(drop=True)
        assert len(selected_pairs) == 2 * len(BEHAVIOR_AXES)
        selected_signature_rows = targeted_program_signatures.merge(
            selected_pairs, on=["axis_id", "program"], how="inner", validate="one_to_one"
        )
        selected_signature_rows["axis_order"] = selected_signature_rows.axis_id.map(
            {axis["axis_id"]: index for index, axis in enumerate(BEHAVIOR_AXES)}
        )
        selected_signature_rows = selected_signature_rows.sort_values(
            ["axis_order", "combined_global_fdr", "program_label"]
        ).reset_index(drop=True)
        forest_parts = []
        for row in selected_signature_rows.itertuples(index=False):
            for estimate_type, label, effect, se in [
                ("combined", "Focal-pair mean", row.combined_effect_sd, row.combined_se),
                ("member1", f"MC{int(row.member1_mc)}", row.member1_effect_sd, row.member1_se),
                ("member2", f"MC{int(row.member2_mc)}", row.member2_effect_sd, row.member2_se),
            ]:
                forest_parts.append({
                    "axis_id": row.axis_id, "axis_label": row.axis_label,
                    "program": row.program, "program_label": row.program_label,
                    "estimate_type": estimate_type, "estimate_label": label,
                    "effect_sd": effect, "standard_error": se,
                    "ci_low": effect - 1.96 * se, "ci_high": effect + 1.96 * se,
                    "combined_global_fdr": row.combined_global_fdr,
                    "conjunction_global_fdr": row.conjunction_global_fdr,
                    "shared_behavioral_signature": row.shared_behavioral_signature,
                    "pooled_only_candidate": row.pooled_only_candidate,
                })
        plot_data = pd.DataFrame(forest_parts)
        plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)
        pair_order = selected_signature_rows[["axis_id", "axis_label", "program", "program_label"]]
        pair_labels = [
            f"{row.axis_label} · {row.program_label}" for row in pair_order.itertuples(index=False)
        ]
        pair_key_to_y = {
            (row.axis_id, row.program): len(pair_order) - 1 - index
            for index, row in enumerate(pair_order.itertuples(index=False))
        }
        style = {
            "combined": {"color": ORANGE, "marker": "D", "offset": 0.0},
            "member1": {"color": INK, "marker": "o", "offset": 0.18},
            "member2": {"color": "#6F93B5", "marker": "s", "offset": -0.18},
        }
        fig, ax = plt.subplots(figsize=(9.2, 7.6))
        for estimate_type in ["member1", "member2", "combined"]:
            subset = plot_data[plot_data.estimate_type == estimate_type]
            y = np.asarray([
                pair_key_to_y[(row.axis_id, row.program)] + style[estimate_type]["offset"]
                for row in subset.itertuples(index=False)
            ])
            ax.errorbar(
                subset.effect_sd, y,
                xerr=np.vstack([
                    subset.effect_sd.to_numpy(float) - subset.ci_low.to_numpy(float),
                    subset.ci_high.to_numpy(float) - subset.effect_sd.to_numpy(float),
                ]), fmt=style[estimate_type]["marker"],
                color=style[estimate_type]["color"], markersize=5.5,
                elinewidth=1.0, capsize=2.2,
                label={
                    "combined": "Focal-pair mean",
                    "member1": "First focal member",
                    "member2": "Second focal member",
                }[estimate_type],
            )
        ax.axvline(0, color=GREY, linewidth=0.9, linestyle="--")
        ax.set_yticks(
            np.arange(len(pair_order)), list(reversed(pair_labels)), fontsize=8.2
        )
        ax.set_xlabel("Target minus equal-MC reference Hallmark score (SD; 95% CI)")
        ax.set_title("Combined and member-specific Hallmark effects", loc="left", pad=23)
        ax.text(
            0, 1.028, "Representative display: two lowest combined-q programs per locked axis",
            transform=ax.transAxes, fontsize=8.4, color=GREY,
        )
        ax.text(
            0, 1.008,
            "Focal pairs: Baso MC2/MC4 · Neu MC0/MC5 · Mono MC0/MC3 · Persistent MC11/MC18",
            transform=ax.transAxes, fontsize=8.1, color=GREY,
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="best", fontsize=8.2)
        fig.tight_layout()
        svg = FIGDIR / f"{plot_stem}.svg"
        png = FIGDIR / f"{plot_stem}.png"
        fig.savefig(svg, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        display(fig)
        print("saved:", svg.relative_to(PROJECT_ROOT), "|", png.relative_to(PROJECT_ROOT))
        plt.close(fig)
        """
    ),
])

cells.extend(targeted_e_f_cells)
cells.extend(targeted_supplement_cells)

cells.extend([
    md(
        r"""
        ## Targeted findings and validation

        Findings are generated from the executed tables. Observed statistical
        evidence, biological interpretation, and design limitations remain
        separate. The final validation also proves that the broad correlation
        sensitivity is numerically unchanged from the dated pre-correction snapshot.
        """
    ),
    code(
        r"""
        # Summarize exact multiplicity families and computed targeted findings
        targeted_bh_family_audit = pd.DataFrame([
            {
                "family": "Hallmark combined across four axes",
                "n_tests": len(targeted_program_signatures),
                "expected_n_tests": 4 * 50,
                "n_fdr_lt_0_05": int(targeted_program_signatures.combined_global_fdr_pass.sum()),
            },
            {
                "family": "Hallmark max-P conjunction across four axes",
                "n_tests": len(targeted_program_signatures),
                "expected_n_tests": 4 * 50,
                "n_fdr_lt_0_05": int(targeted_program_signatures.conjunction_global_fdr_pass.sum()),
            },
            {
                "family": "Gene combined across four axes",
                "n_tests": len(targeted_gene_signatures),
                "expected_n_tests": 4 * 9777,
                "n_fdr_lt_0_05": int(targeted_gene_signatures.combined_global_fdr_pass.sum()),
            },
            {
                "family": "Gene max-P conjunction across four axes",
                "n_tests": len(targeted_gene_signatures),
                "expected_n_tests": 4 * 9777,
                "n_fdr_lt_0_05": int(targeted_gene_signatures.conjunction_global_fdr_pass.sum()),
            },
            {
                "family": "Hallmark secondary pair contrasts",
                "n_tests": len(targeted_secondary_programs),
                "expected_n_tests": 2 * 50,
                "n_fdr_lt_0_05": int((targeted_secondary_programs.secondary_global_fdr < 0.05).sum()),
            },
            {
                "family": "Gene secondary pair contrasts",
                "n_tests": len(targeted_secondary_genes),
                "expected_n_tests": 2 * 9777,
                "n_fdr_lt_0_05": int((targeted_secondary_genes.secondary_global_fdr < 0.05).sum()),
            },
        ])
        assert (targeted_bh_family_audit.n_tests == targeted_bh_family_audit.expected_n_tests).all()
        targeted_bh_family_audit.to_csv(
            TABLEDIR / "targeted_bh_family_audit.csv", index=False
        )

        targeted_key_results = targeted_signature_summary.copy()
        targeted_key_results.to_csv(
            TABLEDIR / "targeted_key_results.csv", index=False
        )
        hallmark_combined_hits = targeted_program_signatures[
            targeted_program_signatures.combined_global_fdr_pass
        ].sort_values(["combined_global_fdr", "combined_pvalue"])
        hallmark_shared_hits = targeted_program_signatures[
            targeted_program_signatures.shared_behavioral_signature
        ].sort_values(["combined_global_fdr", "conjunction_global_fdr"])
        gene_combined_hits = targeted_gene_signatures[
            targeted_gene_signatures.combined_global_fdr_pass
        ].sort_values(["combined_global_fdr", "combined_pvalue"])
        gene_shared_hits = targeted_gene_signatures[
            targeted_gene_signatures.shared_behavioral_signature
        ].sort_values(["combined_global_fdr", "conjunction_global_fdr"])

        finding_lines = [
            "OBSERVED TARGETED METACLONAL-SIGNATURE EVIDENCE",
            "- Primary cohort: 217 Day-2 pooled-HSPC LARRY-clone pseudobulks in seven Multi-lineage metaclones.",
            "- Locked analysis-defined axes: Baso-rich MC2/MC4; Neu-rich MC0/MC5; Mono-rich MC0/MC3; persistent-undifferentiated MC11/MC18.",
            (
                f"- Hallmark combined/global FDR hits: {len(hallmark_combined_hits)}/200; "
                f"full shared-signature hits: {len(hallmark_shared_hits)}/200."
            ),
            (
                f"- Gene combined/global FDR hits: {len(gene_combined_hits)}/39108; "
                f"full shared-signature hits: {len(gene_shared_hits)}/39108."
            ),
            "- Combined-significant Hallmark results:",
        ]
        if hallmark_combined_hits.empty:
            finding_lines.append("  - None.")
        else:
            for row in hallmark_combined_hits.itertuples(index=False):
                finding_lines.append(
                    f"  - {row.axis_label} / {row.program_label}: combined effect={row.combined_effect_sd:+.3f} SD, "
                    f"combined global q={row.combined_global_fdr:.3g}, member effects="
                    f"{row.member1_effect_sd:+.3f}/{row.member2_effect_sd:+.3f} SD, "
                    f"conjunction global q={row.conjunction_global_fdr:.3g}, "
                    f"shared={bool(row.shared_behavioral_signature)}, reason={row.pooled_only_failure_reason}."
                )
        finding_lines.append("- Combined-significant gene results:")
        if gene_combined_hits.empty:
            finding_lines.append("  - None.")
        else:
            for row in gene_combined_hits.head(50).itertuples(index=False):
                finding_lines.append(
                    f"  - {row.axis_label} / {row.gene}: combined log2FC={row.combined_log2fc:+.3f}, "
                    f"combined global q={row.combined_global_fdr:.3g}, member log2FC="
                    f"{row.member1_log2fc:+.3f}/{row.member2_log2fc:+.3f}, "
                    f"conjunction global q={row.conjunction_global_fdr:.3g}, "
                    f"shared={bool(row.shared_behavioral_signature)}, reason={row.pooled_only_failure_reason}."
                )
            if len(gene_combined_hits) > 50:
                finding_lines.append(f"  - {len(gene_combined_hits) - 50} additional rows are in the full table.")
        finding_lines.extend([
            "- Secondary contrasts:",
            *[
                (
                    f"  - {axis_label}: Hallmarks global q<0.05="
                    f"{int((targeted_secondary_programs.loc[targeted_secondary_programs.axis_id == axis_id, 'secondary_global_fdr'] < 0.05).sum())}; "
                    f"genes global q<0.05="
                    f"{int((targeted_secondary_genes.loc[targeted_secondary_genes.axis_id == axis_id, 'secondary_global_fdr'] < 0.05).sum())}."
                )
                for axis_id, axis_label in [
                    ("mc5_vs_mc3_divergence", "MC5 vs MC3 divergence"),
                    ("mc3_vs_mc18_convergence", "MC3 vs MC18 branch-compatibility convergence"),
                ]
            ],
            "",
            "INTERPRETATION BOUNDARIES",
            "- A combined-only result is a pooled candidate and is not a shared behavioral signature.",
            "- One-metaclone-driven is reserved for exactly one nominal same-direction member; lack of global max-P support alone is not assigned that label.",
            "- The four groups were defined retrospectively from later sampled output. Day-2 differences are associations, not causal or prospective predictors.",
            "- Persistent undifferentiated output is an observed compartment fraction, not proof of self-renewal.",
            "- Destructive sampling, incomplete clone recovery, source-sublabel imbalance, and absence of independent biological replicates remain limitations.",
        ])
        targeted_findings_text = "\n".join(finding_lines)
        (REPORTDIR / "targeted_metaclonal_signature_findings_and_boundaries.txt").write_text(
            targeted_findings_text + "\n"
        )
        print(targeted_findings_text)
        display(targeted_bh_family_audit)
        """
    ),
    code(
        r"""
        # Final targeted validation, correlation immutability, plot contract, and hashes
        SNAPSHOT_DIR = (
            PROJECT_ROOT / "analysis_workspace/archive/analyses/"
            "weinreb_day2_hspc_metaclone_gene_programs_pre_targeted_20260903"
        )
        snapshot_correlation_path = (
            SNAPSHOT_DIR / "tables/multilineage_program_output_correlations.csv"
        )
        assert snapshot_correlation_path.exists()
        snapshot_correlations = pd.read_csv(snapshot_correlation_path)
        current_correlations = pd.read_csv(
            TABLEDIR / "multilineage_program_output_correlations.csv"
        )
        correlation_keys = ["temporal_scope", "output_type", "program"]
        assert not snapshot_correlations.duplicated(correlation_keys).any()
        assert not current_correlations.duplicated(correlation_keys).any()
        correlation_comparison = snapshot_correlations.merge(
            current_correlations, on=correlation_keys, how="outer",
            suffixes=("__snapshot", "__current"), indicator=True,
            validate="one_to_one",
        )
        assert (correlation_comparison._merge == "both").all()
        correlation_numeric_columns = [
            column for column in snapshot_correlations.select_dtypes(include=[np.number]).columns
            if column not in correlation_keys
        ]
        correlation_max_errors = {}
        for column in correlation_numeric_columns:
            before = correlation_comparison[f"{column}__snapshot"].to_numpy(float)
            after = correlation_comparison[f"{column}__current"].to_numpy(float)
            equal_nan = np.isnan(before) & np.isnan(after)
            finite = np.isfinite(before) & np.isfinite(after)
            assert np.all(equal_nan | finite)
            error = float(np.max(np.abs(before[finite] - after[finite]))) if finite.any() else 0.0
            correlation_max_errors[column] = error
            assert error < 1e-12
        correlation_boolean_columns = [
            column for column in [
                "lomo_sign_stable", "pooled_direction_agrees",
                "mature_direction_agrees", "stable_endpoint_association",
            ] if column in snapshot_correlations.columns
        ]
        for column in correlation_boolean_columns:
            assert (
                correlation_comparison[f"{column}__snapshot"].astype(str).to_numpy()
                == correlation_comparison[f"{column}__current"].astype(str).to_numpy()
            ).all()
        n_estimable_snapshot = int(snapshot_correlations.exact_permutation_pvalue.notna().sum())
        n_estimable_current = int(current_correlations.exact_permutation_pvalue.notna().sum())
        assert len(snapshot_correlations) == len(current_correlations) == 1500
        assert n_estimable_snapshot == n_estimable_current == 1450
        targeted_correlation_immutability_audit = pd.DataFrame([{
            "snapshot_path": str(snapshot_correlation_path.relative_to(PROJECT_ROOT)),
            "current_path": str((TABLEDIR / "multilineage_program_output_correlations.csv").relative_to(PROJECT_ROOT)),
            "n_grid_rows": len(current_correlations),
            "n_estimable_rows": n_estimable_current,
            "max_numeric_abs_error": max(correlation_max_errors.values()),
            "boolean_columns_equal": True,
            "passed": True,
        }])
        targeted_correlation_immutability_audit.to_csv(
            REPORTDIR / "targeted_correlation_immutability_audit.csv", index=False
        )

        built_notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        notebook_sources = ["".join(cell.get("source", "")) for cell in built_notebook.cells]
        targeted_plot_stems = [
            "targeted_figure_4a_multilineage_day4_day6_output_composition",
            "targeted_figure_4b_behavior_axis_registry",
            "targeted_figure_4c_hallmark_signature_effect_matrix",
            "targeted_figure_4d_member_agreement_forest",
            "targeted_figure_4e_monocyte_matched_output_program_landscape",
            "targeted_figure_4f_neutrophil_matched_output_program_landscape",
            "targeted_supp_baso_rich_gene_volcano",
            "targeted_supp_neu_rich_gene_volcano",
            "targeted_supp_mono_rich_gene_volcano",
            "targeted_supp_persistent_undifferentiated_gene_volcano",
            "targeted_supp_hallmark_preranked_gsea_matrix",
            "targeted_supp_library_loo_hallmark_effect_diagnostic",
        ]
        targeted_plot_sources = [
            source for source in notebook_sources
            if source.lstrip().startswith((
                "# TARGETED_MAIN_PLOT_INLINE_V1",
                "# TARGETED_SUPPLEMENT_PLOT_INLINE_V1",
            ))
        ]
        targeted_plot_contract = all(
            source.count("plt.subplots") == 1
            and source.count("fig.savefig(svg") == 1
            and source.count("fig.savefig(png") == 1
            and source.count("display(fig)") == 1
            and source.count("plt.close(fig)") == 1
            for source in targeted_plot_sources
        )
        targeted_svg_editable = all(
            "<svg" in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
            and "<image" not in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
            for stem in targeted_plot_stems
        )
        active_plot_stems = plot_stems + correlation_plot_stems + targeted_plot_stems
        assert len(active_plot_stems) == len(set(active_plot_stems)) == 24
        all_active_svg_editable = all(
            "<svg" in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
            and "<image" not in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
            for stem in active_plot_stems
        )
        targeted_plot_pairs_complete = all(
            (FIGDIR / f"{stem}.svg").stat().st_size > 0
            and (FIGDIR / f"{stem}.png").stat().st_size > 0
            and (TABLEDIR / f"{stem}_plot_data.csv").stat().st_size > 0
            for stem in targeted_plot_stems
        )

        exact_axis_pairs = {
            row.axis_id: row.focal_mcs
            for row in behavior_axis_definitions.itertuples(index=False)
        } == {
            "baso_rich": "2|4", "neu_rich": "0|5", "mono_rich": "0|3",
            "persistent_undifferentiated": "11|18",
        }
        targeted_validation_rows = [
            ("dated_snapshot_present", SNAPSHOT_DIR.exists(), str(SNAPSHOT_DIR.relative_to(PROJECT_ROOT))),
            ("targeted_clone_units", len(targeted_metadata) == 217, f"observed={len(targeted_metadata)}"),
            ("one_row_per_clone", targeted_metadata.index.is_unique, f"unique={targeted_metadata.index.nunique()}"),
            ("seven_multilineage_metaclones", set(targeted_metadata.metaclone) == set(TARGET_MCS), str(TARGET_MCS)),
            ("exact_behavior_axis_pairs", exact_axis_pairs, str(exact_axis_pairs)),
            ("behavior_rules_reproduced", len(behavior_axis_registry) == 28, f"rows={len(behavior_axis_registry)}"),
            ("population_not_used", "population" not in targeted_metadata.columns and "population" not in targeted_contrast_registry.columns, "metadata and registry checked"),
            ("fixed_gene_universe", targeted_counts.shape[1] == 9777, f"observed={targeted_counts.shape[1]}"),
            ("fixed_hallmark_universe", targeted_program_scores.shape[1] == 50, f"observed={targeted_program_scores.shape[1]}"),
            ("primary_model_full_rank", int(targeted_fit_audit.design_rank.iloc[0]) == int(targeted_fit_audit.design_columns.iloc[0]), targeted_design_formula),
            ("all_targeted_contrasts_oriented", len(targeted_adjusted_mean_orientation_audit) == 14 and (targeted_adjusted_mean_orientation_audit[["program_max_abs_error", "gene_max_abs_error"]].to_numpy(float) < 1e-10).all(), "14/14 specs"),
            ("contrast_vectors_neutral", targeted_vector_neutrality_audit.target_minus_reference_vector_verified.all(), "intercept/library weights cancel"),
            ("hallmark_combined_bh_family_200", len(targeted_program_signatures) == 200, str(len(targeted_program_signatures))),
            ("hallmark_conjunction_bh_family_200", targeted_program_signatures.conjunction_global_fdr.notna().sum() == 200, str(targeted_program_signatures.conjunction_global_fdr.notna().sum())),
            ("gene_combined_bh_family_39108", len(targeted_gene_signatures) == 39108, str(len(targeted_gene_signatures))),
            ("gene_conjunction_bh_family_39108", targeted_gene_signatures.conjunction_global_fdr.notna().sum() == 39108, str(targeted_gene_signatures.conjunction_global_fdr.notna().sum())),
            ("secondary_hallmark_family_100", len(targeted_secondary_programs) == 100, str(len(targeted_secondary_programs))),
            ("secondary_gene_family_19554", len(targeted_secondary_genes) == 19554, str(len(targeted_secondary_genes))),
            ("five_comparator_drops_per_axis", (comparator_program_stability.comparator_drop_n_effects == 15).all(), "5 drops x 3 effects"),
            ("seven_true_library_loo_refits", len(targeted_library_loo_fit_audit) == 7 and targeted_library_loo_fit_audit.estimable.all(), str(len(targeted_library_loo_fit_audit))),
            ("library_loo_removed_cells", (targeted_library_loo_fit_audit.removed_cells > 0).all(), targeted_library_loo_fit_audit.removed_cells.astype(str).str.cat(sep="|")),
            ("library_loo_reaggregated_clone_counts", (targeted_library_loo_fit_audit.remaining_clone_pseudobulks <= 217).all(), targeted_library_loo_fit_audit.remaining_clone_pseudobulks.astype(str).str.cat(sep="|")),
            ("shared_gate_components_exported", all(column in targeted_program_signatures.columns for column in ["combined_global_fdr_pass", "members_same_direction_as_combined", "conjunction_global_fdr_pass", "comparator_drop_all_same_direction", "library_loo_all_estimable_same_direction", "shared_behavioral_signature", "pooled_only_failure_reason"]), "program table"),
            ("broad_correlation_1450_immutable", bool(targeted_correlation_immutability_audit.passed.iloc[0]), f"max_error={targeted_correlation_immutability_audit.max_numeric_abs_error.iloc[0]:.3g}"),
            ("targeted_plot_cell_count", len(targeted_plot_sources) == len(targeted_plot_stems), f"observed={len(targeted_plot_sources)} expected={len(targeted_plot_stems)}"),
            ("targeted_plot_inline_contract", targeted_plot_contract, "one inline plot/save/display/close per cell"),
            ("targeted_plot_triplets_complete", targeted_plot_pairs_complete, f"stems={len(targeted_plot_stems)}"),
            ("targeted_svg_editable", targeted_svg_editable, "no embedded raster images"),
            ("all_active_svg_editable", all_active_svg_editable, f"no embedded raster images across {len(active_plot_stems)} active SVGs"),
            ("input_landing_zone_empty", not any((PROJECT_ROOT / "Input").iterdir()), "Input/ empty"),
        ]
        targeted_validation = pd.DataFrame(
            targeted_validation_rows, columns=["check", "passed", "detail"]
        )
        targeted_validation.to_csv(
            REPORTDIR / "targeted_metaclonal_signature_validation.csv", index=False
        )
        assert targeted_validation.passed.all(), targeted_validation.loc[~targeted_validation.passed]

        targeted_artifact_paths = [BUILDER_PATH]
        for folder in [FIGDIR, TABLEDIR, REPORTDIR, RESOURCEDIR]:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.name not in {
                    "artifact_manifest.csv", "targeted_artifact_manifest.csv"
                } and not path.name.startswith("cnmf_"):
                    targeted_artifact_paths.append(path)
        targeted_artifact_manifest = pd.DataFrame([{
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in sorted(set(targeted_artifact_paths))])
        targeted_artifact_manifest.to_csv(
            REPORTDIR / "targeted_artifact_manifest.csv", index=False
        )
        # Refresh the active broad manifest after every targeted artifact exists.
        targeted_artifact_manifest.to_csv(
            REPORTDIR / "artifact_manifest.csv", index=False
        )
        display(targeted_validation)
        display(targeted_artifact_manifest.tail(20))
        print("All targeted metaclonal-signature validation checks passed.")
        print("Notebook:", NOTEBOOK_PATH)
        print("Environment: chronoclone |", ENV_PATH)
        print("Execution command:", EXECUTION_COMMAND)
        """
    ),
])


cells.extend(build_cnmf_cells(md, code))


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (chronoclone)",
            "language": "python",
            "name": "chronoclone",
        },
        "language_info": {"name": "python", "version": "3.11.15"},
    },
)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(OUT)
