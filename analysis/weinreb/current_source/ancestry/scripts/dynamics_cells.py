"""Notebook cell sources for exact ancestry linkage and observed temporal phenotypes.

The parent notebook provides obs, hspc_cells, MC_TO_FAMILY, TABLEDIR, FIGDIR,
REPORTDIR, numpy as np, pandas as pd, pyplot as plt, and display. All derived
values and figures are calculated visibly in the returned notebook cells.
"""

from textwrap import dedent

import nbformat as nbf


def build_dynamics_cells():
    def md(text):
        return nbf.v4.new_markdown_cell(dedent(text).strip())

    def code(text):
        return nbf.v4.new_code_cell(dedent(text).strip())

    return [
        md(r"""
        ## Link observed clonal behavior to the Day-2 HSPCs

        The unit linking early molecular measurements to later behavior is the
        **LARRY clone identifier**, not an expression-nearest-neighbor match.
        The primary temporal summaries below use exactly the clones with an
        eligible Day-2 `prog_1` or `prog_2` HSPC. Each observed clone contributes
        one value at each sampled day. The complete assigned-clone population
        and the subset observed at both Days 4 and 6 are retained as descriptive
        sensitivities. A missing clone at a snapshot is not assigned zero
        biological output.

        The existing monocyte display order is MC6, MC8, MC7, MC12. It is
        retained for continuity and audited against the recomputed branch
        summaries rather than assumed to be ordered under every estimand.
        The current seven-member multilineage map is retained in its original
        order. Palantir pseudotime and masks are inherited from the existing
        trajectory object; they are not refitted.
        """),
        code(r"""
        DYN_MONO_MCS = [6, 8, 7, 12]
        DYN_MULTI_MCS = [0, 2, 5, 3, 18, 4, 11]
        DYN_DISPLAY_MCS = DYN_MONO_MCS + DYN_MULTI_MCS
        DYN_DAYS = [2, 4, 6]
        DYN_BRANCHES = ["Baso", "Neu", "Mono"]
        assert {mc for mc, family in MC_TO_FAMILY.items() if family == "Monocyte"} == set(DYN_MONO_MCS)
        assert {mc for mc, family in MC_TO_FAMILY.items() if family == "Multi-lineage"} == set(DYN_MULTI_MCS)
        dyn_required = ["LARRY_ID", "metaclone", "day", "cell_type", "Library",
                        "cell_type_weinreb", "palantir_pseudotime"] + ["mask_" + b for b in DYN_BRANCHES]
        assert set(dyn_required).issubset(obs.columns)
        dyn_cells = obs.loc[obs.metaclone.isin(MC_TO_FAMILY), dyn_required].copy()
        dyn_cells.insert(0, "cell_id", dyn_cells.index.astype(str))
        for column in ["LARRY_ID", "Library", "cell_type", "cell_type_weinreb"]:
            dyn_cells[column] = dyn_cells[column].astype(str)
        dyn_cells[["metaclone", "day"]] = dyn_cells[["metaclone", "day"]].astype(int)
        assert dyn_cells.groupby("LARRY_ID", observed=True).metaclone.nunique().eq(1).all()
        assert dyn_cells.day.isin(DYN_DAYS).all()
        assert dyn_cells.cell_id.is_unique
        dyn_day2_cells = hspc_cells.copy()
        assert dyn_day2_cells.day.eq(2).all()
        assert dyn_day2_cells.cell_type.astype(str).isin(["prog_1", "prog_2"]).all()
        assert set(dyn_day2_cells.index.astype(str)).issubset(set(dyn_cells.cell_id))
        dyn_hspc_counts = dyn_day2_cells.groupby("LARRY_ID", observed=True).size()
        dyn_hspc_counts.index = dyn_hspc_counts.index.astype(str)
        dyn_day2_clones = set(dyn_hspc_counts.index)
        dyn_cells["day2_hspc_linked"] = dyn_cells.LARRY_ID.isin(dyn_day2_clones)
        dyn_cells["lineage_family"] = dyn_cells.metaclone.map(MC_TO_FAMILY)
        # Exact source definition: reference_workspace/notebooks/Figure2_klein_pace.ipynb, code cell 8.
        dyn_fine_output_map = {
            "Neu": "Neu", "early_prog_Neu": "Neu", "prog_Neu": "Neu",
            "Mono": "Mono", "prog_Mono": "Mono", "prog_DC_Mono": "Mono", "DC": "Mono",
            "Baso": "Baso", "prog_Baso_Eos": "Baso", "Eos": "Other", "Mast": "Mast",
            "Meg": "Ery/Mk", "Ery": "Ery/Mk", "prog_Meg_Ery": "Ery/Mk", "Ly": "Other", "prog_Ly_pDC": "Other",
            "prog_1": "undiff", "prog_2": "undiff", "prog_3": "undiff", "prog_4": "undiff", "prog_Baso_Meg_Ery_Mast": "undiff",
        }
        dyn_cells["differentiated_output_class"] = dyn_cells.cell_type.map(dyn_fine_output_map)
        assert dyn_cells.differentiated_output_class.notna().all()
        pd.DataFrame(list(dyn_fine_output_map.items()), columns=["cell_type", "differentiated_output_class"]).to_csv(
            TABLEDIR / "ancestry_original_differentiated_output_mapping.csv", index=False)
        dyn_clone_lookup = dyn_cells.groupby("LARRY_ID", observed=True).metaclone.first()
        dyn_clone_days = dyn_cells.groupby(["LARRY_ID", "day"], observed=True).size().rename("n_cells")
        dyn_grid = pd.MultiIndex.from_product(
            [sorted(dyn_clone_lookup.index), DYN_DAYS], names=["LARRY_ID", "day"]
        )
        dyn_clone_day = dyn_clone_days.reindex(dyn_grid, fill_value=0).reset_index()
        dyn_clone_day["metaclone"] = dyn_clone_day.LARRY_ID.map(dyn_clone_lookup).astype(int)
        dyn_clone_day["lineage_family"] = dyn_clone_day.metaclone.map(MC_TO_FAMILY)
        dyn_clone_day["day2_hspc_cells"] = dyn_clone_day.LARRY_ID.map(dyn_hspc_counts).fillna(0).astype(int)
        dyn_clone_day["day2_hspc_linked"] = dyn_clone_day.day2_hspc_cells.gt(0)
        dyn_clone_day["observed"] = dyn_clone_day.n_cells.gt(0)
        dyn_presence = dyn_clone_day.pivot(index="LARRY_ID", columns="day", values="observed")
        dyn_paired_clones = set(dyn_presence.index[dyn_presence[4] & dyn_presence[6]]) & dyn_day2_clones
        dyn_clone_day["observed_day4_and_day6"] = dyn_clone_day.LARRY_ID.isin(dyn_paired_clones)
        dyn_clone_day["missing_snapshot_semantics"] = np.where(
            dyn_clone_day.observed, "observed cells", "no recovered cells; no biological zero imputed"
        )
        dyn_libraries = (dyn_cells.groupby(["LARRY_ID", "day"], observed=True).Library
                         .agg(lambda values: ";".join(sorted(set(values)))).rename("observed_libraries"))
        dyn_clone_day = dyn_clone_day.merge(dyn_libraries, on=["LARRY_ID", "day"], how="left", validate="one_to_one")
        dyn_clone_day["observed_libraries"] = dyn_clone_day.observed_libraries.fillna("")
        dyn_clone_day.to_csv(TABLEDIR / "ancestry_all_clone_day_linkage.csv", index=False)
        dyn_cells.to_csv(TABLEDIR / "ancestry_linked_cell_register.csv.gz", index=False, compression="gzip")
        dyn_support_rows = []
        for mc in MC_TO_FAMILY:
            frame = dyn_clone_day[dyn_clone_day.metaclone.eq(mc)]
            early = frame[frame.day.eq(2) & frame.day2_hspc_linked]
            linked_ids = set(early.LARRY_ID)
            row = {"metaclone": mc, "lineage_family": MC_TO_FAMILY[mc],
                   "n_all_assigned_clones": int(frame.LARRY_ID.nunique()),
                   "n_day2_hspc_clones": int(len(early)),
                   "n_day2_hspc_cells": int(early.day2_hspc_cells.sum()),
                   "n_linked_clones_at_both_day4_and_day6": int(len(linked_ids & dyn_paired_clones))}
            for day in DYN_DAYS:
                dayframe = frame[frame.day.eq(day)]
                linked = dayframe[dayframe.day2_hspc_linked & dayframe.observed]
                row[f"n_all_observed_clones_day{day}"] = int(dayframe.observed.sum())
                row[f"n_linked_observed_clones_day{day}"] = int(len(linked))
                row[f"n_linked_observed_cells_day{day}"] = int(linked.n_cells.sum())
                row[f"fraction_day2_clones_recovered_day{day}"] = len(linked) / len(early) if len(early) else np.nan
            dyn_support_rows.append(row)
        dyn_linkage_support = pd.DataFrame(dyn_support_rows)
        dyn_linkage_support.to_csv(TABLEDIR / "ancestry_linkage_support_by_metaclone.csv", index=False)
        dyn_cohorts = {
            "day2_hspc_linked": dyn_cells[dyn_cells.day2_hspc_linked],
            "all_assigned": dyn_cells,
            "day2_linked_observed_both_later_days": dyn_cells[dyn_cells.LARRY_ID.isin(dyn_paired_clones)],
        }
        dyn_branch_rows = []
        for cohort_name, cohort in dyn_cohorts.items():
            for (mc, clone, day), frame in cohort.groupby(["metaclone", "LARRY_ID", "day"], observed=True):
                for branch in DYN_BRANCHES:
                    selected = frame[frame["mask_" + branch].astype(bool)]
                    values = selected.palantir_pseudotime.dropna()
                    dyn_branch_rows.append({
                        "cohort": cohort_name, "metaclone": int(mc), "LARRY_ID": str(clone), "day": int(day),
                        "branch": branch, "n_total_cells": len(frame), "n_branch_cells": len(selected),
                        "n_finite_branch_pseudotime": len(values),
                        "branch_fraction": len(selected) / len(frame),
                        "branch_pseudotime_median": float(values.median()) if len(values) else np.nan,
                        "day2_hspc_cells": int(dyn_hspc_counts.get(str(clone), 0)),
                    })
        dyn_clone_branch = pd.DataFrame(dyn_branch_rows)
        dyn_clone_branch.to_csv(TABLEDIR / "ancestry_clone_day_branch_measurements.csv", index=False)
        dyn_branch_summary = dyn_clone_branch.groupby(["cohort", "metaclone", "day", "branch"], observed=True).agg(
            n_observed_clones=("LARRY_ID", "nunique"), n_observed_cells=("n_total_cells", "sum"),
            n_branch_cells=("n_branch_cells", "sum"),
            n_branch_informative_clones=("branch_pseudotime_median", "count"),
            clone_balanced_occupancy=("branch_fraction", "mean"),
            median_of_clone_branch_medians=("branch_pseudotime_median", "median"),
            mean_of_clone_branch_medians=("branch_pseudotime_median", "mean"),
        ).reset_index()
        dyn_branch_summary["pooled_cell_occupancy"] = dyn_branch_summary.n_branch_cells / dyn_branch_summary.n_observed_cells
        dyn_branch_summary.to_csv(TABLEDIR / "ancestry_branch_summary_by_cohort_metaclone_day.csv", index=False)
        dyn_order_audit = dyn_branch_summary[
            dyn_branch_summary.metaclone.isin(DYN_MONO_MCS) & dyn_branch_summary.day.eq(6) & dyn_branch_summary.branch.eq("Mono")
        ].copy()
        dyn_order_audit["existing_display_order"] = dyn_order_audit.metaclone.map({mc: i + 1 for i, mc in enumerate(DYN_MONO_MCS)})
        dyn_order_audit["rank_median_of_clone_branch_medians"] = dyn_order_audit.groupby("cohort")["median_of_clone_branch_medians"].rank(method="min")
        dyn_order_audit.to_csv(TABLEDIR / "ancestry_monocyte_day6_order_audit.csv", index=False)
        dyn_checks = {
            "day2_cell_register_exact": set(dyn_day2_cells.index.astype(str)) == set(
                dyn_cells.loc[dyn_cells.day.eq(2) & dyn_cells.cell_type.isin(["prog_1", "prog_2"]), "cell_id"]),
            "unique_clone_day_register": not dyn_clone_day.duplicated(["LARRY_ID", "day"]).any(),
            "three_registered_days_per_clone": dyn_clone_day.groupby("LARRY_ID").size().eq(3).all(),
            "one_mc_per_clone": dyn_cells.groupby("LARRY_ID").metaclone.nunique().eq(1).all(),
            "branch_occupancy_in_unit_interval": dyn_clone_branch.branch_fraction.between(0, 1).all(),
            "absent_branch_position_missing": dyn_clone_branch.loc[dyn_clone_branch.n_branch_cells.eq(0), "branch_pseudotime_median"].isna().all(),
            "clone_day_cell_totals_reconcile": int(dyn_clone_day.n_cells.sum()) == len(dyn_cells),
            "common_later_cohort_has_identical_clone_ids": (
                set(dyn_cohorts["day2_linked_observed_both_later_days"].loc[lambda frame: frame.day.eq(4), "LARRY_ID"])
                == set(dyn_cohorts["day2_linked_observed_both_later_days"].loc[lambda frame: frame.day.eq(6), "LARRY_ID"])
                == dyn_paired_clones
            ),
        }
        assert all(dyn_checks.values()), dyn_checks
        display(dyn_linkage_support[dyn_linkage_support.metaclone.isin(DYN_DISPLAY_MCS)])
        display(dyn_order_audit[["cohort", "metaclone", "n_branch_informative_clones", "median_of_clone_branch_medians", "existing_display_order", "rank_median_of_clone_branch_medians"]])
        """),
        md(r"""
        ### Exact clone support for the ancestry comparison

        This grouped-bar figure follows a fixed Day-2 HSPC clone cohort into
        the two later destructive snapshots. Bar height is a unique-clone
        count. It is an assay-linkage display, not an estimate of survival or
        clone growth. Colors and bar position both identify the snapshot.
        """),
        code(r"""
        plot_stem = "ancestry_01_day2_clone_linkage"
        plot_support = dyn_linkage_support.set_index("metaclone").loc[DYN_DISPLAY_MCS].reset_index()
        plot_rows = []
        for row in plot_support.itertuples(index=False):
            for day in [2, 4, 6]:
                count = row.n_day2_hspc_clones if day == 2 else getattr(row, f"n_linked_observed_clones_day{day}")
                plot_rows.append({"metaclone": row.metaclone, "lineage_family": row.lineage_family, "day": day,
                                  "n_clones": count, "day2_hspc_clone_denominator": row.n_day2_hspc_clones,
                                  "n_day2_hspc_cells": row.n_day2_hspc_cells,
                                  "n_linked_clones_at_both_day4_and_day6": row.n_linked_clones_at_both_day4_and_day6,
                                  "fraction_of_day2_hspc_clones": count / row.n_day2_hspc_clones,
                                  "display_order": DYN_DISPLAY_MCS.index(row.metaclone)})
        plot_data = pd.DataFrame(plot_rows)
        plot_data.to_csv(TABLEDIR / f"{plot_stem}.csv", index=False)
        fig, ax = plt.subplots(figsize=(12.6, 5.4))
        plot_x = np.arange(len(DYN_DISPLAY_MCS), dtype=float)
        plot_x[4:] += 0.6
        day_colors = {2: "#17324D", 4: "#AEB8C2", 6: "#E0913C"}
        for offset, day in zip([-0.25, 0, 0.25], [2, 4, 6]):
            values = plot_data[plot_data.day.eq(day)].set_index("metaclone").loc[DYN_DISPLAY_MCS, "n_clones"].to_numpy()
            bars = ax.bar(plot_x + offset, values, width=0.23, color=day_colors[day],
                          edgecolor="#17324D", linewidth=0.5,
                          label="Day 2: HSPC clones" if day == 2 else f"Day {day}: linked clones recovered")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, value + 0.65, str(value), ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(plot_x, [f"MC{mc}" for mc in DYN_DISPLAY_MCS])
        ax.set_ylim(0, plot_data.n_clones.max() * 1.23)
        ax.set_ylabel("Unique LARRY clones")
        ax.set_title("Day-2 HSPC clones linked to later observed cells", loc="left", pad=29)
        ax.text(0, 1.015, "Fixed Day-2 clone cohort · exact LARRY identifiers · absence of recovery is not biological extinction",
                transform=ax.transAxes, fontsize=9, color="#68717D")
        ax.axvline(3.8, color="#D8DDE3", linewidth=1)
        ax.text(np.mean(plot_x[:4]), -0.14, "Monocyte-output metaclones", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.text(np.mean(plot_x[4:]), -0.14, "Multi-lineage metaclones", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#E6E9EC", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.075, right=0.99, top=0.83, bottom=0.20)
        fig.savefig(FIGDIR / f"{plot_stem}.svg", bbox_inches="tight")
        fig.savefig(FIGDIR / f"{plot_stem}.png", dpi=180, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        print(FIGDIR / f"{plot_stem}.svg")
        """),
        md(r"""
        ### Audit what “similar monocyte output” means

        The existing lineage-family label uses a fixed fine-state-to-output
        mapping pooled across sampling days and excludes the mapped `undiff`
        compartment from that denominator. It does **not** establish identical Day-6 endpoints.
        Below, the original pooled definition is reconstructed alongside full
        Day-6 author-state compositions, with progenitors retained, and
        complete fine-state compositions. In the primary composition estimate
        every observed clone receives equal weight. Pooled-cell fractions are
        saved separately for comparison; zero fractions are inserted only for
        missing states within an observed clone-day, never for an absent clone.
        """),
        code(r"""
        dyn_output_rows = []
        dyn_clone_output_rows = []
        for cohort_name, cohort in dyn_cohorts.items():
            for annotation in ["cell_type_weinreb", "cell_type"]:
                states = sorted(dyn_cells[annotation].unique())
                for (mc, clone, day), frame in cohort.groupby(["metaclone", "LARRY_ID", "day"], observed=True):
                    counts = frame[annotation].value_counts().reindex(states, fill_value=0)
                    for state, count in counts.items():
                        dyn_clone_output_rows.append({"cohort": cohort_name, "annotation": annotation,
                            "metaclone": int(mc), "LARRY_ID": str(clone), "day": int(day), "state": state,
                            "n_state_cells": int(count), "n_clone_day_cells": len(frame), "fraction": count / len(frame)})
        dyn_clone_output = pd.DataFrame(dyn_clone_output_rows)
        dyn_clone_output.to_csv(TABLEDIR / "ancestry_clone_day_full_state_composition.csv.gz", index=False, compression="gzip")
        dyn_output_summary = dyn_clone_output.groupby(["cohort", "annotation", "metaclone", "day", "state"], observed=True).agg(
            fraction=("fraction", "mean"), n_clones=("LARRY_ID", "nunique"), n_state_cells=("n_state_cells", "sum"),
            n_total_cells=("n_clone_day_cells", "sum"),
        ).reset_index()
        dyn_output_summary["pooled_cell_fraction"] = dyn_output_summary.n_state_cells / dyn_output_summary.n_total_cells
        dyn_output_summary.to_csv(TABLEDIR / "ancestry_full_state_composition_by_cohort_metaclone_day.csv", index=False)
        dyn_differentiated_rows = []
        for cohort_name, cohort in dyn_cohorts.items():
            for time_scope, days in [("all_sampling_days", [2, 4, 6]), ("later_days_only", [4, 6]), ("day6_only", [6])]:
                for mc, frame in cohort[cohort.day.isin(days)].groupby("metaclone", observed=True):
                    mature = frame[frame.differentiated_output_class.ne("undiff")]
                    counts = mature.differentiated_output_class.value_counts()
                    for state in ["Neu", "Mono", "Ery/Mk", "Baso", "Mast", "Other"]:
                        count = int(counts.get(state, 0))
                        dyn_differentiated_rows.append({"cohort": cohort_name, "metaclone": int(mc), "time_scope": time_scope,
                            "state": state, "n_state_cells": count, "n_differentiated_cells": len(mature),
                            "n_all_cells": len(frame), "n_observed_clones": frame.LARRY_ID.nunique(),
                            "n_differentiated_informative_clones": mature.LARRY_ID.nunique(),
                            "pooled_differentiated_fraction": count / len(mature) if len(mature) else np.nan,
                            "differentiated_fraction_of_all_cells": len(mature) / len(frame),
                            "denominator": "pooled mapped differentiated cells; fine-state undiff compartment excluded"})
        dyn_differentiated_output = pd.DataFrame(dyn_differentiated_rows)
        dyn_differentiated_output.to_csv(TABLEDIR / "ancestry_differentiated_output_definition_audit.csv", index=False)
        dyn_source_lineage = pd.read_csv(
            "/path/to/ChronoClone/reference_workspace/Figure/fig2/data/fig2h_lineage_bias_map.csv"
        )
        dyn_legacy_audit = dyn_differentiated_output[
            dyn_differentiated_output.cohort.eq("all_assigned") & dyn_differentiated_output.time_scope.eq("all_sampling_days")
        ].merge(dyn_source_lineage[["metaclone", "top_output", "top_frac"]], on="metaclone", validate="many_to_one")
        dyn_legacy_audit = dyn_legacy_audit[dyn_legacy_audit.state.eq(dyn_legacy_audit.top_output)].copy()
        dyn_legacy_audit["matches_original_rounded_fraction"] = (
            dyn_legacy_audit.pooled_differentiated_fraction.round(3) == dyn_legacy_audit.top_frac
        )
        dyn_legacy_audit.to_csv(TABLEDIR / "ancestry_original_output_fraction_reconstruction.csv", index=False)
        dyn_checks["original_pooled_differentiated_fractions_reproduced"] = dyn_legacy_audit.matches_original_rounded_fraction.all()
        dyn_output_sums = dyn_output_summary.groupby(["cohort", "annotation", "metaclone", "day"]).fraction.sum()
        dyn_checks["full_state_compositions_sum_to_one"] = np.allclose(dyn_output_sums, 1.0)
        dyn_checks["clone_full_state_compositions_sum_to_one"] = np.allclose(
            dyn_clone_output.groupby(["cohort", "annotation", "metaclone", "LARRY_ID", "day"]).fraction.sum(), 1.0)
        assert all(dyn_checks.values()), dyn_checks
        display(dyn_differentiated_output[
            dyn_differentiated_output.cohort.isin(["all_assigned", "day2_hspc_linked"])
            & dyn_differentiated_output.time_scope.eq("all_sampling_days")
            & dyn_differentiated_output.metaclone.isin(DYN_MONO_MCS) & dyn_differentiated_output.state.eq("Mono")
        ])
        display(dyn_output_summary[
            dyn_output_summary.cohort.eq("day2_hspc_linked") & dyn_output_summary.annotation.eq("cell_type_weinreb")
            & dyn_output_summary.day.eq(6) & dyn_output_summary.metaclone.isin(DYN_MONO_MCS)
        ].pivot(index="metaclone", columns="state", values="fraction").reindex(DYN_MONO_MCS))
        """),
        md(r"""
        ### Monocyte branch positions in the same ancestry cohort

        Each dot below is one clone's median pseudotime among cells passing
        the inherited monocyte branch mask at that snapshot. Black horizontal
        marks show the median of those clone medians; `n` counts branch-informative
        clones. A clone with no cells on this branch contributes to its
        occupancy denominator but has no defined branch-position value.
        This is a grouped distribution display at three observed snapshots,
        not an interpolated time curve or a direct measurement of velocity.
        The navy/orange grouping follows the existing MC6/MC8 versus MC7/MC12
        comparison; individual metaclones remain separate.
        """),
        code(r"""
        plot_stem = "ancestry_02_monocyte_branch_position"
        plot_data = dyn_clone_branch[
            dyn_clone_branch.cohort.eq("day2_hspc_linked") & dyn_clone_branch.metaclone.isin(DYN_MONO_MCS)
            & dyn_clone_branch.branch.eq("Mono")
        ].copy()
        plot_data["display_order"] = plot_data.metaclone.map({mc: i for i, mc in enumerate(DYN_MONO_MCS)})
        plot_data["position_defined"] = plot_data.branch_pseudotime_median.notna()
        plot_data["x_position"] = plot_data.day.map({2: 0, 4: 5, 6: 10}) + plot_data.display_order
        plot_data = plot_data.sort_values(["day", "display_order", "LARRY_ID"])
        plot_data.to_csv(TABLEDIR / f"{plot_stem}.csv", index=False)
        plot_summary = plot_data.groupby(["metaclone", "day", "x_position"], observed=True).agg(
            n_observed_clones=("LARRY_ID", "size"), n_branch_informative_clones=("branch_pseudotime_median", "count"),
            median_of_clone_medians=("branch_pseudotime_median", "median"),
            lower_quartile=("branch_pseudotime_median", lambda x: x.quantile(0.25)),
            upper_quartile=("branch_pseudotime_median", lambda x: x.quantile(0.75)),
        ).reset_index().sort_values("x_position")
        plot_summary.to_csv(TABLEDIR / f"{plot_stem}_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(12.5, 5.4))
        jitter_rng = np.random.default_rng(20260910)
        for (mc, day), group in plot_data.groupby(["metaclone", "day"], observed=True):
            finite = group[group.position_defined]
            x = float(group.x_position.iloc[0])
            color = "#17324D" if mc in [6, 8] else "#E0913C"
            jitter = jitter_rng.uniform(-0.24, 0.24, len(finite))
            ax.scatter(x + jitter, finite.branch_pseudotime_median, s=17, color=color,
                       alpha=0.65, edgecolors="none", zorder=2)
        for row in plot_summary.itertuples(index=False):
            if np.isfinite(row.median_of_clone_medians):
                ax.vlines(row.x_position, row.lower_quartile, row.upper_quartile, color="#111111", linewidth=2.0, zorder=3)
                ax.hlines(row.median_of_clone_medians, row.x_position - 0.29, row.x_position + 0.29,
                          color="#111111", linewidth=2.0, zorder=4)
        tick_labels = [f"MC{row.metaclone}\nn={row.n_branch_informative_clones}" for row in plot_summary.itertuples(index=False)]
        ax.set_xticks(plot_summary.x_position, tick_labels, fontsize=8)
        for xpos, label in [(1.5, "Day 2"), (6.5, "Day 4"), (11.5, "Day 6")]:
            ax.text(xpos, -0.17, label, transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
        ax.axvline(4, color="#D8DDE3", linewidth=1)
        ax.axvline(9, color="#D8DDE3", linewidth=1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.7, 13.7)
        ax.set_ylabel("Monocyte-mask pseudotime\n(one median per observed clone)")
        ax.set_title("Monocyte-branch positions of Day-2 HSPC-linked clones", loc="left", pad=29)
        ax.text(0, 1.015, "Each clone has equal weight · black marks: median and interquartile range of clone medians",
                transform=ax.transAxes, color="#68717D", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#E6E9EC", linewidth=0.6)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.83, bottom=0.24)
        fig.savefig(FIGDIR / f"{plot_stem}.svg", bbox_inches="tight")
        fig.savefig(FIGDIR / f"{plot_stem}.png", dpi=180, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        print(FIGDIR / f"{plot_stem}.svg")
        """),
        md(r"""
        ### Multilineage branch occupancy changes

        The heatmap compares Day 6 with Day 4 for all seven current multilineage
        metaclones in the Day-2 HSPC-linked cohort. Each value is a difference
        in mean within-clone occupancy, in percentage points. These are
        overlapping branch-compatibility masks, so a cell can contribute to
        several columns and columns do not sum to one. Differences compare
        destructive snapshots. The full-cohort and same-clones-at-both-days
        sensitivities and the pre-existing MC5–MC3 / MC3–MC18 distance
        comparisons are saved alongside the primary values. This plot uses
        the requested positive red / zero white / negative blue palette;
        percentage-point units retain the biological denominator.
        """),
        code(r"""
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        plot_stem = "ancestry_03_multilineage_branch_occupancy_change"
        plot_long = dyn_branch_summary[
            dyn_branch_summary.metaclone.isin(DYN_MULTI_MCS) & dyn_branch_summary.day.isin([4, 6])
        ].copy()
        plot_wide = plot_long.pivot(index=["cohort", "metaclone", "branch"], columns="day",
                                   values=["clone_balanced_occupancy", "n_observed_clones", "n_branch_cells", "n_observed_cells"])
        plot_rows = []
        for (cohort, mc, branch), row in plot_wide.iterrows():
            plot_rows.append({"cohort": cohort, "metaclone": int(mc), "branch": branch,
                "day4_fraction": row[("clone_balanced_occupancy", 4)], "day6_fraction": row[("clone_balanced_occupancy", 6)],
                "day6_minus_day4_percentage_points": 100 * (row[("clone_balanced_occupancy", 6)] - row[("clone_balanced_occupancy", 4)]),
                "n_day4_observed_clones": int(row[("n_observed_clones", 4)]), "n_day6_observed_clones": int(row[("n_observed_clones", 6)]),
                "n_day4_branch_cells": int(row[("n_branch_cells", 4)]), "n_day6_branch_cells": int(row[("n_branch_cells", 6)]),
                "n_day4_total_cells": int(row[("n_observed_cells", 4)]), "n_day6_total_cells": int(row[("n_observed_cells", 6)]),
                "metaclone_display_order": DYN_MULTI_MCS.index(mc), "branch_display_order": DYN_BRANCHES.index(branch)})
        plot_data_all_cohorts = pd.DataFrame(plot_rows)
        plot_data_all_cohorts.to_csv(TABLEDIR / f"{plot_stem}_cohort_sensitivities.csv", index=False)
        plot_data = plot_data_all_cohorts[plot_data_all_cohorts.cohort.eq("day2_hspc_linked")].copy()
        plot_data.to_csv(TABLEDIR / f"{plot_stem}.csv", index=False)
        plot_matrix = plot_data.pivot(index="metaclone", columns="branch", values="day6_minus_day4_percentage_points").reindex(index=DYN_MULTI_MCS, columns=DYN_BRANCHES)
        plot_limit = max(10.0, float(np.ceil(np.nanmax(np.abs(plot_matrix.to_numpy())) / 10) * 10))
        plot_cmap = LinearSegmentedColormap.from_list("ancestry_occupancy_blue_white_red", ["#2166AC", "#FFFFFF", "#B2182B"], N=257)
        fig, ax = plt.subplots(figsize=(7.6, 6.7))
        im = ax.pcolormesh(np.arange(4) - 0.5, np.arange(len(DYN_MULTI_MCS) + 1) - 0.5,
                           plot_matrix.to_numpy(float), cmap=plot_cmap,
                           norm=TwoSlopeNorm(vmin=-plot_limit, vcenter=0, vmax=plot_limit),
                           shading="flat", rasterized=False)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(len(DYN_MULTI_MCS) - 0.5, -0.5)
        for row_index, mc in enumerate(DYN_MULTI_MCS):
            for col_index, branch in enumerate(DYN_BRANCHES):
                value = float(plot_matrix.loc[mc, branch])
                ax.text(col_index, row_index, f"{value:+.1f}", ha="center", va="center", fontsize=11,
                        color="white" if abs(value) > 0.65 * plot_limit else "#17324D")
        labels = []
        for mc in DYN_MULTI_MCS:
            support = plot_data[plot_data.metaclone.eq(mc)].iloc[0]
            labels.append(f"MC{mc}   (n={int(support.n_day4_observed_clones)}→{int(support.n_day6_observed_clones)})")
        ax.set_yticks(range(len(DYN_MULTI_MCS)), labels)
        ax.set_xticks(range(3), ["Basophil", "Neutrophil", "Monocyte"])
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DYN_MULTI_MCS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("Multilineage branch occupancy: Day 6 minus Day 4", loc="left", pad=32, fontsize=12)
        ax.text(0, 1.025, "Equal observed-clone weight · exact Day-2 ancestry cohort\nOverlapping masks; n denotes observed clones at each snapshot",
                transform=ax.transAxes, color="#68717D", fontsize=8.5)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Change in branch occupancy (percentage points)")
        if colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(left=0.28, right=0.89, top=0.83, bottom=0.10)
        fig.savefig(FIGDIR / f"{plot_stem}.svg", bbox_inches="tight")
        fig.savefig(FIGDIR / f"{plot_stem}.png", dpi=180, bbox_inches="tight")
        display(fig)
        plt.close(fig)
        print(FIGDIR / f"{plot_stem}.svg")
        """),
        code(r"""
        # Recompute the two existing temporal examples under every cohort definition.
        dyn_distance_rows = []
        for cohort in dyn_branch_summary.cohort.unique():
            for mc_a, mc_b, example in [(5, 3, "MC5–MC3 previously described divergence"),
                                       (3, 18, "MC3–MC18 previously described convergence")]:
                for estimator in ["clone_balanced_occupancy", "pooled_cell_occupancy"]:
                    distances = {}
                    for day in [4, 6]:
                        frame = dyn_branch_summary[dyn_branch_summary.cohort.eq(cohort) & dyn_branch_summary.day.eq(day)]
                        vectors = frame.pivot(index="metaclone", columns="branch", values=estimator).reindex(columns=DYN_BRANCHES)
                        distances[day] = float(np.linalg.norm(vectors.loc[mc_a].to_numpy() - vectors.loc[mc_b].to_numpy()))
                    dyn_distance_rows.append({"cohort": cohort, "example": example, "metaclone_a": mc_a, "metaclone_b": mc_b,
                        "estimator": estimator, "day4_distance": distances[4], "day6_distance": distances[6],
                        "distance_change": distances[6] - distances[4],
                        "observed_direction": "divergence" if distances[6] > distances[4] else "convergence",
                        "coordinate_system": "Euclidean distance in three overlapping Baso/Neu/Mono branch occupancies"})
        dyn_pair_distances = pd.DataFrame(dyn_distance_rows)
        dyn_pair_distances.to_csv(TABLEDIR / "ancestry_multilineage_branch_pair_distances.csv", index=False)
        # Full state-composition distances answer a different question and are retained explicitly.
        dyn_state_distance_rows = []
        for (cohort, annotation), frame in dyn_output_summary.groupby(["cohort", "annotation"], observed=True):
            for mc_a, mc_b in [(5, 3), (3, 18)]:
                distances = {}
                for day in [4, 6]:
                    vectors = frame[frame.day.eq(day)].pivot(index="metaclone", columns="state", values="fraction")
                    distances[day] = float(np.linalg.norm(vectors.loc[mc_a].to_numpy() - vectors.loc[mc_b].to_numpy()))
                dyn_state_distance_rows.append({"cohort": cohort, "annotation": annotation, "metaclone_a": mc_a,
                    "metaclone_b": mc_b, "day4_distance": distances[4], "day6_distance": distances[6],
                    "distance_change": distances[6] - distances[4], "coordinate_system": "Euclidean distance in complete clone-balanced state composition"})
        dyn_state_pair_distances = pd.DataFrame(dyn_state_distance_rows)
        dyn_state_pair_distances.to_csv(TABLEDIR / "ancestry_multilineage_full_composition_pair_distances.csv", index=False)
        dynamics_summary = dyn_branch_summary
        linkage_summary = dyn_linkage_support
        pair_dynamics_summary = dyn_pair_distances
        pd.DataFrame([{"check": key, "passed": bool(value)} for key, value in dyn_checks.items()]).to_csv(
            TABLEDIR / "ancestry_dynamics_validation.csv", index=False)
        assert all(dyn_checks.values())
        display(dyn_pair_distances[dyn_pair_distances.estimator.eq("clone_balanced_occupancy")])
        display(dyn_state_pair_distances[dyn_state_pair_distances.cohort.eq("day2_hspc_linked")])
        print("Observed ancestry linkage:", len(dyn_day2_clones), "Day-2 HSPC clones;",
              len(dyn_paired_clones), "also recovered at both later snapshots.")
        print("All dynamics above are descriptive associations in an existing temporal partition.",
              "No velocity, survival, prospective fate, or causal effect is estimated.")
        """),
    ]
