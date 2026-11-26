"""Cell-type vocabulary and lineage-identifier handling for the corrected Singh checkpoint.

Item A1 of reference_HANDOFF.md. Three defects in the previous
`divergence_timing_20260910/scripts/dt_core.py` are repaired here, and the repair is
enforced by assertion rather than by convention:

DEFECT 1 -- silent category loss.
    `dt_core.CELLTYPES` contained "Baso". The annotation in
    adata_valid_palantir_20260626.h5ad calls that category "Basophil". Composition
    vectors were built with
        .pivot_table(...).reindex(columns=CELLTYPES, fill_value=0)
    so the 388 Basophil cells were silently replaced by an all-zero "Baso" column and
    then divided away by the row-sum renormalisation. Every composition vector and every
    TVD statistic in the previous run was therefore computed on 7 of 8 categories with
    the basophil mass deleted rather than assigned. `reindex` cannot warn about this: a
    name that does not exist and a name with no cells are indistinguishable to it. The
    repair is an explicit source -> analysis mapping whose domain is asserted equal to
    the set of categories actually present, so a renamed or newly added category raises
    instead of vanishing.

DEFECT 2 -- an annotation the data owner distrusts, used as an analysis axis.
    HSC and MPP were carried as two separate composition columns. The standing
    instruction is to work at the HSPC level because the HSC/MPP split in this
    annotation is not considered reliable. They are pooled to HSPC here. The verbatim
    source label is retained as `ct_source` for provenance only; nothing downstream may
    use it as a feature, ordering axis or stratifier.

DEFECT 3 -- a stringified missing value becoming a clone.
    `clone_id_rebuild` has 25 true-NaN cells. `.astype(str)` mapped them to the string
    "nan", which then behaved as a 1,517th lineage family in every groupby, clone count
    and family-paired join. The registered WT reference is 1,516 clones; the extra
    family was this artefact. Unresolved identifiers are kept as NaN, excluded from all
    clone-level and family-level inference, and counted in `unresolved_lineage_ids.csv`
    so their effect on any cell-level descriptive denominator is auditable.

Nothing else about the previous design is changed. Stratification by
`sample x replicate` is preserved, as is the reason for it: the D27 LK and LSK sort
gates differ in both pseudotime and arm composition, so gate-pooled D27 comparisons are
confounded.
"""
import json
import numpy as np
import pandas as pd
import anndata as ad

PALANTIR = ("/path/to/ChronoClone/reference_workspace/data/"
            "adata_valid_palantir_20260626.h5ad")

# Source annotation label -> analysis category. Domain asserted against the object.
CT_MAP = {
    "HSC":        "HSPC",      # pooled: HSC/MPP split not reliable (standing instruction)
    "MPP":        "HSPC",
    "GMP":        "GMP",
    "Mk":         "Mk",
    "Ery":        "Ery",
    "Basophil":   "Basophil",  # was silently dropped as "Baso"
    "Monocyte":   "Monocyte",
    "Neutrophil": "Neutrophil",
}
# Fixed column order for every composition vector in this run.
CELLTYPES = ["HSPC", "GMP", "Mk", "Ery", "Basophil", "Monocyte", "Neutrophil"]

BRANCHES = ["Mk", "Neu", "Mono", "Ery", "Baso"]   # obsm branch-mask names, a separate namespace
MCS = ["0", "1", "2", "3", "4", "5", "6", "7", "9", "10",
       "11", "12", "13", "15", "16", "17", "19"]
DAYS = ["D7", "D14", "D27"]
MUTANTS = ["Dnmt3a", "Npm1c"]
NA_SENTINELS = ["nan", "NA", "None", "", "-1", "NaN", "na"]


def load_cells(report=None):
    """Cell table with the corrected vocabulary. Raises if any assertion fails.

    `report` : optional path; writes the assertion record as JSON.
    """
    A = ad.read_h5ad(PALANTIR, backed="r")
    o = A.obs
    n_in = A.n_obs
    chk = {"source_object": PALANTIR, "cells_in_object": int(n_in)}

    # --- assertion 1: mapping domain == categories actually present -----------
    present = set(pd.Series(o.cell_type).astype(str).unique())
    unmapped = sorted(present - set(CT_MAP))
    unused = sorted(set(CT_MAP) - present)
    assert not unmapped, (
        f"cell_type categories present in the object but absent from CT_MAP: {unmapped}. "
        "This is the defect that deleted the basophils; refusing to run.")
    chk["source_categories"] = sorted(present)
    chk["mapped_to"] = CELLTYPES
    chk["mapping_keys_with_no_cells"] = unused   # empty is not an error, only reported

    D = pd.DataFrame(index=o.index)
    D["mc"] = o.metaclone.astype(str)
    D["day"] = o.time_point.astype(str)
    D["arm"] = o.condition.astype(str)
    D["rep"] = o.replicate.astype(str)          # biological replicate (mouse)
    D["sample"] = o["sample"].astype(str)       # 10x well x sort gate
    D["ct_source"] = pd.Series(o.cell_type).astype(str).values   # provenance only
    D["ct"] = D["ct_source"].map(CT_MAP)
    D["psi"] = o.palantir_pseudotime.astype(float).values
    D["entropy"] = o.palantir_entropy.astype(float).values

    # --- assertion 2: no cell lost or invented by the mapping -----------------
    assert D["ct"].notna().all(), "unmapped cell_type survived the domain assertion"
    assert len(D) == n_in, f"row count changed: {len(D)} != {n_in}"
    src_n = D.ct_source.value_counts()
    got_n = D.ct.value_counts()
    assert got_n.get("HSPC", 0) == src_n.get("HSC", 0) + src_n.get("MPP", 0), \
        "HSPC pooling did not conserve HSC + MPP"
    assert got_n.get("Basophil", 0) == src_n.get("Basophil", 0) == 388, \
        "basophil count is not the expected 388"
    assert int(got_n.sum()) == n_in, "analysis categories do not sum to the object"
    chk["counts_by_analysis_category"] = {k: int(v) for k, v in got_n.items()}
    chk["counts_by_source_category"] = {k: int(v) for k, v in src_n.items()}
    chk["basophil_cells_recovered"] = int(src_n.get("Basophil", 0))
    chk["hspc_cells_pooled"] = int(got_n.get("HSPC", 0))

    # --- lineage identifiers: keep NaN as NaN --------------------------------
    raw = pd.Series(o.clone_id_rebuild).astype(str).values
    D["clone"] = pd.Series(raw, index=D.index).where(
        ~pd.Series(raw, index=D.index).isin(NA_SENTINELS), np.nan)
    D["clone_resolved"] = D["clone"].notna()
    n_unres = int((~D.clone_resolved).sum())
    chk["cells_with_unresolved_lineage_id"] = n_unres
    chk["distinct_resolved_lineage_families"] = int(D.loc[D.clone_resolved, "clone"].nunique())
    chk["distinct_if_nan_had_been_stringified"] = int(pd.Series(raw).nunique())

    # --- gate / stratum (unchanged from the previous run) --------------------
    D["gate"] = np.where(D["sample"].str.contains("LSK"), "LSK",
                np.where(D["sample"].str.contains("LK"), "LK",
                np.where(D.day == "D7", "HSPC_gate", "pool")))
    # Replicate is excluded from the design by data-owner instruction (2026-09-10):
    # "completely ignore rep". `rep` is retained as an immutable provenance column but
    # is not a stratifier, covariate or reporting axis. `sample` (the 10x well) and
    # `rep` (the mouse) are crossed, not nested, because conditions were multiplexed
    # for sequencing; both arms are present in both replicates at every sampled day
    # (see cohort_units.csv), so pooling replicates within a well does not confound
    # arm with mouse.
    D["stratum"] = D["sample"].astype(str)

    bm = pd.DataFrame(A.obsm["branch_masks"], index=o.index)[BRANCHES].astype(bool)
    for b in BRANCHES:
        D["mask_" + b] = bm[b].values
        D["p_" + b] = o["prob_" + b].astype(float).values
    for c in ["larry_gfp", "larry_scarlet", "larry_Flp", "larry_Cre"]:
        s = pd.Series(o[c]).astype(str)
        D[c] = s.where(~s.isin(NA_SENTINELS), np.nan).values

    if report:
        json.dump(chk, open(report, "w"), indent=2)
    return D, chk


def composition(D, key, require_resolved_clone=True):
    """Normalised analysis-category shares over `key`, with mass conservation asserted.

    Returns rows summing to 1 over CELLTYPES. Groups with no cells are absent, never
    all-zero: an undefined composition must not be representable as a valid vector.
    """
    sub = D[D.clone_resolved] if require_resolved_clone else D
    cnt = (sub.assign(one=1)
              .pivot_table(index=key, columns="ct", values="one",
                           aggfunc="sum", fill_value=0, observed=True)
              .reindex(columns=CELLTYPES, fill_value=0))
    # every cell in `sub` must appear in exactly one cell of the count table
    assert int(cnt.values.sum()) == len(sub), (
        f"composition lost cells: table sums to {int(cnt.values.sum())} of {len(sub)}")
    tot = cnt.sum(1)
    assert (tot > 0).all(), "a composition group has zero cells; it must be absent instead"
    out = cnt.div(tot, axis=0)
    assert np.allclose(out.sum(1), 1), "composition rows do not sum to 1"
    assert np.isfinite(out.values).all(), "non-finite composition entry"
    return out, tot.rename("n_cells")
