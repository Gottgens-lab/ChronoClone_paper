"""Day-7 gene-program analysis for the ChronoClone metaclone response classes.

Unit of observation: the single WT day-7 metaclone-assigned cell (one pooled
HSPC compartment, as specified).

Group comparisons are **cell-level** per the authors' instruction of 2026-09-02: genes
and program scores are tested across cells with `scipy`/`scanpy` rank-sum tests,
and the metaclone-mean permutation contrasts the earlier version of this run used
have been removed (their outputs are archived under
`superseded_metaclone_unit_contrasts/`). Cells of one metaclone are not
independent draws, so these p-values describe the cell populations of the two
groups and are anticonservative as statements about metaclones; group sizes in
metaclones are reported next to every test so that is visible.

The continuous outcome correlations remain metaclone-level, because `fit_*` and
`resp_*` are one number per metaclone (k = 17) and have no cell-level form.

Metaclone response classes are read from the two mutant-arm analyses
(`singh_metaclone_paired_dynamics_20260901`), never retyped here.

Env: micromamba `chronoclone`.
"""
import os

# site-packages is mounted read-only here, so numba cannot write its on-disk
# cache next to scanpy's sources; point it at the run directory before scanpy
# is imported anywhere.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "objects", "numba_cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import rankdata, norm, hypergeom, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- paths
CC = "/path/to/ChronoClone"
RUN = f"{CC}/reference_workspace/Figure/singh_d7_gene_programs_20260902"
H5 = f"{CC}/reference_workspace/data/adata_valid_palantir_20260626.h5ad"
PAIRED = f"{CC}/reference_workspace/Figure/singh_metaclone_paired_dynamics_20260901/tables"
GMT = f"{RUN}/objects/gmt"

N_GENES = 20994          # var indices >= this are Sapphire lineage-barcode features
SEED = 0


# ---------------------------------------------------------------- step 1: classes
def read_group_definitions():
    """The two mutant-arm groupings, as published by the paired-dynamics run."""
    gd = pd.read_csv(f"{PAIRED}/group_definitions.csv")
    out = {}
    for mut, sub in gd.groupby("mutation"):
        out[mut] = {r.group: [int(x[2:]) for x in r.metaclones.split()]
                    for r in sub.itertuples()}
    return out


# Joint response class = the cross of the two independent groupings. Names describe
# which mutation each metaclone responds to, which is the axis the authors asked for.
JOINT_RULES = {
    ("Npm1c-specific", "G1_early_diff"):         "BOTH",
    ("Npm1c-specific", "G3_late_divergent"):     "BOTH",
    ("Npm1c-specific", "G4_primitive_retained"): "BOTH",
    ("Npm1c-specific", "G2_WT-like"):            "NPM1C_ONLY",
    ("other",          "G1_early_diff"):         "DNMT3A_ONLY",
    ("other",          "G3_late_divergent"):     "DNMT3A_ONLY",
    ("other",          "G4_primitive_retained"): "DNMT3A_ONLY",
    ("other",          "G2_WT-like"):            "NEITHER",
}


def joint_classes():
    """One row per metaclone: its Npm1c class, Dnmt3a class and joint class."""
    G = read_group_definitions()
    rows = []
    mcs = sorted({m for d in G.values() for v in d.values() for m in v})
    for mc in mcs:
        npm = next(g for g, v in G["Npm1c"].items() if mc in v)
        dnm = next(g for g, v in G["Dnmt3a"].items() if mc in v)
        rows.append(dict(metaclone=mc, npm1c_group=npm, dnmt3a_group=dnm,
                         joint_class=JOINT_RULES.get((npm, dnm), "UNASSIGNED")))
    return pd.DataFrame(rows)


#: The contrasts actually tested, in priority order. Each entry is
#: (axis column, group A, group B, plain-language description).
#:
#: Per the authors' instruction of 2026-09-02, the only admissible metaclone
#: comparisons are (i) WT vs a given mutant and (ii) metaclone groups *within the
#: same mutant*. Both axes below are of the second kind - each is a grouping
#: defined inside one mutant arm by the paired-dynamics run. Axes that set one
#: mutant against the other (`mutation_specificity`, i.e. Npm1c-responders vs
#: Dnmt3a-responders) and axes that pool the two arms' definitions into one
#: grouping (`joint_class` BOTH vs NEITHER) were removed. `joint_class` itself is
#: retained by `joint_classes()` for metaclone selection and figure annotation
#: only - it is never a tested axis.
CONTRASTS = [
    ("dnmt3a_group", "G4_primitive_retained", "G2_WT-like",
     "primitive-retained vs WT-like metaclone groups within the Dnmt3a arm"),
    ("npm1c_group", "Npm1c-specific", "other",
     "Npm1c-specific vs remaining metaclone groups within the Npm1c arm"),
]


def response_scores():
    """Continuous per-metaclone response magnitude for each mutation.

    Both columns come from the identical code path in the paired-dynamics run, so
    they are comparable between mutations:
      resp_<mut>   = dist_arm_d14, composition distance between the WT and mutant
                     arm of that metaclone at day 14 (response magnitude);
      fit_<mut>    = log2_share_change, day-14 -> day-27 change in the metaclone's
                     share of the mutant arm (expansion outcome), a within-arm
                     comparison of metaclones to each other.

    `resp_` is the WT-vs-that-mutant comparison and `fit_` is a within-that-mutant
    comparison, so both are admissible under the authors' 2026-09-02 instruction. The
    Npm1c-minus-Dnmt3a difference scores that used to be built here
    (`specificity_resp`, `specificity_fit`) set one mutant against the other and
    were removed on that instruction.
    """
    keep = ["metaclone", "dist_arm_d14", "log2_share_change", "n_wt14", "n_mut14"]
    D = pd.read_csv(f"{PAIRED}/metaclone_groups_Dnmt3a.csv")[keep]
    N = pd.read_csv(f"{PAIRED}/metaclone_groups_Npm1c.csv")[keep]
    D.columns = ["metaclone", "resp_Dnmt3a", "fit_Dnmt3a", "n_wt14_Dnmt3a", "n_mut14_Dnmt3a"]
    N.columns = ["metaclone", "resp_Npm1c", "fit_Npm1c", "n_wt14_Npm1c", "n_mut14_Npm1c"]
    R = D.merge(N, on="metaclone", how="outer")
    return R.sort_values("metaclone").reset_index(drop=True)


# ---------------------------------------------------------------- step 2: data
def load_day7():
    """All WT day-7 metaclone-assigned cells as one HSPC compartment.

    Per the authors' 2026-09-02 instruction the HSC/MPP annotation is collapsed: the
    two author labels are re-annotated to a single `HSPC` value in
    `obs.cell_type` and the separation is not used anywhere downstream. The
    original author labels are preserved in `obs.cell_type_author` for
    provenance only and no analysis reads them.

    Expression is the object's own `.X`, which is the matrix every earlier
    ChronoClone day-7 analysis used. Note on the object's transformation state:
    `layers['counts']` in this h5ad does NOT hold raw integer counts (0 % of its
    nonzero values are integers, per-cell sums ~3.7e3 against nCount_RNA ~9e3),
    and `.X` equals log1p(counts_layer / rowsum * 1e4) to r = 0.99999997, i.e.
    `.X` is a second normalisation applied on top of an already-normalised
    matrix. No raw counts exist in this file, so count-based HVG selection and
    count-based models are not applicable here; `.X` is used as the
    log-scale expression matrix and `layers['counts']` is carried through
    unaltered for provenance only.
    """
    A = ad.read_h5ad(H5, backed="r")
    assert np.array(A.var_names)[N_GENES] == "Sapphire1-1"
    o = A.obs
    mc = pd.to_numeric(o.metaclone, errors="coerce")
    m = ((o.condition == "WT").values & mc.notna().values & (o.timepoint == "D7").values)
    sub = A[m, :N_GENES].to_memory()
    sub.obs["mc"] = mc[m].astype(int).values
    assert set(sub.obs.cell_type.unique()) <= {"HSC", "MPP"}, sub.obs.cell_type.unique()
    sub.obs["cell_type_author"] = sub.obs.cell_type.astype(str).values
    sub.obs["cell_type"] = "HSPC"
    X = sparse.csr_matrix(sub.X).astype(np.float32)
    assert X.min() >= 0, "NMF requires a non-negative matrix"
    out = ad.AnnData(X=X, obs=sub.obs.copy(), var=sub.var.copy())
    out.layers["counts_layer_asis"] = sparse.csr_matrix(sub.layers["counts"])
    out.obs["x_rowsum"] = np.asarray(X.sum(1)).ravel()
    return out


# ---------------------------------------------------------------- step 3: cNMF
def hvg_matrix(A, n_top=2000):
    """cNMF input matrix: dispersion-selected HVGs, variance-scaled without
    centering (scaling without centering keeps the matrix non-negative, which
    NMF requires).

    Selection uses scanpy's dispersion flavor on the log-scale `.X` rather than
    the count-based seurat_v3 flavor, because this object carries no raw counts.
    """
    import scanpy as sc
    tmp = ad.AnnData(X=A.X.copy(), obs=A.obs[[]].copy(), var=A.var[[]].copy())
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat")
    gi = np.where(tmp.var.highly_variable.values)[0]
    M = np.asarray(A.X[:, gi].todense(), dtype=np.float64)
    sd = M.std(0)
    sd[sd == 0] = 1.0
    return M / sd, gi


def _nmf_run(M, k, seed):
    from sklearn.decomposition import NMF
    mdl = NMF(n_components=k, init="random", solver="cd", beta_loss="frobenius",
              max_iter=2000, random_state=seed, tol=1e-4)
    W = mdl.fit_transform(M)
    H = mdl.components_
    nrm = np.linalg.norm(H, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return W * nrm.T, H / nrm, mdl.reconstruction_err_


def cnmf_replicates(M, k, n_rep=20, seed=SEED):
    """Pool L2-normalised gene spectra over n_rep random restarts."""
    S, errs = [], []
    for r in range(n_rep):
        _, H, e = _nmf_run(M, k, seed + 1000 * k + r)
        S.append(H)
        errs.append(e)
    return np.vstack(S), np.array(errs)


def consensus_stability(S, k, n_rep, density_frac=0.10):
    """cNMF stability: mean silhouette of the k-way clustering of pooled spectra,
    after dropping the density_frac of components most distant from their local
    neighbourhood (the cNMF outlier filter)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    Dm = cdist(S, S, metric="euclidean")
    nn = max(2, int(round(S.shape[0] / k * 0.30)))
    local = np.sort(Dm, axis=1)[:, 1:nn + 1].mean(1)
    keep = local <= np.quantile(local, 1 - density_frac)
    Sk = S[keep]
    lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Sk)
    return float(silhouette_score(Sk, lab, metric="euclidean")), keep, lab, Sk


def cnmf_consensus_spectra(M, k, n_rep=20, seed=SEED):
    """Consensus spectra for one K, then refit usage by non-negative least squares.

    Returns (usage n_cells x k, spectra k x n_hvg, stability, median_error).
    """
    from scipy.optimize import nnls
    S, errs = cnmf_replicates(M, k, n_rep=n_rep, seed=seed)
    stab, keep, lab, Sk = consensus_stability(S, k, n_rep)
    spec = np.vstack([Sk[lab == c].mean(0) for c in range(k)])
    nrm = np.linalg.norm(spec, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    spec = spec / nrm
    U = np.vstack([nnls(spec.T, M[i])[0] for i in range(M.shape[0])])
    return U, spec, stab, float(np.median(errs))


def usage_fractions(U):
    """Per-cell program usage as a fraction of that cell's total usage."""
    s = U.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return U / s


def program_gene_scores(A, spec, gi):
    """Consensus gene loadings as (loading, specificity) frames, programs x HVGs.

    `loading` is the consensus spectrum scaled so each program sums to 1, which
    makes programs comparable despite differing usage magnitudes. `specificity`
    is that loading z-scored across programs for each gene, i.e. how much more a
    gene is used by one program than by the others; it is the quantity used to
    pick marker genes, because a highly expressed housekeeping gene loads on
    every program and so has a high loading but a low specificity.
    """
    genes = np.asarray(A.var_names)[gi]
    L = spec / np.maximum(spec.sum(1, keepdims=True), 1e-12)
    idx = [f"P{i + 1}" for i in range(L.shape[0])]
    loading = pd.DataFrame(L, index=idx, columns=genes)
    z = (loading - loading.mean(0)) / loading.std(0).replace(0, np.nan)
    return loading, z.fillna(0.0)


def top_program_genes(loading, specificity, n_top=30, loading_pct=90):
    """Long table of marker genes per program.

    Genes are ranked by specificity, but only among genes the program actually
    uses: loading at or above `loading_pct` of that program's own loadings.
    Ranking on specificity alone promotes near-zero, sparsely detected genes,
    whose loadings are numerically tiny in every program and so give a large
    z-score off one slightly larger value.
    """
    rows = []
    for p in loading.index:
        lo = loading.loc[p]
        keep = lo >= np.percentile(lo, loading_pct)
        s = specificity.loc[p][keep].sort_values(ascending=False)
        for rank, (g, v) in enumerate(s.head(n_top).items(), start=1):
            rows.append(dict(program=p, rank=rank, gene=g, specificity=v,
                             loading=float(lo[g])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 4: reference
UCELL_COLS = [
    "dHSC_aHSC_single-cell_Cabezas-Wallscheid_et_al_UCell",
    "dHSC_vs_aHSC_Cabezas-Wallscheid_et_al_UCell",
    "High-output_UCell", "HSC_Pietras_et_al_UCell",
    "HSC_vs_MPP4_Cabezas-Wallscheid_et_al_UCell",
    "HSC1 cluster_UCell", "HSC2 cluster_UCell", "HSC3 cluster_UCell",
    "HSC4 cluster_UCell", "Low-output_UCell", "Mk-bias_UCell",
    "MolO_Wilson_et_al_UCell", "Multilineage_UCell",
    "RACFP_DIM_Lauridsen_et_al_UCell", "Serial-engraftment_UCell",
    "Stemscore_Giladi_et_al_UCell", "SuMO_Wilson_et_al_UCell",
    "aging_up_UCell", "aging_down_UCell",
]

GMT_REF_SETS = {
    "CellCycle_G2M": ("MSigDB_Hallmark_2020", "G2-M Checkpoint"),
    "CellCycle_E2F": ("MSigDB_Hallmark_2020", "E2F Targets"),
    "MYC_targets": ("MSigDB_Hallmark_2020", "Myc Targets V1"),
    "Interferon_alpha": ("MSigDB_Hallmark_2020", "Interferon Alpha Response"),
    "Interferon_gamma": ("MSigDB_Hallmark_2020", "Interferon Gamma Response"),
    "TNFa_NFkB": ("MSigDB_Hallmark_2020", "TNF-alpha Signaling via NF-kB"),
    "Inflammatory": ("MSigDB_Hallmark_2020", "Inflammatory Response"),
    "OxPhos": ("MSigDB_Hallmark_2020", "Oxidative Phosphorylation"),
    "Hypoxia": ("MSigDB_Hallmark_2020", "Hypoxia"),
    "Heme_metabolism": ("MSigDB_Hallmark_2020", "heme Metabolism"),
    "Ribosome": ("KEGG_2019_Mouse", "Ribosome"),
    "Myeloid_priming": ("KEGG_2019_Mouse", "Hematopoietic cell lineage"),
}


def read_gmt(name, genes=None, lo=5, hi=500):
    """One GMT as {term: [genes]}, restricted to a gene universe.

    These Enrichr libraries carry uppercase human-style symbols while the object
    carries mouse symbols, so membership is resolved case-insensitively and the
    object's own spelling is returned. This is symbol-level ortholog matching,
    identical to the August/September ChronoClone enrichment pipeline, and it
    silently drops genes whose mouse and human symbols differ.
    """
    back = {g.upper(): g for g in genes} if genes is not None else None
    out = {}
    with open(f"{GMT}/{name}.gmt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            g = {x.split(",")[0].strip().upper() for x in p[2:] if x.strip()}
            g = sorted({back[x] for x in g if x in back}) if back is not None else sorted(g)
            if lo <= len(g) <= hi:
                out[p[0]] = g
    return out


def reference_scores(A):
    """Reference program scores per cell: the stored UCell columns plus scanpy
    score_genes on curated Hallmark/Reactome sets."""
    import scanpy as sc
    out = A.obs[UCELL_COLS].copy()
    out.columns = [c.replace("_UCell", "").replace(" ", "_") for c in UCELL_COLS]
    genes = set(A.var_names)
    libs, tmp, used = {}, A.copy(), {}
    for label, (lib, term) in GMT_REF_SETS.items():
        if lib not in libs:
            libs[lib] = read_gmt(lib, genes, hi=2000)
        L = libs[lib]
        assert term in L, f"{term!r} not resolvable in {lib}"
        sc.tl.score_genes(tmp, L[term], score_name=label, random_state=SEED)
        out[label] = tmp.obs[label].values
        used[label] = L[term]
    return out, used


# ------------------------------------------------- step 5: metaclone aggregation
def aggregate(scores, obs):
    """Metaclone-level program means over all of the metaclone's HSPC cells.

    Cells are one compartment (see `load_day7`): the metaclone mean is the plain
    mean across its cells, so a metaclone's value is weighted by the cells it
    actually contributed and no cell-type stratification is applied.
    """
    S = pd.DataFrame(np.asarray(scores), index=obs.index,
                     columns=list(getattr(scores, "columns", range(np.shape(scores)[1]))))
    S["mc"] = obs.mc.values
    cols = [c for c in S.columns if c != "mc"]
    return S.groupby("mc")[cols].mean()


# ---------------------------------------------------------------- step 6: tests
def _perm_labels(g, nperm, rng, exhaustive_cap=20000):
    """All distinct label assignments if the group sizes allow it, else nperm draws."""
    from itertools import combinations
    from math import comb
    g = np.asarray(g)
    idx = np.arange(len(g))
    nA = int((g == 1).sum())
    total = comb(len(g), nA)
    if total <= exhaustive_cap:
        L = np.zeros((total, len(g)), dtype=np.int8)
        for i, c in enumerate(combinations(idx, nA)):
            L[i, list(c)] = 1
        return L, int(total), True
    L = np.zeros((nperm, len(g)), dtype=np.int8)
    for i in range(nperm):
        L[i, rng.choice(idx, nA, replace=False)] = 1
    return L, int(total), False


def group_contrast(usage_mc, classes, group_a, group_b, key="joint_class",
                   nperm=10000, seed=SEED):
    """Difference in metaclone means between two response classes, with an
    exact-where-possible metaclone-label permutation null."""
    rng = np.random.default_rng(seed)
    cls = classes.set_index("metaclone")[key]
    mcs = [m for m in usage_mc.index if cls.get(m) in (group_a, group_b)]
    Y = usage_mc.loc[mcs].values
    g = (cls.loc[mcs].values == group_a).astype(np.int8)
    L, total, exact = _perm_labels(g, nperm, rng)
    nA, nB = g.sum(), (1 - g).sum()
    obs = Y[g == 1].mean(0) - Y[g == 0].mean(0)
    tot = Y.sum(0)
    null = (L @ Y) / nA - (tot - L @ Y) / nB
    p = (np.abs(null) >= np.abs(obs)[None, :] - 1e-12).mean(0)
    p = np.maximum(p, 1.0 / len(L))
    sd = Y.std(0, ddof=1)
    return pd.DataFrame(dict(
        program=usage_mc.columns, mean_A=Y[g == 1].mean(0), mean_B=Y[g == 0].mean(0),
        diff=obs, cohens_d=np.where(sd > 0, obs / sd, np.nan),
        p_perm=p, q_perm=multipletests(p, method="fdr_bh")[1],
        null_sd=null.std(0), n_mc_A=int(nA), n_mc_B=int(nB),
        n_perm=len(L), n_distinct=total, exact=exact)), null


def clone_block_contrast(scores, obs, classes, group_a, group_b, key="joint_class",
                         nperm=1000, seed=SEED):
    """Same contrast with the label permuted over CLONES instead of metaclones.
    Reported only to expose the inflation the wrong unit produces."""
    rng = np.random.default_rng(seed)
    cls = classes.set_index("metaclone")[key]
    m = obs.mc.map(lambda x: cls.get(x) in (group_a, group_b)).values
    S = pd.DataFrame(np.asarray(scores), index=obs.index,
                     columns=list(getattr(scores, "columns", range(np.shape(scores)[1]))))[m]
    ob = obs[m]
    lab = (ob.mc.map(cls).values == group_a).astype(float)
    obsd = S.values[lab == 1].mean(0) - S.values[lab == 0].mean(0)
    cl = ob.clone_id.astype(str).values
    uc, inv = np.unique(cl, return_inverse=True)
    clone_lab = pd.Series(lab, index=inv).groupby(level=0).first().reindex(range(len(uc))).values
    null = np.empty((nperm, S.shape[1]))
    for i in range(nperm):
        cp = rng.permutation(clone_lab)[inv]
        null[i] = S.values[cp == 1].mean(0) - S.values[cp == 0].mean(0)
    p = np.maximum((np.abs(null) >= np.abs(obsd)[None, :] - 1e-12).mean(0), 1.0 / nperm)
    return pd.DataFrame(dict(program=S.columns, diff_cellmean=obsd, p_clone_block=p,
                             q_clone_block=multipletests(p, method="fdr_bh")[1],
                             null_sd_clone=null.std(0),
                             null_q975_clone=np.quantile(np.abs(null), 0.95, axis=0)))


# ---------------------------------------------------------------- step 7: continuous
def _axis_labels(classes, obs, axis):
    """Per-cell group label for one within-arm axis, aligned to `obs`."""
    return classes.set_index("metaclone")[axis].reindex(obs.mc).values


def cell_level_de(A, classes, axis, group_a, group_b, min_pct=0.05,
                  tie_correct=True):
    """Cell-level Wilcoxon rank-sum DE between two metaclone groups of one arm.

    `group_a` is the test group and `group_b` the reference, so a positive
    log2 fold change means higher in `group_a`. Genes are pre-filtered to those
    detected in at least `min_pct` of the tested cells, which sets the multiple-
    testing universe; p-values are Benjamini-Hochberg adjusted by scanpy across
    that universe. Expression is the object's `.X` (log scale - see
    `load_day7`), which is what a rank-sum test expects.

    Returns (table, meta): the per-gene table and a one-row frame recording the
    cell and metaclone counts behind it.
    """
    import scanpy as sc
    lab = _axis_labels(classes, A.obs, axis)
    m = np.isin(lab, [group_a, group_b])
    S = A[m].copy()
    S.obs["_grp"] = pd.Categorical(lab[m], categories=[group_a, group_b])
    det = np.asarray((S.X > 0).mean(0)).ravel()
    keep = det >= min_pct
    sc.tl.rank_genes_groups(S, "_grp", groups=[group_a], reference=group_b,
                            method="wilcoxon", tie_correct=tie_correct,
                            corr_method="benjamini-hochberg", pts=True,
                            mask_var=keep)
    T = sc.get.rank_genes_groups_df(S, group=group_a)
    T = T.rename(columns={"names": "gene", "logfoldchanges": "log2fc",
                          "pvals": "p", "pvals_adj": "q",
                          "pct_nz_group": "pct_A"})
    # scanpy only returns the reference detection fraction when reference="rest",
    # so compute it here for the explicit reference group.
    ib = (S.obs._grp == group_b).values
    pct_b = pd.Series(np.asarray((S[ib].X > 0).mean(0)).ravel(), index=S.var_names)
    T["pct_B"] = pct_b.reindex(T.gene).values
    T["axis"], T["group_A"], T["group_B"] = axis, group_a, group_b
    meta = pd.DataFrame([dict(
        axis=axis, group_A=group_a, group_B=group_b,
        n_cells_A=int((S.obs._grp == group_a).sum()),
        n_cells_B=int((S.obs._grp == group_b).sum()),
        n_mc_A=int((classes[axis] == group_a).sum()),
        n_mc_B=int((classes[axis] == group_b).sum()),
        n_genes_tested=int(keep.sum()), min_pct=min_pct)])
    return T.sort_values("p").reset_index(drop=True), meta


def cell_level_program_test(scores, obs, classes, axis, group_a, group_b):
    """Cell-level Mann-Whitney test of each program score between two groups.

    AUROC is the rank-based effect size (0.5 = no separation); Cohen's d is
    reported alongside on the pooled within-group SD. FDR is Benjamini-Hochberg
    across the programs of one call.
    """
    lab = _axis_labels(classes, obs, axis)
    ia, ib = lab == group_a, lab == group_b
    rows = []
    for prog in scores.columns:
        x = np.asarray(scores.loc[ia, prog], dtype=float)
        y = np.asarray(scores.loc[ib, prog], dtype=float)
        u, pv = mannwhitneyu(x, y, alternative="two-sided")
        sd = np.sqrt((((len(x) - 1) * x.var(ddof=1)) +
                      ((len(y) - 1) * y.var(ddof=1))) / (len(x) + len(y) - 2))
        rows.append(dict(program=prog, mean_A=x.mean(), mean_B=y.mean(),
                         diff=x.mean() - y.mean(),
                         cohens_d=(x.mean() - y.mean()) / sd,
                         auroc=u / (len(x) * len(y)), p=pv,
                         n_cells_A=len(x), n_cells_B=len(y)))
    T = pd.DataFrame(rows)
    T["q"] = multipletests(T.p, method="fdr_bh")[1]
    T["axis"], T["group_A"], T["group_B"] = axis, group_a, group_b
    T["n_mc_A"] = int((classes[axis] == group_a).sum())
    T["n_mc_B"] = int((classes[axis] == group_b).sum())
    return T.sort_values("p").reset_index(drop=True)


def response_correlation(usage_mc, R, score_col, nperm=10000, seed=SEED,
                         min_mc_cells=None, obs=None):
    """Spearman of metaclone program usage against a continuous response score,
    with a metaclone-label permutation null."""
    rng = np.random.default_rng(seed)
    r = R.set_index("metaclone")[score_col].dropna()
    mcs = [m for m in usage_mc.index if m in r.index]
    if min_mc_cells is not None:
        n = obs.groupby("mc").size()
        mcs = [m for m in mcs if n.get(m, 0) >= min_mc_cells]
    Y = usage_mc.loc[mcs].values
    x = r.loc[mcs].values
    Yr = np.apply_along_axis(rankdata, 0, Y)
    xr = rankdata(x)
    Yz = (Yr - Yr.mean(0)) / Yr.std(0, ddof=0)
    xz = (xr - xr.mean()) / xr.std(ddof=0)
    rho = (Yz * xz[:, None]).mean(0)
    P = np.vstack([rng.permutation(xz) for _ in range(nperm)])
    null = (P @ Yz) / len(xz)
    p = np.maximum((np.abs(null) >= np.abs(rho)[None, :] - 1e-12).mean(0), 1.0 / nperm)
    return pd.DataFrame(dict(program=usage_mc.columns, score=score_col, rho=rho,
                             p_perm=p, q_perm=multipletests(p, method="fdr_bh")[1],
                             n_mc=len(mcs))), null, np.array(mcs), x


# ---------------------------------------------------------------- step 8: enrichment
def enrich(hits, universe, sets, min_overlap=3):
    """Hypergeometric overrepresentation of a gene list in a GMT library."""
    N, K = len(universe), len(set(hits) & set(universe))
    rows = []
    hs = set(hits)
    for term, g in sets.items():
        n = len(g)
        k = len(hs & set(g))
        if k < min_overlap:
            continue
        rows.append(dict(term=term, n_set=n, n_hit=k,
                         expected=K * n / N,
                         p=hypergeom.sf(k - 1, N, n, K),
                         genes=";".join(sorted(hs & set(g)))))
    E = pd.DataFrame(rows)
    if len(E):
        E["q"] = multipletests(E.p, method="fdr_bh")[1]
        E = E.sort_values("p")
    return E


def matched_background(A, gi, hits, genes_hvg, n_bins=20, seed=SEED):
    """Expression-matched universe for a program's top genes."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(A.X.mean(0)).ravel()[gi]
    b = pd.qcut(pd.Series(mu, index=genes_hvg), n_bins, labels=False, duplicates="drop")
    return list(genes_hvg), b
