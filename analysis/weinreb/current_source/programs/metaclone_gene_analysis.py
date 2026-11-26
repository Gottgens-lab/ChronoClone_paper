"""Clone-pseudobulk gene analysis for the ChronoClone Singh/STRACK figures.

The updated H5AD stores RNA expression as log1p-normalized values in the
``counts`` layer. Count-like integers are reconstructed using retained
``nCount_RNA`` values::

    raw_count = round(expm1(value) * nCount_RNA / 10_000)

QC verifies reconstructed per-cell library totals and sparse-cache consistency;
without an independently supplied raw UMI matrix, it cannot establish
element-wise identity to original raw counts. All hypothesis tests in this module use one pseudobulk observation per clone,
condition, and timepoint. Cells are never treated as independent replicates.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

from paper_figures.project_paths import (
    PALANTIR_PATH,
    PROJECT_ROOT,
    resolve_original_study_dir,
)

NORMALIZATION_SCALE = 10_000.0
BARCODE_PREFIXES = ("Sapphire1-", "Sapphire2-", "GFP-", "Cre-", "Flp-")
SEX_LINKED_EXCLUSIONS = {
    "Xist",
    "Ddx3y",
    "Eif2s3y",
    "Kdm5d",
    "Uty",
    "Usp9y",
    "Zfy1",
    "Zfy2",
    "Rbmy",
    "Sry",
}

CURATED_MODULES = {
    "AP1_early_response": [
        "Fos",
        "Fosb",
        "Jun",
        "Junb",
        "Jund",
        "Egr1",
        "Egr2",
        "Egr3",
        "Ier2",
        "Ier3",
        "Dusp1",
        "Dusp2",
        "Dusp5",
        "Nr4a1",
        "Nr4a2",
        "Sik1",
        "Zfp36",
    ],
    "HSC_stemness": [
        "Procr",
        "Hlf",
        "Mecom",
        "Msi2",
        "Mllt3",
        "Prdm16",
        "Meis1",
        "Pbx1",
        "Tcf15",
        "Slamf1",
    ],
    "myeloid_maturation": [
        "Mpo",
        "Ctsg",
        "Elane",
        "Prtn3",
        "Plac8",
        "Zeb2",
        "Zbtb16",
        "Lyz2",
        "S100a8",
        "S100a9",
    ],
    "reported_ancestry_memory": ["Pcdh7", "Dnajc6", "Slc22a3", "Cdk6", "Cd48"],
}


@dataclass
class PseudobulkData:
    counts: sparse.csr_matrix
    metadata: pd.DataFrame
    genes: pd.Index


@dataclass
class SnapshotData:
    counts: sparse.csr_matrix
    metadata: pd.DataFrame
    genes: pd.Index


def _bh(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    mask = np.isfinite(values)
    if mask.any():
        out[mask] = multipletests(values[mask], method="fdr_bh")[1]
    return out


def rna_feature_count(var_names: Sequence[str]) -> int:
    """Return the number of leading RNA features before barcode features."""
    for i, name in enumerate(map(str, var_names)):
        if name.startswith(BARCODE_PREFIXES):
            return i
    raise ValueError("No barcode feature boundary was found in var_names.")


def _to_csr(matrix) -> sparse.csr_matrix:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return sparse.csr_matrix(matrix)


def recover_raw_counts(
    obj: ad.AnnData,
    cell_indices: np.ndarray,
    *,
    n_rna_features: int | None = None,
) -> sparse.csr_matrix:
    """Invert Seurat-style log normalization for selected cells."""
    cell_indices = np.asarray(cell_indices, dtype=int)
    n_rna_features = n_rna_features or rna_feature_count(obj.var_names)
    matrix = _to_csr(obj.layers["counts"][cell_indices, :n_rna_features]).astype(float)
    totals = pd.to_numeric(obj.obs.iloc[cell_indices]["nCount_RNA"], errors="raise").to_numpy(float)

    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        matrix.data[start:end] = np.rint(
            np.expm1(matrix.data[start:end]) * totals[row] / NORMALIZATION_SCALE
        )
    matrix.data = matrix.data.astype(np.int64)
    matrix.eliminate_zeros()

    reconstructed = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.array_equal(reconstructed, totals.astype(np.int64)):
        delta = np.abs(reconstructed - totals).max()
        raise AssertionError(f"Raw RNA reconstruction failed; maximum library-size error={delta}.")
    return matrix


def validate_raw_reconstruction(
    path: Path = PALANTIR_PATH,
    *,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Validate exact count recovery for every assigned cell in chunks."""
    obj = ad.read_h5ad(path, backed="r")
    rows = []
    try:
        n_rna = rna_feature_count(obj.var_names)
        for start in range(0, obj.n_obs, chunk_size):
            stop = min(start + chunk_size, obj.n_obs)
            indices = np.arange(start, stop)
            matrix = recover_raw_counts(obj, indices, n_rna_features=n_rna)
            expected = obj.obs.iloc[indices]["nCount_RNA"].to_numpy(int)
            observed = np.asarray(matrix.sum(axis=1)).ravel().astype(int)
            rows.append(
                {
                    "cell_start": start,
                    "cell_stop": stop,
                    "n_cells": len(indices),
                    "n_rna_features": n_rna,
                    "all_library_sizes_exact": bool(np.array_equal(expected, observed)),
                    "maximum_absolute_library_size_error": int(np.abs(expected - observed).max()),
                }
            )
    finally:
        obj.file.close()
    return pd.DataFrame(rows)


def build_hsc_clone_pseudobulk(
    path: Path = PALANTIR_PATH,
    *,
    cell_type: str = "HSC",
) -> PseudobulkData:
    """Aggregate reconstructed raw RNA counts to one HSC pseudobulk per clone snapshot."""
    obj = ad.read_h5ad(path, backed="r")
    try:
        obs = obj.obs.copy()
        mask = obs["cell_type"].astype(str).eq(cell_type).to_numpy()
        cell_indices = np.flatnonzero(mask)
        sub = obs.iloc[cell_indices].copy()
        sub["metaclone"] = pd.to_numeric(sub["metaclone"], errors="raise").astype(int)
        group_cols = ["metaclone", "clone_id", "condition", "time_point"]
        group_index = pd.MultiIndex.from_frame(sub[group_cols])
        unique_index = group_index.drop_duplicates()
        codes = unique_index.get_indexer(group_index)

        raw = recover_raw_counts(obj, cell_indices)
        indicator = sparse.csr_matrix(
            (np.ones(len(codes), dtype=np.int8), (codes, np.arange(len(codes)))),
            shape=(len(unique_index), len(codes)),
        )
        counts = (indicator @ raw).tocsr()

        metadata = unique_index.to_frame(index=False)
        metadata["n_cells"] = np.bincount(codes, minlength=len(unique_index)).astype(int)
        metadata["library_size"] = np.asarray(counts.sum(axis=1)).ravel().astype(int)
        for column in ["replicate", "Sex"]:
            values = sub.groupby(group_cols, observed=True, sort=False)[column].first()
            metadata[column] = values.reindex(unique_index).to_numpy()
        metadata["analysis_unit"] = "one raw-count HSC pseudobulk per clone-condition-time snapshot"
        metadata["raw_count_recovery"] = "round(expm1(counts_layer) * nCount_RNA / 10000)"
        genes = pd.Index(obj.var_names[: counts.shape[1]].astype(str), name="gene")
    finally:
        obj.file.close()
    return PseudobulkData(counts=counts, metadata=metadata, genes=genes)


def build_metaclone_snapshot_pseudobulk(pb: PseudobulkData) -> SnapshotData:
    """Pool every observed clone into replicate-level metaclonal snapshots.

    A snapshot is one metaclone x condition x timepoint x replicate observation.
    This is the primary condition-comparison grain: no clone is required to be
    observed at another timepoint or in another condition.
    """
    group_cols = ["metaclone", "condition", "time_point", "replicate"]
    group_index = pd.MultiIndex.from_frame(pb.metadata[group_cols])
    unique_index = group_index.drop_duplicates()
    codes = unique_index.get_indexer(group_index)
    indicator = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.int8), (codes, np.arange(len(codes)))),
        shape=(len(unique_index), len(codes)),
    )
    counts = (indicator @ pb.counts).tocsr()
    metadata = unique_index.to_frame(index=False)
    metadata["n_clones"] = np.bincount(codes, minlength=len(unique_index)).astype(int)
    metadata["n_cells"] = (
        pb.metadata.groupby(group_cols, observed=True, sort=False)["n_cells"]
        .sum()
        .reindex(unique_index)
        .to_numpy(int)
    )
    metadata["library_size"] = np.asarray(counts.sum(axis=1)).ravel().astype(int)
    metadata["analysis_unit"] = (
        "one raw-count HSC metaclone-condition-time-replicate snapshot"
    )
    metadata["clone_inclusion_rule"] = (
        "all clones observed in this snapshot; no cross-time complete-case filter"
    )
    return SnapshotData(counts=counts, metadata=metadata, genes=pb.genes.copy())


def _log_cpm(counts: sparse.csr_matrix, prior_count: float = 0.5) -> np.ndarray:
    library_size = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    dense = counts.toarray().astype(float)
    return np.log2((dense + prior_count) / (library_size[:, None] + 1.0) * 1_000_000.0)


def _technical_class(gene: str) -> str:
    if gene in SEX_LINKED_EXCLUSIONS:
        return "sex_linked_excluded"
    if gene.startswith("mt-"):
        return "mitochondrial"
    if gene.startswith(("Rpl", "Rps")):
        return "ribosomal"
    return "biological"


def _fit_coefficient(
    expression: np.ndarray,
    design: np.ndarray,
    coefficient: int,
    *,
    clusters: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit all genes at once and return coefficient, robust SE, and p-value."""
    design = np.asarray(design, dtype=float)
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError("Design matrix is rank deficient.")
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ expression
    residual = expression - design @ beta
    contrast_weights = design @ bread[:, coefficient]

    if clusters is None:
        hat = np.sum((design @ bread) * design, axis=1)
        adjusted = residual / np.clip(1.0 - hat[:, None], 1e-6, None)
        variance = np.sum((contrast_weights[:, None] * adjusted) ** 2, axis=0)
        df = max(len(design) - rank, 1)
    else:
        cluster_codes, unique = pd.factorize(pd.Series(clusters, dtype="string"), sort=True)
        scores = sparse.csr_matrix(
            (
                np.ones(len(cluster_codes)),
                (cluster_codes, np.arange(len(cluster_codes))),
            ),
            shape=(len(unique), len(cluster_codes)),
        ) @ (contrast_weights[:, None] * residual)
        correction = (len(unique) / max(len(unique) - 1, 1)) * (
            (len(design) - 1) / max(len(design) - rank, 1)
        )
        variance = correction * np.sum(np.asarray(scores) ** 2, axis=0)
        df = max(len(unique) - 1, 1)

    standard_error = np.sqrt(np.clip(variance, 0, None))
    statistic = np.divide(
        beta[coefficient],
        standard_error,
        out=np.zeros_like(beta[coefficient]),
        where=standard_error > 0,
    )
    p_value = 2 * stats.t.sf(np.abs(statistic), df=df)
    return beta[coefficient], standard_error, p_value


def _module_annotations(genes: Iterable[str]) -> list[str]:
    membership = {gene: [] for gene in genes}
    for module, members in CURATED_MODULES.items():
        for gene in members:
            if gene in membership:
                membership[gene].append(module)
    return [";".join(membership[gene]) for gene in genes]


def _base_result(
    pb: PseudobulkData,
    row_indices: np.ndarray,
    group: np.ndarray,
    effect: np.ndarray,
    standard_error: np.ndarray,
    p_value: np.ndarray,
) -> pd.DataFrame:
    counts = pb.counts[row_indices]
    expression = _log_cpm(counts)
    group = np.asarray(group, dtype=bool)
    reference = ~group
    detected = counts > 0
    result = pd.DataFrame(
        {
            "gene": pb.genes,
            "effect_log2_cpm": effect,
            "standard_error": standard_error,
            "p_value": p_value,
            "comparison_mean_log2_cpm": expression[group].mean(axis=0),
            "reference_mean_log2_cpm": expression[reference].mean(axis=0),
            "comparison_median_log2_cpm": np.median(expression[group], axis=0),
            "reference_median_log2_cpm": np.median(expression[reference], axis=0),
            "comparison_detection_fraction": np.asarray(detected[group].mean(axis=0)).ravel(),
            "reference_detection_fraction": np.asarray(detected[reference].mean(axis=0)).ravel(),
            "comparison_total_counts": np.asarray(counts[group].sum(axis=0)).ravel(),
            "reference_total_counts": np.asarray(counts[reference].sum(axis=0)).ravel(),
        }
    )
    result["n_comparison_clone_pseudobulks"] = int(group.sum())
    result["n_reference_clone_pseudobulks"] = int(reference.sum())
    result["technical_class"] = result["gene"].map(_technical_class)
    result["module"] = _module_annotations(result["gene"])
    testable = (
        (result["comparison_total_counts"] + result["reference_total_counts"] >= 20)
        & (
            (result["comparison_detection_fraction"] >= 0.10)
            | (result["reference_detection_fraction"] >= 0.10)
        )
        & ~result["gene"].isin(SEX_LINKED_EXCLUSIONS)
    )
    result["tested"] = testable
    result["q_value"] = np.nan
    result.loc[testable, "q_value"] = _bh(result.loc[testable, "p_value"])
    result["direction"] = np.where(result["effect_log2_cpm"] >= 0, "up", "down")
    return result.sort_values(["q_value", "p_value", "gene"], na_position="last")


def two_group_de(
    pb: PseudobulkData,
    *,
    metaclone: int,
    time_point: str,
    comparison: str,
    reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate-adjusted state-matched DE within one metaclone and timepoint."""
    metadata = pb.metadata
    mask = (
        metadata["metaclone"].eq(metaclone)
        & metadata["time_point"].eq(time_point)
        & metadata["condition"].isin([reference, comparison])
    )
    row_indices = np.flatnonzero(mask.to_numpy())
    meta = metadata.iloc[row_indices].reset_index(drop=True)
    group = meta["condition"].eq(comparison).to_numpy(float)
    replicate = pd.get_dummies(meta["replicate"].astype(str), drop_first=True, dtype=float)
    design = np.column_stack([np.ones(len(meta)), group, replicate.to_numpy()])
    expression = _log_cpm(pb.counts[row_indices])
    effect, standard_error, p_value = _fit_coefficient(expression, design, 1)
    result = _base_result(
        pb,
        row_indices,
        group.astype(bool),
        effect,
        standard_error,
        p_value,
    )
    result.insert(1, "contrast", f"MC{metaclone} {time_point} {comparison} - {reference}")
    result.insert(2, "metaclone", metaclone)
    result.insert(3, "time_point", time_point)
    result.insert(4, "comparison", comparison)
    result.insert(5, "reference", reference)
    return result, meta


def temporal_interaction_de(
    pb: PseudobulkData,
    *,
    metaclone: int,
    comparison: str,
    reference: str = "WT",
    early: str = "D14",
    late: str = "D27",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condition-by-time DE with clone-clustered uncertainty."""
    metadata = pb.metadata
    mask = (
        metadata["metaclone"].eq(metaclone)
        & metadata["condition"].isin([reference, comparison])
        & metadata["time_point"].isin([early, late])
    )
    row_indices = np.flatnonzero(mask.to_numpy())
    meta = metadata.iloc[row_indices].reset_index(drop=True)
    condition = meta["condition"].eq(comparison).to_numpy(float)
    time = meta["time_point"].eq(late).to_numpy(float)
    interaction = condition * time
    replicate = pd.get_dummies(meta["replicate"].astype(str), drop_first=True, dtype=float)
    design = np.column_stack(
        [np.ones(len(meta)), condition, time, interaction, replicate.to_numpy()]
    )
    expression = _log_cpm(pb.counts[row_indices])
    effect, standard_error, p_value = _fit_coefficient(
        expression,
        design,
        3,
        clusters=meta["clone_id"].astype(str),
    )

    counts = pb.counts[row_indices]
    result = pd.DataFrame(
        {
            "gene": pb.genes,
            "effect_log2_cpm": effect,
            "standard_error": standard_error,
            "p_value": p_value,
        }
    )
    for condition_name in [reference, comparison]:
        for time_name in [early, late]:
            arm = meta["condition"].eq(condition_name) & meta["time_point"].eq(time_name)
            arm_counts = counts[arm.to_numpy()]
            arm_expression = _log_cpm(arm_counts)
            prefix = f"{condition_name}_{time_name}"
            result[f"{prefix}_mean_log2_cpm"] = arm_expression.mean(axis=0)
            result[f"{prefix}_detection_fraction"] = np.asarray(
                (arm_counts > 0).mean(axis=0)
            ).ravel()
            result[f"{prefix}_total_counts"] = np.asarray(arm_counts.sum(axis=0)).ravel()
            result[f"{prefix}_clone_pseudobulks"] = int(arm.sum())

    result.insert(1, "contrast", f"MC{metaclone} ({comparison} {late}-{early}) - ({reference} {late}-{early})")
    result.insert(2, "metaclone", metaclone)
    result["technical_class"] = result["gene"].map(_technical_class)
    result["module"] = _module_annotations(result["gene"])
    count_columns = [column for column in result if column.endswith("_total_counts")]
    detection_columns = [column for column in result if column.endswith("_detection_fraction")]
    testable = (
        result[count_columns].sum(axis=1).ge(20)
        & result[detection_columns].max(axis=1).ge(0.10)
        & ~result["gene"].isin(SEX_LINKED_EXCLUSIONS)
    )
    result["tested"] = testable
    result["q_value"] = np.nan
    result.loc[testable, "q_value"] = _bh(result.loc[testable, "p_value"])
    result["direction"] = np.where(result["effect_log2_cpm"] >= 0, "up", "down")
    return result.sort_values(["q_value", "p_value", "gene"], na_position="last"), meta


def metaclone_specificity_de(
    pb: PseudobulkData,
    *,
    metaclone: int,
    condition: str,
    time_point: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare one metaclone with all other final metaclones in a state-matched arm."""
    metadata = pb.metadata
    mask = metadata["condition"].eq(condition) & metadata["time_point"].eq(time_point)
    row_indices = np.flatnonzero(mask.to_numpy())
    meta = metadata.iloc[row_indices].reset_index(drop=True)
    group = meta["metaclone"].eq(metaclone).to_numpy(float)
    replicate = pd.get_dummies(meta["replicate"].astype(str), drop_first=True, dtype=float)
    design = np.column_stack([np.ones(len(meta)), group, replicate.to_numpy()])
    expression = _log_cpm(pb.counts[row_indices])
    effect, standard_error, p_value = _fit_coefficient(expression, design, 1)
    result = _base_result(
        pb,
        row_indices,
        group.astype(bool),
        effect,
        standard_error,
        p_value,
    )
    result.insert(1, "contrast", f"MC{metaclone} vs other metaclones, {condition} {time_point}")
    result.insert(2, "metaclone", metaclone)
    result.insert(3, "condition", condition)
    result.insert(4, "time_point", time_point)
    return result, meta


def molecular_memory_table(
    baseline: pd.DataFrame,
    postmutation: pd.DataFrame,
    *,
    baseline_label: str,
    postmutation_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join pre-mutation and post-mutation metaclone-specific gene effects."""
    columns = [
        "gene",
        "effect_log2_cpm",
        "standard_error",
        "p_value",
        "q_value",
        "comparison_detection_fraction",
        "reference_detection_fraction",
        "technical_class",
        "module",
        "tested",
    ]
    left = baseline[columns].rename(
        columns={column: f"baseline_{column}" for column in columns if column != "gene"}
    )
    right = postmutation[columns].rename(
        columns={column: f"postmutation_{column}" for column in columns if column != "gene"}
    )
    merged = left.merge(right, on="gene", how="inner")
    merged["baseline_label"] = baseline_label
    merged["postmutation_label"] = postmutation_label
    merged["technical_class"] = merged["baseline_technical_class"]
    merged["module"] = merged["baseline_module"].where(
        merged["baseline_module"].ne(""), merged["postmutation_module"]
    )

    x = merged["baseline_effect_log2_cpm"]
    y = merged["postmutation_effect_log2_cpm"]
    merged["memory_class"] = "other"
    merged.loc[
        (x >= 0.35)
        & (y >= 0.35)
        & merged["baseline_q_value"].lt(0.05)
        & merged["postmutation_q_value"].lt(0.05),
        "memory_class",
    ] = "retained_up"
    merged.loc[
        (x <= -0.35)
        & (y <= -0.35)
        & merged["baseline_q_value"].lt(0.05)
        & merged["postmutation_q_value"].lt(0.05),
        "memory_class",
    ] = "retained_down"
    merged.loc[
        x.abs().lt(0.20)
        & y.ge(0.50)
        & merged["postmutation_q_value"].lt(0.05),
        "memory_class",
    ] = "acquired_up"
    merged.loc[
        x.abs().lt(0.20)
        & y.le(-0.50)
        & merged["postmutation_q_value"].lt(0.05),
        "memory_class",
    ] = "acquired_down"
    merged.loc[
        x.ge(0.35)
        & merged["baseline_q_value"].lt(0.05)
        & y.abs().lt(0.20),
        "memory_class",
    ] = "erased_up"
    merged.loc[
        x.le(-0.35)
        & merged["baseline_q_value"].lt(0.05)
        & y.abs().lt(0.20),
        "memory_class",
    ] = "erased_down"

    tested = merged[
        merged["baseline_tested"].fillna(False)
        & merged["postmutation_tested"].fillna(False)
        & merged["technical_class"].eq("biological")
    ]
    correlation = stats.spearmanr(
        tested["baseline_effect_log2_cpm"],
        tested["postmutation_effect_log2_cpm"],
    )
    summary = pd.DataFrame(
        [
            {
                "baseline_label": baseline_label,
                "postmutation_label": postmutation_label,
                "n_tested_genes": len(tested),
                "spearman_rho": float(correlation.statistic),
                "spearman_p_value": float(correlation.pvalue),
                **{
                    f"n_{label}": int(merged["memory_class"].eq(label).sum())
                    for label in [
                        "retained_up",
                        "retained_down",
                        "acquired_up",
                        "acquired_down",
                        "erased_up",
                        "erased_down",
                    ]
                },
            }
        ]
    )
    return merged.sort_values(
        ["memory_class", "postmutation_q_value", "baseline_q_value", "gene"],
        na_position="last",
    ), summary


def original_study_gene_sets() -> pd.DataFrame:
    """Read Singh Table S4 sets and append explicitly ChronoClone-curated sets.

    The original spreadsheet is preferred. Public release clones may omit that
    third-party workbook, so the exact previously extracted, provenance-labeled
    CSV distributed with the release is an explicit reproducibility fallback.
    The curated modules are analysis definitions, not signatures copied from
    Singh Table S4; their provenance label is normalized on both code paths.
    """
    curated_source = (
        "ChronoClone-curated gene set; biological rationale informed by "
        "Singh et al. 2025"
    )
    try:
        signature_path = (
            resolve_original_study_dir()
            / "1-s2.0-S1934590925000128-mmc5.xlsx"
        )
    except FileNotFoundError:
        fallback = (
            PROJECT_ROOT
            / "final_release"
            / "source_tables"
            / "figure5"
            / "fig5_gene_original_study_gene_sets.csv"
        )
        if not fallback.is_file():
            raise
        result = pd.read_csv(fallback)
        result = result[~result["gene_set"].eq("Npm1c_HOX_PBX_MEIS")].copy()
        result.loc[result["gene_set"].isin(CURATED_MODULES), "source"] = (
            curated_source
        )
        return result.drop_duplicates(["gene_set", "gene"]).sort_values(
            ["gene_set", "gene"]
        ).reset_index(drop=True)
    rows = []
    table = pd.read_excel(signature_path)
    for column in table.columns:
        for gene in table[column].dropna().astype(str):
            rows.append(
                {
                    "gene_set": str(column),
                    "gene": gene,
                    "source": "Singh et al. 2025 Table S4",
                }
            )
    for module, genes in CURATED_MODULES.items():
        for gene in genes:
            rows.append(
                {
                    "gene_set": module,
                    "gene": gene,
                    "source": curated_source,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["gene_set", "gene"]).sort_values(
        ["gene_set", "gene"]
    )


def _deseq_filter(
    counts: sparse.csr_matrix,
    genes: pd.Index,
    *,
    minimum_total_count: int,
    minimum_detection_fraction: float,
) -> np.ndarray:
    total = np.asarray(counts.sum(axis=0)).ravel()
    detection = np.asarray((counts > 0).mean(axis=0)).ravel()
    return (
        (total >= minimum_total_count)
        & (detection >= minimum_detection_fraction)
        & ~genes.isin(SEX_LINKED_EXCLUSIONS)
    )


def _annotate_deseq_result(
    result: pd.DataFrame,
    *,
    counts: sparse.csr_matrix,
    genes: pd.Index,
    group: np.ndarray,
    comparison: str,
    reference: str,
) -> pd.DataFrame:
    expression = _log_cpm(counts)
    comparison_mask = np.asarray(group).astype(str) == str(comparison)
    reference_mask = np.asarray(group).astype(str) == str(reference)
    gene_positions = genes.get_indexer(result["gene"])
    if (gene_positions < 0).any():
        raise KeyError("A DESeq2 result gene was not found in the filtered count matrix.")
    result["comparison_mean_log2_cpm"] = expression[comparison_mask].mean(axis=0)[
        gene_positions
    ]
    result["reference_mean_log2_cpm"] = expression[reference_mask].mean(axis=0)[
        gene_positions
    ]
    detected = counts > 0
    result["comparison_detection_fraction"] = np.asarray(
        detected[comparison_mask].mean(axis=0)
    ).ravel()[gene_positions]
    result["reference_detection_fraction"] = np.asarray(
        detected[reference_mask].mean(axis=0)
    ).ravel()[gene_positions]
    result["n_comparison_samples"] = int(comparison_mask.sum())
    result["n_reference_samples"] = int(reference_mask.sum())
    result["technical_class"] = result["gene"].map(_technical_class)
    result["module"] = _module_annotations(result["gene"])
    result["tested"] = True
    result["direction"] = np.where(result["effect_log2_fc"] >= 0, "up", "down")
    return result


def deseq2_binary_contrast(
    data: PseudobulkData | SnapshotData,
    *,
    row_mask: Sequence[bool],
    group: Sequence[str] | pd.Series,
    comparison: str,
    reference: str,
    contrast_label: str,
    covariates: Sequence[str] = ("replicate",),
    minimum_total_count: int = 20,
    minimum_detection_fraction: float = 0.25,
    n_cpus: int = 1,
    random_seed: int = 20_260_714,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a count-aware DESeq2 contrast on clone or snapshot pseudobulks.

    PyDESeq2 is imported lazily so the data-preparation code remains usable in
    lightweight environments. The ChronoClone ``LARRY_env`` provides PyDESeq2.
    """
    try:
        import pydeseq2
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise ImportError(
            "PyDESeq2 is required. Execute with the LARRY_env Python kernel."
        ) from exc

    row_mask = np.asarray(row_mask, dtype=bool)
    row_indices = np.flatnonzero(row_mask)
    if not len(row_indices):
        raise ValueError(f"No observations selected for {contrast_label}.")
    metadata = data.metadata.iloc[row_indices].copy().reset_index(drop=True)
    all_group = np.asarray(group).astype(str)
    selected_group = all_group[row_indices]
    valid_group = np.isin(selected_group, [reference, comparison])
    if not valid_group.all():
        row_indices = row_indices[valid_group]
        metadata = metadata.loc[valid_group].reset_index(drop=True)
        selected_group = selected_group[valid_group]
    observed_levels = set(selected_group)
    if observed_levels != {reference, comparison}:
        raise ValueError(
            f"{contrast_label} requires {reference!r} and {comparison!r}; "
            f"observed {sorted(observed_levels)}."
        )

    selected_counts = data.counts[row_indices].tocsr()
    keep = _deseq_filter(
        selected_counts,
        data.genes,
        minimum_total_count=minimum_total_count,
        minimum_detection_fraction=minimum_detection_fraction,
    )
    filtered_genes = data.genes[keep]
    sample_ids = [f"sample_{i:04d}" for i in range(len(metadata))]
    count_frame = pd.DataFrame(
        selected_counts[:, keep].toarray().astype(np.int64),
        index=sample_ids,
        columns=filtered_genes,
    )
    design_metadata = pd.DataFrame(index=sample_ids)
    design_factors = []
    for covariate in covariates:
        values = metadata[covariate].astype(str).to_numpy()
        if len(np.unique(values)) > 1:
            design_metadata[covariate] = values
            design_factors.append(covariate)
    design_metadata["group"] = selected_group
    design_factors.append("group")

    np.random.seed(random_seed)
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=count_frame,
        metadata=design_metadata,
        design_factors=design_factors,
        ref_level=["group", reference],
        refit_cooks=False,
        inference=inference,
        quiet=True,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dds.deseq2()
    stats_model = DeseqStats(
        dds,
        contrast=["group", comparison, reference],
        cooks_filter=False,
        inference=inference,
        quiet=True,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        stats_model.summary()
    result = (
        stats_model.results_df.reset_index(names="gene")
        .rename(
            columns={
                "baseMean": "base_mean",
                "log2FoldChange": "effect_log2_fc",
                "lfcSE": "standard_error",
                "stat": "statistic",
                "pvalue": "p_value",
                "padj": "q_value",
            }
        )
        .copy()
    )
    result.insert(1, "contrast", contrast_label)
    result["comparison"] = comparison
    result["reference"] = reference
    result["model"] = "PyDESeq2 negative-binomial GLM"
    result["pydeseq2_version"] = pydeseq2.__version__
    result["random_seed"] = random_seed
    result = _annotate_deseq_result(
        result,
        counts=selected_counts[:, keep],
        genes=filtered_genes,
        group=selected_group,
        comparison=comparison,
        reference=reference,
    )

    support_columns = [
        column
        for column in ["metaclone", "condition", "time_point", "replicate"]
        if column in metadata
    ]
    support = metadata[support_columns].copy()
    support["comparison_group"] = selected_group
    for column in ["n_clones", "n_cells", "library_size"]:
        if column in metadata:
            support[column] = metadata[column].to_numpy()
    support["contrast"] = contrast_label
    support["analysis_unit"] = metadata["analysis_unit"].astype(str).to_numpy()
    return result.sort_values(["q_value", "p_value", "gene"], na_position="last"), support


def deseq2_temporal_interaction(
    snapshots: SnapshotData,
    *,
    metaclone: int,
    comparison: str,
    reference: str = "WT",
    early: str = "D14",
    late: str = "D27",
    minimum_total_count: int = 20,
    minimum_detection_fraction: float = 0.25,
    n_cpus: int = 1,
    random_seed: int = 20_260_714,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test a driver-by-time interaction on metaclonal HSC snapshots."""
    try:
        import pydeseq2
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise ImportError(
            "PyDESeq2 is required. Execute with the LARRY_env Python kernel."
        ) from exc

    metadata = snapshots.metadata
    mask = (
        metadata["metaclone"].eq(metaclone)
        & metadata["condition"].isin([reference, comparison])
        & metadata["time_point"].isin([early, late])
    )
    row_indices = np.flatnonzero(mask.to_numpy())
    selected_metadata = metadata.iloc[row_indices].copy().reset_index(drop=True)
    selected_counts = snapshots.counts[row_indices].tocsr()
    expected_arms = {
        f"{reference}-{early}",
        f"{reference}-{late}",
        f"{comparison}-{early}",
        f"{comparison}-{late}",
    }
    arm = (
        selected_metadata["condition"].astype(str)
        + "-"
        + selected_metadata["time_point"].astype(str)
    )
    if set(arm) != expected_arms:
        raise ValueError(
            f"Temporal interaction requires all four arms; observed {sorted(set(arm))}."
        )
    keep = _deseq_filter(
        selected_counts,
        snapshots.genes,
        minimum_total_count=minimum_total_count,
        minimum_detection_fraction=minimum_detection_fraction,
    )
    filtered_genes = snapshots.genes[keep]
    sample_ids = [f"sample_{i:04d}" for i in range(len(selected_metadata))]
    count_frame = pd.DataFrame(
        selected_counts[:, keep].toarray().astype(np.int64),
        index=sample_ids,
        columns=filtered_genes,
    )
    design_metadata = pd.DataFrame(
        {
            "replicate": selected_metadata["replicate"].astype(str).to_numpy(),
            "arm": arm.to_numpy(),
        },
        index=sample_ids,
    )
    np.random.seed(random_seed)
    inference = DefaultInference(n_cpus=n_cpus)
    reference_arm = f"{reference}-{early}"
    dds = DeseqDataSet(
        counts=count_frame,
        metadata=design_metadata,
        design_factors=["replicate", "arm"],
        ref_level=["arm", reference_arm],
        refit_cooks=False,
        inference=inference,
        quiet=True,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dds.deseq2()
    design = dds.obsm["design_matrix"]
    arm_design = design.groupby(design_metadata["arm"]).mean()
    contrast_vector = (
        arm_design.loc[f"{comparison}-{late}"]
        - arm_design.loc[f"{comparison}-{early}"]
        - arm_design.loc[f"{reference}-{late}"]
        + arm_design.loc[f"{reference}-{early}"]
    ).to_numpy(float)
    stats_model = DeseqStats(
        dds,
        contrast=["arm", f"{comparison}-{late}", reference_arm],
        cooks_filter=False,
        inference=inference,
        quiet=True,
    )
    stats_model.contrast_vector = contrast_vector
    stats_model.contrast = [
        "temporal_interaction",
        f"{comparison}-{late}-minus-{early}",
        f"{reference}-{late}-minus-{early}",
    ]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        stats_model.summary()
    result = stats_model.results_df.reset_index(names="gene").rename(
        columns={
            "baseMean": "base_mean",
            "log2FoldChange": "effect_log2_fc",
            "lfcSE": "standard_error",
            "stat": "statistic",
            "pvalue": "p_value",
            "padj": "q_value",
        }
    )
    contrast_label = (
        f"MC{metaclone} ({comparison} {late}-{early}) - "
        f"({reference} {late}-{early})"
    )
    result.insert(1, "contrast", contrast_label)
    result.insert(2, "metaclone", metaclone)
    result["comparison"] = comparison
    result["reference"] = reference
    result["model"] = "PyDESeq2 negative-binomial GLM temporal interaction"
    result["pydeseq2_version"] = pydeseq2.__version__
    result["random_seed"] = random_seed
    result["technical_class"] = result["gene"].map(_technical_class)
    result["module"] = _module_annotations(result["gene"])
    result["tested"] = True
    result["direction"] = np.where(result["effect_log2_fc"] >= 0, "up", "down")

    expression = _log_cpm(selected_counts[:, keep])
    filtered_positions = filtered_genes.get_indexer(result["gene"])
    for arm_name in sorted(expected_arms):
        arm_mask = arm.eq(arm_name).to_numpy()
        prefix = arm_name.replace("-", "_")
        result[f"{prefix}_mean_log2_cpm"] = expression[arm_mask].mean(axis=0)[
            filtered_positions
        ]
        result[f"{prefix}_n_snapshots"] = int(arm_mask.sum())
    support = selected_metadata.copy()
    support["arm"] = arm.to_numpy()
    support["contrast"] = contrast_label
    return result.sort_values(["q_value", "p_value", "gene"], na_position="last"), support


def gene_set_scores(de: pd.DataFrame, gene_sets: pd.DataFrame) -> pd.DataFrame:
    """Summarize directional DE statistics for supplied gene sets."""
    work = de[de["tested"] & de["technical_class"].eq("biological")].copy()
    if "statistic" in work:
        work["z_score"] = pd.to_numeric(work["statistic"], errors="coerce")
    elif "effect_log2_fc" in work:
        work["z_score"] = work["effect_log2_fc"] / work["standard_error"].replace(
            0, np.nan
        )
    else:
        work["z_score"] = work["effect_log2_cpm"] / work["standard_error"].replace(
            0, np.nan
        )
    background = work["z_score"].dropna().to_numpy(float)
    rows = []
    for gene_set, members in gene_sets.groupby("gene_set", observed=True):
        member_genes = set(members["gene"])
        inside = work[work["gene"].isin(member_genes)]["z_score"].dropna().to_numpy(float)
        outside = work[~work["gene"].isin(member_genes)]["z_score"].dropna().to_numpy(float)
        if len(inside) < 3 or not len(outside):
            continue
        test = stats.mannwhitneyu(
            inside,
            outside,
            alternative="two-sided",
            method="asymptotic",
        )
        rows.append(
            {
                "gene_set": gene_set,
                "n_genes_in_data": len(inside),
                "mean_z_score": float(np.mean(inside)),
                "median_z_score": float(np.median(inside)),
                "rank_biserial": float(2 * test.statistic / (len(inside) * len(outside)) - 1),
                "p_value": float(test.pvalue),
                "mean_z_direction": "up" if np.mean(inside) >= 0 else "down",
                "genes": ";".join(sorted(work[work["gene"].isin(member_genes)]["gene"])),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["q_value"] = _bh(result["p_value"])
        result["direction"] = np.where(
            result["rank_biserial"].ge(0),
            "positive_rank_enrichment",
            "negative_rank_enrichment",
        )
    return result.sort_values(["q_value", "p_value"], na_position="last")
