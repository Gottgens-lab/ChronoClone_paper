#!/usr/bin/env python3
# =============================================================================
# export_mast_bundle.py  --  build the MAST input bundle from the AnnData object
# =============================================================================
#
# This is step 1 of the two-step MAST provenance for Figure 5 panels 5D/5F:
#
#   (1) export_mast_bundle.py   adata_valid_palantir_20260626.h5ad  ->  bundle/
#   (2) run_mast.R              bundle/  ->  mast_out/   (Seurat + MAST DE)
#
# It reads the primitive (HSC + MPP) compartment out of the AnnData object,
# recovers integer raw counts by inverting Seurat LogNormalize, and writes the
# gzipped MatrixMarket + genes + cell-metadata + manifest that run_mast.R
# consumes.  The panels do NOT re-run this: the preserved MAST CSVs are loaded
# by Figure5_molecular_response.ipynb.  This script documents how those inputs
# were produced from the h5ad, for the paper's code-availability section.
#
# This is the reference_workspace self-contained rewrite of the authoritative research
# implementation at:
#   analysis_workspace/paper_figures/mast_gene_analysis.py::export_mast_bundle
#   analysis_workspace/paper_figures/metaclone_gene_analysis.py::recover_raw_counts
# The count-recovery math and the exported columns are identical to those
# modules (the paper's MAST inputs come from them).
#
# WHY COUNT RECOVERY IS NEEDED
# ----------------------------
# The AnnData `counts` layer stores Seurat log1p-normalized values, not raw
# integer counts.  MAST expects raw counts (it does its own normalization via
# Seurat downstream).  For each cell we invert the normalization exactly:
#     raw = round( expm1(log1p_value) * nCount_RNA / 10000 )
# and assert that the recovered library size matches nCount_RNA to the integer,
# so the recovery is verified per cell (no silent drift).
#
# USAGE
# -----
#   python export_mast_bundle.py H5AD_PATH OUTPUT_DIR
#   # e.g.
#   python export_mast_bundle.py \
#       ../data/adata_valid_palantir_20260626.h5ad  ./mast_bundle
# =============================================================================

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse

# --- Constants (identical to research metaclone_gene_analysis.py) --------------
NORMALIZATION_SCALE = 10_000.0
# var_names lead with RNA features, then LARRY/reporter barcode features; the
# first barcode name marks the RNA/barcode boundary.
BARCODE_PREFIXES = ("Sapphire1-", "Sapphire2-", "GFP-", "Cre-", "Flp-")
# Primitive compartment the MAST contrasts are defined on.
PRIMITIVE_CELL_TYPES = ("HSC", "MPP")


def rna_feature_count(var_names) -> int:
    """Number of leading RNA features before the first barcode feature."""
    for i, name in enumerate(map(str, var_names)):
        if name.startswith(BARCODE_PREFIXES):
            return i
    raise ValueError("No barcode feature boundary was found in var_names.")


def _to_csr(matrix) -> sparse.csr_matrix:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return sparse.csr_matrix(matrix)


def recover_raw_counts(obj, cell_indices, n_rna_features=None) -> sparse.csr_matrix:
    """Invert Seurat LogNormalize to integer counts for selected cells."""
    cell_indices = np.asarray(cell_indices, dtype=int)
    n_rna_features = n_rna_features or rna_feature_count(obj.var_names)
    matrix = _to_csr(
        obj.layers["counts"][cell_indices, :n_rna_features]
    ).astype(float)
    totals = pd.to_numeric(
        obj.obs.iloc[cell_indices]["nCount_RNA"], errors="raise"
    ).to_numpy(float)

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
        raise AssertionError(
            f"Raw RNA reconstruction failed; max library-size error={delta}."
        )
    return matrix


def _write_matrix_market_gzip(matrix: sparse.spmatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        io.mmwrite(handle, matrix)


def export_mast_bundle(h5ad_path: Path, output_dir: Path,
                       cell_types=PRIMITIVE_CELL_TYPES) -> None:
    """Export raw counts + aligned metadata for all observed primitive cells.

    Every assigned cell in the requested states is included; no clone is
    required to occur at another timepoint or in another condition
    (no complete-case filter).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    obj = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = obj.obs.copy()
        mask = obs["cell_type"].astype(str).isin(cell_types).to_numpy()
        cell_indices = np.flatnonzero(mask)
        metadata = obs.iloc[cell_indices].copy()
        metadata.insert(0, "cell_id", obj.obs_names[cell_indices].astype(str))
        metadata["metaclone"] = pd.to_numeric(
            metadata["metaclone"], errors="raise"
        ).astype(int)
        for col in ("clone_id", "condition", "time_point", "cell_type",
                    "replicate"):
            metadata[col] = metadata[col].astype(str)
        if "Sex" in metadata.columns:
            metadata["Sex"] = metadata["Sex"].astype(str)

        n_rna = rna_feature_count(obj.var_names)
        raw_counts = recover_raw_counts(
            obj, cell_indices, n_rna_features=n_rna
        ).transpose().tocoo()  # -> genes x cells
        genes = pd.Index(obj.var_names[:n_rna].astype(str), name="gene")
    finally:
        obj.file.close()

    # Integrity guards — fail loud rather than feed MAST a corrupt bundle.
    if metadata["cell_id"].duplicated().any():
        raise AssertionError("MAST metadata contains duplicate cell IDs.")
    if genes.duplicated().any():
        raise AssertionError("MAST input contains duplicate RNA feature names.")
    if raw_counts.shape != (len(genes), len(metadata)):
        raise AssertionError(
            f"MAST matrix/metadata mismatch: {raw_counts.shape}, "
            f"{len(genes)} genes, {len(metadata)} cells."
        )
    if 99 in set(metadata["metaclone"]):
        raise AssertionError("Unassigned MC99 cells entered the MAST input.")

    matrix_path = output_dir / "primitive_raw_counts.mtx.gz"
    genes_path = output_dir / "primitive_genes.tsv.gz"
    metadata_path = output_dir / "primitive_cell_metadata.csv.gz"
    manifest_path = output_dir / "primitive_mast_input_manifest.csv"

    _write_matrix_market_gzip(raw_counts, matrix_path)
    pd.Series(genes, name="gene").to_csv(
        genes_path, sep="\t", index=False, header=False, compression="gzip"
    )
    keep_cols = [c for c in ["cell_id", "metaclone", "clone_id", "condition",
                             "time_point", "cell_type", "replicate", "Sex"]
                 if c in metadata.columns]
    metadata[keep_cols].to_csv(metadata_path, index=False, compression="gzip")

    manifest = pd.DataFrame([{
        "source_h5ad": h5ad_path.name,
        "source_h5ad_size_bytes": h5ad_path.stat().st_size,
        "n_cells": len(metadata),
        "n_genes": len(genes),
        "n_nonzero_counts": int(raw_counts.nnz),
        "n_clones": int(metadata["clone_id"].nunique()),
        "n_metaclones": int(metadata["metaclone"].nunique()),
        "cell_types": ";".join(cell_types),
        "matrix_orientation": "genes_by_cells",
        "raw_count_recovery": "round(expm1(counts_layer) * nCount_RNA / 10000)",
        "clone_inclusion_rule": (
            "all cells observed in each state-condition-time snapshot; "
            "no cross-time or cross-condition complete-case filter"
        ),
        "matrix_path": matrix_path.name,
        "genes_path": genes_path.name,
        "metadata_path": metadata_path.name,
    }])
    manifest.to_csv(manifest_path, index=False)
    print(
        f"Wrote MAST bundle to {output_dir}: "
        f"{len(genes)} genes x {len(metadata)} cells "
        f"({int(raw_counts.nnz)} nonzero), {metadata['clone_id'].nunique()} "
        f"clones, {metadata['metaclone'].nunique()} metaclones."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export the MAST input bundle from the AnnData object."
    )
    parser.add_argument("h5ad_path", type=Path,
                        help="Path to adata_valid_palantir_20260626.h5ad")
    parser.add_argument("output_dir", type=Path,
                        help="Directory to write the bundle into")
    ns = parser.parse_args()
    export_mast_bundle(ns.h5ad_path.resolve(), ns.output_dir.resolve())
