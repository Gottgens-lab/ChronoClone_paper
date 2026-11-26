"""De novo programs learned on day-7 WT HSPCs only, to compare metaclones.

Day 7 is one arm (WT), one compartment (HSC+MPP), one timepoint, so any structure NMF finds is
metaclone/cell-state structure and not day, arm, or lineage structure. Sapphire* LARRY barcode
features, mt- genes and sex-chromosome genes (Xist/Tsix and the Y-linked genes) are excluded
BEFORE variable-gene selection: 87,071 of the object's 144,976
features are barcode reads, and left in they dominate HVG selection and NMF learns clone identity
rather than transcriptional programs.
"""
import json, os
import numpy as np, pandas as pd, scanpy as sc
from sklearn.decomposition import NMF

ROOT = "/path/to/ChronoClone"
RUN = f"{ROOT}/reference_workspace/Figure/metaclone_day7_programs_20260909"
os.makedirs(f"{RUN}/tables", exist_ok=True)

A = sc.read_h5ad(f"{ROOT}/reference_workspace/data/adata_valid_palantir_20260626.h5ad")
day = A.obs.time_point.astype(str).str.extract(r"(\d+)")[0].astype(int).values
A = A[day == 7].copy()                                     # day-7 HSPCs, all WT
A.obs["mc"] = A.obs.metaclone.astype(str).str.extract(r"(\d+)")[0].astype(int).values

# sex genes: recorded per cell first (for the composition QC), then dropped
SEX = ["Xist", "Tsix", "Ddx3y", "Eif2s3y", "Kdm5d", "Usp9y", "Uty"]
SEX = [g for g in SEX if g in A.var_names]
Y = [g for g in SEX if g not in ("Xist", "Tsix")]
sx = pd.DataFrame(np.asarray(A[:, SEX].X.todense()), index=A.obs_names, columns=SEX)
QC = pd.DataFrame(index=A.obs_names,
                  data={"mc": A.obs.mc.values, "clone_id": A.obs.clone_id.values,
                        "replicate": A.obs.replicate.astype(str).values,
                        "xist": sx.Xist.values, "y_sum": sx[Y].sum(1).values})
QC["sex"] = np.where(QC.y_sum > QC.xist, "male", np.where(QC.xist > 0, "female", "unassigned"))
QC.to_csv(f"{RUN}/tables/day7_sex_percell.csv")
print(QC.sex.value_counts().to_dict(), flush=True)

vn = pd.Index(A.var_names)
keep = ~(vn.str.startswith("Sapphire") | vn.str.startswith("mt-") | vn.isin(SEX))
A = A[:, np.asarray(keep)].copy()
print(f"dropped {len(SEX)} sex genes: {SEX}", flush=True)
sc.pp.highly_variable_genes(A, n_top_genes=2000, flavor="seurat")
V = A[:, A.var.highly_variable].copy()
X = np.asarray(V.X.todense() if hasattr(V.X, "todense") else V.X, dtype=np.float64)
X[X < 0] = 0
print(f"day-7 matrix for NMF: {X.shape[0]} cells x {X.shape[1]} HVGs", flush=True)

# K by reconstruction-error elbow, averaged over 3 seeds
rows = []
for k in range(4, 17, 2):
    for sd in (0, 1, 2):
        m = NMF(n_components=k, init="nndsvd", random_state=sd, max_iter=400, tol=1e-4)
        m.fit(X)
        rows.append({"k": k, "seed": sd, "err": float(m.reconstruction_err_)})
    print(f"  k={k} done", flush=True)
KS = pd.DataFrame(rows).groupby("k").err.agg(["mean", "std"]).reset_index()
KS["drop"] = -KS["mean"].diff()
KS.to_csv(f"{RUN}/tables/nmf_k_selection.csv", index=False)

K = int(KS.k[KS["drop"].idxmax()]) if KS["drop"].notna().any() else 10
K = max(K, 8)
mod = NMF(n_components=K, init="nndsvd", random_state=0, max_iter=1200, tol=1e-5).fit(X)
W = mod.transform(X)
U = pd.DataFrame(W / np.clip(W.sum(1, keepdims=True), 1e-12, None),
                 index=A.obs_names, columns=[f"P{i+1}" for i in range(K)])
U["mc"] = A.obs.mc.values
U["clone_id"] = A.obs.clone_id.values
U.to_csv(f"{RUN}/tables/nmf_day7_usage.csv")

LG = pd.DataFrame(
    {f"P{i+1}": V.var_names[np.argsort(-mod.components_[i])[:25]].tolist() for i in range(K)})
LG.to_csv(f"{RUN}/tables/nmf_day7_leading_genes.csv", index=False)

json.dump({"K": K, "n_cells": int(X.shape[0]), "n_hvg": int(X.shape[1]),
           "excluded_feature_classes": ["Sapphire* (LARRY barcodes)", "mt-", "sex genes"],
           "sex_genes_excluded": SEX,
           "n_features_before_exclusion": 144976,
           "recon_err": float(mod.reconstruction_err_)},
          open(f"{RUN}/tables/nmf_day7_meta.json", "w"), indent=2)
print("K =", K, "| leading genes P1:", LG.P1.head(6).tolist(), flush=True)
