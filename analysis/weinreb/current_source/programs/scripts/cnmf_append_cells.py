#!/usr/bin/env python3
"""Notebook cells for the independent Weinreb Day-2 consensus-NMF analysis."""


def build_cnmf_cells(md, code):
    """Return append-only notebook cells for the cNMF program family."""
    return [
        md(
            r"""
            ## Independent de novo consensus-NMF program family

            This section learns a second program family from **all 1,041** Day-2
            `prog_1` + `prog_2` HSPC cells. Cells with no current metaclone are
            included in discovery. Metaclone, lineage-bias, later-output,
            pseudotime, fate-probability, and library fields are not supplied to
            feature selection, NMF, consensus construction, or K selection.
            `prog_1`/`prog_2` is used only to define the pooled input compartment;
            it is revisited after the programs are frozen for sensitivity analysis.

            ### Predeclared discovery contract and pre-freeze K-selection amendment

            - Recovered counts are normalized to 10,000 counts per cell. Features
              detected in at least 10 cells with at least 20 recovered counts are
              eligible. Exact synthetic barcode/readout features, mitochondrial
              genes, and genes in the fixed KEGG mouse Ribosome set are excluded
              before dispersion selection. Hemoglobin, sex-associated,
              cell-cycle, stress/interferon, and uncharacterized endogenous genes
              remain eligible so that such factors can be identified and flagged.
              **Feature-audit clarification dated 2026-09-09, before the accepted
              definitive run:** `Malat1` is also hard-excluded from the primary
              and every sensitivity factorization matrix. Its single-gene mask is
              applied after within-bin dispersion calibration, so excluding it
              cannot change the calibrated z-scores of unrelated genes; it is
              never eligible for the top-2,000 selection. The post-freeze full-gene
              refit retains it only as an explicit nuclear-QC audit reference.
            - The 2,000 highest within-mean-bin dispersion z-scores are scaled to
              unit variance without centering, preserving non-negativity.
            - K = 3,...,20 is swept with 30 genuinely random, distinct initializations
              per K (`init="random"`). Converged L2-normalized spectra are pooled;
              local-density outliers above median + 3 scaled MAD are removed;
              retained spectra are clustered into K consensus groups and the
              within-cluster median is frozen. Per-cell usage is then estimated by
              non-negative least squares against those fixed spectra.
            - A K is eligible only if at least 90% of restarts converge, every
              consensus cluster is represented in at least 40% of converged
              restarts, overall cosine silhouette is at least 0.80 and the worst
              cluster-mean silhouette is at least 0.30. Every program must have
              mean fractional usage at least 25% of the equal-share expectation
              (0.25/K) and usage at least 0.5/K in at least 10% of cells. No
              spectrum pair may have cosine similarity >= 0.35 or top-50-gene
              Jaccard >= 0.25.
            - Among eligible K values, the reconstruction-error elbow is the K
              with greatest distance below the endpoint chord after scaling K and
              error to [0,1]. The selected K is the highest-stability eligible K
              within elbow +/- 1; an exact tie chooses the smaller K. Program
              annotations and any outcome information are unavailable to this rule.
            - **Protocol amendment dated 2026-09-09, before any program was frozen
              or any biological annotation/outcome was examined:** a dry execution
              produced only two eligible K values, so a three-point error elbow was
              undefined. If the definitive sweep has one or two eligible values,
              retain eligible K values within 0.01 of the best consensus stability,
              choose the lowest consensus reconstruction error, and use smaller K
              only for an exact error tie. If no K is eligible, stop without fitting
              downstream analyses. This fallback remains entirely diagnostic-driven.

            After freezing K and the HVG consensus spectra, normalized-expression
            spectra are refitted over the full expressed non-readout gene universe
            using non-negative regression. All statistical comparisons use one
            mean fractional-usage value per LARRY clone and HC3 inference; there
            are no cell-level differential tests. This remains retrospective
            ancestry analysis, not classification, prediction, or evidence that a
            program prospectively determines fate.
            """
        ),
        code(
            r"""
            # cNMF feature audit and label-blind matrix construction
            import ast
            import re
            import zlib
            from textwrap import dedent

            from scipy.optimize import linear_sum_assignment, nnls
            from scipy.spatial.distance import cdist
            from scipy.stats import hypergeom, spearmanr
            from sklearn.cluster import KMeans
            from sklearn.decomposition import NMF
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.metrics import silhouette_samples

            OBJECTDIR = ANALYSIS_DIR / "objects"
            OBJECTDIR.mkdir(parents=True, exist_ok=True)

            CNMF_RUN_ID = "Weinreb_D2_prog1plusprog2_cNMF_20260909"
            CNMF_K_GRID = list(range(3, 21))
            CNMF_N_RESTARTS = 30
            CNMF_N_HVG = 2000
            CNMF_MAX_ITER = 1000
            CNMF_TOL = 1e-4
            CNMF_ROOT_SEED = 20260909
            CNMF_DENSITY_MAD = 3.0
            CNMF_MIN_CONVERGED = 0.90
            CNMF_MIN_RESTART_COVERAGE = 0.40
            CNMF_MIN_SILHOUETTE = 0.80
            CNMF_MIN_CLUSTER_MEAN_SILHOUETTE = 0.30
            CNMF_MIN_USAGE_VS_EQUAL_SHARE = 0.25
            CNMF_MIN_ACTIVE_FRACTION = 0.10
            CNMF_ACTIVE_USAGE_VS_EQUAL_SHARE = 0.50
            CNMF_DUPLICATE_COSINE = 0.35
            CNMF_DUPLICATE_TOP50_JACCARD = 0.25

            cnmf_environment_addendum = pd.DataFrame({
                "field": [
                    "environment_name", "environment_path", "kernel", "python",
                    "anndata", "numpy", "scipy", "scikit-learn", "statsmodels",
                    "adjustText",
                    "official_cnmf_package", "implementation", "execution_command",
                ],
                "value": [
                    "chronoclone", str(ENV_PATH), "chronoclone", sys.version.split()[0],
                    im.version("anndata"), im.version("numpy"), im.version("scipy"),
                    im.version("scikit-learn"), im.version("statsmodels"),
                    im.version("adjustText"),
                    "not installed; not required",
                    "sklearn random NMF + consensus clustering + scipy NNLS",
                    EXECUTION_COMMAND,
                ],
            })
            cnmf_environment_addendum.to_csv(
                REPORTDIR / "cnmf_environment_addendum.csv", index=False
            )
            display(cnmf_environment_addendum)

            KEGG_SOURCE = PROJECT_ROOT / "reference_workspace/data/gene_sets/KEGG_2019_Mouse.gmt"
            KEGG_LOCAL = RESOURCEDIR / "KEGG_2019_Mouse.gmt"
            EXPECTED_KEGG_SHA256 = "6aabbf753f6d14dd2205f47fc5e461688fa550a9685dffc77d27c28c7fb13df4"
            assert sha256(KEGG_SOURCE) == EXPECTED_KEGG_SHA256
            if (not KEGG_LOCAL.exists()) or sha256(KEGG_LOCAL) != EXPECTED_KEGG_SHA256:
                shutil.copy2(KEGG_SOURCE, KEGG_LOCAL)
            assert sha256(KEGG_LOCAL) == EXPECTED_KEGG_SHA256

            kegg_sets = {}
            with open(KEGG_LOCAL) as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    kegg_sets[fields[0]] = {gene.upper() for gene in fields[2:] if gene}
            ribosome_term = next(term for term in kegg_sets if term.lower() == "ribosome")
            hematopoietic_term = next(
                term for term in kegg_sets if term.lower() == "hematopoietic cell lineage"
            )
            KEGG_RIBOSOME_GENES = kegg_sets[ribosome_term]

            discovery_cells = all_day2_hspc.index.copy()
            discovery_indices = POS.loc[discovery_cells].to_numpy()
            discovery_counts = COUNTS[discovery_indices].tocsr().astype(np.float64)
            discovery_genes = np.asarray(A.var_names.astype(str))
            discovery_upper = np.char.upper(discovery_genes.astype(str))
            assert discovery_counts.shape == (1041, 25289)
            assert len(discovery_cells) == 1041
            assert all_day2_hspc.LARRY_ID.nunique() == 784

            synthetic_pattern = re.compile(
                r"^(?:Sapphire(?:1|2)?[-_:].*|LARRY(?:[-_:].*)?|Barcode(?:[-_:].*)?|"
                r"Guide(?:[-_:].*)?|gRNA(?:[-_:].*)?|(?:GFP|RFP|mCherry|tdTomato|"
                r"Luciferase|Puro|Hygro|Cre|Flp)(?:[-_:].*|$))$",
                flags=re.IGNORECASE,
            )
            SEX_AUDIT_GENES = {
                "XIST", "TSIX", "JPX", "FTX", "DDX3Y", "EIF2S3Y", "UTY",
                "KDM5D", "USP9Y", "SRY", "ZFY1", "ZFY2", "XLR3A", "XLR3B",
                "XLR4A", "XLR4B",
            }
            HEMOGLOBIN_PATTERN = re.compile(
                r"^HB(?:A|B|D|E|G|Q|Z)(?:[-0-9].*)?$"
            )
            synthetic_flag = np.array([
                bool(synthetic_pattern.match(gene)) for gene in discovery_genes
            ])
            mitochondrial_flag = np.char.startswith(
                np.char.lower(discovery_genes.astype(str)), "mt-"
            )
            ribosome_flag = np.array([gene in KEGG_RIBOSOME_GENES for gene in discovery_upper])
            nuclear_qc_flag = discovery_upper == "MALAT1"
            sex_flag = np.array([gene in SEX_AUDIT_GENES for gene in discovery_upper])
            hemoglobin_flag = np.array([
                bool(HEMOGLOBIN_PATTERN.match(gene)) for gene in discovery_upper
            ])
            assert int(nuclear_qc_flag.sum()) == 1
            assert not hemoglobin_flag[discovery_upper == "HBEGF"].any()

            detected_cells = np.asarray((discovery_counts > 0).sum(axis=0)).ravel().astype(int)
            total_recovered = np.asarray(discovery_counts.sum(axis=0)).ravel()
            dispersion_exclusion = synthetic_flag | mitochondrial_flag | ribosome_flag
            hard_exclusion = dispersion_exclusion | nuclear_qc_flag
            # MALAT1 is excluded from every factorization matrix after the
            # within-bin dispersion calibration. This prevents a one-gene
            # technical policy change from shifting the z-scores of unrelated
            # genes while still guaranteeing that MALAT1 cannot be selected.
            dispersion_calibration_eligible = (
                (detected_cells >= 10) & (total_recovered >= 20)
                & ~dispersion_exclusion
            )
            basic_eligible = (
                dispersion_calibration_eligible & ~nuclear_qc_flag
            )
            row_totals = np.asarray(discovery_counts.sum(axis=1)).ravel()
            assert (row_totals > 0).all()
            normalized_sparse = discovery_counts.multiply((1e4 / row_totals)[:, None]).tocsr()
            normalized_mean = np.asarray(normalized_sparse.mean(axis=0)).ravel()
            normalized_second = np.asarray(normalized_sparse.power(2).mean(axis=0)).ravel()
            normalized_variance = np.maximum(
                (normalized_second - normalized_mean ** 2) * len(discovery_cells) / (len(discovery_cells) - 1),
                0.0,
            )
            log_dispersion = np.log1p(
                normalized_variance / np.maximum(normalized_mean, 1e-12)
            )
            dispersion_z = np.full(A.n_vars, np.nan, dtype=float)
            eligible_indices = np.flatnonzero(dispersion_calibration_eligible)
            mean_bins = pd.qcut(
                pd.Series(np.log1p(normalized_mean[eligible_indices])),
                q=20, duplicates="drop",
            )
            dispersion_frame = pd.DataFrame({
                "index": eligible_indices,
                "log_dispersion": log_dispersion[eligible_indices],
                "mean_bin": mean_bins.astype(str).to_numpy(),
            })
            for _, frame in dispersion_frame.groupby("mean_bin", observed=True):
                scale = float(frame.log_dispersion.std(ddof=0))
                scale = scale if scale > 1e-12 else 1.0
                dispersion_z[frame["index"].to_numpy()] = (
                    frame.log_dispersion.to_numpy() - frame.log_dispersion.mean()
                ) / scale
            hvg_order = np.lexsort((discovery_genes, -np.nan_to_num(dispersion_z, nan=-np.inf)))
            cnmf_hvg_indices = hvg_order[basic_eligible[hvg_order]][:CNMF_N_HVG]
            assert basic_eligible[cnmf_hvg_indices].all()
            cnmf_hvg_genes = discovery_genes[cnmf_hvg_indices]
            selected_feature = np.zeros(A.n_vars, dtype=bool)
            selected_feature[cnmf_hvg_indices] = True

            cnmf_hvg_unscaled = normalized_sparse[:, cnmf_hvg_indices].toarray().astype(np.float64)
            cnmf_hvg_sd = cnmf_hvg_unscaled.std(axis=0, ddof=1)
            assert (cnmf_hvg_sd > 1e-12).all()
            CNMF_MATRIX = (cnmf_hvg_unscaled / cnmf_hvg_sd).astype(np.float32)
            assert CNMF_MATRIX.shape == (1041, CNMF_N_HVG)
            assert np.isfinite(CNMF_MATRIX).all() and (CNMF_MATRIX >= 0).all()

            exclusion_reason = np.full(A.n_vars, "", dtype=object)
            exclusion_reason[synthetic_flag] = "synthetic_barcode_or_readout"
            exclusion_reason[mitochondrial_flag & ~synthetic_flag] = "mitochondrial"
            exclusion_reason[ribosome_flag & ~synthetic_flag & ~mitochondrial_flag] = "KEGG_ribosome"
            exclusion_reason[nuclear_qc_flag & (exclusion_reason == "")] = "nuclear_QC_Malat1"
            exclusion_reason[(detected_cells < 10) & (exclusion_reason == "")] = "detected_lt_10_cells"
            exclusion_reason[(total_recovered < 20) & (exclusion_reason == "")] = "total_counts_lt_20"
            exclusion_reason[(~selected_feature) & basic_eligible & (exclusion_reason == "")] = "eligible_not_top_2000"
            exclusion_reason[selected_feature] = "selected_HVG"
            cnmf_feature_audit = pd.DataFrame({
                "gene": discovery_genes,
                "detected_cells": detected_cells,
                "total_recovered_counts": total_recovered,
                "normalized_mean": normalized_mean,
                "normalized_variance": normalized_variance,
                "dispersion_z_within_mean_bin": dispersion_z,
                "synthetic_barcode_or_readout": synthetic_flag,
                "mitochondrial": mitochondrial_flag,
                "KEGG_ribosome": ribosome_flag,
                "nuclear_QC_Malat1": nuclear_qc_flag,
                "sex_associated_audit": sex_flag,
                "hemoglobin_audit": hemoglobin_flag,
                "hard_excluded": hard_exclusion,
                "eligible_for_dispersion_calibration": dispersion_calibration_eligible,
                "eligible_before_HVG_rank": basic_eligible,
                "selected_for_cNMF": selected_feature,
                "disposition": exclusion_reason,
            })
            cnmf_feature_audit.to_csv(
                TABLEDIR / "cnmf_feature_matrix_audit.csv.gz", index=False, compression="gzip"
            )
            cnmf_feature_summary = pd.DataFrame([
                ["all_features", A.n_vars],
                ["synthetic_barcode_or_readout", int(synthetic_flag.sum())],
                ["mitochondrial_excluded", int(mitochondrial_flag.sum())],
                ["KEGG_ribosome_excluded", int((ribosome_flag & ~mitochondrial_flag).sum())],
                ["nuclear_QC_Malat1_excluded", int(nuclear_qc_flag.sum())],
                ["sex_associated_retained_for_postfit_flagging", int(sex_flag.sum())],
                ["hemoglobin_retained_for_postfit_flagging", int(hemoglobin_flag.sum())],
                ["detected_in_at_least_10_cells", int((detected_cells >= 10).sum())],
                ["eligible_after_hard_filters", int(basic_eligible.sum())],
                ["selected_HVG", int(selected_feature.sum())],
            ], columns=["feature_audit_item", "n_features"])
            cnmf_feature_summary.to_csv(TABLEDIR / "cnmf_feature_matrix_summary.csv", index=False)

            forbidden_discovery_fields = {
                "metaclone", "lineage_bias", "cell_type_weinreb", "pseudotime",
                "palantir_pseudotime", "neutrophil_pseudotime", "Library",
            }
            discovery_fields_used = {"day", "cell_type"}
            cnmf_discovery_firewall = pd.DataFrame([
                ["input_cells", "all Day-2 prog_1 + prog_2 HSPCs", 1041],
                ["expression_input", "recovered counts only", discovery_counts.shape[1]],
                ["metadata_used_in_discovery", "day|cell_type (eligibility gate only)", 2],
                ["forbidden_fields_used", "", 0],
                ["unassigned_current_metaclone_cells_included", "yes", int(all_day2_hspc.metaclone.isna().sum() + (all_day2_hspc.metaclone == -1).sum())],
            ], columns=["audit_item", "value", "n"])
            assert not (forbidden_discovery_fields & discovery_fields_used)
            assert not synthetic_flag.any()
            cnmf_discovery_firewall.to_csv(REPORTDIR / "cnmf_discovery_firewall.csv", index=False)
            display(cnmf_feature_summary)
            display(cnmf_discovery_firewall)
            """
        ),
        code(
            r"""
            # Random-restart cNMF, consensus construction, and complete K sweep
            def cnmf_seed_registry(context, k, n_restarts):
                context_crc = zlib.crc32(str(context).encode("utf-8")) & 0xFFFFFFFF
                sequence = np.random.SeedSequence([CNMF_ROOT_SEED, context_crc, int(k)])
                children = sequence.spawn(int(n_restarts))
                seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
                assert len(seeds) == len(set(seeds))
                return seeds

            def fit_random_nmf(matrix, k, seed):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model = NMF(
                        n_components=int(k), init="random", solver="cd",
                        beta_loss="frobenius", tol=CNMF_TOL,
                        max_iter=CNMF_MAX_ITER, random_state=int(seed),
                        shuffle=True,
                    )
                    usage = model.fit_transform(matrix)
                spectra = model.components_.astype(np.float64)
                norms = np.linalg.norm(spectra, axis=1, keepdims=True)
                norms[norms <= 1e-15] = 1.0
                spectra /= norms
                usage = usage * norms.ravel()[None, :]
                convergence_warnings = [
                    item for item in caught if issubclass(item.category, ConvergenceWarning)
                ]
                converged = len(convergence_warnings) == 0 and model.n_iter_ < CNMF_MAX_ITER
                return usage, spectra, float(model.reconstruction_err_), int(model.n_iter_), converged

            def fixed_spectrum_nnls(matrix, spectra):
                return np.vstack([nnls(spectra.T, row)[0] for row in matrix])

            def fractional_usage(usage):
                denominator = usage.sum(axis=1, keepdims=True)
                denominator[denominator <= 1e-15] = 1.0
                return usage / denominator

            def pairwise_top_jaccard(spectra, n_top=50):
                top = [set(np.argsort(row)[-n_top:]) for row in spectra]
                values = []
                for left, right in itertools.combinations(range(len(top)), 2):
                    union = top[left] | top[right]
                    values.append(len(top[left] & top[right]) / len(union))
                return max(values) if values else 0.0

            def run_consensus_cnmf(matrix, k, n_restarts, context):
                seeds = cnmf_seed_registry(context, k, n_restarts)
                pooled = []
                pooled_restart = []
                restart_rows = []
                for restart, seed in enumerate(seeds):
                    _, spectra, error, n_iter, converged = fit_random_nmf(matrix, k, seed)
                    restart_rows.append({
                        "context": context, "K": int(k), "restart": restart,
                        "seed": seed, "init": "random", "solver": "cd",
                        "reconstruction_error": error,
                        "relative_reconstruction_error": error / np.linalg.norm(matrix),
                        "n_iter": n_iter, "converged": bool(converged),
                    })
                    if converged:
                        pooled.append(spectra)
                        pooled_restart.extend([restart] * int(k))
                restart_diagnostics = pd.DataFrame(restart_rows)
                n_converged = int(restart_diagnostics.converged.sum())
                assert n_converged >= max(2, int(np.ceil(0.50 * n_restarts)))
                pooled_spectra = np.vstack(pooled)
                pooled_restart = np.asarray(pooled_restart, dtype=int)

                distances = cdist(pooled_spectra, pooled_spectra, metric="cosine")
                neighbor_count = max(3, int(round(0.30 * n_converged)))
                local_distance = np.sort(distances, axis=1)[:, 1:neighbor_count + 1].mean(axis=1)
                local_median = float(np.median(local_distance))
                scaled_mad = float(1.4826 * np.median(np.abs(local_distance - local_median)))
                density_cutoff = local_median + CNMF_DENSITY_MAD * max(scaled_mad, 1e-12)
                retained = local_distance <= density_cutoff
                retained_spectra = pooled_spectra[retained]
                assert len(retained_spectra) >= int(k)
                clustering_seed = cnmf_seed_registry(f"{context}__consensus_cluster", k, 1)[0]
                clusterer = KMeans(
                    n_clusters=int(k), n_init=50, random_state=clustering_seed
                )
                cluster_labels = clusterer.fit_predict(retained_spectra)
                consensus = np.vstack([
                    np.median(retained_spectra[cluster_labels == cluster], axis=0)
                    for cluster in range(int(k))
                ])
                consensus_norm = np.linalg.norm(consensus, axis=1, keepdims=True)
                assert (consensus_norm > 1e-15).all()
                consensus /= consensus_norm
                usage = fixed_spectrum_nnls(matrix, consensus)
                usage_fraction = fractional_usage(usage)
                reconstruction = usage @ consensus

                assignment_rows = []
                retained_counter = 0
                for component in range(len(pooled_spectra)):
                    cluster = int(cluster_labels[retained_counter]) if retained[component] else -1
                    if retained[component]:
                        retained_counter += 1
                    assignment_rows.append({
                        "context": context, "K": int(k), "pooled_component": component,
                        "restart": int(pooled_restart[component]),
                        "component_within_restart": int(component % int(k)),
                        "local_cosine_distance": float(local_distance[component]),
                        "density_cutoff": density_cutoff,
                        "retained": bool(retained[component]),
                        "consensus_cluster": cluster,
                    })
                assignments = pd.DataFrame(assignment_rows)
                coverage = (
                    assignments[assignments.retained]
                    .groupby("consensus_cluster", observed=True).restart.nunique()
                    .reindex(range(int(k)), fill_value=0)
                )
                cluster_sizes = (
                    assignments[assignments.retained]
                    .groupby("consensus_cluster", observed=True).size()
                    .reindex(range(int(k)), fill_value=0)
                )
                cosine_matrix = consensus @ consensus.T
                np.fill_diagonal(cosine_matrix, -np.inf)
                max_pair_cosine = float(np.max(cosine_matrix)) if int(k) > 1 else 0.0
                sample_silhouette = silhouette_samples(
                    retained_spectra, cluster_labels, metric="cosine"
                )
                stability = float(sample_silhouette.mean())
                cluster_mean_silhouette = pd.Series(sample_silhouette).groupby(
                    cluster_labels
                ).mean().reindex(range(int(k)))
                mean_usage = usage_fraction.mean(axis=0)
                active_threshold = CNMF_ACTIVE_USAGE_VS_EQUAL_SHARE / int(k)
                active_fraction = (usage_fraction >= active_threshold).mean(axis=0)
                summary = {
                    "K": int(k), "context": context,
                    "n_restarts_requested": int(n_restarts),
                    "n_restarts_converged": n_converged,
                    "converged_fraction": n_converged / n_restarts,
                    "n_pooled_components": int(len(pooled_spectra)),
                    "n_density_outliers": int((~retained).sum()),
                    "density_cutoff": density_cutoff,
                    "min_cluster_size": int(cluster_sizes.min()),
                    "min_restart_coverage": int(coverage.min()),
                    "min_restart_coverage_fraction": float(coverage.min() / n_converged),
                    "consensus_silhouette_cosine": stability,
                    "minimum_cluster_mean_silhouette": float(cluster_mean_silhouette.min()),
                    "median_restart_relative_error": float(
                        restart_diagnostics.loc[
                            restart_diagnostics.converged, "relative_reconstruction_error"
                        ].median()
                    ),
                    "consensus_relative_error": float(
                        np.linalg.norm(matrix - reconstruction) / np.linalg.norm(matrix)
                    ),
                    "min_program_mean_usage": float(mean_usage.min()),
                    "min_mean_usage_vs_equal_share": float(mean_usage.min() * int(k)),
                    "max_program_mean_usage": float(mean_usage.max()),
                    "min_program_active_fraction": float(active_fraction.min()),
                    "active_usage_threshold": float(active_threshold),
                    "max_pairwise_consensus_cosine": max_pair_cosine,
                    "max_pairwise_top50_jaccard": float(pairwise_top_jaccard(consensus, 50)),
                }
                return {
                    "summary": summary, "restart_diagnostics": restart_diagnostics,
                    "assignments": assignments, "spectra": consensus,
                    "usage": usage, "usage_fraction": usage_fraction,
                    "seeds": seeds,
                }

            cnmf_k_results = {}
            cnmf_sweep_rows = []
            cnmf_restart_parts = []
            cnmf_assignment_parts = []
            for k in CNMF_K_GRID:
                result = run_consensus_cnmf(
                    CNMF_MATRIX, k, CNMF_N_RESTARTS, f"primary_K_sweep_K{k:02d}"
                )
                cnmf_k_results[k] = result
                cnmf_sweep_rows.append(result["summary"])
                cnmf_restart_parts.append(result["restart_diagnostics"])
                cnmf_assignment_parts.append(result["assignments"])
                np.savez_compressed(
                    OBJECTDIR / f"cnmf_K{k:02d}_consensus.npz",
                    spectra_scaled=result["spectra"].astype(np.float32),
                    usage_raw=result["usage"].astype(np.float32),
                    usage_fraction=result["usage_fraction"].astype(np.float32),
                    hvg_genes=cnmf_hvg_genes,
                    hvg_indices=cnmf_hvg_indices,
                )
                print(
                    f"K={k:02d} | stability={result['summary']['consensus_silhouette_cosine']:.3f} | "
                    f"error={result['summary']['consensus_relative_error']:.4f} | "
                    f"min usage={result['summary']['min_program_mean_usage']:.3f}"
                )

            cnmf_k_diagnostics = pd.DataFrame(cnmf_sweep_rows).sort_values("K").reset_index(drop=True)
            cnmf_restart_diagnostics = pd.concat(cnmf_restart_parts, ignore_index=True)
            cnmf_component_assignments = pd.concat(cnmf_assignment_parts, ignore_index=True)
            cnmf_restart_diagnostics.to_csv(
                TABLEDIR / "cnmf_K_sweep_restart_diagnostics.csv.gz",
                index=False, compression="gzip",
            )
            cnmf_component_assignments.to_csv(
                TABLEDIR / "cnmf_K_sweep_component_consensus_assignments.csv.gz",
                index=False, compression="gzip",
            )

            cnmf_k_diagnostics["convergence_gate"] = (
                cnmf_k_diagnostics.converged_fraction >= CNMF_MIN_CONVERGED
            )
            cnmf_k_diagnostics["restart_coverage_gate"] = (
                cnmf_k_diagnostics.min_restart_coverage_fraction >= CNMF_MIN_RESTART_COVERAGE
            )
            cnmf_k_diagnostics["stability_gate"] = (
                cnmf_k_diagnostics.consensus_silhouette_cosine >= CNMF_MIN_SILHOUETTE
            )
            cnmf_k_diagnostics["worst_cluster_stability_gate"] = (
                cnmf_k_diagnostics.minimum_cluster_mean_silhouette
                >= CNMF_MIN_CLUSTER_MEAN_SILHOUETTE
            )
            cnmf_k_diagnostics["minimum_usage_gate"] = (
                cnmf_k_diagnostics.min_mean_usage_vs_equal_share
                >= CNMF_MIN_USAGE_VS_EQUAL_SHARE
            )
            cnmf_k_diagnostics["active_usage_gate"] = (
                cnmf_k_diagnostics.min_program_active_fraction >= CNMF_MIN_ACTIVE_FRACTION
            )
            cnmf_k_diagnostics["no_duplicate_cosine_gate"] = (
                cnmf_k_diagnostics.max_pairwise_consensus_cosine < CNMF_DUPLICATE_COSINE
            )
            cnmf_k_diagnostics["no_duplicate_top50_gate"] = (
                cnmf_k_diagnostics.max_pairwise_top50_jaccard < CNMF_DUPLICATE_TOP50_JACCARD
            )
            gate_columns = [
                "convergence_gate", "restart_coverage_gate", "stability_gate",
                "worst_cluster_stability_gate",
                "minimum_usage_gate", "active_usage_gate",
                "no_duplicate_cosine_gate", "no_duplicate_top50_gate",
            ]
            cnmf_k_diagnostics["eligible"] = cnmf_k_diagnostics[gate_columns].all(axis=1)
            eligible_k = cnmf_k_diagnostics[cnmf_k_diagnostics.eligible].copy()
            assert len(eligible_k) >= 1, cnmf_k_diagnostics
            best_eligible_stability = float(
                eligible_k.consensus_silhouette_cosine.max()
            )
            fallback_stability_cutoff = best_eligible_stability - 0.01
            eligible_k["reconstruction_elbow_distance"] = np.nan
            if len(eligible_k) >= 3:
                k_scaled = (
                    eligible_k.K - eligible_k.K.min()
                ) / (eligible_k.K.max() - eligible_k.K.min())
                error_scaled = (
                    eligible_k.consensus_relative_error
                    - eligible_k.consensus_relative_error.min()
                ) / (
                    eligible_k.consensus_relative_error.max()
                    - eligible_k.consensus_relative_error.min()
                )
                endpoint_line = error_scaled.iloc[0] + (
                    error_scaled.iloc[-1] - error_scaled.iloc[0]
                ) * k_scaled
                eligible_k["reconstruction_elbow_distance"] = (
                    endpoint_line - error_scaled.to_numpy()
                )
                elbow_k = int(
                    eligible_k.sort_values(
                        ["reconstruction_elbow_distance", "K"],
                        ascending=[False, True],
                    ).iloc[0].K
                )
                selection_candidates = eligible_k[
                    (eligible_k.K - elbow_k).abs() <= 1
                ].copy()
                selected_row = selection_candidates.sort_values(
                    ["consensus_silhouette_cosine", "K"],
                    ascending=[False, True],
                ).iloc[0]
                selection_mode = "eligible_error_elbow_then_stability"
            else:
                elbow_k = np.nan
                selection_candidates = eligible_k[
                    eligible_k.consensus_silhouette_cosine
                    >= fallback_stability_cutoff
                ].copy()
                selected_row = selection_candidates.sort_values(
                    ["consensus_relative_error", "K"],
                    ascending=[True, True],
                ).iloc[0]
                selection_mode = (
                    "protocol_amendment_fewer_than_3_eligible__"
                    "stability_within_0.01_then_min_error"
                )
            CNMF_SELECTED_K = int(selected_row.K)
            cnmf_k_diagnostics["reconstruction_elbow_distance"] = cnmf_k_diagnostics.K.map(
                eligible_k.set_index("K").reconstruction_elbow_distance
            )
            cnmf_k_diagnostics["error_elbow_K"] = elbow_k
            cnmf_k_diagnostics["within_elbow_plusminus_one"] = (
                False if not np.isfinite(elbow_k) else (
                    cnmf_k_diagnostics.eligible
                    & ((cnmf_k_diagnostics.K - elbow_k).abs() <= 1)
                )
            )
            cnmf_k_diagnostics["selection_candidate"] = (
                cnmf_k_diagnostics.K.isin(selection_candidates.K)
            )
            cnmf_k_diagnostics["best_eligible_stability"] = best_eligible_stability
            cnmf_k_diagnostics["fallback_stability_cutoff"] = fallback_stability_cutoff
            cnmf_k_diagnostics["selection_mode"] = selection_mode
            cnmf_k_diagnostics["selected_K"] = cnmf_k_diagnostics.K == CNMF_SELECTED_K
            assert cnmf_k_diagnostics.selected_K.sum() == 1
            cnmf_k_diagnostics.to_csv(TABLEDIR / "cnmf_K_selection_diagnostics.csv", index=False)
            protocol_note = (
                "cNMF K-selection protocol record\n"
                f"selection_mode={selection_mode}\n"
                f"eligible_K={','.join(map(str, eligible_k.K.tolist()))}\n"
                f"best_eligible_stability={best_eligible_stability:.12g}\n"
                f"fallback_stability_cutoff={fallback_stability_cutoff:.12g}\n"
                f"selection_candidate_K={','.join(map(str, selection_candidates.K.tolist()))}\n"
                f"selected_K={CNMF_SELECTED_K}\n"
                "No biological annotations, metaclone labels, lineage outputs, or pseudotime entered selection.\n"
            )
            (REPORTDIR / "cnmf_K_selection_protocol_record.txt").write_text(protocol_note)
            display(cnmf_k_diagnostics)
            print("Frozen K:", CNMF_SELECTED_K, "| selection mode:", selection_mode)
            """
        ),
        code(
            r"""
            # CNMF_K_SELECTION_PLOT_INLINE_V1 — predeclared diagnostic rule
            plot_stem = "cnmf_01_K_selection_diagnostics"
            plot_data = cnmf_k_diagnostics.copy()
            metric_map = {
                "Consensus stability": "consensus_silhouette_cosine",
                "1 - reconstruction error": "consensus_relative_error",
                "Minimum mean usage": "min_program_mean_usage",
                "Program distinctness": "max_pairwise_consensus_cosine",
            }
            plot_rows = []
            for label, column in metric_map.items():
                values = plot_data[column].to_numpy(float)
                if label in {"1 - reconstruction error", "Program distinctness"}:
                    values = -values
                scaled = (values - values.min()) / max(values.max() - values.min(), 1e-12)
                for k, value in zip(plot_data.K, scaled):
                    plot_rows.append({"K": int(k), "diagnostic": label, "scaled_utility": float(value)})
            plot_long = pd.DataFrame(plot_rows)
            plot_long.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)

            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            colors = [INK, ORANGE, "#3F9E4D", "#7A5AA6"]
            for (label, frame), color in zip(plot_long.groupby("diagnostic", sort=False), colors):
                ax.plot(frame.K, frame.scaled_utility, marker="o", linewidth=1.6,
                        markersize=4, color=color, label=label)
            failed = plot_data.loc[~plot_data.eligible, "K"]
            ax.scatter(failed, np.full(len(failed), -0.06), marker="x", s=28,
                       color="#B23A48", label="Failed a hard gate", clip_on=False)
            if np.isfinite(elbow_k):
                ax.axvline(elbow_k, color=GREY, linestyle="--", linewidth=1.0,
                           label=f"Error elbow K={int(elbow_k)}")
            ax.axvline(CNMF_SELECTED_K, color=INK, linewidth=2.4,
                       label=f"Frozen K={CNMF_SELECTED_K}")
            ax.set(xlabel="K", ylabel="Scaled diagnostic utility", ylim=(-0.10, 1.05),
                   xticks=CNMF_K_GRID)
            ax.set_title("cNMF K selection is diagnostic-driven and label-blind", loc="left")
            ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
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
            # Freeze selected programs, refit full-gene spectra, and save complete matrices
            selected_cnmf = cnmf_k_results[CNMF_SELECTED_K]
            cnmf_selected_spectra_scaled = selected_cnmf["spectra"].copy()
            cnmf_selected_usage_raw = selected_cnmf["usage"].copy()
            cnmf_selected_usage_fraction = selected_cnmf["usage_fraction"].copy()

            # A label-blind, deterministic abundance ordering gives stable program IDs.
            abundance_order = np.argsort(
                -cnmf_selected_usage_fraction.mean(axis=0), kind="mergesort"
            )
            cnmf_selected_spectra_scaled = cnmf_selected_spectra_scaled[abundance_order]
            cnmf_selected_usage_raw = cnmf_selected_usage_raw[:, abundance_order]
            cnmf_selected_usage_fraction = cnmf_selected_usage_fraction[:, abundance_order]
            cnmf_program_ids = [
                f"Weinreb_D2_cNMF_K{CNMF_SELECTED_K:02d}_P{index:02d}"
                for index in range(1, CNMF_SELECTED_K + 1)
            ]
            assert np.allclose(cnmf_selected_usage_fraction.sum(axis=1), 1.0)

            cnmf_hvg_spectra = pd.DataFrame(
                cnmf_selected_spectra_scaled,
                index=pd.Index(cnmf_program_ids, name="program"),
                columns=cnmf_hvg_genes,
            )
            cnmf_hvg_spectra.to_csv(
                TABLEDIR / "cnmf_selected_consensus_spectra_HVG_scaled.csv.gz",
                compression="gzip",
            )
            cnmf_cell_usage_raw = pd.DataFrame(
                cnmf_selected_usage_raw,
                index=pd.Index(discovery_cells, name="cell"),
                columns=cnmf_program_ids,
            )
            cnmf_cell_usage_fraction = pd.DataFrame(
                cnmf_selected_usage_fraction,
                index=pd.Index(discovery_cells, name="cell"),
                columns=cnmf_program_ids,
            )
            cnmf_cell_usage_raw.to_csv(
                TABLEDIR / "cnmf_selected_per_cell_usage_raw.csv.gz", compression="gzip"
            )
            cnmf_cell_usage_fraction.to_csv(
                TABLEDIR / "cnmf_selected_per_cell_usage_fraction.csv.gz", compression="gzip"
            )
            cnmf_postfreeze_cell_metadata = all_day2_hspc[[
                "LARRY_ID", "Library", "cell_type", "day", "metaclone"
            ]].copy()
            cnmf_postfreeze_cell_metadata.index.name = "cell"
            cnmf_postfreeze_cell_metadata.rename(
                columns={"cell_type": "source_sublabel_postfreeze"}
            ).to_csv(
                TABLEDIR / "cnmf_discovery_cell_metadata_postfreeze.csv.gz",
                compression="gzip",
            )

            # Refit normalized-expression spectra over every expressed, non-readout
            # gene while keeping the selected cell-usage matrix fixed. Excluded
            # mt/ribosomal genes can therefore diagnose technical alignment without
            # influencing discovery or K selection.
            full_spectrum_mask = (detected_cells > 0) & ~synthetic_flag
            cnmf_full_genes = discovery_genes[full_spectrum_mask]
            full_normalized = normalized_sparse[:, full_spectrum_mask].tocsc()
            cnmf_full_spectra_coeff = np.zeros(
                (CNMF_SELECTED_K, int(full_spectrum_mask.sum())), dtype=np.float64
            )
            for gene_index in range(full_normalized.shape[1]):
                response = full_normalized[:, gene_index].toarray().ravel()
                cnmf_full_spectra_coeff[:, gene_index] = nnls(
                    cnmf_selected_usage_raw, response
                )[0]
                if (gene_index + 1) % 3000 == 0:
                    print(f"full-gene frozen-usage refit: {gene_index + 1}/{full_normalized.shape[1]}")
            row_sum = cnmf_full_spectra_coeff.sum(axis=1, keepdims=True)
            assert (row_sum > 0).all()
            cnmf_full_loading = cnmf_full_spectra_coeff / row_sum
            across_program_mean = cnmf_full_loading.mean(axis=0)
            across_program_sd = cnmf_full_loading.std(axis=0, ddof=0)
            cnmf_full_specificity = np.zeros_like(cnmf_full_loading)
            variable_genes = across_program_sd > 1e-15
            cnmf_full_specificity[:, variable_genes] = (
                cnmf_full_loading[:, variable_genes] - across_program_mean[variable_genes]
            ) / across_program_sd[variable_genes]

            cnmf_full_spectra = pd.DataFrame(
                cnmf_full_spectra_coeff,
                index=pd.Index(cnmf_program_ids, name="program"),
                columns=cnmf_full_genes,
            )
            cnmf_full_spectra.to_csv(
                TABLEDIR / "cnmf_selected_consensus_spectra_all_expressed_genes.csv.gz",
                compression="gzip",
            )
            np.savez_compressed(
                OBJECTDIR / "cnmf_selected_frozen_programs.npz",
                selected_K=CNMF_SELECTED_K,
                program_ids=np.asarray(cnmf_program_ids),
                hvg_genes=cnmf_hvg_genes,
                hvg_spectra_scaled=cnmf_selected_spectra_scaled.astype(np.float32),
                cell_usage_raw=cnmf_selected_usage_raw.astype(np.float32),
                cell_usage_fraction=cnmf_selected_usage_fraction.astype(np.float32),
                full_genes=cnmf_full_genes,
                full_gene_spectra=cnmf_full_spectra_coeff.astype(np.float32),
            )

            loading_frame = pd.DataFrame(
                cnmf_full_loading, index=cnmf_program_ids, columns=cnmf_full_genes
            )
            specificity_frame = pd.DataFrame(
                cnmf_full_specificity, index=cnmf_program_ids, columns=cnmf_full_genes
            )
            loading_rank = loading_frame.rank(axis=1, ascending=False, method="first")
            specificity_rank = specificity_frame.rank(axis=1, ascending=False, method="first")
            ranking_parts = []
            for p_index, program in enumerate(cnmf_program_ids):
                floor = float(np.quantile(cnmf_full_loading[p_index], 0.90))
                eligible_for_leading = cnmf_full_loading[p_index] >= floor
                eligible_ranking = pd.DataFrame({
                    "gene": cnmf_full_genes[eligible_for_leading],
                    "specificity": cnmf_full_specificity[p_index, eligible_for_leading],
                    "loading": cnmf_full_loading[p_index, eligible_for_leading],
                }).sort_values(
                    ["specificity", "loading", "gene"],
                    ascending=[False, False, True], kind="mergesort",
                )
                floor_specificity = pd.Series(
                    np.arange(1, len(eligible_ranking) + 1),
                    index=eligible_ranking.gene,
                )
                frame = pd.DataFrame({
                    "program": program,
                    "gene": cnmf_full_genes,
                    "spectrum_coefficient": cnmf_full_spectra_coeff[p_index],
                    "normalized_loading": cnmf_full_loading[p_index],
                    "specificity_z_across_programs": cnmf_full_specificity[p_index],
                    "loading_rank": loading_rank.loc[program].to_numpy(int),
                    "specificity_rank_all_genes": specificity_rank.loc[program].to_numpy(int),
                    "passes_program_loading_floor": eligible_for_leading,
                    "specificity_rank_with_loading_floor": np.nan,
                })
                frame.loc[eligible_for_leading, "specificity_rank_with_loading_floor"] = (
                    frame.loc[eligible_for_leading, "gene"].map(floor_specificity).to_numpy()
                )
                ranking_parts.append(frame)
            cnmf_gene_rankings = pd.concat(ranking_parts, ignore_index=True)
            cnmf_gene_rankings = cnmf_gene_rankings.merge(
                cnmf_feature_audit[[
                    "gene", "synthetic_barcode_or_readout", "mitochondrial",
                    "KEGG_ribosome", "nuclear_QC_Malat1",
                    "sex_associated_audit", "hemoglobin_audit",
                    "selected_for_cNMF",
                ]], on="gene", how="left", validate="many_to_one",
            )
            cnmf_leading_genes = (
                cnmf_gene_rankings[
                    cnmf_gene_rankings.specificity_rank_with_loading_floor <= 100
                ]
                .sort_values(["program", "specificity_rank_with_loading_floor"])
                .reset_index(drop=True)
            )
            assert cnmf_leading_genes.groupby("program").size().eq(100).all()
            cnmf_gene_rankings.to_csv(
                TABLEDIR / "cnmf_selected_program_complete_gene_rankings.csv.gz",
                index=False, compression="gzip",
            )
            cnmf_leading_genes.to_csv(
                TABLEDIR / "cnmf_selected_program_top100_genes.csv", index=False
            )

            frozen_program_registry = pd.DataFrame({
                "program": cnmf_program_ids,
                "selected_K": CNMF_SELECTED_K,
                "mean_cell_usage_fraction": cnmf_selected_usage_fraction.mean(axis=0),
                "active_cell_fraction_usage_ge_0_05": (
                    cnmf_selected_usage_fraction >= 0.05
                ).mean(axis=0),
                "top_gene_preannotation": [
                    cnmf_leading_genes.loc[
                        cnmf_leading_genes.program == program, "gene"
                    ].iloc[0]
                    for program in cnmf_program_ids
                ],
                "program_order_rule": "descending mean fractional usage; stable sort",
            })
            frozen_program_registry.to_csv(TABLEDIR / "cnmf_frozen_program_registry.csv", index=False)
            display(frozen_program_registry)
            display(cnmf_leading_genes.groupby("program", observed=True).head(10))
            """
        ),
        code(
            r"""
            # CNMF_SPECTRA_PLOT_INLINE_V1 — frozen program leading genes
            plot_stem = "cnmf_02_frozen_program_leading_genes"
            plot_data = (
                cnmf_leading_genes[cnmf_leading_genes.specificity_rank_with_loading_floor <= 5]
                .copy()
            )
            gene_order = list(dict.fromkeys(plot_data.gene.tolist()))
            plot_data["gene_x"] = plot_data.gene.map({gene: i for i, gene in enumerate(gene_order)})
            plot_data["program_y"] = plot_data.program.map({
                program: i for i, program in enumerate(cnmf_program_ids[::-1])
            })
            plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)

            fig, ax = plt.subplots(figsize=(max(9.5, 0.25 * len(gene_order)),
                                             max(4.2, 0.42 * CNMF_SELECTED_K)))
            color_values = np.clip(plot_data.specificity_z_across_programs, -2.5, 2.5)
            size_values = 35 + 180 * (
                plot_data.normalized_loading / max(plot_data.normalized_loading.max(), 1e-12)
            )
            points = ax.scatter(
                plot_data.gene_x, plot_data.program_y, c=color_values,
                s=size_values, marker="s", cmap=PROGRAM_CMAP, vmin=-2.5, vmax=2.5,
                edgecolors=WHITE, linewidths=0.35,
            )
            ax.set_xticks(range(len(gene_order)), gene_order, rotation=90, fontsize=7)
            ax.set_yticks(range(CNMF_SELECTED_K), cnmf_program_ids[::-1], fontsize=7)
            ax.set(xlabel="Top five specificity-ranked genes per program", ylabel="Frozen cNMF program")
            ax.set_title("De novo program spectra before biological annotation", loc="left")
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.tick_params(length=0)
            colorbar = fig.colorbar(points, ax=ax, pad=0.01, fraction=0.025)
            colorbar.solids.set_rasterized(False)
            colorbar.set_label("Across-program specificity z", fontsize=8)
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
            ### Frozen program usage at the LARRY-clone level

            Fractional usage is averaged over the Day-2 cells of each LARRY clone.
            The all-cell discovery export contains 784 clones. Only after this
            aggregation are the 569 clones in the current 20-metaclone partition
            joined to the existing library-fraction designs. The exact 94 broad
            contrasts, orientation vectors, support gates, HC3 covariance, and BH
            families from the curated-program analysis are reused.
            """
        ),
        code(
            r"""
            # One mean fractional-usage record per LARRY clone; no cell-level inference
            postfreeze_usage_metadata = all_day2_hspc[[
                "LARRY_ID", "Library", "cell_type"
            ]].copy()
            postfreeze_usage_metadata.index = discovery_cells
            usage_with_clone = cnmf_cell_usage_fraction.join(
                postfreeze_usage_metadata[["LARRY_ID", "cell_type"]]
            )
            cnmf_clone_usage_all = usage_with_clone.groupby("LARRY_ID", observed=True)[
                cnmf_program_ids
            ].mean()
            clone_support_all = usage_with_clone.groupby("LARRY_ID", observed=True).agg(
                n_day2_cells=("cell_type", "size"),
                n_source_sublabels=("cell_type", "nunique"),
                n_prog_1_cells=("cell_type", lambda values: int((values == "prog_1").sum())),
                n_prog_2_cells=("cell_type", lambda values: int((values == "prog_2").sum())),
            )
            cnmf_clone_usage_all = clone_support_all.join(cnmf_clone_usage_all)
            assert len(cnmf_clone_usage_all) == 784
            assert np.allclose(cnmf_clone_usage_all[cnmf_program_ids].sum(axis=1), 1.0)
            cnmf_clone_usage_all.to_csv(
                TABLEDIR / "cnmf_clone_usage_all_discovery_clones.csv.gz", compression="gzip"
            )

            cnmf_clone_usage_assigned_raw = (
                cnmf_clone_usage_all[cnmf_program_ids]
                .reindex(pseudobulk_metadata.index)
            )
            assert cnmf_clone_usage_assigned_raw.notna().all().all()
            assert len(cnmf_clone_usage_assigned_raw) == 569
            cnmf_clone_scores = (
                cnmf_clone_usage_assigned_raw - cnmf_clone_usage_assigned_raw.mean(axis=0)
            ) / cnmf_clone_usage_assigned_raw.std(axis=0, ddof=1)
            assert np.isfinite(cnmf_clone_scores.to_numpy()).all()
            cnmf_clone_usage_export = pseudobulk_metadata[[
                "clone", "metaclone", "mc_group", "lineage_bias", "lineage_bias_key",
                "n_cells", "n_prog_1_cells", "n_prog_2_cells",
            ]].join(cnmf_clone_usage_assigned_raw.add_suffix("__usage_fraction"))
            cnmf_clone_usage_export = cnmf_clone_usage_export.join(
                cnmf_clone_scores.add_suffix("__z")
            )
            cnmf_clone_usage_export.to_csv(
                TABLEDIR / "cnmf_clone_usage_current_metaclones.csv.gz", compression="gzip"
            )

            cnmf_fits_mc = {
                program: sm.OLS(
                    cnmf_clone_scores[program].to_numpy(float), mc_design.to_numpy(float)
                ).fit(cov_type="HC3")
                for program in cnmf_program_ids
            }
            cnmf_fits_family = {
                program: sm.OLS(
                    cnmf_clone_scores[program].to_numpy(float), family_design.to_numpy(float)
                ).fit(cov_type="HC3")
                for program in cnmf_program_ids
            }
            cnmf_broad_rows = []
            for spec in contrast_specs:
                fits = cnmf_fits_mc if spec["model"] == "metaclone" else cnmf_fits_family
                for program, fit_result in fits.items():
                    test = fit_result.t_test(np.asarray(spec["vector"], dtype=float))
                    cnmf_broad_rows.append({
                        "program": program, "model": spec["model"], "tier": spec["tier"],
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
            cnmf_broad_tests = pd.DataFrame(cnmf_broad_rows)
            cnmf_broad_tests["contrast_fdr"] = (
                cnmf_broad_tests.groupby("contrast_id", observed=True, group_keys=False).pvalue
                .transform(bh_adjust)
            )
            cnmf_broad_tests["tier_fdr"] = (
                cnmf_broad_tests.groupby("tier", observed=True, group_keys=False).pvalue
                .transform(bh_adjust)
            )
            metaclone_search_mask = (
                cnmf_broad_tests.tier.isin({
                    "all_metaclones_ovr", "within_lineage_ovr",
                    "within_lineage_pairwise",
                })
                & cnmf_broad_tests.support_gate
            )
            cnmf_broad_tests["metaclone_search_global_fdr"] = np.nan
            cnmf_broad_tests.loc[
                metaclone_search_mask, "metaclone_search_global_fdr"
            ] = bh_adjust(
                cnmf_broad_tests.loc[metaclone_search_mask, "pvalue"].to_numpy(float)
            )
            assert len(cnmf_broad_tests) == 94 * CNMF_SELECTED_K
            cnmf_broad_tests.to_csv(TABLEDIR / "cnmf_clone_level_broad_contrasts.csv", index=False)

            cnmf_orientation_rows = []
            for spec in contrast_specs:
                dds = dds_mc if spec["model"] == "metaclone" else dds_family
                factor = "mc_group" if spec["model"] == "metaclone" else "lineage_bias_key"
                fits = cnmf_fits_mc if spec["model"] == "metaclone" else cnmf_fits_family
                target_row = condition_mean_vector(dds, factor, spec["target_levels"])
                reference_row = condition_mean_vector(dds, factor, spec["reference_levels"])
                observed_effect = cnmf_broad_tests[
                    cnmf_broad_tests.contrast_id == spec["contrast_id"]
                ].set_index("program").effect_sd
                expected_effect = pd.Series({
                    program: float((target_row - reference_row) @ np.asarray(fit.params))
                    for program, fit in fits.items()
                })
                error = float((observed_effect.reindex(expected_effect.index) - expected_effect).abs().max())
                cnmf_orientation_rows.append({
                    "contrast_id": spec["contrast_id"], "max_abs_error": error,
                    "target_minus_reference_verified": error < 1e-10,
                })
            cnmf_orientation_audit = pd.DataFrame(cnmf_orientation_rows)
            assert cnmf_orientation_audit.target_minus_reference_verified.all()
            cnmf_orientation_audit.to_csv(
                TABLEDIR / "cnmf_clone_level_contrast_orientation_audit.csv", index=False
            )

            def cnmf_adjusted_means(design, fits, metadata, factor, levels, dds):
                coefficient_matrix = np.column_stack([
                    fits[program].params for program in cnmf_program_ids
                ])
                rows = []
                for level in levels:
                    sample = metadata.index[metadata[factor].astype(str) == str(level)][0]
                    design_row = design.loc[sample].copy()
                    design_row.loc[[c for c in design_row.index if c in library_covariates]] = 0.0
                    estimates = design_row.to_numpy(float) @ coefficient_matrix
                    rows.append({"group": str(level), **dict(zip(cnmf_program_ids, estimates))})
                return pd.DataFrame(rows)

            cnmf_adjusted_mc_means = cnmf_adjusted_means(
                mc_design, cnmf_fits_mc, pseudobulk_metadata, "mc_group",
                [f"MC{mc}" for mc in MC_ORDER], dds_mc,
            )
            cnmf_adjusted_mc_means["metaclone"] = cnmf_adjusted_mc_means.group.str[2:].astype(int)
            cnmf_adjusted_mc_means["lineage_bias"] = cnmf_adjusted_mc_means.metaclone.map(MC_TO_FAMILY)
            cnmf_adjusted_family_means = cnmf_adjusted_means(
                family_design, cnmf_fits_family, pseudobulk_metadata, "lineage_bias_key",
                FAMILY_KEY_ORDER, dds_family,
            )
            cnmf_adjusted_family_means["lineage_bias"] = cnmf_adjusted_family_means.group.map(FAMILY_DISPLAY)
            cnmf_adjusted_mc_means.to_csv(
                TABLEDIR / "cnmf_adjusted_usage_by_metaclone.csv", index=False
            )
            cnmf_adjusted_family_means.to_csv(
                TABLEDIR / "cnmf_adjusted_usage_by_lineage_bias.csv", index=False
            )
            display(
                cnmf_broad_tests.groupby("tier", observed=True).agg(
                    n_tests=("program", "size"),
                    n_supported=("support_gate", "sum"),
                    n_tier_fdr_lt_0_05=("tier_fdr", lambda values: int((values < 0.05).sum())),
                ).reset_index()
            )
            """
        ),
        code(
            r"""
            # CNMF_METACLONE_PLOT_INLINE_V1 — library-adjusted clone usage
            plot_stem = "cnmf_03_adjusted_usage_by_metaclone"
            plot_data = cnmf_adjusted_mc_means.melt(
                id_vars=["group", "metaclone", "lineage_bias"],
                value_vars=cnmf_program_ids,
                var_name="program", value_name="adjusted_usage_z",
            )
            plot_data["mc_x"] = plot_data.metaclone.map({mc: i for i, mc in enumerate(MC_ORDER)})
            plot_data["program_y"] = plot_data.program.map({
                program: i for i, program in enumerate(cnmf_program_ids[::-1])
            })
            plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)

            limit = max(1.0, float(np.nanquantile(np.abs(plot_data.adjusted_usage_z), 0.98)))
            fig, ax = plt.subplots(figsize=(11.0, max(4.2, 0.43 * CNMF_SELECTED_K)))
            points = ax.scatter(
                plot_data.mc_x, plot_data.program_y,
                c=np.clip(plot_data.adjusted_usage_z, -limit, limit),
                s=150, marker="s", cmap=PROGRAM_CMAP, vmin=-limit, vmax=limit,
                edgecolors=WHITE, linewidths=0.5,
            )
            ax.set_xticks(range(len(MC_ORDER)), [f"MC{mc}" for mc in MC_ORDER], rotation=90)
            ax.set_yticks(range(CNMF_SELECTED_K), cnmf_program_ids[::-1], fontsize=7)
            ax.set(xlabel="Metaclone (manual lineage-family order)", ylabel="Frozen cNMF program")
            ax.set_title("Clone-level cNMF usage after library-fraction adjustment", loc="left")
            family_boundaries = np.cumsum([len(FAMILY_GROUPS[name]) for name in FAMILY_ORDER])[:-1] - 0.5
            for boundary in family_boundaries:
                ax.axvline(boundary, color=GREY, linewidth=0.8)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.tick_params(length=0)
            colorbar = fig.colorbar(points, ax=ax, pad=0.01, fraction=0.025)
            colorbar.solids.set_rasterized(False)
            colorbar.set_label("Adjusted usage (SD)", fontsize=8)
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
            ### Output-matched Multi-lineage comparisons

            The four locked Day-6 behavior axes and two secondary pairs are copied
            exactly from the curated-program registry. Programs remain fixed.
            Fractional usage is re-standardized over the same 217 Multi-lineage
            clone units, then fitted with the same 13-column seven-metaclone plus
            library-fraction design. Primary combined and max-P member-conjunction
            FDR families contain 4 x K tests; the secondary family contains 2 x K.
            """
        ),
        code(
            r"""
            # Exact targeted 14-contrast mirror for frozen cNMF clone usage
            targeted_cnmf_usage_raw = cnmf_clone_usage_assigned_raw.loc[targeted_metadata.index]
            targeted_cnmf_scores = (
                targeted_cnmf_usage_raw - targeted_cnmf_usage_raw.mean(axis=0)
            ) / targeted_cnmf_usage_raw.std(axis=0, ddof=1)
            assert len(targeted_cnmf_scores) == 217
            assert targeted_cnmf_scores.index.tolist() == targeted_program_design.index.tolist()
            targeted_cnmf_fits = {
                program: sm.OLS(
                    targeted_cnmf_scores[program].to_numpy(float),
                    targeted_program_design.to_numpy(float),
                ).fit(cov_type="HC3")
                for program in cnmf_program_ids
            }
            targeted_cnmf_rows = []
            for spec in targeted_specs:
                for program, fit_result in targeted_cnmf_fits.items():
                    test = fit_result.t_test(np.asarray(spec["vector"], dtype=float))
                    targeted_cnmf_rows.append({
                        "analysis": spec["analysis"], "axis_id": spec["axis_id"],
                        "axis_label": spec["axis_label"], "role": spec["role"],
                        "member_mc": spec["member_mc"],
                        "contrast_id": spec["contrast_id"],
                        "contrast_label": spec["contrast_label"],
                        "program": program,
                        "effect_sd": float(np.asarray(test.effect).squeeze()),
                        "standard_error": float(np.asarray(test.sd).squeeze()),
                        "t_stat": float(np.asarray(test.tvalue).squeeze()),
                        "pvalue": float(np.asarray(test.pvalue).squeeze()),
                    })
            targeted_cnmf_all = pd.DataFrame(targeted_cnmf_rows)
            assert len(targeted_cnmf_all) == 14 * CNMF_SELECTED_K
            targeted_cnmf_all.to_csv(
                TABLEDIR / "cnmf_targeted_output_matched_effects_all.csv", index=False
            )

            primary_combined = targeted_cnmf_all[
                (targeted_cnmf_all.analysis == "primary_behavior_axis")
                & (targeted_cnmf_all.role == "combined")
            ][[
                "axis_id", "axis_label", "program", "effect_sd", "standard_error",
                "t_stat", "pvalue",
            ]].rename(columns={
                "effect_sd": "combined_effect_sd", "standard_error": "combined_se",
                "t_stat": "combined_t", "pvalue": "combined_pvalue",
            })
            primary_members = targeted_cnmf_all[
                (targeted_cnmf_all.analysis == "primary_behavior_axis")
                & (targeted_cnmf_all.role == "member")
            ].copy()
            member_parts = []
            for axis in BEHAVIOR_AXES:
                axis_frame = primary_members[primary_members.axis_id == axis["axis_id"]]
                left_mc, right_mc = axis["focal_mcs"]
                left = axis_frame[axis_frame.member_mc == left_mc][[
                    "program", "effect_sd", "standard_error", "t_stat", "pvalue"
                ]].rename(columns={
                    "effect_sd": "member1_effect_sd", "standard_error": "member1_se",
                    "t_stat": "member1_t", "pvalue": "member1_pvalue",
                })
                right = axis_frame[axis_frame.member_mc == right_mc][[
                    "program", "effect_sd", "standard_error", "t_stat", "pvalue"
                ]].rename(columns={
                    "effect_sd": "member2_effect_sd", "standard_error": "member2_se",
                    "t_stat": "member2_t", "pvalue": "member2_pvalue",
                })
                merged = left.merge(right, on="program", validate="one_to_one")
                merged["axis_id"] = axis["axis_id"]
                merged["member1_mc"] = left_mc
                merged["member2_mc"] = right_mc
                member_parts.append(merged)
            member_wide = pd.concat(member_parts, ignore_index=True)
            cnmf_targeted_signatures = primary_combined.merge(
                member_wide, on=["axis_id", "program"], validate="one_to_one"
            )
            cnmf_targeted_signatures["combined_global_fdr"] = bh_adjust(
                cnmf_targeted_signatures.combined_pvalue.to_numpy(float)
            )
            cnmf_targeted_signatures["conjunction_pvalue"] = cnmf_targeted_signatures[[
                "member1_pvalue", "member2_pvalue"
            ]].max(axis=1)
            cnmf_targeted_signatures["conjunction_global_fdr"] = bh_adjust(
                cnmf_targeted_signatures.conjunction_pvalue.to_numpy(float)
            )
            cnmf_targeted_signatures["members_same_direction_as_combined"] = (
                (np.sign(cnmf_targeted_signatures.member1_effect_sd)
                 == np.sign(cnmf_targeted_signatures.combined_effect_sd))
                & (np.sign(cnmf_targeted_signatures.member2_effect_sd)
                   == np.sign(cnmf_targeted_signatures.combined_effect_sd))
                & (cnmf_targeted_signatures.combined_effect_sd != 0)
            )
            secondary_cnmf = targeted_cnmf_all[
                targeted_cnmf_all.analysis == "secondary_pair"
            ].copy()
            secondary_cnmf["secondary_global_fdr"] = bh_adjust(
                secondary_cnmf.pvalue.to_numpy(float)
            )
            assert len(cnmf_targeted_signatures) == 4 * CNMF_SELECTED_K
            assert len(secondary_cnmf) == 2 * CNMF_SELECTED_K
            secondary_cnmf.to_csv(
                TABLEDIR / "cnmf_targeted_secondary_pair_effects.csv", index=False
            )

            # Same five comparator drops per primary axis as the curated analysis.
            comparator_rows = []
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
                        for program, fit_result in targeted_cnmf_fits.items():
                            comparator_rows.append({
                                "axis_id": axis["axis_id"],
                                "axis_label": axis["axis_label"],
                                "omitted_comparator_mc": omitted_mc,
                                "remaining_comparator_mcs": "|".join(map(str, remaining)),
                                "role": role, "member_mc": member_mc,
                                "program": program,
                                "effect_sd": float(np.asarray(fit_result.params) @ vector),
                            })
            cnmf_comparator_effects = pd.DataFrame(comparator_rows)
            assert len(cnmf_comparator_effects) == 4 * 5 * 3 * CNMF_SELECTED_K
            primary_direction = cnmf_targeted_signatures[[
                "axis_id", "program", "combined_effect_sd"
            ]]
            comparator_check = cnmf_comparator_effects.merge(
                primary_direction, on=["axis_id", "program"], validate="many_to_one"
            )
            comparator_check["direction_agrees"] = (
                (np.sign(comparator_check.effect_sd)
                 == np.sign(comparator_check.combined_effect_sd))
                & (comparator_check.combined_effect_sd != 0)
            )
            cnmf_comparator_stability = (
                comparator_check.groupby(["axis_id", "program"], observed=True)
                .agg(
                    comparator_drop_n_effects=("direction_agrees", "size"),
                    comparator_drop_n_direction_agree=("direction_agrees", "sum"),
                    comparator_drop_all_same_direction=("direction_agrees", "all"),
                ).reset_index()
            )
            assert cnmf_comparator_stability.comparator_drop_n_effects.eq(15).all()
            cnmf_comparator_effects.to_csv(
                TABLEDIR / "cnmf_targeted_comparator_drop_effects.csv", index=False
            )
            display(cnmf_targeted_signatures.sort_values("combined_global_fdr").head(20))
            """
        ),
        md(
            r"""
            ### Program-recovery stability

            Program discovery is rerun at the frozen K after removing each of seven
            libraries, including a fresh group-blind 2,000-HVG selection and fresh
            scaling inside each retained subset. A second
            sensitivity repeats a within-library balanced sample of `prog_1` and
            `prog_2` cells ten times. These labels are read only here, after the
            primary K, spectra, usage, and program IDs have been frozen. Each refit
            uses 20 new random restarts and is aligned to the primary spectra by a
            label-blind Hungarian maximum-cosine assignment after undoing each
            fit's gene-specific variance scaling and zero-filling over the union of
            primary/refit HVGs. Intersection size and common-feature spectral norm
            coverage are retained as audits. Predeclared recovery support requires
            every expected refit to return finite metrics, minimum matched spectral
            cosine >= 0.75 in both sensitivity families, and median retained-cell
            raw-usage Spearman rho >= 0.50. Raw usage is used for this rank
            diagnostic because component-wise scale changes do not alter Spearman
            ranks; fractional usage would inherit cross-component normalization.
            Both matched spectra must also retain at least `sqrt(0.5)` = 0.7071
            of their L2 norm on common genes, guaranteeing at least 50% of squared
            spectral energy is shared-feature evaluable.
            """
        ),
        code(
            r"""
            # Leave-one-library-out and prog-composition-balanced program recovery
            CNMF_SENSITIVITY_RESTARTS = 20
            CNMF_BALANCE_REPEATS = 10
            CNMF_RECOVERY_COSINE_GATE = 0.75
            CNMF_RECOVERY_USAGE_RHO_GATE = 0.50
            CNMF_RECOVERY_COMMON_NORM_GATE = float(np.sqrt(0.5))

            def build_sensitivity_feature_matrix(row_indices):
                subset_counts = discovery_counts[row_indices].tocsr()
                subset_totals = np.asarray(subset_counts.sum(axis=1)).ravel()
                subset_normalized = subset_counts.multiply(
                    (1e4 / subset_totals)[:, None]
                ).tocsr()
                subset_detected = np.asarray((subset_counts > 0).sum(axis=0)).ravel()
                subset_recovered = np.asarray(subset_counts.sum(axis=0)).ravel()
                subset_dispersion_eligible = (
                    (subset_detected >= 10) & (subset_recovered >= 20)
                    & ~dispersion_exclusion
                )
                subset_eligible = subset_dispersion_eligible & ~nuclear_qc_flag
                calibration_indices = np.flatnonzero(subset_dispersion_eligible)
                assert int(subset_eligible.sum()) >= CNMF_N_HVG
                subset_mean = np.asarray(subset_normalized.mean(axis=0)).ravel()
                subset_second = np.asarray(
                    subset_normalized.power(2).mean(axis=0)
                ).ravel()
                n_subset = len(row_indices)
                subset_variance = np.maximum(
                    (subset_second - subset_mean ** 2) * n_subset / (n_subset - 1),
                    0.0,
                )
                subset_dispersion = np.log1p(
                    subset_variance / np.maximum(subset_mean, 1e-12)
                )
                subset_z = np.full(A.n_vars, np.nan, dtype=float)
                subset_bins = pd.qcut(
                    pd.Series(np.log1p(subset_mean[calibration_indices])), q=20,
                    duplicates="drop",
                )
                subset_frame = pd.DataFrame({
                    "index": calibration_indices,
                    "dispersion": subset_dispersion[calibration_indices],
                    "mean_bin": subset_bins.astype(str).to_numpy(),
                })
                for _, frame in subset_frame.groupby("mean_bin", observed=True):
                    scale = float(frame.dispersion.std(ddof=0))
                    scale = scale if scale > 1e-12 else 1.0
                    subset_z[frame["index"].to_numpy()] = (
                        frame.dispersion.to_numpy() - frame.dispersion.mean()
                    ) / scale
                order = np.lexsort((
                    discovery_genes,
                    -np.nan_to_num(subset_z, nan=-np.inf),
                ))
                hvg_indices = order[subset_eligible[order]][:CNMF_N_HVG]
                assert subset_eligible[hvg_indices].all()
                unscaled = subset_normalized[:, hvg_indices].toarray().astype(np.float64)
                feature_sd = unscaled.std(axis=0, ddof=1)
                assert (feature_sd > 1e-12).all()
                matrix = (unscaled / feature_sd).astype(np.float32)
                return matrix, hvg_indices, subset_z[hvg_indices], feature_sd

            def align_cnmf_refit(
                refit_result, refit_hvg_indices, refit_feature_sd,
                row_indices, sensitivity, replicate
            ):
                refit_spectra = refit_result["spectra"]
                primary_position = {
                    int(gene_index): position
                    for position, gene_index in enumerate(cnmf_hvg_indices)
                }
                refit_position = {
                    int(gene_index): position
                    for position, gene_index in enumerate(refit_hvg_indices)
                }
                common_indices = sorted(set(primary_position) & set(refit_position))
                assert len(common_indices) >= 500
                primary_native = (
                    cnmf_selected_spectra_scaled * cnmf_hvg_sd[None, :]
                )
                refit_native = refit_spectra * refit_feature_sd[None, :]
                primary_common = primary_native[:, [
                    primary_position[index] for index in common_indices
                ]]
                refit_common = refit_native[:, [
                    refit_position[index] for index in common_indices
                ]]
                primary_common_coverage = (
                    np.linalg.norm(primary_common, axis=1)
                    / np.maximum(np.linalg.norm(primary_native, axis=1), 1e-15)
                )
                refit_common_coverage = (
                    np.linalg.norm(refit_common, axis=1)
                    / np.maximum(np.linalg.norm(refit_native, axis=1), 1e-15)
                )
                union_indices = sorted(set(primary_position) | set(refit_position))
                union_position = {
                    gene_index: position
                    for position, gene_index in enumerate(union_indices)
                }
                primary_union = np.zeros(
                    (CNMF_SELECTED_K, len(union_indices)), dtype=float
                )
                refit_union = np.zeros_like(primary_union)
                for gene_index, source_position in primary_position.items():
                    primary_union[:, union_position[gene_index]] = (
                        primary_native[:, source_position]
                    )
                for gene_index, source_position in refit_position.items():
                    refit_union[:, union_position[gene_index]] = (
                        refit_native[:, source_position]
                    )
                primary_union /= np.maximum(
                    np.linalg.norm(primary_union, axis=1, keepdims=True), 1e-15
                )
                refit_union /= np.maximum(
                    np.linalg.norm(refit_union, axis=1, keepdims=True), 1e-15
                )
                cosine = primary_union @ refit_union.T
                full_index, refit_index = linear_sum_assignment(-cosine)
                mapping = dict(zip(full_index, refit_index))
                assert set(mapping) == set(range(CNMF_SELECTED_K))
                rows = []
                for program_index, program in enumerate(cnmf_program_ids):
                    matched_index = mapping[program_index]
                    rho, _ = spearmanr(
                        cnmf_selected_usage_raw[row_indices, program_index],
                        refit_result["usage"][:, matched_index],
                    )
                    rows.append({
                        "sensitivity": sensitivity, "replicate": replicate,
                        "program": program, "matched_refit_component": int(matched_index),
                        "matched_spectrum_cosine": float(cosine[program_index, matched_index]),
                        "retained_cell_raw_usage_spearman_rho": float(rho),
                        "n_cells": int(len(row_indices)),
                        "n_common_HVGs": int(len(common_indices)),
                        "n_union_HVGs": int(len(union_indices)),
                        "spectral_comparison_coordinates": (
                            "normalized_expression_units_union_zero_filled"
                        ),
                        "refit_reselected_HVGs": True,
                        "primary_common_spectral_norm_fraction": float(
                            primary_common_coverage[program_index]
                        ),
                        "refit_common_spectral_norm_fraction": float(
                            refit_common_coverage[matched_index]
                        ),
                    })
                return pd.DataFrame(rows)

            recovery_parts = []
            recovery_restart_parts = []
            recovery_assignment_parts = []
            recovery_feature_parts = []
            discovery_library = all_day2_hspc.Library.astype(str).to_numpy()
            for omitted_library in sorted(pd.unique(discovery_library)):
                keep_rows = np.flatnonzero(discovery_library != omitted_library)
                sensitivity_matrix, sensitivity_hvgs, sensitivity_dispersion_z, sensitivity_sd = (
                    build_sensitivity_feature_matrix(keep_rows)
                )
                result = run_consensus_cnmf(
                    sensitivity_matrix, CNMF_SELECTED_K,
                    CNMF_SENSITIVITY_RESTARTS,
                    f"LOLO_discovery__{safe_name(omitted_library)}",
                )
                recovery_parts.append(align_cnmf_refit(
                    result, sensitivity_hvgs, sensitivity_sd, keep_rows,
                    "leave_one_library_out", omitted_library
                ))
                recovery_feature_parts.append(pd.DataFrame({
                    "sensitivity": "leave_one_library_out",
                    "replicate": str(omitted_library),
                    "feature_rank": np.arange(1, CNMF_N_HVG + 1),
                    "gene": discovery_genes[sensitivity_hvgs],
                    "gene_index": sensitivity_hvgs,
                    "dispersion_z": sensitivity_dispersion_z,
                    "also_primary_HVG": np.isin(sensitivity_hvgs, cnmf_hvg_indices),
                }))
                recovery_restart_parts.append(result["restart_diagnostics"])
                recovery_assignment_parts.append(result["assignments"])
                print("program recovery LOO:", omitted_library)

            discovery_source = all_day2_hspc.cell_type.astype(str).to_numpy()
            for repeat in range(CNMF_BALANCE_REPEATS):
                balanced_rows = []
                for library in sorted(pd.unique(discovery_library)):
                    library_rows = np.flatnonzero(discovery_library == library)
                    prog1 = library_rows[discovery_source[library_rows] == "prog_1"]
                    prog2 = library_rows[discovery_source[library_rows] == "prog_2"]
                    n_take = min(len(prog1), len(prog2))
                    assert n_take > 0
                    seed = cnmf_seed_registry(
                        f"prog_balance_sampling__{safe_name(library)}__repeat{repeat:02d}",
                        CNMF_SELECTED_K, 1,
                    )[0]
                    rng = np.random.default_rng(seed)
                    take1 = prog1 if len(prog1) == n_take else rng.choice(prog1, n_take, replace=False)
                    take2 = prog2 if len(prog2) == n_take else rng.choice(prog2, n_take, replace=False)
                    balanced_rows.extend(take1.tolist())
                    balanced_rows.extend(take2.tolist())
                balanced_rows = np.asarray(sorted(balanced_rows), dtype=int)
                sensitivity_matrix, sensitivity_hvgs, sensitivity_dispersion_z, sensitivity_sd = (
                    build_sensitivity_feature_matrix(balanced_rows)
                )
                result = run_consensus_cnmf(
                    sensitivity_matrix, CNMF_SELECTED_K,
                    CNMF_SENSITIVITY_RESTARTS,
                    f"prog_composition_balance__repeat{repeat:02d}",
                )
                recovery_parts.append(align_cnmf_refit(
                    result, sensitivity_hvgs, sensitivity_sd, balanced_rows,
                    "within_library_prog1_prog2_balance", repeat
                ))
                recovery_feature_parts.append(pd.DataFrame({
                    "sensitivity": "within_library_prog1_prog2_balance",
                    "replicate": str(repeat),
                    "feature_rank": np.arange(1, CNMF_N_HVG + 1),
                    "gene": discovery_genes[sensitivity_hvgs],
                    "gene_index": sensitivity_hvgs,
                    "dispersion_z": sensitivity_dispersion_z,
                    "also_primary_HVG": np.isin(sensitivity_hvgs, cnmf_hvg_indices),
                }))
                recovery_restart_parts.append(result["restart_diagnostics"])
                recovery_assignment_parts.append(result["assignments"])
                print("program recovery composition balance:", repeat, "cells:", len(balanced_rows))

            cnmf_program_recovery = pd.concat(recovery_parts, ignore_index=True)
            cnmf_recovery_restart_diagnostics = pd.concat(
                recovery_restart_parts, ignore_index=True
            )
            cnmf_recovery_component_assignments = pd.concat(
                recovery_assignment_parts, ignore_index=True
            )
            cnmf_recovery_feature_selection = pd.concat(
                recovery_feature_parts, ignore_index=True
            )
            cnmf_program_recovery.to_csv(
                TABLEDIR / "cnmf_postfreeze_program_recovery.csv", index=False
            )
            cnmf_recovery_restart_diagnostics.to_csv(
                TABLEDIR / "cnmf_postfreeze_recovery_restart_diagnostics.csv.gz",
                index=False, compression="gzip",
            )
            cnmf_recovery_component_assignments.to_csv(
                TABLEDIR / "cnmf_postfreeze_recovery_component_assignments.csv.gz",
                index=False, compression="gzip",
            )
            cnmf_recovery_feature_selection.to_csv(
                TABLEDIR / "cnmf_postfreeze_recovery_feature_selection.csv.gz",
                index=False, compression="gzip",
            )
            cnmf_program_recovery_summary = (
                cnmf_program_recovery.groupby(["sensitivity", "program"], observed=True)
                .agg(
                    n_refits=("replicate", "size"),
                    n_finite_matched_spectrum_cosine=(
                        "matched_spectrum_cosine",
                        lambda values: int(np.isfinite(values).sum()),
                    ),
                    n_finite_usage_spearman_rho=(
                        "retained_cell_raw_usage_spearman_rho",
                        lambda values: int(np.isfinite(values).sum()),
                    ),
                    minimum_matched_spectrum_cosine=("matched_spectrum_cosine", "min"),
                    median_matched_spectrum_cosine=("matched_spectrum_cosine", "median"),
                    minimum_usage_spearman_rho=("retained_cell_raw_usage_spearman_rho", "min"),
                    median_usage_spearman_rho=("retained_cell_raw_usage_spearman_rho", "median"),
                    minimum_common_HVGs=("n_common_HVGs", "min"),
                    minimum_primary_common_spectral_norm_fraction=(
                        "primary_common_spectral_norm_fraction", "min"
                    ),
                    minimum_refit_common_spectral_norm_fraction=(
                        "refit_common_spectral_norm_fraction", "min"
                    ),
                ).reset_index()
            )
            expected_recovery_refits = {
                "leave_one_library_out": len(day2_libraries),
                "within_library_prog1_prog2_balance": CNMF_BALANCE_REPEATS,
            }
            cnmf_program_recovery_summary["expected_refits"] = (
                cnmf_program_recovery_summary.sensitivity.map(expected_recovery_refits)
            )
            cnmf_program_recovery_summary["recovery_supported"] = (
                cnmf_program_recovery_summary.n_refits.eq(
                    cnmf_program_recovery_summary.expected_refits
                )
                & cnmf_program_recovery_summary.n_finite_matched_spectrum_cosine.eq(
                    cnmf_program_recovery_summary.expected_refits
                )
                & cnmf_program_recovery_summary.n_finite_usage_spearman_rho.eq(
                    cnmf_program_recovery_summary.expected_refits
                )
                & (cnmf_program_recovery_summary.minimum_matched_spectrum_cosine
                 >= CNMF_RECOVERY_COSINE_GATE)
                & (cnmf_program_recovery_summary.median_usage_spearman_rho
                   >= CNMF_RECOVERY_USAGE_RHO_GATE)
                & (cnmf_program_recovery_summary.minimum_primary_common_spectral_norm_fraction
                   >= CNMF_RECOVERY_COMMON_NORM_GATE)
                & (cnmf_program_recovery_summary.minimum_refit_common_spectral_norm_fraction
                   >= CNMF_RECOVERY_COMMON_NORM_GATE)
            )
            cnmf_program_recovery_summary.to_csv(
                TABLEDIR / "cnmf_postfreeze_program_recovery_summary.csv", index=False
            )
            display(cnmf_program_recovery_summary)
            """
        ),
        code(
            r"""
            # CNMF_RECOVERY_PLOT_INLINE_V1 — spectral recovery sensitivities
            plot_stem = "cnmf_04_program_recovery_stability"
            plot_data = cnmf_program_recovery.copy()
            sensitivity_offset = {
                "leave_one_library_out": -0.14,
                "within_library_prog1_prog2_balance": 0.14,
            }
            program_x = {program: i for i, program in enumerate(cnmf_program_ids)}
            plot_data["x"] = plot_data.program.map(program_x) + plot_data.sensitivity.map(sensitivity_offset)
            plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)

            fig, ax = plt.subplots(figsize=(max(8.5, 0.75 * CNMF_SELECTED_K), 4.8))
            style = {
                "leave_one_library_out": (INK, "Library LOO"),
                "within_library_prog1_prog2_balance": (ORANGE, "Within-library source balance"),
            }
            for sensitivity, frame in plot_data.groupby("sensitivity", sort=False):
                color, label = style[sensitivity]
                ax.scatter(frame.x, frame.matched_spectrum_cosine, s=24, alpha=0.65,
                           color=color, edgecolors=WHITE, linewidths=0.3, label=label)
                medians = frame.groupby("program", observed=True).matched_spectrum_cosine.median()
                xs = np.array([program_x[p] + sensitivity_offset[sensitivity] for p in medians.index])
                ax.scatter(xs, medians.to_numpy(), s=70, marker="_", linewidths=2.2,
                           color=color)
            ax.axhline(CNMF_RECOVERY_COSINE_GATE, color="#B23A48", linestyle="--",
                       linewidth=1.0, label=f"Predeclared cosine gate {CNMF_RECOVERY_COSINE_GATE:.2f}")
            ax.set_xticks(range(CNMF_SELECTED_K), cnmf_program_ids, rotation=90, fontsize=7)
            ax.set(xlabel="Frozen cNMF program", ylabel="Hungarian-matched spectral cosine",
                   ylim=(0, 1.03))
            ax.set_title("Frozen programs under library removal and source balancing", loc="left")
            ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
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
            ### Clone-level sensitivity to library removal and source composition

            Library leave-one-out inference removes cells, re-averages the frozen
            per-cell usage within surviving LARRY clones, recomputes library
            fractions, rebuilds full-rank designs, and refits HC3 models. It covers
            all 94 broad contrasts and the exact 14 targeted contrasts. Separately,
            clone `prog_2` fraction is added to the full-data designs only as a
            sensitivity covariate; because source composition and metaclone are
            strongly associated, this conditional estimand is not promoted over
            the prespecified pooled-HSPC analysis.
            """
        ),
        code(
            r"""
            # Reusable clone-level design for true cell-removal cNMF inference
            def aggregate_frozen_usage_by_clone(cell_frame):
                usage = cnmf_cell_usage_fraction.loc[cell_frame.index, cnmf_program_ids]
                clone_usage = usage.join(cell_frame[["LARRY_ID"]]).groupby(
                    "LARRY_ID", observed=True
                )[cnmf_program_ids].mean()
                rows = []
                for clone, clone_cells in cell_frame.groupby("LARRY_ID", observed=True):
                    assert clone_cells.metaclone.nunique() == 1
                    mc = int(clone_cells.metaclone.iloc[0])
                    library_counts = clone_cells.Library.astype(str).value_counts()
                    source_counts = clone_cells.cell_type.astype(str).value_counts()
                    record = {
                        "sample": str(clone), "clone": str(clone), "metaclone": mc,
                        "mc_group": f"MC{mc}", "multi_mc_group": f"MC{mc}",
                        "lineage_bias": MC_TO_FAMILY[mc],
                        "lineage_bias_key": MC_TO_FAMILY_KEY[mc],
                        "n_cells": int(len(clone_cells)),
                        "n_prog_1_cells": int(source_counts.get("prog_1", 0)),
                        "n_prog_2_cells": int(source_counts.get("prog_2", 0)),
                    }
                    for library in day2_libraries:
                        record[f"libfrac__{safe_name(library)}"] = float(
                            library_counts.get(library, 0) / len(clone_cells)
                        )
                    rows.append(record)
                metadata = pd.DataFrame(rows).set_index("sample").loc[clone_usage.index]
                return clone_usage, metadata

            def prepare_usage_design_metadata(metadata, selection_factor, suffix):
                frame = metadata.copy()
                canonical_library_fractions = {
                    f"libfrac__{safe_name(library)}" for library in day2_libraries
                }
                generated_library_columns = [
                    column for column in frame.columns
                    if column.startswith("libfrac__")
                    and column not in canonical_library_fractions
                ]
                frame = frame.drop(columns=generated_library_columns)
                assert canonical_library_fractions.issubset(frame.columns)
                frame[selection_factor] = frame[selection_factor].astype(str)
                frame, covariates, reference, dropped = select_library_covariates(
                    frame, selection_factor, suffix
                )
                return frame, covariates, reference, dropped

            def manual_usage_design(metadata, factor, levels, covariates):
                factor_values = pd.Categorical(
                    metadata[factor].astype(str), categories=list(levels), ordered=True
                )
                assert not pd.isna(factor_values).any()
                dummies = pd.get_dummies(factor_values, drop_first=True, dtype=float)
                dummies.index = metadata.index
                dummies.columns = [f"{factor}::{level}" for level in levels[1:]]
                design = pd.DataFrame({"Intercept": np.ones(len(metadata))}, index=metadata.index)
                for column in covariates:
                    design[column] = metadata[column].to_numpy(float)
                design = pd.concat([design, dummies], axis=1)
                assert np.linalg.matrix_rank(design.to_numpy(float)) == design.shape[1]
                return design

            def manual_group_row(design_columns, factor, levels, level):
                row = pd.Series(0.0, index=design_columns)
                row["Intercept"] = 1.0
                if str(level) != str(levels[0]):
                    row[f"{factor}::{level}"] = 1.0
                return row.to_numpy(float)

            def manual_contrast_vector(design, factor, levels, targets, references):
                target = np.mean(np.vstack([
                    manual_group_row(design.columns, factor, levels, value)
                    for value in targets
                ]), axis=0)
                reference = np.mean(np.vstack([
                    manual_group_row(design.columns, factor, levels, value)
                    for value in references
                ]), axis=0)
                vector = target - reference
                nonfactor = [
                    index for index, column in enumerate(design.columns)
                    if not column.startswith(f"{factor}::")
                ]
                assert np.allclose(vector[nonfactor], 0.0, atol=1e-12)
                return vector

            def fit_manual_usage_models(scores, design):
                assert scores.index.tolist() == design.index.tolist()
                return {
                    program: sm.OLS(
                        scores[program].to_numpy(float), design.to_numpy(float)
                    ).fit(cov_type="HC3")
                    for program in cnmf_program_ids
                }

            def usage_fit_is_estimable(fit, expected_n):
                return bool(
                    int(fit.nobs) == int(expected_n)
                    and float(fit.df_resid) > 0
                    and np.isfinite(np.asarray(fit.params, dtype=float)).all()
                    and np.isfinite(np.asarray(fit.bse, dtype=float)).all()
                    and np.isfinite(np.asarray(fit.pvalues, dtype=float)).all()
                )

            def loo_pair_support(metadata, target_mc, reference_mc):
                qualifying = []
                for library in day2_libraries:
                    column = f"libfrac__{safe_name(library)}"
                    target_n = int(
                        ((metadata.metaclone == int(target_mc)) & (metadata[column] > 0)).sum()
                    )
                    reference_n = int(
                        ((metadata.metaclone == int(reference_mc)) & (metadata[column] > 0)).sum()
                    )
                    if target_n >= 3 and reference_n >= 3:
                        qualifying.append(str(library))
                return qualifying

            # Independent equivalence check on the complete data before LOO use.
            full_manual_metadata, full_manual_covariates, full_manual_reference, _ = (
                prepare_usage_design_metadata(pseudobulk_metadata, "mc_group", "cnmf_fullcheck")
            )
            full_manual_mc = manual_usage_design(
                full_manual_metadata, "mc_group", [f"MC{mc}" for mc in MC_ORDER],
                full_manual_covariates,
            )
            full_manual_family = manual_usage_design(
                full_manual_metadata, "lineage_bias_key", FAMILY_KEY_ORDER,
                full_manual_covariates,
            )
            manual_mc_fits = fit_manual_usage_models(cnmf_clone_scores, full_manual_mc)
            manual_family_fits = fit_manual_usage_models(cnmf_clone_scores, full_manual_family)
            equivalence_errors = []
            for spec in contrast_specs:
                factor = "mc_group" if spec["model"] == "metaclone" else "lineage_bias_key"
                levels = [f"MC{mc}" for mc in MC_ORDER] if spec["model"] == "metaclone" else FAMILY_KEY_ORDER
                design = full_manual_mc if spec["model"] == "metaclone" else full_manual_family
                fits = manual_mc_fits if spec["model"] == "metaclone" else manual_family_fits
                vector = manual_contrast_vector(
                    design, factor, levels, spec["target_levels"], spec["reference_levels"]
                )
                observed = cnmf_broad_tests[
                    cnmf_broad_tests.contrast_id == spec["contrast_id"]
                ].set_index("program").effect_sd
                projected = pd.Series({
                    program: float(np.asarray(fit.params) @ vector)
                    for program, fit in fits.items()
                })
                equivalence_errors.append({
                    "contrast_id": spec["contrast_id"],
                    "max_abs_effect_error": float((observed - projected).abs().max()),
                })
            cnmf_manual_design_equivalence = pd.DataFrame(equivalence_errors)
            assert (cnmf_manual_design_equivalence.max_abs_effect_error < 1e-9).all()
            cnmf_manual_design_equivalence.to_csv(
                TABLEDIR / "cnmf_manual_design_equivalence_audit.csv", index=False
            )

            cnmf_loo_broad_parts = []
            cnmf_loo_targeted_parts = []
            cnmf_loo_fit_rows = []
            for omitted_library in day2_libraries:
                surviving_cells = hspc_cells[hspc_cells.Library != omitted_library].copy()
                clone_usage, metadata = aggregate_frozen_usage_by_clone(surviving_cells)
                assert set(metadata.mc_group) == {f"MC{mc}" for mc in MC_ORDER}
                scores = (clone_usage - clone_usage.mean(axis=0)) / clone_usage.std(axis=0, ddof=1)
                prepared, covariates, reference, dropped = prepare_usage_design_metadata(
                    metadata, "mc_group", f"cnmf_loo_{safe_name(omitted_library)}"
                )
                mc_levels = [f"MC{mc}" for mc in MC_ORDER]
                mc_loo_design = manual_usage_design(prepared, "mc_group", mc_levels, covariates)
                family_loo_design = manual_usage_design(
                    prepared, "lineage_bias_key", FAMILY_KEY_ORDER, covariates
                )
                mc_loo_fits = fit_manual_usage_models(scores, mc_loo_design)
                family_loo_fits = fit_manual_usage_models(scores, family_loo_design)
                broad_program_variance_ok = scores.std(axis=0, ddof=1).gt(1e-12)
                broad_rows = []
                for spec in contrast_specs:
                    factor = "mc_group" if spec["model"] == "metaclone" else "lineage_bias_key"
                    levels = mc_levels if spec["model"] == "metaclone" else FAMILY_KEY_ORDER
                    design = mc_loo_design if spec["model"] == "metaclone" else family_loo_design
                    fits = mc_loo_fits if spec["model"] == "metaclone" else family_loo_fits
                    vector = manual_contrast_vector(
                        design, factor, levels, spec["target_levels"], spec["reference_levels"]
                    )
                    qualifying_shared_libraries = (
                        loo_pair_support(metadata, spec["target_mc"], spec["reference_mc"])
                        if spec["tier"] == "within_lineage_pairwise"
                        else []
                    )
                    loo_support_gate = bool(qualifying_shared_libraries) if (
                        spec["tier"] == "within_lineage_pairwise"
                    ) else bool(spec["support_gate"])
                    for program, fit in fits.items():
                        test = fit.t_test(vector)
                        test_values = np.asarray([
                            np.asarray(test.effect).squeeze(),
                            np.asarray(test.sd).squeeze(),
                            np.asarray(test.pvalue).squeeze(),
                        ], dtype=float)
                        estimable = bool(
                            broad_program_variance_ok.loc[program]
                            and usage_fit_is_estimable(fit, len(scores))
                            and np.isfinite(test_values).all()
                        )
                        broad_rows.append({
                            "omitted_library": omitted_library,
                            "program": program, "model": spec["model"], "tier": spec["tier"],
                            "contrast_id": spec["contrast_id"],
                            "support_gate": loo_support_gate,
                            "qualifying_shared_libraries_ge3_per_arm": "|".join(
                                qualifying_shared_libraries
                            ),
                            "effect_sd": float(np.asarray(test.effect).squeeze()),
                            "standard_error": float(np.asarray(test.sd).squeeze()),
                            "pvalue": float(np.asarray(test.pvalue).squeeze()),
                            "estimable": estimable,
                        })
                cnmf_loo_broad_parts.append(pd.DataFrame(broad_rows))

                target_mask = prepared.metaclone.isin(TARGET_MCS)
                target_metadata = prepared.loc[target_mask].copy()
                target_usage = clone_usage.loc[target_metadata.index]
                target_scores = (
                    target_usage - target_usage.mean(axis=0)
                ) / target_usage.std(axis=0, ddof=1)
                target_prepared, target_covariates, target_reference, target_dropped = (
                    prepare_usage_design_metadata(
                        target_metadata, "multi_mc_group",
                        f"cnmf_target_loo_{safe_name(omitted_library)}",
                    )
                )
                target_levels = [f"MC{mc}" for mc in TARGET_MCS]
                target_loo_design = manual_usage_design(
                    target_prepared, "multi_mc_group", target_levels, target_covariates
                )
                target_loo_fits = fit_manual_usage_models(target_scores, target_loo_design)
                target_program_variance_ok = target_scores.std(axis=0, ddof=1).gt(1e-12)
                targeted_rows = []
                for spec in targeted_specs:
                    vector = manual_contrast_vector(
                        target_loo_design, "multi_mc_group", target_levels,
                        spec["target_levels"], spec["reference_levels"],
                    )
                    role = spec["role"]
                    if role == "member":
                        axis = next(axis for axis in BEHAVIOR_AXES if axis["axis_id"] == spec["axis_id"])
                        role = "member1" if int(spec["member_mc"]) == axis["focal_mcs"][0] else "member2"
                    for program, fit in target_loo_fits.items():
                        test = fit.t_test(vector)
                        test_values = np.asarray([
                            np.asarray(test.effect).squeeze(),
                            np.asarray(test.sd).squeeze(),
                            np.asarray(test.pvalue).squeeze(),
                        ], dtype=float)
                        estimable = bool(
                            target_program_variance_ok.loc[program]
                            and usage_fit_is_estimable(fit, len(target_scores))
                            and np.isfinite(test_values).all()
                        )
                        targeted_rows.append({
                            "omitted_library": omitted_library,
                            "analysis": spec["analysis"], "axis_id": spec["axis_id"],
                            "axis_label": spec["axis_label"], "role": role,
                            "member_mc": spec["member_mc"], "contrast_id": spec["contrast_id"],
                            "program": program,
                            "effect_sd": float(np.asarray(test.effect).squeeze()),
                            "standard_error": float(np.asarray(test.sd).squeeze()),
                            "pvalue": float(np.asarray(test.pvalue).squeeze()),
                            "estimable": estimable,
                        })
                cnmf_loo_targeted_parts.append(pd.DataFrame(targeted_rows))
                broad_frame = pd.DataFrame(broad_rows)
                targeted_frame = pd.DataFrame(targeted_rows)
                cnmf_loo_fit_rows.append({
                    "omitted_library": omitted_library,
                    "removed_assigned_cells": int((hspc_cells.Library == omitted_library).sum()),
                    "remaining_assigned_cells": int(len(surviving_cells)),
                    "remaining_clone_units": int(len(metadata)),
                    "remaining_targeted_clone_units": int(len(target_metadata)),
                    "broad_reference_library": reference,
                    "broad_library_covariates": "|".join(covariates),
                    "broad_rank_dropped": "|".join(dropped),
                    "target_reference_library": target_reference,
                    "target_library_covariates": "|".join(target_covariates),
                    "target_rank_dropped": "|".join(target_dropped),
                    "broad_mc_design_rank": int(np.linalg.matrix_rank(mc_loo_design)),
                    "broad_mc_design_columns": int(mc_loo_design.shape[1]),
                    "target_design_rank": int(np.linalg.matrix_rank(target_loo_design)),
                    "target_design_columns": int(target_loo_design.shape[1]),
                    "broad_all_programs_variable": bool(broad_program_variance_ok.all()),
                    "target_all_programs_variable": bool(target_program_variance_ok.all()),
                    "broad_all_fit_nobs_match": bool(all(
                        int(fit.nobs) == len(scores)
                        for fit in list(mc_loo_fits.values()) + list(family_loo_fits.values())
                    )),
                    "target_all_fit_nobs_match": bool(all(
                        int(fit.nobs) == len(target_scores)
                        for fit in target_loo_fits.values()
                    )),
                    "broad_all_tests_estimable": bool(broad_frame.estimable.all()),
                    "target_all_tests_estimable": bool(targeted_frame.estimable.all()),
                    "broad_outcome_index_matches_metadata": bool(
                        scores.index.tolist() == prepared.index.tolist()
                    ),
                    "target_outcome_index_matches_metadata": bool(
                        target_scores.index.tolist() == target_prepared.index.tolist()
                    ),
                    "estimable": bool(
                        broad_frame.estimable.all() and targeted_frame.estimable.all()
                    ),
                })
                print("clone-level cNMF inference LOO:", omitted_library)

            cnmf_library_loo_broad = pd.concat(cnmf_loo_broad_parts, ignore_index=True)
            cnmf_library_loo_targeted = pd.concat(cnmf_loo_targeted_parts, ignore_index=True)
            cnmf_library_loo_fit_audit = pd.DataFrame(cnmf_loo_fit_rows)
            assert len(cnmf_library_loo_broad) == 7 * 94 * CNMF_SELECTED_K
            assert len(cnmf_library_loo_targeted) == 7 * 14 * CNMF_SELECTED_K
            assert len(cnmf_library_loo_fit_audit) == len(day2_libraries) == 7
            assert cnmf_library_loo_fit_audit.estimable.notna().all()
            cnmf_library_loo_broad.to_csv(
                TABLEDIR / "cnmf_library_loo_broad_clone_effects.csv.gz",
                index=False, compression="gzip",
            )
            cnmf_library_loo_targeted.to_csv(
                TABLEDIR / "cnmf_library_loo_targeted_clone_effects.csv.gz",
                index=False, compression="gzip",
            )
            cnmf_library_loo_fit_audit.to_csv(
                TABLEDIR / "cnmf_library_loo_clone_model_audit.csv", index=False
            )

            broad_primary_direction = cnmf_broad_tests[[
                "contrast_id", "program", "effect_sd"
            ]].rename(columns={"effect_sd": "full_effect_sd"})
            broad_loo_check = cnmf_library_loo_broad.merge(
                broad_primary_direction, on=["contrast_id", "program"], validate="many_to_one"
            )
            broad_loo_check["direction_agrees"] = (
                np.sign(broad_loo_check.effect_sd) == np.sign(broad_loo_check.full_effect_sd)
            ) & (broad_loo_check.full_effect_sd != 0) & broad_loo_check.estimable
            cnmf_library_loo_broad_summary = (
                broad_loo_check.groupby(["contrast_id", "program"], observed=True)
                .agg(
                    n_library_loo=("omitted_library", "size"),
                    n_library_loo_estimable=("estimable", "sum"),
                    n_library_loo_supported=("support_gate", "sum"),
                    n_direction_agree=("direction_agrees", "sum"),
                    all_library_loo_effects_finite=("estimable", "all"),
                    all_library_loo_supported=("support_gate", "all"),
                    all_library_loo_same_direction=("direction_agrees", "all"),
                    min_abs_effect_sd=("effect_sd", lambda values: float(np.min(np.abs(values)))),
                ).reset_index()
            )
            cnmf_library_loo_broad_summary["complete_supported_stability"] = (
                cnmf_library_loo_broad_summary.n_library_loo.eq(len(day2_libraries))
                & cnmf_library_loo_broad_summary.all_library_loo_effects_finite
                & cnmf_library_loo_broad_summary.all_library_loo_supported
                & cnmf_library_loo_broad_summary.all_library_loo_same_direction
            )
            cnmf_library_loo_broad_summary.to_csv(
                TABLEDIR / "cnmf_library_loo_broad_effect_stability.csv", index=False
            )

            loo_primary = cnmf_library_loo_targeted[
                cnmf_library_loo_targeted.analysis == "primary_behavior_axis"
            ].merge(primary_direction, on=["axis_id", "program"], validate="many_to_one")
            loo_primary["direction_agrees"] = (
                (np.sign(loo_primary.effect_sd) == np.sign(loo_primary.combined_effect_sd))
                & (loo_primary.combined_effect_sd != 0) & loo_primary.estimable
            )
            cnmf_targeted_loo_stability = (
                loo_primary.groupby(["axis_id", "program"], observed=True)
                .agg(
                    library_loo_n_requested_effects=("direction_agrees", "size"),
                    library_loo_n_estimable_effects=("estimable", "sum"),
                    library_loo_n_direction_agree=("direction_agrees", "sum"),
                    library_loo_all_effects_finite=("estimable", "all"),
                    library_loo_all_estimable_same_direction=("direction_agrees", "all"),
                ).reset_index()
            )
            assert cnmf_targeted_loo_stability.library_loo_n_requested_effects.eq(21).all()
            display(cnmf_library_loo_fit_audit)
            """
        ),
        code(
            r"""
            # Clone prog_2-fraction covariate sensitivity on the exact full-data designs
            prog2_fraction = (
                pseudobulk_metadata.n_prog_2_cells
                / (pseudobulk_metadata.n_prog_1_cells + pseudobulk_metadata.n_prog_2_cells)
            ).astype(float)
            prog2_fraction_z = (
                prog2_fraction - prog2_fraction.mean()
            ) / prog2_fraction.std(ddof=0)

            def append_source_covariate(design, source_z):
                out = design.copy()
                out["prog2_fraction_postfreeze_z"] = source_z.loc[out.index].to_numpy(float)
                full_rank = np.linalg.matrix_rank(out.to_numpy(float)) == out.shape[1]
                assert full_rank
                return out

            mc_source_design = append_source_covariate(mc_design, prog2_fraction_z)
            family_source_design = append_source_covariate(family_design, prog2_fraction_z)
            targeted_source_design = append_source_covariate(
                targeted_program_design, prog2_fraction_z.loc[targeted_program_design.index]
            )
            source_fit_sets = {
                "metaclone": fit_manual_usage_models(cnmf_clone_scores, mc_source_design),
                "family": fit_manual_usage_models(cnmf_clone_scores, family_source_design),
            }
            source_sensitivity_rows = []
            for spec in contrast_specs:
                fits = source_fit_sets[spec["model"]]
                vector = np.append(np.asarray(spec["vector"], dtype=float), 0.0)
                for program, fit in fits.items():
                    test = fit.t_test(vector)
                    source_sensitivity_rows.append({
                        "scope": "broad", "contrast_id": spec["contrast_id"],
                        "tier": spec["tier"], "program": program,
                        "source_adjusted_effect_sd": float(np.asarray(test.effect).squeeze()),
                        "source_adjusted_pvalue": float(np.asarray(test.pvalue).squeeze()),
                    })
            targeted_source_fits = fit_manual_usage_models(
                targeted_cnmf_scores, targeted_source_design
            )
            for spec in targeted_specs:
                vector = np.append(np.asarray(spec["vector"], dtype=float), 0.0)
                for program, fit in targeted_source_fits.items():
                    test = fit.t_test(vector)
                    source_sensitivity_rows.append({
                        "scope": "targeted", "contrast_id": spec["contrast_id"],
                        "tier": spec["analysis"], "program": program,
                        "source_adjusted_effect_sd": float(np.asarray(test.effect).squeeze()),
                        "source_adjusted_pvalue": float(np.asarray(test.pvalue).squeeze()),
                    })
            cnmf_source_covariate_effects = pd.DataFrame(source_sensitivity_rows)
            primary_effect_lookup = pd.concat([
                cnmf_broad_tests[["contrast_id", "program", "effect_sd"]].assign(scope="broad"),
                targeted_cnmf_all[["contrast_id", "program", "effect_sd"]].assign(scope="targeted"),
            ], ignore_index=True).rename(columns={"effect_sd": "primary_effect_sd"})
            cnmf_source_covariate_effects = cnmf_source_covariate_effects.merge(
                primary_effect_lookup, on=["scope", "contrast_id", "program"],
                validate="one_to_one",
            )
            cnmf_source_covariate_effects["direction_agrees"] = (
                np.sign(cnmf_source_covariate_effects.source_adjusted_effect_sd)
                == np.sign(cnmf_source_covariate_effects.primary_effect_sd)
            ) & (cnmf_source_covariate_effects.primary_effect_sd != 0)
            cnmf_source_covariate_effects["absolute_effect_change"] = (
                cnmf_source_covariate_effects.source_adjusted_effect_sd
                - cnmf_source_covariate_effects.primary_effect_sd
            ).abs()
            cnmf_source_covariate_effects.to_csv(
                TABLEDIR / "cnmf_prog2_fraction_covariate_effect_sensitivity.csv", index=False
            )
            cnmf_source_design_audit = pd.DataFrame([
                ["broad_metaclone", len(mc_source_design), mc_source_design.shape[1],
                 np.linalg.matrix_rank(mc_source_design), np.linalg.cond(mc_source_design)],
                ["broad_lineage_family", len(family_source_design), family_source_design.shape[1],
                 np.linalg.matrix_rank(family_source_design), np.linalg.cond(family_source_design)],
                ["targeted_seven_MC", len(targeted_source_design), targeted_source_design.shape[1],
                 np.linalg.matrix_rank(targeted_source_design), np.linalg.cond(targeted_source_design)],
            ], columns=["model", "n_clone_units", "n_columns", "rank", "condition_number"])
            cnmf_source_design_audit.to_csv(
                TABLEDIR / "cnmf_prog2_fraction_covariate_design_audit.csv", index=False
            )
            display(cnmf_source_design_audit)
            display(
                cnmf_source_covariate_effects.groupby("scope", observed=True).agg(
                    n_effects=("program", "size"),
                    direction_agreement=("direction_agrees", "mean"),
                ).reset_index()
            )
            """
        ),
        md(
            r"""
            ### Post-fit annotation and conservative novelty test

            Annotation begins only now. The frozen top-100 genes are compared with
            all 50 fixed Hallmarks and a fixed hematopoietic reference assembled
            from two pre-existing, hash-locked research marker/module registries plus
            KEGG Hematopoietic Cell Lineage. The nearby Singh UCell program names
            are not used because their gene memberships are unavailable in this
            dataset. Technical/context flags take precedence over biological names.

            Post-fit overlap flags use global overlap FDR < 0.05 with at least
            three leading genes for cell-cycle, mitochondrial, ribosomal, and AP1
            stress sets, and at least two for sex-associated or hemoglobin sets.
            A library flag requires the metaclone-adjusted joint library test FDR
            < 0.05 and partial R-squared >= 0.03; a residual QC flag requires FDR
            < 0.05 and absolute rho >= 0.30 with detected-HVG complexity.

            A program is considered captured by the reference family if either
            (a) its top-100 overlap has global FDR < 0.05 with at least three genes,
            or (b) clone usage has positive Spearman rho >= 0.50 with any fixed
            Hallmark/hematopoietic score. Strong negative correlations are reported
            separately because closed fractional usages can induce anticorrelation;
            they are not treated as capture. A metaclone-associated program is called a
            conservative *not-captured candidate* only when it additionally has a
            metaclone effect passing one global BH correction across every supported
            MC-specific search test, retains that direction in all seven supported,
            finite library-removal refits and after `prog_2`-fraction adjustment,
            passes both program-recovery sensitivities, and has no technical or
            context flag. This is novelty relative to the audited reference panel,
            not proof of a new biological mechanism.
            """
        ),
        code(
            r"""
            # Hash-locked curated hematopoietic memberships, loaded only post-fit
            CURATED_MODULE_SOURCE = (
                PROJECT_ROOT / "analysis_workspace/paper_figures/metaclone_gene_analysis.py"
            )
            CURATED_MARKER_SOURCE = (
                PROJECT_ROOT
                / "analysis_workspace/analyses/singh_full_scrna_landscape_20260806/build_notebooks.py"
            )
            EXPECTED_CURATED_MODULE_SHA256 = "21438bd1152dc1863068873883d33843fb9bb6eb2f9563bda90ade95114dabc1"
            EXPECTED_CURATED_MARKER_SHA256 = "14662bfc08a1003e8400a80e727598a0355a03ba2eee204ecc16e97929a97a5a"
            assert sha256(CURATED_MODULE_SOURCE) == EXPECTED_CURATED_MODULE_SHA256
            assert sha256(CURATED_MARKER_SOURCE) == EXPECTED_CURATED_MARKER_SHA256

            def literal_assignments(path, variable):
                outer_tree = ast.parse(path.read_text())
                trees = [outer_tree]
                for node in ast.walk(outer_tree):
                    if (
                        isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and variable in node.value
                    ):
                        try:
                            trees.append(ast.parse(dedent(node.value)))
                        except SyntaxError:
                            pass
                values = []
                for tree in trees:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            if any(
                                isinstance(target, ast.Name) and target.id == variable
                                for target in node.targets
                            ):
                                try:
                                    values.append(ast.literal_eval(node.value))
                                except Exception:
                                    pass
                return values

            curated_modules = literal_assignments(CURATED_MODULE_SOURCE, "CURATED_MODULES")
            assert len(curated_modules) == 1
            curated_modules = curated_modules[0]
            marker_candidates = literal_assignments(CURATED_MARKER_SOURCE, "marker_panel")
            marker_candidates = [
                value for value in marker_candidates
                if isinstance(value, dict) and {"HSC", "MPP", "Mk", "GMP"}.issubset(value)
            ]
            assert len(marker_candidates) >= 1
            curated_marker_panel = marker_candidates[0]

            curated_hematopoietic_sets = {
                f"Module::{name}": {str(gene).upper() for gene in genes}
                for name, genes in curated_modules.items()
            }
            curated_hematopoietic_sets.update({
                f"State_marker::{state}": {str(gene).upper() for gene in genes}
                for state, genes in curated_marker_panel.items()
            })
            curated_hematopoietic_sets["KEGG::Hematopoietic_cell_lineage"] = set(
                kegg_sets[hematopoietic_term]
            )
            curated_membership_rows = []
            for program, genes in curated_hematopoietic_sets.items():
                if program.startswith("Module::"):
                    source = str(CURATED_MODULE_SOURCE.relative_to(PROJECT_ROOT))
                    source_hash = EXPECTED_CURATED_MODULE_SHA256
                elif program.startswith("State_marker::"):
                    source = str(CURATED_MARKER_SOURCE.relative_to(PROJECT_ROOT))
                    source_hash = EXPECTED_CURATED_MARKER_SHA256
                else:
                    source = str(KEGG_SOURCE.relative_to(PROJECT_ROOT))
                    source_hash = EXPECTED_KEGG_SHA256
                for gene in sorted(genes):
                    curated_membership_rows.append({
                        "program": program, "gene_upper": gene,
                        "source_path": source, "source_sha256": source_hash,
                        "loaded_after_cnmf_freeze": True,
                    })
            curated_hematopoietic_membership = pd.DataFrame(curated_membership_rows)
            curated_hematopoietic_membership.to_csv(
                RESOURCEDIR / "curated_hematopoietic_program_membership_20260909.csv",
                index=False,
            )
            curated_resource_audit = (
                curated_hematopoietic_membership.groupby(
                    ["program", "source_path", "source_sha256"], observed=True
                ).gene_upper.nunique().rename("n_genes").reset_index()
            )
            curated_resource_audit.to_csv(
                REPORTDIR / "cnmf_curated_hematopoietic_resource_audit.csv", index=False
            )
            display(curated_resource_audit)
            """
        ),
        code(
            r"""
            # Top-gene overlap, clone-score correlations, and technical/context flags
            full_upper = np.char.upper(cnmf_full_genes.astype(str))
            universe = set(full_upper)
            upper_to_display = {}
            for gene, upper in zip(cnmf_full_genes, full_upper):
                upper_to_display.setdefault(upper, gene)

            reference_sets = {}
            for name, genes in hallmark_sets.items():
                reference_sets[("Hallmark", name)] = set(genes) & universe
            for name, genes in curated_hematopoietic_sets.items():
                reference_sets[("Curated_hematopoietic", name)] = set(genes) & universe
            technical_reference_sets = {
                "Mitochondrial": {
                    upper for gene, upper in zip(cnmf_full_genes, full_upper)
                    if str(gene).lower().startswith("mt-")
                },
                "Ribosomal": KEGG_RIBOSOME_GENES & universe,
                "Sex_associated": SEX_AUDIT_GENES & universe,
                "Hemoglobin": {
                    upper for upper in universe if HEMOGLOBIN_PATTERN.match(upper)
                },
                "Nuclear_QC_Malat1": {"MALAT1"} & universe,
            }
            for name, genes in technical_reference_sets.items():
                reference_sets[("Technical_audit", name)] = genes

            top100_by_program = {
                program: set(
                    cnmf_leading_genes.loc[
                        cnmf_leading_genes.program == program, "gene"
                    ].astype(str).str.upper()
                )
                for program in cnmf_program_ids
            }
            overlap_rows = []
            universe_size = len(universe)
            for program in cnmf_program_ids:
                top = top100_by_program[program] & universe
                for (reference_type, reference), genes in reference_sets.items():
                    overlap = top & genes
                    pvalue = float(hypergeom.sf(
                        len(overlap) - 1, universe_size, len(genes), len(top)
                    )) if genes else 1.0
                    overlap_rows.append({
                        "program": program, "reference_type": reference_type,
                        "reference_program": reference,
                        "universe_genes": universe_size,
                        "reference_genes_in_universe": len(genes),
                        "leading_genes": len(top), "overlap_n": len(overlap),
                        "jaccard": len(overlap) / max(len(top | genes), 1),
                        "hypergeom_pvalue": pvalue,
                        "overlap_genes": "|".join(sorted(overlap)),
                    })
            cnmf_reference_overlap = pd.DataFrame(overlap_rows)
            cnmf_reference_overlap["global_fdr"] = bh_adjust(
                cnmf_reference_overlap.hypergeom_pvalue.to_numpy(float)
            )
            cnmf_reference_overlap.to_csv(
                TABLEDIR / "cnmf_program_reference_gene_overlap.csv", index=False
            )

            # Clone-level scores for the fixed curated memberships use the same
            # DESeq-normalized, gene-z matrix as the existing Hallmark analysis.
            curated_score_data = {}
            curated_mapping_rows = []
            for reference, genes in curated_hematopoietic_sets.items():
                mapped = sorted({
                    index for gene in genes for index in upper_to_indices.get(gene, [])
                })
                curated_mapping_rows.append({
                    "reference_program": reference, "n_requested": len(genes),
                    "n_mapped_filtered_genes": len(mapped),
                })
                if len(mapped) >= 2:
                    curated_score_data[reference] = gene_z[:, mapped].mean(axis=1)
            curated_clone_scores = pd.DataFrame(
                curated_score_data, index=pseudobulk_metadata.index
            )
            curated_clone_scores = (
                curated_clone_scores - curated_clone_scores.mean(axis=0)
            ) / curated_clone_scores.std(axis=0, ddof=1)
            pd.DataFrame(curated_mapping_rows).to_csv(
                TABLEDIR / "cnmf_curated_hematopoietic_score_mapping.csv", index=False
            )
            curated_clone_scores.to_csv(
                TABLEDIR / "cnmf_curated_hematopoietic_clone_scores.csv.gz",
                compression="gzip",
            )

            correlation_rows = []
            capture_score_families = {
                "Hallmark": program_scores,
                "Curated_hematopoietic": curated_clone_scores,
            }
            for cnmf_program in cnmf_program_ids:
                for reference_type, reference_scores in capture_score_families.items():
                    for reference_program in reference_scores.columns:
                        rho, pvalue = spearmanr(
                            cnmf_clone_scores[cnmf_program],
                            reference_scores[reference_program],
                        )
                        correlation_rows.append({
                            "program": cnmf_program, "reference_type": reference_type,
                            "reference_program": reference_program,
                            "spearman_rho": float(rho), "pvalue": float(pvalue),
                        })
            cnmf_reference_correlations = pd.DataFrame(correlation_rows)
            cnmf_reference_correlations["global_fdr"] = bh_adjust(
                cnmf_reference_correlations.pvalue.to_numpy(float)
            )
            cnmf_reference_correlations.to_csv(
                TABLEDIR / "cnmf_program_clone_score_reference_correlations.csv", index=False
            )

            # Library-fraction joint association and residual QC association both
            # use one record per clone and adjust for current metaclone.
            library_column_indices = [
                mc_design.columns.get_loc(column) for column in library_covariates
            ]
            joint_rows = []
            reduced_columns = [
                index for index, column in enumerate(mc_design.columns)
                if column not in library_covariates
            ]
            selected_detected_per_cell = np.asarray(
                (discovery_counts[:, cnmf_hvg_indices] > 0).sum(axis=1)
            ).ravel()
            detected_series = pd.Series(selected_detected_per_cell, index=discovery_cells)
            clone_detected = detected_series.loc[hspc_cells.index].groupby(
                hspc_cells.LARRY_ID.astype(str), observed=True
            ).mean().reindex(pseudobulk_metadata.index)
            qc_residual = sm.OLS(
                clone_detected.to_numpy(float), mc_design.to_numpy(float)
            ).fit().resid
            for program, fit in cnmf_fits_mc.items():
                restriction = np.zeros((len(library_column_indices), mc_design.shape[1]))
                for row_index, column_index in enumerate(library_column_indices):
                    restriction[row_index, column_index] = 1.0
                joint_test = fit.f_test(restriction)
                reduced_fit = sm.OLS(
                    cnmf_clone_scores[program].to_numpy(float),
                    mc_design.to_numpy(float)[:, reduced_columns],
                ).fit()
                partial_r2 = max(
                    0.0, float((reduced_fit.ssr - fit.ssr) / max(reduced_fit.ssr, 1e-12))
                )
                program_residual = sm.OLS(
                    cnmf_clone_scores[program].to_numpy(float), mc_design.to_numpy(float)
                ).fit().resid
                qc_rho, qc_pvalue = spearmanr(program_residual, qc_residual)
                joint_rows.append({
                    "program": program,
                    "library_joint_pvalue": float(np.asarray(joint_test.pvalue).squeeze()),
                    "library_partial_r2": partial_r2,
                    "residual_detected_HVG_spearman_rho": float(qc_rho),
                    "residual_detected_HVG_pvalue": float(qc_pvalue),
                })
            cnmf_technical_association = pd.DataFrame(joint_rows)
            cnmf_technical_association["library_joint_fdr"] = bh_adjust(
                cnmf_technical_association.library_joint_pvalue.to_numpy(float)
            )
            cnmf_technical_association["residual_detected_HVG_fdr"] = bh_adjust(
                cnmf_technical_association.residual_detected_HVG_pvalue.to_numpy(float)
            )
            cnmf_technical_association.to_csv(
                TABLEDIR / "cnmf_program_technical_association_audit.csv", index=False
            )

            def normalized_program_label(value):
                return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")

            CELL_CYCLE_HALLMARK_KEYS = {
                "E2F_TARGETS", "G2_M_CHECKPOINT", "G2M_CHECKPOINT",
                "MITOTIC_SPINDLE",
            }
            # Fail closed if a future resource/display-name change would make the
            # annotation rules silently match no fixed reference sets.
            hallmark_reference_labels = (
                cnmf_reference_overlap.loc[
                    cnmf_reference_overlap.reference_type == "Hallmark",
                    "reference_program",
                ].drop_duplicates()
            )
            normalized_hallmark_labels = set(
                hallmark_reference_labels.map(normalized_program_label)
            )
            required_hallmark_labels = {
                "E2F_TARGETS", "G2_M_CHECKPOINT", "MITOTIC_SPINDLE",
            }
            assert required_hallmark_labels.issubset(normalized_hallmark_labels), (
                required_hallmark_labels - normalized_hallmark_labels
            )
            curated_reference_labels = set(
                cnmf_reference_overlap.loc[
                    cnmf_reference_overlap.reference_type == "Curated_hematopoietic",
                    "reference_program",
                ].drop_duplicates()
            )
            assert "Module::AP1_early_response" in curated_reference_labels
            technical_reference_labels = set(
                cnmf_reference_overlap.loc[
                    cnmf_reference_overlap.reference_type == "Technical_audit",
                    "reference_program",
                ].drop_duplicates()
            )
            required_technical_labels = {
                "Mitochondrial", "Ribosomal", "Sex_associated", "Hemoglobin",
                "Nuclear_QC_Malat1",
            }
            assert required_technical_labels.issubset(technical_reference_labels), (
                required_technical_labels - technical_reference_labels
            )
            program_annotation_rows = []
            for program in cnmf_program_ids:
                overlaps = cnmf_reference_overlap[cnmf_reference_overlap.program == program]
                technical = overlaps[overlaps.reference_type == "Technical_audit"].set_index(
                    "reference_program"
                )
                biological = overlaps[overlaps.reference_type != "Technical_audit"].copy()
                best = biological.sort_values(
                    ["global_fdr", "hypergeom_pvalue", "reference_type", "reference_program"]
                ).iloc[0]
                assoc = cnmf_technical_association.set_index("program").loc[program]
                def tech_overlap(name, minimum):
                    return bool(
                        name in technical.index
                        and technical.loc[name, "global_fdr"] < 0.05
                        and technical.loc[name, "overlap_n"] >= minimum
                    )
                cell_cycle_rows = overlaps[
                    (overlaps.reference_type == "Hallmark")
                    & overlaps.reference_program.map(normalized_program_label).isin(
                        CELL_CYCLE_HALLMARK_KEYS
                    )
                ]
                cell_cycle = bool(
                    ((cell_cycle_rows.global_fdr < 0.05)
                     & (cell_cycle_rows.overlap_n >= 3)).any()
                )
                stress_rows = overlaps[
                    (overlaps.reference_type == "Curated_hematopoietic")
                    & (overlaps.reference_program == "Module::AP1_early_response")
                ]
                stress = bool(
                    len(stress_rows)
                    and (stress_rows.global_fdr.iloc[0] < 0.05)
                    and (stress_rows.overlap_n.iloc[0] >= 3)
                )
                flags = {
                    "cell_cycle_flag": cell_cycle,
                    "mitochondrial_flag": tech_overlap("Mitochondrial", 3),
                    "ribosomal_flag": tech_overlap("Ribosomal", 3),
                    "sex_associated_flag": tech_overlap("Sex_associated", 2),
                    "hemoglobin_context_flag": tech_overlap("Hemoglobin", 2),
                    "nuclear_qc_malat1_flag": tech_overlap("Nuclear_QC_Malat1", 1),
                    "stress_immediate_early_flag": stress,
                    "library_associated_flag": bool(
                        assoc.library_joint_fdr < 0.05 and assoc.library_partial_r2 >= 0.03
                    ),
                    "qc_associated_flag": bool(
                        assoc.residual_detected_HVG_fdr < 0.05
                        and abs(assoc.residual_detected_HVG_spearman_rho) >= 0.30
                    ),
                }
                flagged_names = [name.replace("_flag", "") for name, value in flags.items() if value]
                any_flag = bool(flagged_names)
                biological_supported = bool(best.global_fdr < 0.05 and best.overlap_n >= 3)
                annotation = (
                    "Flagged: " + "; ".join(flagged_names)
                    if any_flag else
                    (f"{best.reference_type}: {best.reference_program}" if biological_supported
                     else "Unannotated by fixed references")
                )
                program_annotation_rows.append({
                    "program": program,
                    **flags,
                    "any_technical_or_context_flag": any_flag,
                    "best_reference_type": best.reference_type,
                    "best_reference_program": best.reference_program,
                    "best_reference_overlap_n": int(best.overlap_n),
                    "best_reference_global_fdr": float(best.global_fdr),
                    "postfit_annotation": annotation,
                })
            cnmf_program_annotations = pd.DataFrame(program_annotation_rows)
            expected_cell_cycle_programs = set(
                cnmf_reference_overlap[
                    (cnmf_reference_overlap.reference_type == "Hallmark")
                    & cnmf_reference_overlap.reference_program.map(
                        normalized_program_label
                    ).isin(CELL_CYCLE_HALLMARK_KEYS)
                    & (cnmf_reference_overlap.global_fdr < 0.05)
                    & (cnmf_reference_overlap.overlap_n >= 3)
                ].program
            )
            observed_cell_cycle_programs = set(
                cnmf_program_annotations.loc[
                    cnmf_program_annotations.cell_cycle_flag, "program"
                ]
            )
            assert observed_cell_cycle_programs == expected_cell_cycle_programs
            cnmf_program_annotations.to_csv(
                TABLEDIR / "cnmf_postfit_program_annotations.csv", index=False
            )
            display(cnmf_program_annotations)
            """
        ),
        code(
            r"""
            # Complete targeted shared-signature gates and conservative novelty decision
            cnmf_targeted_signatures = (
                cnmf_targeted_signatures
                .merge(cnmf_comparator_stability, on=["axis_id", "program"], validate="one_to_one")
                .merge(cnmf_targeted_loo_stability, on=["axis_id", "program"], validate="one_to_one")
            )
            cnmf_targeted_signatures["combined_global_fdr_pass"] = (
                cnmf_targeted_signatures.combined_global_fdr < 0.05
            )
            cnmf_targeted_signatures["conjunction_global_fdr_pass"] = (
                cnmf_targeted_signatures.conjunction_global_fdr < 0.05
            )
            cnmf_targeted_signatures["shared_behavioral_signature"] = (
                cnmf_targeted_signatures.combined_global_fdr_pass
                & cnmf_targeted_signatures.members_same_direction_as_combined
                & cnmf_targeted_signatures.conjunction_global_fdr_pass
                & cnmf_targeted_signatures.comparator_drop_all_same_direction
                & cnmf_targeted_signatures.library_loo_all_estimable_same_direction
            )
            cnmf_targeted_signatures["pooled_only_candidate"] = (
                cnmf_targeted_signatures.combined_global_fdr_pass
                & ~cnmf_targeted_signatures.shared_behavioral_signature
            )
            cnmf_targeted_signatures["pooled_only_failure_reason"] = np.select(
                [
                    ~cnmf_targeted_signatures.combined_global_fdr_pass,
                    ~cnmf_targeted_signatures.members_same_direction_as_combined,
                    ~cnmf_targeted_signatures.conjunction_global_fdr_pass,
                    ~cnmf_targeted_signatures.comparator_drop_all_same_direction,
                    ~cnmf_targeted_signatures.library_loo_all_estimable_same_direction,
                ],
                [
                    "combined_global_fdr_not_supported",
                    "member_direction_disagreement",
                    "global_max_p_conjunction_not_supported",
                    "comparator_drop_direction_unstable",
                    "library_loo_direction_unstable",
                ],
                default="all_shared_signature_gates_pass",
            )
            cnmf_targeted_signatures.to_csv(
                TABLEDIR / "cnmf_targeted_behavior_program_signatures.csv", index=False
            )

            recovery_wide = cnmf_program_recovery_summary.pivot(
                index="program", columns="sensitivity", values="recovery_supported"
            ).rename(columns={
                "leave_one_library_out": "library_recovery_supported",
                "within_library_prog1_prog2_balance": "source_balance_recovery_supported",
            }).reset_index()
            capture_correlation = (
                cnmf_reference_correlations.sort_values(
                    ["program", "spearman_rho", "reference_type", "reference_program"],
                    ascending=[True, False, True, True],
                ).groupby("program", observed=True).head(1)[[
                    "program", "reference_type", "reference_program", "spearman_rho"
                ]].rename(columns={
                    "reference_type": "max_positive_correlation_reference_type",
                    "reference_program": "max_positive_correlation_reference_program",
                    "spearman_rho": "max_positive_reference_rho",
                })
            )
            inverse_correlation = (
                cnmf_reference_correlations.sort_values(
                    ["program", "spearman_rho", "reference_type", "reference_program"],
                    ascending=[True, True, True, True],
                ).groupby("program", observed=True).head(1)[[
                    "program", "reference_type", "reference_program", "spearman_rho"
                ]].rename(columns={
                    "reference_type": "most_negative_correlation_reference_type",
                    "reference_program": "most_negative_correlation_reference_program",
                    "spearman_rho": "most_negative_reference_rho",
                })
            )
            capture_overlap = (
                cnmf_reference_overlap[
                    cnmf_reference_overlap.reference_type != "Technical_audit"
                ].sort_values(
                    ["program", "global_fdr", "overlap_n", "reference_program"],
                    ascending=[True, True, False, True],
                ).groupby("program", observed=True).head(1)[[
                    "program", "reference_type", "reference_program", "overlap_n",
                    "jaccard", "global_fdr",
                ]].rename(columns={
                    "reference_type": "best_overlap_reference_type",
                    "reference_program": "best_overlap_reference_program",
                    "overlap_n": "best_overlap_n", "jaccard": "best_overlap_jaccard",
                    "global_fdr": "best_overlap_global_fdr",
                })
            )
            metaclone_tiers = {
                "all_metaclones_ovr", "within_lineage_ovr", "within_lineage_pairwise"
            }
            best_metaclone_effect = (
                cnmf_broad_tests[
                    cnmf_broad_tests.tier.isin(metaclone_tiers)
                    & cnmf_broad_tests.support_gate
                ].sort_values(
                    ["program", "metaclone_search_global_fdr", "pvalue", "contrast_id"]
                ).groupby("program", observed=True).head(1)[[
                    "program", "contrast_id", "contrast_label", "tier", "effect_sd",
                    "pvalue", "tier_fdr", "metaclone_search_global_fdr",
                ]].rename(columns={
                    "contrast_id": "best_metaclone_contrast_id",
                    "contrast_label": "best_metaclone_contrast_label",
                    "tier": "best_metaclone_tier",
                    "effect_sd": "best_metaclone_effect_sd",
                    "pvalue": "best_metaclone_pvalue",
                    "tier_fdr": "best_metaclone_tier_fdr",
                    "metaclone_search_global_fdr": "best_metaclone_search_global_fdr",
                })
            )
            novelty = (
                pd.DataFrame({"program": cnmf_program_ids})
                .merge(cnmf_program_annotations, on="program", validate="one_to_one")
                .merge(recovery_wide, on="program", validate="one_to_one")
                .merge(capture_correlation, on="program", validate="one_to_one")
                .merge(inverse_correlation, on="program", validate="one_to_one")
                .merge(capture_overlap, on="program", validate="one_to_one")
                .merge(best_metaclone_effect, on="program", validate="one_to_one")
            )
            novelty = novelty.merge(
                cnmf_library_loo_broad_summary.rename(columns={
                    "contrast_id": "best_metaclone_contrast_id"
                })[[
                    "best_metaclone_contrast_id", "program",
                    "all_library_loo_same_direction", "all_library_loo_effects_finite",
                    "all_library_loo_supported", "complete_supported_stability",
                ]],
                on=["best_metaclone_contrast_id", "program"], validate="one_to_one",
            )
            source_best = cnmf_source_covariate_effects[
                cnmf_source_covariate_effects.scope == "broad"
            ].rename(columns={"contrast_id": "best_metaclone_contrast_id"})[[
                "best_metaclone_contrast_id", "program", "direction_agrees",
            ]].rename(columns={"direction_agrees": "source_covariate_direction_agrees"})
            novelty = novelty.merge(
                source_best, on=["best_metaclone_contrast_id", "program"],
                validate="one_to_one",
            )
            novelty["captured_by_clone_score_correlation"] = (
                novelty.max_positive_reference_rho >= 0.50
            )
            novelty["captured_by_gene_overlap"] = (
                (novelty.best_overlap_global_fdr < 0.05)
                & (novelty.best_overlap_n >= 3)
            )
            novelty["metaclone_association_supported"] = (
                novelty.best_metaclone_search_global_fdr < 0.05
            )
            novelty["novel_not_captured_candidate"] = (
                novelty.metaclone_association_supported
                & ~novelty.captured_by_clone_score_correlation
                & ~novelty.captured_by_gene_overlap
                & novelty.library_recovery_supported
                & novelty.source_balance_recovery_supported
                & novelty.complete_supported_stability
                & novelty.source_covariate_direction_agrees
                & ~novelty.any_technical_or_context_flag
            )
            novelty["novelty_decision_reason"] = np.select(
                [
                    ~novelty.metaclone_association_supported,
                    novelty.any_technical_or_context_flag,
                    novelty.captured_by_clone_score_correlation,
                    novelty.captured_by_gene_overlap,
                    ~novelty.library_recovery_supported,
                    ~novelty.source_balance_recovery_supported,
                    ~novelty.all_library_loo_effects_finite,
                    ~novelty.all_library_loo_supported,
                    ~novelty.all_library_loo_same_direction,
                    ~novelty.source_covariate_direction_agrees,
                ],
                [
                    "no multiplicity-controlled metaclone association",
                    "technical/context flag takes precedence",
                    "captured by fixed clone-score correlation",
                    "captured by fixed gene-set overlap",
                    "library-removal program recovery failed",
                    "prog-composition program recovery failed",
                    "library-removal effect not finite/estimable",
                    "library-removal support gate failed",
                    "library-removal effect direction unstable",
                    "prog2-fraction-adjusted effect direction changed",
                ],
                default="passes conservative not-captured criteria",
            )
            cnmf_novelty_assessment = novelty
            cnmf_novelty_assessment.to_csv(
                TABLEDIR / "cnmf_metaclone_program_novelty_assessment.csv", index=False
            )
            display(cnmf_novelty_assessment[[
                "program", "postfit_annotation", "best_metaclone_contrast_label",
                "best_metaclone_effect_sd", "best_metaclone_search_global_fdr",
                "max_positive_reference_rho", "most_negative_reference_rho",
                "best_overlap_global_fdr",
                "novel_not_captured_candidate", "novelty_decision_reason",
            ]])
            """
        ),
        code(
            r"""
            # CNMF_ANNOTATION_PLOT_INLINE_V1 — reference capture and flags
            from adjustText import adjust_text

            plot_stem = "cnmf_05_reference_capture_and_novelty"
            plot_data = cnmf_novelty_assessment.copy()
            plot_data["max_positive_reference_rho_plot"] = (
                plot_data.max_positive_reference_rho
            )
            plot_data["overlap_strength"] = -np.log10(
                np.clip(plot_data.best_overlap_global_fdr.to_numpy(float), 1e-300, 1.0)
            )
            plot_data["plot_class"] = np.select(
                [
                    plot_data.novel_not_captured_candidate,
                    plot_data.any_technical_or_context_flag,
                    (
                        plot_data.metaclone_association_supported
                        & (
                            plot_data.captured_by_clone_score_correlation
                            | plot_data.captured_by_gene_overlap
                        )
                    ),
                    plot_data.metaclone_association_supported,
                ],
                [
                    "Not-captured candidate", "Technical/context flag",
                    "Metaclone-associated; captured",
                    "Associated; uncaptured but stability-limited",
                ],
                default="No supported metaclone association",
            )
            plot_data.to_csv(TABLEDIR / f"{plot_stem}_plot_data.csv", index=False)

            class_colors = {
                "Not-captured candidate": "#3F9E4D",
                "Technical/context flag": "#B23A48",
                "Metaclone-associated; captured": ORANGE,
                "Associated; uncaptured but stability-limited": "#7A5AA6",
                "No supported metaclone association": GREY,
            }
            fig, ax = plt.subplots(figsize=(7.4, 5.6))
            label_texts = []
            for plot_class, frame in plot_data.groupby("plot_class", sort=False):
                ax.scatter(
                    frame.max_positive_reference_rho_plot, frame.overlap_strength,
                    s=65, color=class_colors[plot_class], alpha=0.85,
                    edgecolors=WHITE, linewidths=0.6, label=plot_class,
                )
                for row in frame.itertuples(index=False):
                    label_texts.append(ax.text(
                        row.max_positive_reference_rho_plot,
                        row.overlap_strength,
                        row.program.rsplit("_", 1)[-1],
                        fontsize=7, color=INK,
                    ))
            ax.axvline(0.50, color=INK, linestyle="--", linewidth=1.0,
                       label="Clone-score capture rho=0.50")
            ax.axhline(-np.log10(0.05), color="#7A5AA6", linestyle=":", linewidth=1.0,
                       label="Gene-overlap global FDR=0.05")
            ax.set(xlabel="Maximum positive Spearman rho with fixed program scores",
                   ylabel="Best fixed-set overlap (-log10 global FDR)")
            ax.set_title("Reference capture is evaluated only after cNMF programs freeze", loc="left")
            ax.grid(color=LIGHT_GREY, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=7.2, loc="best")
            np.random.seed(CNMF_ROOT_SEED + 5)
            adjust_text(
                label_texts, ax=ax, ensure_inside_axes=True,
                expand=(1.25, 1.35), force_text=(0.35, 0.55),
                force_static=(0.15, 0.25),
                arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.5},
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
            # Computed cNMF findings and interpretation boundaries
            selected_diagnostics = cnmf_k_diagnostics.loc[
                cnmf_k_diagnostics.selected_K
            ].iloc[0]
            broad_supported_hits = cnmf_broad_tests[
                cnmf_broad_tests.support_gate & (cnmf_broad_tests.tier_fdr < 0.05)
            ].sort_values(["tier_fdr", "pvalue"])
            targeted_combined_hits = cnmf_targeted_signatures[
                cnmf_targeted_signatures.combined_global_fdr < 0.05
            ].sort_values("combined_global_fdr")
            targeted_shared_hits = cnmf_targeted_signatures[
                cnmf_targeted_signatures.shared_behavioral_signature
            ].sort_values("combined_global_fdr")
            targeted_hit_audit = (
                targeted_combined_hits
                .merge(
                    cnmf_program_annotations[[
                        "program", "postfit_annotation",
                        "any_technical_or_context_flag",
                        "library_associated_flag",
                    ]],
                    on="program", validate="many_to_one",
                )
                .merge(recovery_wide, on="program", validate="many_to_one")
            )
            targeted_hit_audit["fully_stable_unflagged_shared_signal"] = (
                targeted_hit_audit.shared_behavioral_signature
                & targeted_hit_audit.library_recovery_supported
                & targeted_hit_audit.source_balance_recovery_supported
                & ~targeted_hit_audit.any_technical_or_context_flag
            )
            targeted_hit_audit.to_csv(
                TABLEDIR / "cnmf_targeted_combined_hit_interpretation.csv", index=False
            )
            novel_candidates = cnmf_novelty_assessment[
                cnmf_novelty_assessment.novel_not_captured_candidate
            ].sort_values("best_metaclone_tier_fdr")

            finding_lines = [
                "INDEPENDENT DE NOVO cNMF PROGRAM FAMILY — COMPUTED FINDINGS",
                "",
                "DISCOVERY (LABEL-BLIND)",
                f"- Frozen K={CNMF_SELECTED_K}; selection mode={selection_mode}; "
                f"eligible K={','.join(map(str, eligible_k.K.tolist()))}; "
                f"stability={selected_diagnostics.consensus_silhouette_cosine:.3f}; "
                f"worst-cluster stability={selected_diagnostics.minimum_cluster_mean_silhouette:.3f}; "
                f"consensus relative error={selected_diagnostics.consensus_relative_error:.4f}.",
                f"- {CNMF_N_RESTARTS} distinct random restarts per K; "
                f"selected-K convergence={selected_diagnostics.converged_fraction:.1%}; "
                f"minimum restart coverage={selected_diagnostics.min_restart_coverage_fraction:.1%}.",
                f"- Discovery used 1,041 cells / 784 clones and 2,000 audited genes; "
                f"synthetic readout features found={int(synthetic_flag.sum())}.",
                "",
                "CLONE-LEVEL METACLONE INFERENCE",
                f"- Broad supported tier-FDR<0.05 effects: {len(broad_supported_hits)} "
                f"of {int(cnmf_broad_tests.support_gate.sum())} supported program-contrast tests.",
                f"- Targeted combined global-FDR<0.05 effects: {len(targeted_combined_hits)} "
                f"of {len(cnmf_targeted_signatures)}; full shared-signature gate: "
                f"{len(targeted_shared_hits)}.",
                f"- Post-freeze program recovery: library-removal gate "
                f"{int(recovery_wide.library_recovery_supported.sum())}/{CNMF_SELECTED_K}; "
                f"within-library prog_1/prog_2-balance gate "
                f"{int(recovery_wide.source_balance_recovery_supported.sum())}/{CNMF_SELECTED_K}.",
            ]
            if len(targeted_hit_audit):
                finding_lines.append("- Targeted combined hits:")
                for row in targeted_hit_audit.itertuples(index=False):
                    finding_lines.append(
                        f"  - {row.axis_label} / {row.program}: effect={row.combined_effect_sd:+.3f} SD, "
                        f"combined q={row.combined_global_fdr:.3g}, conjunction q={row.conjunction_global_fdr:.3g}, "
                        f"shared={bool(row.shared_behavioral_signature)}, "
                        f"annotation={row.postfit_annotation}, "
                        f"library-associated={bool(row.library_associated_flag)}, "
                        f"library-recovery={bool(row.library_recovery_supported)}, "
                        f"source-balance-recovery={bool(row.source_balance_recovery_supported)}, "
                        f"fully stable/unflagged={bool(row.fully_stable_unflagged_shared_signal)}."
                    )
                finding_lines.append(
                    f"- Targeted hits passing shared behavior, both program-recovery gates, "
                    f"and no technical/context flag: "
                    f"{int(targeted_hit_audit.fully_stable_unflagged_shared_signal.sum())}."
                )
            else:
                finding_lines.append("- No targeted combined effect passes global FDR 0.05.")
            finding_lines.extend([
                "",
                "POST-FIT ANNOTATION AND REFERENCE CAPTURE",
                f"- Programs with a technical/context flag: "
                f"{int(cnmf_program_annotations.any_technical_or_context_flag.sum())}/{CNMF_SELECTED_K}.",
                f"- Conservative metaclone-associated not-captured candidates: "
                f"{len(novel_candidates)}/{CNMF_SELECTED_K}.",
            ])
            if len(novel_candidates):
                for row in novel_candidates.itertuples(index=False):
                    finding_lines.append(
                        f"  - {row.program}: {row.best_metaclone_contrast_label}; "
                        f"effect={row.best_metaclone_effect_sd:+.3f} SD, global search q={row.best_metaclone_search_global_fdr:.3g}; "
                        f"max positive fixed-score rho={row.max_positive_reference_rho:.3f}."
                    )
            else:
                finding_lines.append(
                    "  - None pass the complete association, capture, technical, library, and source-composition gates."
                )
            finding_lines.extend([
                "",
                "INTERPRETATION BOUNDARIES",
                "- cNMF discovery is cell-resolved, but every comparison and p-value uses one LARRY-clone usage value.",
                "- Program usage is compositional; increasing one fractional usage necessarily reduces others.",
                "- Metaclone and behavior-axis labels summarize later realized descendants; associations are retrospective.",
                "- No classifier or prediction model is fitted, and no result shows that a Day-2 program prospectively determines fate.",
                "- Novelty is assessed only against the fixed, hash-audited Hallmark/hematopoietic reference family.",
            ])
            cnmf_findings_text = "\n".join(finding_lines)
            (REPORTDIR / "cnmf_findings_and_boundaries.txt").write_text(
                cnmf_findings_text + "\n"
            )
            print(cnmf_findings_text)
            """
        ),
        md(
            r"""
            ### Z-scored clone-level cNMF usage heatmaps

            These three descriptive views use the same library-adjusted means
            estimated from one fractional-usage value per LARRY clone. For each
            frozen program, the first heatmap standardizes across all 20
            metaclones, the second across the five pooled lineage-bias families,
            and the third separately across metaclones within each lineage-bias
            family. Positive z-scores are red, zero is white, and negative
            z-scores are blue. The Basophil family contains only MC13, so its
            within-family z-score is not estimable and remains grey/NE rather
            than being converted to zero. Rings mark supported tier-wide FDR
            below 0.05. All three panels share a display range of -2.5 to +2.5;
            unclipped z-scores and clipping flags remain in each plot-source
            table. The color scale itself is descriptive and does not encode a
            p-value or create a new inferential test.
            """
        ),
        code(
            r"""
            # CNMF_ALL_MC_ZSCORE_HEATMAP_INLINE_V1 — descriptive row z-scores across 20 MCs
            plot_stem = "cnmf_06_all_metaclone_usage_row_zscore_heatmap"
            cnmf_all_mc_adjusted_matrix = (
                cnmf_adjusted_mc_means.set_index("metaclone")
                .reindex(MC_ORDER)[cnmf_program_ids]
                .T
            )
            cnmf_all_mc_row_mean = cnmf_all_mc_adjusted_matrix.mean(axis=1)
            cnmf_all_mc_row_sd = cnmf_all_mc_adjusted_matrix.std(axis=1, ddof=0)
            assert cnmf_all_mc_row_sd.gt(1e-12).all()
            cnmf_all_mc_zscore_matrix = (
                cnmf_all_mc_adjusted_matrix
                .sub(cnmf_all_mc_row_mean, axis=0)
                .div(cnmf_all_mc_row_sd, axis=0)
            )
            cnmf_all_mc_heatmap_long = (
                cnmf_all_mc_zscore_matrix.rename_axis("program").reset_index()
                .melt(id_vars="program", var_name="metaclone", value_name="row_zscore")
            )
            cnmf_all_mc_heatmap_long["metaclone"] = (
                cnmf_all_mc_heatmap_long.metaclone.astype(int)
            )
            cnmf_all_mc_adjusted_long = (
                cnmf_all_mc_adjusted_matrix.rename_axis("program").reset_index()
                .melt(
                    id_vars="program", var_name="metaclone",
                    value_name="adjusted_clone_usage_sd",
                )
            )
            cnmf_all_mc_adjusted_long["metaclone"] = (
                cnmf_all_mc_adjusted_long.metaclone.astype(int)
            )
            cnmf_all_mc_heatmap_long = cnmf_all_mc_heatmap_long.merge(
                cnmf_all_mc_adjusted_long,
                on=["program", "metaclone"], validate="one_to_one",
            )
            cnmf_all_mc_heatmap_long["lineage_bias"] = (
                cnmf_all_mc_heatmap_long.metaclone.map(MC_TO_FAMILY)
            )
            cnmf_all_mc_heatmap_long["program_order"] = (
                cnmf_all_mc_heatmap_long.program.map(
                    {program: index for index, program in enumerate(cnmf_program_ids)}
                )
            )
            cnmf_all_mc_heatmap_long["metaclone_order"] = (
                cnmf_all_mc_heatmap_long.metaclone.map(
                    {metaclone: index for index, metaclone in enumerate(MC_ORDER)}
                )
            )
            cnmf_all_mc_heatmap_long["z_reference"] = "all_20_metaclones"
            cnmf_all_mc_heatmap_long["z_center"] = (
                cnmf_all_mc_heatmap_long.program.map(cnmf_all_mc_row_mean)
            )
            cnmf_all_mc_heatmap_long["z_scale"] = (
                cnmf_all_mc_heatmap_long.program.map(cnmf_all_mc_row_sd)
            )
            cnmf_all_mc_heatmap_long["z_ddof"] = 0
            cnmf_all_mc_heatmap_long["z_estimable"] = True
            cnmf_all_mc_heatmap_long["undefined_reason"] = ""
            cnmf_all_mc_heatmap_long["display_z"] = (
                cnmf_all_mc_heatmap_long.row_zscore.clip(-2.5, 2.5)
            )
            cnmf_all_mc_heatmap_long["was_clipped"] = (
                cnmf_all_mc_heatmap_long.row_zscore.abs().gt(2.5)
            )
            cnmf_all_mc_stats = cnmf_broad_tests[
                cnmf_broad_tests.tier.eq("all_metaclones_ovr")
            ][[
                "program", "target_mc", "contrast_id", "effect_sd",
                "support_gate", "tier_fdr",
            ]].copy()
            cnmf_all_mc_stats["metaclone"] = cnmf_all_mc_stats.target_mc.astype(int)
            cnmf_all_mc_stats = cnmf_all_mc_stats.rename(
                columns={"effect_sd": "one_vs_rest_effect_sd"}
            ).drop(columns="target_mc")
            cnmf_all_mc_heatmap_long = cnmf_all_mc_heatmap_long.merge(
                cnmf_all_mc_stats,
                on=["program", "metaclone"], how="left", validate="one_to_one",
            )
            assert cnmf_all_mc_heatmap_long.contrast_id.notna().all()
            cnmf_all_mc_heatmap_long["tier_fdr_lt_0_05"] = (
                cnmf_all_mc_heatmap_long.support_gate
                & cnmf_all_mc_heatmap_long.tier_fdr.lt(0.05)
            )
            cnmf_all_mc_heatmap_long.to_csv(
                TABLEDIR / f"{plot_stem}_plot_data.csv", index=False
            )

            cnmf_flagged_programs = set(
                cnmf_program_annotations.loc[
                    cnmf_program_annotations.any_technical_or_context_flag, "program"
                ]
            )
            cnmf_program_tick_labels = [
                f"{program.rsplit('_', 1)[-1]}{'†' if program in cnmf_flagged_programs else ''}"
                for program in cnmf_program_ids
            ]
            cnmf_all_mc_limit = 2.5
            cnmf_all_mc_cmap = LinearSegmentedColormap.from_list(
                "all_mc_negative_blue_zero_white_positive_red",
                ["#2166AC", "#FFFFFF", "#B2182B"], N=257,
            )
            cnmf_all_mc_cmap.set_bad("#D9DEE5")
            cnmf_all_mc_norm = mpl.colors.TwoSlopeNorm(
                vmin=-cnmf_all_mc_limit, vcenter=0.0, vmax=cnmf_all_mc_limit
            )

            fig, ax = plt.subplots(figsize=(12.4, 6.4))
            image_artist = ax.pcolormesh(
                np.arange(len(MC_ORDER) + 1),
                np.arange(CNMF_SELECTED_K + 1),
                cnmf_all_mc_zscore_matrix.to_numpy(float),
                cmap=cnmf_all_mc_cmap, norm=cnmf_all_mc_norm,
                shading="flat", edgecolors=WHITE, linewidth=0.25,
            )
            cnmf_all_mc_sig = cnmf_all_mc_heatmap_long[
                cnmf_all_mc_heatmap_long.tier_fdr_lt_0_05
            ]
            cnmf_all_mc_x = {metaclone: index for index, metaclone in enumerate(MC_ORDER)}
            cnmf_program_y = {program: index for index, program in enumerate(cnmf_program_ids)}
            ax.scatter(
                [cnmf_all_mc_x[int(value)] + 0.5 for value in cnmf_all_mc_sig.metaclone],
                [cnmf_program_y[value] + 0.5 for value in cnmf_all_mc_sig.program],
                s=34, marker="o", facecolors=WHITE, edgecolors=INK,
                linewidths=0.9, zorder=4,
            )
            cnmf_family_boundaries = np.cumsum(
                [len(FAMILY_GROUPS[family]) for family in FAMILY_ORDER]
            )[:-1]
            for boundary in cnmf_family_boundaries:
                ax.axvline(boundary, color=INK, linewidth=1.0)
            ax.set_xticks(
                np.arange(len(MC_ORDER)) + 0.5,
                [f"MC{metaclone}" for metaclone in MC_ORDER], rotation=90,
            )
            for tick, metaclone in zip(ax.get_xticklabels(), MC_ORDER):
                tick.set_color(FAMILY_COLORS[MC_TO_FAMILY[metaclone]])
                tick.set_fontweight("bold")
            ax.set_yticks(
                np.arange(CNMF_SELECTED_K) + 0.5,
                cnmf_program_tick_labels, fontsize=8,
            )
            ax.set_xlim(0, len(MC_ORDER))
            ax.set_ylim(0, CNMF_SELECTED_K)
            ax.invert_yaxis()
            ax.set_xlabel("Metaclone (tick color denotes current lineage-bias family)")
            ax.set_ylabel("Frozen cNMF program")
            ax.set_title("Relative cNMF usage across all metaclones", loc="left", pad=12)
            ax.text(
                0, 1.012,
                "Library-adjusted clone means; row z-score across 20 MCs · ring: supported tier FDR < 0.05 · \u2020 technical/context flag",
                transform=ax.transAxes, fontsize=8.4, color=GREY,
            )
            colorbar = fig.colorbar(
                image_artist, ax=ax, fraction=0.027, pad=0.02, extend="both"
            )
            colorbar.solids.set_rasterized(False)
            colorbar.set_label("Row z-score (+ red / - blue)")
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
            # CNMF_LINEAGE_BIAS_ZSCORE_HEATMAP_INLINE_V1 — descriptive row z-scores across 5 families
            plot_stem = "cnmf_07_lineage_bias_usage_row_zscore_heatmap"
            cnmf_family_adjusted_matrix = (
                cnmf_adjusted_family_means.set_index("lineage_bias")
                .reindex(FAMILY_ORDER)[cnmf_program_ids]
                .T
            )
            cnmf_family_row_mean = cnmf_family_adjusted_matrix.mean(axis=1)
            cnmf_family_row_sd = cnmf_family_adjusted_matrix.std(axis=1, ddof=0)
            assert cnmf_family_row_sd.gt(1e-12).all()
            cnmf_family_zscore_matrix = (
                cnmf_family_adjusted_matrix
                .sub(cnmf_family_row_mean, axis=0)
                .div(cnmf_family_row_sd, axis=0)
            )
            cnmf_family_heatmap_long = (
                cnmf_family_zscore_matrix.rename_axis("program").reset_index()
                .melt(
                    id_vars="program", var_name="lineage_bias",
                    value_name="row_zscore",
                )
            )
            cnmf_family_adjusted_long = (
                cnmf_family_adjusted_matrix.rename_axis("program").reset_index()
                .melt(
                    id_vars="program", var_name="lineage_bias",
                    value_name="adjusted_clone_usage_sd",
                )
            )
            cnmf_family_heatmap_long = cnmf_family_heatmap_long.merge(
                cnmf_family_adjusted_long,
                on=["program", "lineage_bias"], validate="one_to_one",
            )
            cnmf_family_heatmap_long["program_order"] = (
                cnmf_family_heatmap_long.program.map(
                    {program: index for index, program in enumerate(cnmf_program_ids)}
                )
            )
            cnmf_family_heatmap_long["family_order"] = (
                cnmf_family_heatmap_long.lineage_bias.map(
                    {family: index for index, family in enumerate(FAMILY_ORDER)}
                )
            )
            cnmf_family_heatmap_long["z_reference"] = "five_lineage_bias_families"
            cnmf_family_heatmap_long["z_center"] = (
                cnmf_family_heatmap_long.program.map(cnmf_family_row_mean)
            )
            cnmf_family_heatmap_long["z_scale"] = (
                cnmf_family_heatmap_long.program.map(cnmf_family_row_sd)
            )
            cnmf_family_heatmap_long["z_ddof"] = 0
            cnmf_family_heatmap_long["z_estimable"] = True
            cnmf_family_heatmap_long["undefined_reason"] = ""
            cnmf_family_heatmap_long["display_z"] = (
                cnmf_family_heatmap_long.row_zscore.clip(-2.5, 2.5)
            )
            cnmf_family_heatmap_long["was_clipped"] = (
                cnmf_family_heatmap_long.row_zscore.abs().gt(2.5)
            )
            cnmf_family_stats = cnmf_broad_tests[
                cnmf_broad_tests.tier.eq("lineage_bias_ovr")
            ][[
                "program", "lineage_bias", "contrast_id", "effect_sd",
                "support_gate", "tier_fdr",
            ]].copy().rename(columns={"effect_sd": "one_vs_rest_effect_sd"})
            cnmf_family_heatmap_long = cnmf_family_heatmap_long.merge(
                cnmf_family_stats,
                on=["program", "lineage_bias"], how="left", validate="one_to_one",
            )
            assert cnmf_family_heatmap_long.contrast_id.notna().all()
            cnmf_family_heatmap_long["tier_fdr_lt_0_05"] = (
                cnmf_family_heatmap_long.support_gate
                & cnmf_family_heatmap_long.tier_fdr.lt(0.05)
            )
            cnmf_family_heatmap_long.to_csv(
                TABLEDIR / f"{plot_stem}_plot_data.csv", index=False
            )

            cnmf_flagged_programs = set(
                cnmf_program_annotations.loc[
                    cnmf_program_annotations.any_technical_or_context_flag, "program"
                ]
            )
            cnmf_program_tick_labels = [
                f"{program.rsplit('_', 1)[-1]}{'†' if program in cnmf_flagged_programs else ''}"
                for program in cnmf_program_ids
            ]
            cnmf_family_limit = 2.5
            cnmf_family_cmap = LinearSegmentedColormap.from_list(
                "family_negative_blue_zero_white_positive_red",
                ["#2166AC", "#FFFFFF", "#B2182B"], N=257,
            )
            cnmf_family_cmap.set_bad("#D9DEE5")
            cnmf_family_norm = mpl.colors.TwoSlopeNorm(
                vmin=-cnmf_family_limit, vcenter=0.0, vmax=cnmf_family_limit
            )

            fig, ax = plt.subplots(figsize=(8.0, 6.4))
            image_artist = ax.pcolormesh(
                np.arange(len(FAMILY_ORDER) + 1),
                np.arange(CNMF_SELECTED_K + 1),
                cnmf_family_zscore_matrix.to_numpy(float),
                cmap=cnmf_family_cmap, norm=cnmf_family_norm,
                shading="flat", edgecolors=WHITE, linewidth=0.35,
            )
            cnmf_family_sig = cnmf_family_heatmap_long[
                cnmf_family_heatmap_long.tier_fdr_lt_0_05
            ]
            cnmf_family_x = {family: index for index, family in enumerate(FAMILY_ORDER)}
            cnmf_program_y = {program: index for index, program in enumerate(cnmf_program_ids)}
            ax.scatter(
                [cnmf_family_x[value] + 0.5 for value in cnmf_family_sig.lineage_bias],
                [cnmf_program_y[value] + 0.5 for value in cnmf_family_sig.program],
                s=34, marker="o", facecolors=WHITE, edgecolors=INK,
                linewidths=0.9, zorder=4,
            )
            ax.set_xticks(
                np.arange(len(FAMILY_ORDER)) + 0.5,
                FAMILY_ORDER, rotation=28, ha="right",
            )
            for tick, family in zip(ax.get_xticklabels(), FAMILY_ORDER):
                tick.set_color(FAMILY_COLORS[family])
                tick.set_fontweight("bold")
            ax.set_yticks(
                np.arange(CNMF_SELECTED_K) + 0.5,
                cnmf_program_tick_labels, fontsize=8,
            )
            ax.set_xlim(0, len(FAMILY_ORDER))
            ax.set_ylim(0, CNMF_SELECTED_K)
            ax.invert_yaxis()
            ax.set_xlabel("Current lineage-bias family")
            ax.set_ylabel("Frozen cNMF program")
            ax.set_title("Relative cNMF usage across lineage-bias families", loc="left", pad=12)
            ax.text(
                0, 1.012,
                "Library-adjusted clone means; row z-score across five families · ring: supported tier FDR < 0.05 · \u2020 technical/context flag",
                transform=ax.transAxes, fontsize=8.2, color=GREY,
            )
            colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.043, pad=0.03)
            colorbar.solids.set_rasterized(False)
            colorbar.set_label("Row z-score (+ red / - blue)")
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
            # CNMF_WITHIN_BIAS_ZSCORE_HEATMAP_INLINE_V1 — z-scores within each family
            plot_stem = "cnmf_08_within_lineage_bias_usage_row_zscore_heatmap"
            cnmf_within_family_heatmap_long = (
                cnmf_adjusted_mc_means.set_index("metaclone")
                .reindex(MC_ORDER)[cnmf_program_ids]
                .T.rename_axis("program").reset_index()
                .melt(
                    id_vars="program", var_name="metaclone",
                    value_name="adjusted_clone_usage_sd",
                )
            )
            cnmf_within_family_heatmap_long["metaclone"] = (
                cnmf_within_family_heatmap_long.metaclone.astype(int)
            )
            cnmf_within_family_heatmap_long["lineage_bias"] = (
                cnmf_within_family_heatmap_long.metaclone.map(MC_TO_FAMILY)
            )
            cnmf_within_groups = cnmf_within_family_heatmap_long.groupby(
                ["program", "lineage_bias"], observed=True
            ).adjusted_clone_usage_sd
            cnmf_within_family_heatmap_long["family_n_metaclones"] = (
                cnmf_within_groups.transform("size")
            )
            cnmf_within_family_heatmap_long["family_program_mean"] = (
                cnmf_within_groups.transform("mean")
            )
            cnmf_within_family_heatmap_long["family_program_sd_ddof0"] = (
                cnmf_within_groups.transform(lambda values: values.std(ddof=0))
            )
            cnmf_within_family_heatmap_long["zscore_defined"] = (
                cnmf_within_family_heatmap_long.family_n_metaclones.ge(2)
                & cnmf_within_family_heatmap_long.family_program_sd_ddof0.gt(1e-12)
            )
            cnmf_within_family_heatmap_long["within_family_row_zscore"] = np.where(
                cnmf_within_family_heatmap_long.zscore_defined,
                (
                    cnmf_within_family_heatmap_long.adjusted_clone_usage_sd
                    - cnmf_within_family_heatmap_long.family_program_mean
                ) / cnmf_within_family_heatmap_long.family_program_sd_ddof0,
                np.nan,
            )
            cnmf_within_family_heatmap_long["program_order"] = (
                cnmf_within_family_heatmap_long.program.map(
                    {program: index for index, program in enumerate(cnmf_program_ids)}
                )
            )
            cnmf_within_family_heatmap_long["metaclone_order"] = (
                cnmf_within_family_heatmap_long.metaclone.map(
                    {metaclone: index for index, metaclone in enumerate(MC_ORDER)}
                )
            )
            cnmf_within_family_heatmap_long["family_order"] = (
                cnmf_within_family_heatmap_long.lineage_bias.map(
                    {family: index for index, family in enumerate(FAMILY_ORDER)}
                )
            )
            cnmf_within_family_heatmap_long["z_reference"] = (
                "metaclones_within_" + cnmf_within_family_heatmap_long.lineage_bias
            )
            cnmf_within_family_heatmap_long["z_center"] = (
                cnmf_within_family_heatmap_long.family_program_mean
            )
            cnmf_within_family_heatmap_long["z_scale"] = (
                cnmf_within_family_heatmap_long.family_program_sd_ddof0
            )
            cnmf_within_family_heatmap_long["z_ddof"] = 0
            cnmf_within_family_heatmap_long["z_estimable"] = (
                cnmf_within_family_heatmap_long.zscore_defined
            )
            cnmf_within_family_heatmap_long["undefined_reason"] = np.select(
                [
                    cnmf_within_family_heatmap_long.family_n_metaclones.lt(2),
                    cnmf_within_family_heatmap_long.family_program_sd_ddof0.le(1e-12),
                ],
                ["one metaclone in lineage-bias family", "zero within-family variance"],
                default="",
            )
            cnmf_within_family_heatmap_long["display_z"] = (
                cnmf_within_family_heatmap_long.within_family_row_zscore.clip(-2.5, 2.5)
            )
            cnmf_within_family_heatmap_long["was_clipped"] = (
                cnmf_within_family_heatmap_long.within_family_row_zscore.abs().gt(2.5)
            )
            cnmf_within_mc_stats = cnmf_broad_tests[
                cnmf_broad_tests.tier.eq("within_lineage_ovr")
            ][[
                "program", "target_mc", "contrast_id", "effect_sd",
                "support_gate", "tier_fdr",
            ]].copy()
            cnmf_within_mc_stats["metaclone"] = cnmf_within_mc_stats.target_mc.astype(int)
            cnmf_within_mc_stats = cnmf_within_mc_stats.rename(
                columns={"effect_sd": "within_family_one_vs_rest_effect_sd"}
            ).drop(columns="target_mc")
            cnmf_within_family_heatmap_long = cnmf_within_family_heatmap_long.merge(
                cnmf_within_mc_stats,
                on=["program", "metaclone"], how="left", validate="one_to_one",
            )
            cnmf_within_family_heatmap_long["tier_fdr_lt_0_05"] = (
                cnmf_within_family_heatmap_long.support_gate.eq(True)
                & cnmf_within_family_heatmap_long.tier_fdr.lt(0.05)
            )
            cnmf_within_family_zscore_matrix = (
                cnmf_within_family_heatmap_long.pivot(
                    index="program", columns="metaclone",
                    values="within_family_row_zscore",
                ).reindex(index=cnmf_program_ids, columns=MC_ORDER)
            )
            cnmf_within_family_heatmap_long.to_csv(
                TABLEDIR / f"{plot_stem}_plot_data.csv", index=False
            )

            cnmf_flagged_programs = set(
                cnmf_program_annotations.loc[
                    cnmf_program_annotations.any_technical_or_context_flag, "program"
                ]
            )
            cnmf_program_tick_labels = [
                f"{program.rsplit('_', 1)[-1]}{'†' if program in cnmf_flagged_programs else ''}"
                for program in cnmf_program_ids
            ]
            cnmf_within_family_limit = 2.5
            cnmf_within_family_cmap = LinearSegmentedColormap.from_list(
                "within_family_negative_blue_zero_white_positive_red",
                ["#2166AC", "#FFFFFF", "#B2182B"], N=257,
            )
            cnmf_within_family_cmap.set_bad("#D9DEE5")
            cnmf_within_family_norm = mpl.colors.TwoSlopeNorm(
                vmin=-cnmf_within_family_limit,
                vcenter=0.0,
                vmax=cnmf_within_family_limit,
            )

            fig, ax = plt.subplots(figsize=(12.4, 6.4))
            image_artist = ax.pcolormesh(
                np.arange(len(MC_ORDER) + 1),
                np.arange(CNMF_SELECTED_K + 1),
                np.ma.masked_invalid(cnmf_within_family_zscore_matrix.to_numpy(float)),
                cmap=cnmf_within_family_cmap, norm=cnmf_within_family_norm,
                shading="flat", edgecolors=WHITE, linewidth=0.25,
            )
            cnmf_within_sig = cnmf_within_family_heatmap_long[
                cnmf_within_family_heatmap_long.tier_fdr_lt_0_05
            ]
            cnmf_within_x = {metaclone: index for index, metaclone in enumerate(MC_ORDER)}
            cnmf_program_y = {program: index for index, program in enumerate(cnmf_program_ids)}
            ax.scatter(
                [cnmf_within_x[int(value)] + 0.5 for value in cnmf_within_sig.metaclone],
                [cnmf_program_y[value] + 0.5 for value in cnmf_within_sig.program],
                s=34, marker="o", facecolors=WHITE, edgecolors=INK,
                linewidths=0.9, zorder=4,
            )
            cnmf_family_boundaries = np.cumsum(
                [len(FAMILY_GROUPS[family]) for family in FAMILY_ORDER]
            )[:-1]
            for boundary in cnmf_family_boundaries:
                ax.axvline(boundary, color=INK, linewidth=1.0)
            cnmf_singleton_metaclones = {
                members[0]
                for members in FAMILY_GROUPS.values()
                if len(members) == 1
            }
            ax.set_xticks(
                np.arange(len(MC_ORDER)) + 0.5,
                [
                    f"MC{metaclone}\nNE" if metaclone in cnmf_singleton_metaclones
                    else f"MC{metaclone}"
                    for metaclone in MC_ORDER
                ],
                rotation=90,
            )
            for tick, metaclone in zip(ax.get_xticklabels(), MC_ORDER):
                tick.set_color(FAMILY_COLORS[MC_TO_FAMILY[metaclone]])
                tick.set_fontweight("bold")
            ax.set_yticks(
                np.arange(CNMF_SELECTED_K) + 0.5,
                cnmf_program_tick_labels, fontsize=8,
            )
            ax.set_xlim(0, len(MC_ORDER))
            ax.set_ylim(0, CNMF_SELECTED_K)
            ax.invert_yaxis()
            ax.set_xlabel("Metaclone, standardized only against peers in the same lineage-bias family")
            ax.set_ylabel("Frozen cNMF program")
            ax.set_title("Relative cNMF usage within lineage-bias families", loc="left", pad=12)
            ax.text(
                0, 1.012,
                "Within-family row z-score · MC13 is singleton and grey/NE · ring: supported tier FDR < 0.05 · \u2020 technical/context flag",
                transform=ax.transAxes, fontsize=8.25, color=GREY,
            )
            colorbar = fig.colorbar(image_artist, ax=ax, fraction=0.027, pad=0.02)
            colorbar.solids.set_rasterized(False)
            colorbar.set_label("Within-family row z-score (+ red / - blue)")
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
            # cNMF validation, editable-plot contract, and active manifest refresh
            cnmf_plot_stems = [
                "cnmf_01_K_selection_diagnostics",
                "cnmf_02_frozen_program_leading_genes",
                "cnmf_03_adjusted_usage_by_metaclone",
                "cnmf_04_program_recovery_stability",
                "cnmf_05_reference_capture_and_novelty",
                "cnmf_06_all_metaclone_usage_row_zscore_heatmap",
                "cnmf_07_lineage_bias_usage_row_zscore_heatmap",
                "cnmf_08_within_lineage_bias_usage_row_zscore_heatmap",
            ]
            built_notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
            notebook_sources = ["".join(cell.get("source", "")) for cell in built_notebook.cells]
            cnmf_plot_sources = [
                source for source in notebook_sources
                if source.lstrip().startswith((
                    "# CNMF_K_SELECTION_PLOT_INLINE_V1",
                    "# CNMF_SPECTRA_PLOT_INLINE_V1",
                    "# CNMF_METACLONE_PLOT_INLINE_V1",
                    "# CNMF_RECOVERY_PLOT_INLINE_V1",
                    "# CNMF_ANNOTATION_PLOT_INLINE_V1",
                    "# CNMF_ALL_MC_ZSCORE_HEATMAP_INLINE_V1",
                    "# CNMF_LINEAGE_BIAS_ZSCORE_HEATMAP_INLINE_V1",
                    "# CNMF_WITHIN_BIAS_ZSCORE_HEATMAP_INLINE_V1",
                ))
            ]
            cnmf_plot_contract = all(
                source.count("plt.subplots") == 1
                and source.count("fig.savefig(svg") == 1
                and source.count("fig.savefig(png") == 1
                and source.count("display(fig)") == 1
                and source.count("plt.close(fig)") == 1
                for source in cnmf_plot_sources
            )
            cnmf_svg_editable = all(
                "<svg" in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
                and "<image" not in (FIGDIR / f"{stem}.svg").read_text(errors="ignore")
                for stem in cnmf_plot_stems
            )
            cnmf_plot_triplets = all(
                (FIGDIR / f"{stem}.svg").stat().st_size > 0
                and (FIGDIR / f"{stem}.png").stat().st_size > 0
                and (TABLEDIR / f"{stem}_plot_data.csv").stat().st_size > 0
                for stem in cnmf_plot_stems
            )
            unique_restart_seeds = (
                cnmf_restart_diagnostics.groupby(["context", "K"], observed=True)
                .agg(n_rows=("seed", "size"), n_unique_seeds=("seed", "nunique"))
            )
            recovery_counts = cnmf_program_recovery.groupby("sensitivity", observed=True).replicate.nunique()
            targeted_manifest_path = REPORTDIR / "targeted_artifact_manifest.csv"
            targeted_manifest_before_refresh = pd.read_csv(targeted_manifest_path)
            assert targeted_manifest_before_refresh.path.is_unique
            targeted_manifest_scope = sorted(targeted_manifest_before_refresh.path.tolist())
            targeted_manifest_scope_paths = [
                PROJECT_ROOT / relative_path for relative_path in targeted_manifest_scope
            ]
            assert all(path.is_file() for path in targeted_manifest_scope_paths)
            targeted_manifest_scope_is_targeted = not any(
                path.name.startswith("cnmf_") for path in targeted_manifest_scope_paths
            )
            targeted_artifact_manifest_refreshed = pd.DataFrame([{
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            } for path in targeted_manifest_scope_paths])
            assert targeted_artifact_manifest_refreshed.path.tolist() == targeted_manifest_scope
            targeted_artifact_manifest_refreshed.to_csv(targeted_manifest_path, index=False)
            targeted_manifest_refresh_exact = all(
                (PROJECT_ROOT / row.path).stat().st_size == row.bytes
                and sha256(PROJECT_ROOT / row.path) == row.sha256
                for row in targeted_artifact_manifest_refreshed.itertuples(index=False)
            )
            cnmf_zscore_heatmap_sources = [
                source for source in cnmf_plot_sources
                if "ZSCORE_HEATMAP_INLINE_V1" in source
            ]
            cnmf_all_mc_zscore_audit = cnmf_all_mc_heatmap_long.groupby(
                "program", observed=True
            ).row_zscore.agg(
                row_mean="mean", row_sd_ddof0=lambda values: values.std(ddof=0)
            )
            cnmf_family_zscore_audit = cnmf_family_heatmap_long.groupby(
                "program", observed=True
            ).row_zscore.agg(
                row_mean="mean", row_sd_ddof0=lambda values: values.std(ddof=0)
            )
            cnmf_within_defined = cnmf_within_family_heatmap_long[
                cnmf_within_family_heatmap_long.zscore_defined
            ]
            cnmf_within_zscore_audit = cnmf_within_defined.groupby(
                ["program", "lineage_bias"], observed=True
            ).within_family_row_zscore.agg(
                row_mean="mean", row_sd_ddof0=lambda values: values.std(ddof=0)
            )
            cnmf_within_undefined = cnmf_within_family_heatmap_long[
                ~cnmf_within_family_heatmap_long.zscore_defined
            ]
            cnmf_heatmap_palette_contract = all(
                "LinearSegmentedColormap.from_list" in source
                and "[\"#2166AC\", \"#FFFFFF\", \"#B2182B\"]" in source
                and "N=257" in source
                and "TwoSlopeNorm" in source
                and "vcenter=0.0" in source
                for source in cnmf_zscore_heatmap_sources
            )
            cnmf_heatmap_color_contract = all(
                [
                    mpl.colors.to_hex(cmap(0.0)).lower(),
                    mpl.colors.to_hex(cmap(0.5)).lower(),
                    mpl.colors.to_hex(cmap(1.0)).lower(),
                ] == ["#2166ac", "#ffffff", "#b2182b"]
                for cmap in [
                    cnmf_all_mc_cmap, cnmf_family_cmap, cnmf_within_family_cmap
                ]
            )
            cnmf_heatmap_clone_level_only = not any(
                token in source
                for source in cnmf_zscore_heatmap_sources
                for token in ["cnmf_cell_usage_raw", "cnmf_cell_usage_fraction"]
            )
            cnmf_all_mc_reconstructed = (
                cnmf_all_mc_heatmap_long.pivot(
                    index="program", columns="metaclone",
                    values="adjusted_clone_usage_sd",
                ).reindex(index=cnmf_program_ids, columns=MC_ORDER)
            )
            cnmf_family_reconstructed = (
                cnmf_family_heatmap_long.pivot(
                    index="program", columns="lineage_bias",
                    values="adjusted_clone_usage_sd",
                ).reindex(index=cnmf_program_ids, columns=FAMILY_ORDER)
            )
            cnmf_heatmap_adjusted_means_exact = (
                np.allclose(
                    cnmf_all_mc_reconstructed.to_numpy(float),
                    cnmf_all_mc_adjusted_matrix.to_numpy(float),
                    atol=1e-12,
                )
                and np.allclose(
                    cnmf_family_reconstructed.to_numpy(float),
                    cnmf_family_adjusted_matrix.to_numpy(float),
                    atol=1e-12,
                )
                and np.allclose(
                    cnmf_within_family_heatmap_long.pivot(
                        index="program", columns="metaclone",
                        values="adjusted_clone_usage_sd",
                    ).reindex(index=cnmf_program_ids, columns=MC_ORDER).to_numpy(float),
                    cnmf_all_mc_adjusted_matrix.to_numpy(float),
                    atol=1e-12,
                )
            )
            cnmf_heatmap_keys_complete = (
                len(cnmf_all_mc_heatmap_long) == CNMF_SELECTED_K * len(MC_ORDER)
                and len(cnmf_family_heatmap_long) == CNMF_SELECTED_K * len(FAMILY_ORDER)
                and len(cnmf_within_family_heatmap_long) == CNMF_SELECTED_K * len(MC_ORDER)
                and not cnmf_all_mc_heatmap_long.duplicated(["program", "metaclone"]).any()
                and not cnmf_family_heatmap_long.duplicated(["program", "lineage_bias"]).any()
                and not cnmf_within_family_heatmap_long.duplicated(
                    ["program", "metaclone"]
                ).any()
                and cnmf_all_mc_adjusted_matrix.index.tolist() == cnmf_program_ids
                and cnmf_all_mc_adjusted_matrix.columns.tolist() == MC_ORDER
                and cnmf_family_adjusted_matrix.index.tolist() == cnmf_program_ids
                and cnmf_family_adjusted_matrix.columns.tolist() == FAMILY_ORDER
                and cnmf_within_family_zscore_matrix.index.tolist() == cnmf_program_ids
                and cnmf_within_family_zscore_matrix.columns.tolist() == MC_ORDER
            )
            cnmf_heatmap_display_clipping_audit = all(
                frame.loc[frame.z_estimable, "display_z"].abs().le(2.5).all()
                and frame.loc[frame.z_estimable, "was_clipped"].eq(
                    frame.loc[frame.z_estimable, z_column].abs().gt(2.5)
                ).all()
                for frame, z_column in [
                    (cnmf_all_mc_heatmap_long, "row_zscore"),
                    (cnmf_family_heatmap_long, "row_zscore"),
                    (cnmf_within_family_heatmap_long, "within_family_row_zscore"),
                ]
            )
            required_annotation_flags = {
                "cell_cycle_flag", "mitochondrial_flag", "ribosomal_flag",
                "sex_associated_flag", "nuclear_qc_malat1_flag",
                "library_associated_flag",
            }
            cnmf_inference_unit_rows = [
                {
                    "model": "broad_metaclone", "unit": "LARRY_clone",
                    "expected_units": len(cnmf_clone_scores),
                    "unique_clone_units": pseudobulk_metadata.index.nunique(),
                    "all_fit_nobs_match": all(
                        int(fit.nobs) == len(cnmf_clone_scores)
                        for fit in cnmf_fits_mc.values()
                    ),
                    "outcome_index_matches_clone_metadata": (
                        cnmf_clone_scores.index.tolist() == pseudobulk_metadata.index.tolist()
                    ),
                },
                {
                    "model": "broad_lineage_family", "unit": "LARRY_clone",
                    "expected_units": len(cnmf_clone_scores),
                    "unique_clone_units": pseudobulk_metadata.index.nunique(),
                    "all_fit_nobs_match": all(
                        int(fit.nobs) == len(cnmf_clone_scores)
                        for fit in cnmf_fits_family.values()
                    ),
                    "outcome_index_matches_clone_metadata": (
                        cnmf_clone_scores.index.tolist() == pseudobulk_metadata.index.tolist()
                    ),
                },
                {
                    "model": "targeted_multilineage", "unit": "LARRY_clone",
                    "expected_units": len(targeted_cnmf_scores),
                    "unique_clone_units": targeted_metadata.index.nunique(),
                    "all_fit_nobs_match": all(
                        int(fit.nobs) == len(targeted_cnmf_scores)
                        for fit in targeted_cnmf_fits.values()
                    ),
                    "outcome_index_matches_clone_metadata": (
                        targeted_cnmf_scores.index.tolist() == targeted_metadata.index.tolist()
                    ),
                },
            ]
            for audit_row in cnmf_library_loo_fit_audit.itertuples(index=False):
                cnmf_inference_unit_rows.extend([
                    {
                        "model": f"library_LOO_broad::{audit_row.omitted_library}",
                        "unit": "LARRY_clone",
                        "expected_units": int(audit_row.remaining_clone_units),
                        "unique_clone_units": int(audit_row.remaining_clone_units),
                        "all_fit_nobs_match": bool(audit_row.broad_all_fit_nobs_match),
                        "outcome_index_matches_clone_metadata": bool(
                            audit_row.broad_outcome_index_matches_metadata
                        ),
                    },
                    {
                        "model": f"library_LOO_targeted::{audit_row.omitted_library}",
                        "unit": "LARRY_clone",
                        "expected_units": int(audit_row.remaining_targeted_clone_units),
                        "unique_clone_units": int(audit_row.remaining_targeted_clone_units),
                        "all_fit_nobs_match": bool(audit_row.target_all_fit_nobs_match),
                        "outcome_index_matches_clone_metadata": bool(
                            audit_row.target_outcome_index_matches_metadata
                        ),
                    },
                ])
            cnmf_inference_unit_audit = pd.DataFrame(cnmf_inference_unit_rows)
            cnmf_inference_unit_audit["passed"] = (
                cnmf_inference_unit_audit.unit.eq("LARRY_clone")
                & cnmf_inference_unit_audit.expected_units.eq(
                    cnmf_inference_unit_audit.unique_clone_units
                )
                & cnmf_inference_unit_audit.all_fit_nobs_match
                & cnmf_inference_unit_audit.outcome_index_matches_clone_metadata
            )
            cnmf_inference_unit_audit.to_csv(
                REPORTDIR / "cnmf_inference_unit_audit.csv", index=False
            )
            loo_support_audit_frame = cnmf_library_loo_broad.merge(
                cnmf_broad_tests[["contrast_id", "support_gate"]]
                .drop_duplicates()
                .rename(columns={"support_gate": "full_data_support_gate"}),
                on="contrast_id", validate="many_to_one",
            )
            cnmf_loo_support_audit = (
                loo_support_audit_frame.groupby(
                    ["omitted_library", "contrast_id"], observed=True
                ).agg(
                    full_data_support_gate=("full_data_support_gate", "first"),
                    loo_support_gate=("support_gate", "first"),
                    all_program_rows_agree=("support_gate", "nunique"),
                ).reset_index()
            )
            cnmf_loo_support_audit["support_lost_after_removal"] = (
                cnmf_loo_support_audit.full_data_support_gate
                & ~cnmf_loo_support_audit.loo_support_gate
            )
            cnmf_loo_support_audit.to_csv(
                REPORTDIR / "cnmf_library_loo_support_recalculation_audit.csv",
                index=False,
            )
            expected_metaclone_global_fdr = bh_adjust(
                cnmf_broad_tests.loc[
                    metaclone_search_mask, "pvalue"
                ].to_numpy(float)
            )
            if len(eligible_k) >= 3:
                audit_elbow_k = int(
                    eligible_k.sort_values(
                        ["reconstruction_elbow_distance", "K"],
                        ascending=[False, True],
                    ).iloc[0].K
                )
                audit_selection_candidates = eligible_k[
                    (eligible_k.K - audit_elbow_k).abs() <= 1
                ]
                audit_selected_k = int(
                    audit_selection_candidates.sort_values(
                        ["consensus_silhouette_cosine", "K"],
                        ascending=[False, True],
                    ).iloc[0].K
                )
            else:
                audit_selection_candidates = eligible_k[
                    eligible_k.consensus_silhouette_cosine
                    >= eligible_k.consensus_silhouette_cosine.max() - 0.01
                ]
                audit_selected_k = int(
                    audit_selection_candidates.sort_values(
                        ["consensus_relative_error", "K"],
                        ascending=[True, True],
                    ).iloc[0].K
                )
            cnmf_validation_rows = [
                ("discovery_all_1041_cells", len(cnmf_cell_usage_fraction) == 1041, str(len(cnmf_cell_usage_fraction))),
                ("discovery_all_784_clones", len(cnmf_clone_usage_all) == 784, str(len(cnmf_clone_usage_all))),
                ("inference_569_current_MC_clones", len(cnmf_clone_scores) == 569, str(len(cnmf_clone_scores))),
                ("targeted_217_multilineage_clones", len(targeted_cnmf_scores) == 217, str(len(targeted_cnmf_scores))),
                ("no_forbidden_discovery_metadata", not (forbidden_discovery_fields & discovery_fields_used), str(sorted(discovery_fields_used))),
                ("zero_synthetic_features_entered", int(synthetic_flag[cnmf_hvg_indices].sum()) == 0, str(int(synthetic_flag[cnmf_hvg_indices].sum()))),
                ("zero_mitochondrial_features_entered", int(mitochondrial_flag[cnmf_hvg_indices].sum()) == 0, str(int(mitochondrial_flag[cnmf_hvg_indices].sum()))),
                ("zero_KEGG_ribosome_features_entered", int(ribosome_flag[cnmf_hvg_indices].sum()) == 0, str(int(ribosome_flag[cnmf_hvg_indices].sum()))),
                ("zero_Malat1_features_entered", int(nuclear_qc_flag[cnmf_hvg_indices].sum()) == 0, str(int(nuclear_qc_flag[cnmf_hvg_indices].sum()))),
                ("Malat1_absent_from_recovery_HVGs", ~cnmf_recovery_feature_selection.gene.astype(str).str.upper().eq("MALAT1").any(), str(int(cnmf_recovery_feature_selection.gene.astype(str).str.upper().eq("MALAT1").sum()))),
                ("Hbegf_not_misclassified_as_hemoglobin", not hemoglobin_flag[discovery_upper == "HBEGF"].any(), str(int(hemoglobin_flag[discovery_upper == "HBEGF"].sum()))),
                ("K_grid_complete", cnmf_k_diagnostics.K.tolist() == CNMF_K_GRID, str(CNMF_K_GRID)),
                ("random_restart_init_only", set(cnmf_restart_diagnostics.init) == {"random"}, str(set(cnmf_restart_diagnostics.init))),
                ("restart_seeds_unique_within_fit", (unique_restart_seeds.n_rows == unique_restart_seeds.n_unique_seeds).all(), str(len(unique_restart_seeds))),
                ("selected_K_passes_all_gates", bool(cnmf_k_diagnostics.loc[cnmf_k_diagnostics.selected_K, "eligible"].iloc[0]), str(CNMF_SELECTED_K)),
                ("one_selected_K", int(cnmf_k_diagnostics.selected_K.sum()) == 1, str(int(cnmf_k_diagnostics.selected_K.sum()))),
                ("K_selection_rule_reproduced", audit_selected_k == CNMF_SELECTED_K, str({"audit": audit_selected_k, "selected": CNMF_SELECTED_K, "candidates": audit_selection_candidates.K.tolist()})),
                ("frozen_HVG_spectra_complete", cnmf_hvg_spectra.shape == (CNMF_SELECTED_K, 2000), str(cnmf_hvg_spectra.shape)),
                ("frozen_all_gene_spectra_complete", cnmf_full_spectra.shape[0] == CNMF_SELECTED_K and cnmf_full_spectra.shape[1] == int(full_spectrum_mask.sum()), str(cnmf_full_spectra.shape)),
                ("top100_every_program", cnmf_leading_genes.groupby("program").size().eq(100).all(), str(cnmf_leading_genes.groupby("program").size().min())),
                ("fractional_usage_sums_one", np.allclose(cnmf_cell_usage_fraction.sum(axis=1), 1.0), str(float(np.max(np.abs(cnmf_cell_usage_fraction.sum(axis=1) - 1))))),
                ("broad_94_registry_mirror", len(cnmf_broad_tests) == 94 * CNMF_SELECTED_K, str(len(cnmf_broad_tests))),
                ("targeted_14_registry_mirror", len(targeted_cnmf_all) == 14 * CNMF_SELECTED_K, str(len(targeted_cnmf_all))),
                ("novelty_search_global_BH_exact", np.allclose(cnmf_broad_tests.loc[metaclone_search_mask, "metaclone_search_global_fdr"], expected_metaclone_global_fdr), str(int(metaclone_search_mask.sum()))),
                ("manual_design_equivalent", (cnmf_manual_design_equivalence.max_abs_effect_error < 1e-9).all(), str(cnmf_manual_design_equivalence.max_abs_effect_error.max())),
                ("seven_library_program_refits", int(recovery_counts.get("leave_one_library_out", 0)) == 7, str(recovery_counts.to_dict())),
                ("ten_source_balance_refits", int(recovery_counts.get("within_library_prog1_prog2_balance", 0)) == 10, str(recovery_counts.to_dict())),
                ("sensitivity_HVGs_reselected", cnmf_program_recovery.refit_reselected_HVGs.all() and len(cnmf_recovery_feature_selection) == 17 * CNMF_N_HVG, str(len(cnmf_recovery_feature_selection))),
                ("sensitivity_common_HVG_support", int(cnmf_program_recovery.n_common_HVGs.min()) >= 500, str(int(cnmf_program_recovery.n_common_HVGs.min()))),
                ("seven_library_inference_refits_audited", len(cnmf_library_loo_fit_audit) == 7 and cnmf_library_loo_fit_audit.estimable.notna().all(), str({"refits": len(cnmf_library_loo_fit_audit), "fully_estimable": int(cnmf_library_loo_fit_audit.estimable.sum())})),
                ("library_LOO_support_recomputed", cnmf_loo_support_audit.all_program_rows_agree.eq(1).all() and cnmf_loo_support_audit.support_lost_after_removal.any(), str(int(cnmf_loo_support_audit.support_lost_after_removal.sum()))),
                ("targeted_LOO_library_candidates_canonical", ~cnmf_library_loo_fit_audit.target_rank_dropped.str.contains("__cnmf_loo_", regex=False).any(), "generated standardized columns excluded"),
                ("source_covariate_models_full_rank", (cnmf_source_design_audit.n_columns == cnmf_source_design_audit["rank"]).all(), cnmf_source_design_audit.n_columns.astype(str).str.cat(sep="|")),
                ("annotation_flags_explicit", required_annotation_flags.issubset(cnmf_program_annotations.columns), str(sorted(required_annotation_flags))),
                ("annotation_reference_labels_resolved", required_hallmark_labels.issubset(normalized_hallmark_labels) and "Module::AP1_early_response" in curated_reference_labels and required_technical_labels.issubset(technical_reference_labels), "fixed Hallmark/AP1/technical labels present"),
                ("cell_cycle_flags_match_fixed_Hallmark_rule", observed_cell_cycle_programs == expected_cell_cycle_programs, str(sorted(observed_cell_cycle_programs))),
                ("targeted_hit_interpretation_complete", len(targeted_hit_audit) == len(targeted_combined_hits) and targeted_hit_audit[["postfit_annotation", "library_recovery_supported", "source_balance_recovery_supported"]].notna().all().all(), str(len(targeted_hit_audit))),
                ("curated_membership_loaded_postfreeze", curated_hematopoietic_membership.loaded_after_cnmf_freeze.all(), str(len(curated_hematopoietic_membership))),
                ("all_inferential_models_use_unique_clone_units", cnmf_inference_unit_audit.passed.all(), f"{int(cnmf_inference_unit_audit.passed.sum())}/{len(cnmf_inference_unit_audit)} model contexts"),
                ("all_metaclone_heatmap_rows_zscored", np.allclose(cnmf_all_mc_zscore_audit.row_mean, 0.0, atol=1e-12) and np.allclose(cnmf_all_mc_zscore_audit.row_sd_ddof0, 1.0, atol=1e-12), str(cnmf_all_mc_zscore_audit.shape)),
                ("lineage_bias_heatmap_rows_zscored", np.allclose(cnmf_family_zscore_audit.row_mean, 0.0, atol=1e-12) and np.allclose(cnmf_family_zscore_audit.row_sd_ddof0, 1.0, atol=1e-12), str(cnmf_family_zscore_audit.shape)),
                ("within_lineage_heatmap_rows_zscored", np.allclose(cnmf_within_zscore_audit.row_mean, 0.0, atol=1e-12) and np.allclose(cnmf_within_zscore_audit.row_sd_ddof0, 1.0, atol=1e-12), str(cnmf_within_zscore_audit.shape)),
                ("within_lineage_singleton_is_NE", set(cnmf_within_undefined.lineage_bias) == {"Basophil"} and set(cnmf_within_undefined.metaclone) == {13} and cnmf_within_undefined.within_family_row_zscore.isna().all() and len(cnmf_within_undefined) == CNMF_SELECTED_K, str(len(cnmf_within_undefined))),
                ("zscore_heatmap_dimensions_and_keys", cnmf_heatmap_keys_complete, str({"all_mc": len(cnmf_all_mc_heatmap_long), "families": len(cnmf_family_heatmap_long), "within_family": len(cnmf_within_family_heatmap_long)})),
                ("zscore_heatmap_adjusted_means_exact", cnmf_heatmap_adjusted_means_exact, "plot sources reconstruct fitted clone-level adjusted means"),
                ("zscore_heatmap_shared_display_clip_exact", cnmf_heatmap_display_clipping_audit, str({"limit": 2.5, "all_mc_clipped": int(cnmf_all_mc_heatmap_long.was_clipped.sum()), "families_clipped": int(cnmf_family_heatmap_long.was_clipped.sum()), "within_family_clipped": int(cnmf_within_family_heatmap_long.was_clipped.sum())})),
                ("zscore_heatmap_palette_declared", cnmf_heatmap_palette_contract, "custom blue-white-red with zero-centered normalization"),
                ("zscore_heatmap_palette_exact", cnmf_heatmap_color_contract, "negative=#2166AC; zero=#FFFFFF; positive=#B2182B"),
                ("zscore_heatmaps_clone_level_only", cnmf_heatmap_clone_level_only, "no per-cell usage object accessed by heatmap cells"),
                ("targeted_manifest_refreshed_and_exact", targeted_manifest_refresh_exact, str(len(targeted_artifact_manifest_refreshed))),
                ("targeted_manifest_excludes_cnmf_scope", targeted_manifest_scope_is_targeted, "no cnmf_* artifact in targeted-only manifest"),
                ("cnmf_plot_cell_count", len(cnmf_plot_sources) == len(cnmf_plot_stems), f"{len(cnmf_plot_sources)}/{len(cnmf_plot_stems)}"),
                ("cnmf_inline_plot_contract", cnmf_plot_contract, "one plot/save/display/close per cell"),
                ("cnmf_plot_triplets_complete", cnmf_plot_triplets, str(len(cnmf_plot_stems))),
                ("cnmf_SVGs_editable", cnmf_svg_editable, "no embedded raster images"),
                ("input_landing_zone_empty", not any((PROJECT_ROOT / "Input").iterdir()), "Input/ empty"),
            ]
            cnmf_validation = pd.DataFrame(
                cnmf_validation_rows, columns=["check", "passed", "detail"]
            )
            cnmf_validation.to_csv(REPORTDIR / "cnmf_validation.csv", index=False)
            assert cnmf_validation.passed.all(), cnmf_validation.loc[~cnmf_validation.passed]

            helper_path = ANALYSIS_DIR / "scripts/cnmf_append_cells.py"
            artifact_paths = [BUILDER_PATH, helper_path]
            for folder in [FIGDIR, TABLEDIR, REPORTDIR, RESOURCEDIR, OBJECTDIR]:
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.name not in {
                        "artifact_manifest.csv", "targeted_artifact_manifest.csv",
                        "cnmf_artifact_manifest.csv",
                    }:
                        artifact_paths.append(path)
            cnmf_artifact_manifest = pd.DataFrame([{
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            } for path in sorted(set(artifact_paths))])
            cnmf_artifact_manifest.to_csv(
                REPORTDIR / "cnmf_artifact_manifest.csv", index=False
            )
            cnmf_artifact_manifest.to_csv(
                REPORTDIR / "artifact_manifest.csv", index=False
            )
            display(cnmf_validation)
            display(cnmf_artifact_manifest.tail(25))
            print("All cNMF validation checks passed.")
            print("Notebook:", NOTEBOOK_PATH)
            print("Environment: chronoclone |", ENV_PATH)
            print("Execution command:", EXECUTION_COMMAND)
            """
        ),
    ]
