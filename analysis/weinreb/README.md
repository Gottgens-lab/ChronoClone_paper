# Weinreb LARRY analyses

Start with [the executed evidence and figure notebook](notebooks/01_v5_evidence_and_figures.ipynb). It reads only bundled tables, recalculates the complete within-day pseudotime and Hallmark correction families, and recreates the branch-restricted pseudotime displays (Figures 2F and 2H). The figure panels are collected in [figures/](../../figures/README.md); generated rerun output goes under `generated/weinreb/`.

## Find the evidence

| Paper content | Scientific record | Tables and checkpoints |
|---|---|---|
| Figures 2 and S2: landscape, reference metaclones and temporal maps | `current_source/upstream/`, `current_source/v3_display/`, `current_source/clone_embedding/` | `inputs/reference_figure2/`, `inputs/clone_embedding/` |
| Figures 2F and 2H: branch-restricted pseudotime | `current_source/v5_repair/` and the portable notebook | `tables/v5_repair/` |
| Figure 3A and 3B: multilineage output and branch occupancy | `current_source/dynamics/`, `current_source/ancestry/`, `current_source/v3_display/` | `tables/dynamics/`, `tables/ancestry/`, `tables/v3_display/` |
| Figures 3C–3E and S3: Day-2 clone-level Hallmark scores, gene contributions and sensitivity checks | `current_source/ancestry/` | `tables/ancestry/`, `tables/programs/` |
| Supporting analysis (Table S3): label-blind cNMF programs | `current_source/programs/` | `objects/`, `resources/`, `tables/programs/cnmf_*` |

[The figure index](../../figures/figure_index.csv) links each panel to its file and notebook. [The source file map](provenance/source_file_map.json) provides neutral source-layout paths, original and current checksums, destination and role for every copied file. [The original execution register](provenance/source_notebook_execution.json) separates preserved execution from the newly executed portable notebook.

## Reproduce at the appropriate level

The portable notebook runs without the large H5AD inputs. It verifies stored evidence and reproduces the two repaired figures; it is not a rerun of the entire scientific pipeline. Each notebook in `current_source/` preserves its saved scientific execution record. Publication paths and labels have been adapted; the source hash and current publication hash are separately recorded. Scientific computations were not rerun as part of that adaptation. The figure-production sources include manuscript styling and SVG transformations; their original relative dependencies are documented by the source map. Original author notebooks retain some unexecuted historical cells; these are identified in the execution register and are not new execution claims.

Full downstream expression/trajectory recomputation needs the exact corrected `klein_metaclone_palantir_trajectories.h5ad` in [the external input registry](provenance/external_inputs.json). Follow the repository data acquisition instructions to populate `data/external/weinreb/klein_metaclone_palantir_trajectories.h5ad`; its checksum identifies the retained manuscript input. The old paper's 775-cell/6,887-gene compact expression object is insufficient: cNMF discovery uses every Day-2 HSPC, including cells without a retained metaclone assignment. Original reference construction additionally needs the optional registered upstream objects and the author workflow. These upstream objects are unnecessary for the current analysis conditional on the retained reference. Public CLADES source attribution and the exact local-to-deposited object equivalence are separate provenance questions; no byte-equivalence is inferred from the dataset name.

[The common source-layout preparer](../singh/supporting_code/prepare_reproduction.py) accepts this study's source map with `--no-default-map --additional-map analysis/weinreb/provenance/source_file_map.json` and its corrected release input with `--additional-map analysis/weinreb/provenance/release_input_map.json`. It prepares fresh files and can explicitly adapt original path prefixes; preparation does not execute the notebooks or establish a successful full scientific rerun. Use the exact-input checks before writing a prepared workspace. The optional `--link-external` avoids copying the large verified H5AD inputs.

The same local resources used by the current workflow are retained, with their hashes: Hallmark GMT, KEGG mouse GMT and the curated hematopoietic membership table. They keep their upstream attribution/terms and are not relicensed by this repository's code license. No gene list was submitted to an external service during curation.

## Scientific scope

The current model uses clone-level Day-2 HSPC summaries. Original full testing families, library/source/normalization sensitivities and cNMF recovery criteria are preserved. HSPCs are pooled for biological interpretation. Source annotations remain only where needed to reproduce historical representation or explicit composition-sensitivity diagnostics. Metaclone numbers are Weinreb-specific.

The monocyte program association is retrospective and conditional on the selected reference. The cNMF output does not establish a stable uncaptured metaclone-specific program. Branch masks overlap; lower pseudotime is earlier transcriptional position, not a measured differentiation rate. Sparse or missing recovery remains explicit.

Superseded cell-DE manuscripts, the older source-state-split DE notebooks, unused legacy gene-volcano tables, document-transfer backups and review previews are outside this release. Original workspace copies remain intact; the [exclusion register](provenance/excluded_workspace_outputs.json) records the specifically omitted current-directory outputs.

The portable notebooks were run with Python 3.11.15 in the environment described by `environment.yml`. Publication paths and labels were adapted; scientific results were retained. [The adaptation record](provenance/publication_adaptation.json) distinguishes historical provenance hashes from current payload hashes.

## Figure panel folders

These folders make panels added after the first release. Each keeps its executed notebook, build script, panels and
tables, with neutral path names; [provenance/figure_panels.json](provenance/figure_panels.json) lists every
file with its source and current SHA-256.

| Folder | Panels |
|---|---|
| [`figure_panels/pseudotime_branch_panels_20260917`](figure_panels/pseudotime_branch_panels_20260917) | Figures 2C and S2B (cells on each branch) |
| [`figure_panels/complete_metaclone_overview_20260916`](figure_panels/complete_metaclone_overview_20260916) | Figures 2D and S2A; also the multilineage outputs and the 12 pseudotime tests in Table S2 |
| [`figure_panels/square_output_panels_20260915`](figure_panels/square_output_panels_20260915) | Figures 2E and 2G |
| [`figure_panels/fast_slow_clone_size_20261113`](figure_panels/fast_slow_clone_size_20261113) | Figure S2C (cells per clone, fast- vs slow-maturing metaclones) |
| [`figure_panels/figure3a_mc11_order_20261113`](figure_panels/figure3a_mc11_order_20261113) | Figure 3A |
| [`figure_panels/figure4_uniform_labels_20260918`](figure_panels/figure4_uniform_labels_20260918) | Figures 3C and 3D |
