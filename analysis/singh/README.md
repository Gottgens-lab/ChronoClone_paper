# Singh / STRACK analyses

This directory holds the Singh analyses behind Figures 4 and S4. The figure panels
are collected in [figures/](../../figures/README.md). Dataset-specific metaclone
numbers refer only to this study.

Start with [the executed evidence check](0001_validate_published_evidence.ipynb).
It runs using shipped tables and checks resolved family identities, fixed
multiple-testing families, program means, preserved notebook execution, and all
individual atlas maps. It does not refit the reference partition or NMF.

| Evidence | Preserved source entry point | Shipped inputs and results |
| --- | --- | --- |
| Pooled WT-relative state/composition | `source_notebooks/dynamics/0001_family_aware_responses.ipynb` | `tables/dynamics/` |
| Original MC3 branch calculation | `source_notebooks/dynamics/0002_mc3_branch_responses.ipynb` | `tables/dynamics/` |
| Complete lineage comparison family | `source_notebooks/lineage_responses/0001_lineage_state_responses.ipynb` | `tables/lineage_responses/` |
| RNA-normalized early NMF | `source_notebooks/programs/0002_rna_normalized_day7_programs.ipynb` | `tables/programs/`, `checkpoints/programs/` |
| Gene-grounded program labels | `source_notebooks/program_annotations/0001_gene_grounded_program_annotations.ipynb` | `tables/program_annotations/` |
| Full-cell landscape and WT profiles | `source_notebooks/landscape/0001_singh_landscape_and_reference.ipynb` | `tables/landscape/` |
| Individual maps | `source_notebooks/atlas/0001_singh_weinreb_style_umaps.ipynb` | `atlas/svg/`, `tables/atlas/` |
| Map grids | `source_notebooks/grid_layout/0001_figure2b_grid_layout.ipynb` | `tables/grid_layout/` |

The current analytical registry contains 14,896 cells from 1,516 resolved
lineage families, including 2,713 Day-7 HSPCs from 1,514 families. Each family has
one fixed WT metaclone. HSC and MPP annotations are pooled; unresolved family
identifiers are excluded. Every family remains linked across sister-clones and
sequencing samples for uncertainty estimation. The state and composition
analyses have separate 68-test correction families; branch states use the
complete 340-test family. The NMF is ordinary fixed-rank NMF, not consensus NMF.

## Reproduction and provenance

Source notebooks retain their recorded scientific outputs. Publication path aliases
and environment labels have been adapted without changing numerical checkpoints
or embedded image bytes. [The copy plan](provenance/copy_plan.json) maps each
source-layout path alias to its published location and current SHA-256, with a
separate historical source digest. These path aliases require preparation before a fresh source
rerun. The preserved environment records give exact package versions;
`environment.yml` describes the environment used.

[The compact RNA export](0002_prepare_compact_rna_input.ipynb) supplies the
RNA-normalized sparse matrix, exact selected-gene input, ordered cells/genes,
and extraction receipt under `checkpoints/rna_input/`. With no external-path
environment variable it checks the shipped exports. With
`CHRONOCLONE_SINGH_H5AD` set to the exact registered source H5AD it recreates them.
The extraction reproduces feature selection and reconciles the retained model's
reconstruction error without refitting the model.

For historical source reruns, `supporting_code/prepare_reproduction.py` first
checks every copied source and configured external input, then creates a **new**
workspace with the recorded source layout. It refuses to overwrite an existing
directory. Supply an external-map JSON object mapping relative source aliases
in `provenance/external_inputs.json` to exact local files. `--adapt-paths`
optionally replaces the recorded workspace/environment prefixes in newly
prepared Python/notebook copies, clears notebook outputs, and writes a receipt.
It never executes prepared notebooks. Source snapshots here remain unchanged.
Use `--link-external` to keep verified large inputs at their supplied locations
and link them into the prepared workspace. Keep those external files available
and read-only. Run primary analyses before their dependent visual-review
notebooks, so runtime manifests and figure companions are regenerated together.

```sh
# Run from the repository root.
python analysis/singh/supporting_code/prepare_reproduction.py /new/reproduction/workspace \
  --external-map /path/to/external_inputs_map.json --check-only
python analysis/singh/supporting_code/prepare_reproduction.py /new/reproduction/workspace \
  --external-map /path/to/external_inputs_map.json --adapt-paths \
  --environment-prefix /path/to/installed/chronoclone/environment
```

For another study, use `--no-default-map` with its explicit `--additional-map`
files. This excludes every Singh source and external input. For example, the
Weinreb source map plus `analysis/weinreb/provenance/release_input_map.json`
require only the corrected release H5AD in addition to bundled source files.
Upstream objects recorded solely for historical audits are optional unless
those particular notebooks are selected for replay.

The source notebooks are not claimed to have been rerun from the prepared
workspace. The executed evidence check is the portable entry point that was
tested for this release. Source-preparation checkpoints are separate from
analysis recomputation.

The curation delivery receipt records a fresh-location preparation check with
large inputs linked, plus execution of both portable evidence notebooks from a
temporary repository copy. The temporary copies were removed after validation.

[Publication label validation](provenance/publication_label_validation.json)
records the subsequent label adaptation, unchanged numerical and image contents,
current source-map hashes, and notebook execution under the neutral runtime.

## Input and scientific boundaries

The original study is Singh et al., Cell Stem Cell (2025),
[doi:10.1016/j.stem.2025.01.012](https://doi.org/10.1016/j.stem.2025.01.012).
Original public data and code remain at
[GSE266232](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266232) and
[Zenodo](https://doi.org/10.5281/zenodo.14261507). The multi-gigabyte H5AD files are
listed as external inputs. The exact relationship between local processed H5ADs
and members of that upstream deposit must be checked before substituting files.
Small, derived figure-source tables and RNA checkpoints are included here.

The exact original fitting recipe for the retained 17-metaclone partition has
not been recovered. This release uses the explicit frozen assignment registry;
it does not claim end-to-end reconstruction of that original fitting step.
Likewise, the analyses condition on retained Palantir coordinates and overlapping
branch masks. Their original construction must not be inferred from a later
cell-neighbor graph. The provenance note under `supporting_inputs/` records the
bounded source search. Different tuned or de novo partitions are not replacements.

Results describe this observed experiment conditional on its fixed partition,
support and sample weights. Sequencing samples are not independent biological
replicates. Missing recovery is not extinction; density is not abundance; branch
masks are not exclusive fates. Program annotations characterize gene content and
do not establish causal mechanisms or prediction of mutation response.

Historical reporter-containing NMF fits, unresolved-family analyses, retired
survival/response classes, and the older 20-test branch stars are not the current
manuscript evidence. Necessary prior spectra remain only as explicitly registered
identity-alignment dependencies; source bytes are retained for provenance.

## Figure panels

Figure 4B comes from `source_notebooks/landscape/`, and Figures 4C and 4D from
`source_notebooks/dynamics/0001_family_aware_responses.ipynb`. The folders below make the
remaining panels. Each keeps its executed notebook, build script, panels and tables, with neutral
path names; [provenance/figure_panels.json](provenance/figure_panels.json) lists every
file with its source and current SHA-256. [The figure index](../../figures/figure_index.csv) links
each panel to its file and notebook.

| Folder | Panels |
|---|---|
| [`figure_panels/mc_condition_grid_flipped_20261021`](figure_panels/mc_condition_grid_flipped_20261021) | Figure 4E (MC0, MC3 and MC7 maps) |
| [`figure_panels/condition_grid_flipped_20261021`](figure_panels/condition_grid_flipped_20261021) | Figure 4E (density key) |
| [`figure_panels/s5_equal_umap_atlas_20260915`](figure_panels/s5_equal_umap_atlas_20260915) | Figure S4A–S4C (WT, Dnmt3a and Npm1c maps of every metaclone) |
