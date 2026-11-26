# Weinreb pseudotime branches

`pseudotime_branch_panels.ipynb` (built by `build_notebook.py`) draws the cells on each Palantir branch on the
landscape, using the stored coordinates, pseudotime and branch masks; nothing is refitted.

- `panels/figure_2c_branch_cells_neu_mono`: Figure 2C (neutrophil branch, n = 5,488 cells; monocyte branch, n = 3,329)
- `panels/figure_s2_branch_cells_meg_mast_baso`: Figure S2B (megakaryocyte, n = 1,045; mast cell, n = 1,002;
  basophil, n = 1,739)

Branches overlap, because a cell can be assigned to more than one branch. `panels/figure_2b_pseudotime_landscape` is a
record of the analysis and is not shown in the paper.
