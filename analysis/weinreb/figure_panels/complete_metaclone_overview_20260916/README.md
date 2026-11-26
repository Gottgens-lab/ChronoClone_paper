# Weinreb metaclone overview

`complete_overview_and_analysis.ipynb` (built by `build_notebook.py`) draws the time-resolved density maps of every
metaclone and runs the multilineage composition and pseudotime analyses.

Panels used in the paper ([figure index](../../../../figures/figure_index.csv)):

- `figures/overview_neutrophil` and `figures/overview_monocyte`: Figure 2D
- `figures/overview_erythroid_mk`, `figures/overview_basophil` and `figures/overview_multilineage`: Figure S2A

Columns are metaclones and rows are Days 2, 4 and 6; every map has the same size and coordinate limits. `n` is the
number of cells and `C` the number of clones. Density counts cells of the same metaclone and day among the 100 nearest
neighbours. Colour maxima are fixed within each metaclone across days, so colours do not compare abundance between
metaclones. The grey background is rasterized; points and text are editable in the SVG files.

The notebook also runs the 12 within-day pseudotime tests reported in Table S2 (clone medians, at least three clones
per tested metaclone, Benjamini–Hochberg correction; `tables/pseudotime_within_day_omnibus_tests.csv`). The other
plots in `figures/` are records of the analysis and are not shown in the paper.
