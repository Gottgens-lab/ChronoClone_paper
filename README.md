# ChronoClone paper

Notebooks, results and figures for the paper **ChronoClone links temporal clonal behavior to lineage output and
mutation responses**. The ChronoClone software is in the separate
[ChronoClone repository](https://github.com/hiromitsuhari/ChronoClone); this repository holds the analyses for the paper.

| Folder | Contents |
|---|---|
| [`figures/`](figures/README.md) | Every panel of Figures 1–4 and S1–S4 (SVG and PNG), each linked to the notebook that made it |
| [`supplementary_tables/`](supplementary_tables/README.md) | Tables S1–S6 and the Key Resources Table |
| [`benchmarks/ode_noise/`](benchmarks/ode_noise/README.md) | Simulation benchmark (Figures 1 and S1) |
| [`analysis/weinreb/`](analysis/weinreb/README.md) | Weinreb et al. LARRY data (Figures 2, 3, S2 and S3) |
| [`analysis/singh/`](analysis/singh/README.md) | Singh et al. mutation-tracing data (Figures 4 and S4) |
| `data/` | Registry of the large input files and how to download them |
| `scripts/`, `provenance/` | Repository checks and the origin and checksum of every file |

## Citation

Harimoto H, Zheng W, Göttgens B, Barile M, Wilson NK. ChronoClone links temporal clonal behavior to lineage output and mutation responses. *bioRxiv* (2026). [doi:10.64898/2026.11.24.768931](https://doi.org/10.64898/2026.11.24.768931)

## Checking and rerunning

`python scripts/verify_repository.py` checks every file against `MANIFEST.sha256.tsv`, confirms that each figure panel
and its notebook are present, and checks the notebooks' saved state. It does not rerun the analyses.
[REPRODUCING.md](REPRODUCING.md) explains how to rerun the notebooks, and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)
lists the input data.

The notebooks keep their saved outputs. File paths inside them use neutral workspace names; the original and current
checksum of every copied file are recorded in `provenance/` and in each study's `provenance/` folder.

Two inputs limit what can be rebuilt from scratch: the recipe that produced the Singh 17-metaclone reference is not
fully registered, so the Singh analyses start from the saved metaclone labels; and the Weinreb analyses start from a
registered processed object whose exact relationship to the public CLADES download is not established.
