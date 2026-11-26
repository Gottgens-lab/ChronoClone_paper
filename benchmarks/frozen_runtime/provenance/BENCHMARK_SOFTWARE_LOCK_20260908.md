# ODE benchmark production software lock

The completed 51 × 51 aggregate records the historical production experiment.
Its scientific cores are in `../production/` and `../native_20260803/`. The table
below preserves historical input hashes; current publication hashes are in
`../../curation/publication_frozen_source_manifest.sha256`. Ancillary path labels
have been normalized, so some current file hashes differ from this table.

| Artifact | SHA-256 |
|---|---|
| `benchmark_package/scripts/benchmark_core.py` | `4641e28b317308473a65472fbf2ebf533db41e37f2954d9702991c110bdb5ca1` |
| `benchmark_package/environment.yml` | `b271432465ad809c47e9d37994e87430627cdba442694c11edef03eb8ca00d16` |
| `benchmark_package/R/validate_cyclone_core.R` | `211be308929ca1d68d575ea5a79f002882c93bdd167cd843f82b59c3c143c9a2` |
| `results/provenance/environment_manifest.json` | `9c642ec81219966897cdf2015f2eff3288668900d70c6f031461aad5ae87febd` |
| `results/provenance/local_conda_records.json` | `73ba2a1da7f6968c27dfd392ef7872a6724767305ef703980f99e8658b968ccf` |
| `results/provenance/wheels/0b09e41a381e4e2219160a3ea513aae008cae14d/chronoclone-0.1.0-py3-none-any.whl` | `2661c65f9137768fa107e59204f308dd001c501c1b503e0d82f4a6f6def499c4` |

The wheel hash identifies the original ChronoClone 0.1.0 artifact recorded for Git commit
`0b09e41a381e4e2219160a3ea513aae008cae14d`. The original wheel is excluded from the publication tree; see `numerical_runtime.json`. The current bounded wrapper loads
the unchanged numerical module extracted from that distribution in an isolated Python subprocess and fits the clean ODE
reference. The paper contains no maintained package API. Package development
belongs in the separate software repository.

The production scripts retain historical activation structure with neutral
path labels. They are inspectable provenance, not a validated portable full-grid
launcher. Full-grid regeneration requires establishing the production topology
and exact software environment. The separately registered 153,477,867-byte
repeat-level aggregate supports inspection of the historical results. Ordinary manuscript reproduction
uses the compact aggregate and reader-facing notebooks instead.
