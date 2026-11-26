# Manuscript benchmark

The paper benchmark measures recovery of a known reference partition under
controlled post-start count variation and state redistribution. It is synthetic
evidence separate from the Weinreb and Singh biological analyses.

- [`ode_noise/`](ode_noise/) contains the reference, complete summary grid,
  executed result notebooks, individual editable SVG panels, and a portable
  evidence check.
- [`frozen_runtime/`](frozen_runtime/) preserves historical production
  cores and the exact numerical normalization module. Ancillary path labels are normalized. It is provenance for this experiment,
  not a second maintained ChronoClone package.
- [`curation/0001_curate_benchmark.ipynb`](curation/0001_curate_benchmark.ipynb)
  records the local assembly, neutral source labels, historical and publication hashes, and
  execution checks. It needs the original author workspace; ordinary readers
  start with the evidence notebook below.
- [`curation/0002_preservation_checks.ipynb`](curation/0002_preservation_checks.ipynb)
  records exact figure preservation and retained notebook execution outputs.
  Its original comparison needs the author workspace. Current preservation checks
  are in `curation/0003_publication_preservation.ipynb`.

The maintained package home is
[hiromitsuhari/ChronoClone](https://github.com/hiromitsuhari/ChronoClone).
The paper does not install a distribution named `chronoclone` from this repository.

## Reproduce the retained evidence

Use the recorded software versions in [`environment.yml`](environment.yml).
A reader may create a micromamba environment from this specification and register
its Jupyter kernel with a name of their choice; pass that name to `--kernel`. Portable checks discover the current checkout and use the active Python prefix.

From the paper repository root:

```bash
python benchmarks/run_notebook.py benchmarks/ode_noise/notebooks/0006_benchmark_evidence_and_smoke.ipynb --kernel chronoclone-paper
python benchmarks/run_notebook.py benchmarks/ode_noise/notebooks/0005_t1_locked_four_method_noise_results.ipynb --kernel chronoclone-paper
python benchmarks/run_notebook.py benchmarks/ode_noise/notebooks/0003_t1_locked_method_noise_results.ipynb --kernel chronoclone-paper
```

The first command checks summary hashes and all method/grid keys, then executes
one clean reference condition using the exact historical numerical module and cores. The
other commands regenerate individual plots from the released summaries. They
replace notebook outputs and timestamp-bearing SVG metadata; run them in a
working copy if you want to retain a pristine publication checkout.

The evidence notebook intentionally uses tables. The two result notebooks retain
the scientific plots, each in a dedicated calculation/construction/export cell.

## Run the bounded fit directly

```bash
python benchmarks/ode_noise/scripts/run_smoke.py --output benchmarks/ode_noise/reports/portable_clean_smoke.json
```

This wrapper fits only one condition, across ChronoClone, CLADES preprocessing,
K-means, and Ward. It verifies exact source hashes and the recorded dependency
versions, copies unchanged cores into a temporary layout, imports the included
unchanged numerical module directly, and applies the original eligibility and silhouette
rules. It writes a separate smoke record and cannot append to the historical
aggregate. Optional `--gaussian`, `--uniform`, and `--repeat` select one other
configured condition; only the clean condition is part of the retained
publication portability check.

## Full-grid boundary

The full historical experiment used 100 repeats of a 51-by-51 grid and all five
recorded adapters. The first sampling time is immutable. Cyclone remains in the
full aggregate as an abundance-only applicability diagnostic; estimator failures
have missing metrics and are not converted to zero accuracy. The manuscript's
accuracy display uses the four estimator-compatible methods.

The original five-method production controller, shard runner, strict schema,
aggregator, tests, environment records, and source hashes are preserved under
`frozen_runtime/production/` and `frozen_runtime/native_20260803/`. Scientific cores remain byte-identical; author path strings in ancillary
launchers and records use neutral labels. Historical experiment hashes are
retained separately from publication-file hashes. They are inspectable
production provenance, not a newly validated portable full-grid launcher. The
bounded wrapper does not weaken those guards or claim to rerun that controller.
A full independent rerun must establish its own runtime identity and output root;
do not merge its shards into the historical run.

The complete repeat-level aggregate is a separate versioned release asset because
it exceeds ordinary Git's per-file limit. See
[`ode_noise/LARGE_ASSET.json`](ode_noise/LARGE_ASSET.json) for its exact identity.
The paper repository's data/download manifest records the release location.
Summary plotting and the bounded smoke do not require this download; independent
repeat-level inspection and the historical Cyclone diagnostic do.

This benchmark does not establish biological generalization, condition-transfer
accuracy, or prospective prediction. Its percentile ribbons are empirical
Monte Carlo ranges under the declared generator.

The original wheel is excluded from the publication tree. The sole package function used by the benchmark is retained byte-for-byte under `frozen_runtime/normalization_runtime/`, with a minimal import entrypoint and its original license. See [`frozen_runtime/provenance/numerical_runtime.json`](frozen_runtime/provenance/numerical_runtime.json) for the extraction hashes and scope.
