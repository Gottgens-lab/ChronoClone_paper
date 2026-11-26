# Fixed-start ODE benchmark

Begin with [`../README.md`](../README.md) for the verified execution commands and
the boundary between released summaries, one portable smoke condition, and the
historical full production run.

| Location | Purpose |
|---|---|
| `notebooks/0001_simulation.ipynb` | Exact ODE generator, rate parameters and reference equivalence |
| `notebooks/0002_noise_generation.ipynb` | Fixed-start Gaussian and redistribution counts, saved tensors and historical identity checks |
| `notebooks/0004_cyclone_applicability_diagnostic.ipynb` | Portable abundance-only diagnostic; requires the repeat-level asset |
| `notebooks/0005_t1_locked_four_method_noise_results.ipynb` | Main four-method display |
| `notebooks/0003_t1_locked_method_noise_results.ipynb` | Complete five-method summary and applicability display |
| `notebooks/0006_benchmark_evidence_and_smoke.ipynb` | Hash/grid checks and one fresh clean-reference fit |
| `results/aggregate/` | Checksum-registered compact aggregate tables |
| `figures/manuscript_four_method/` | Individual SVG and PNG output stems from the main result notebook |
| `figures/validation_five_method/` | Individual SVGs from the complete summary notebook |
| `reports/` | Current evidence and bounded-fit execution records |
| `provenance/` | Historical artwork and Cyclone diagnostic notebooks |
| `../frozen_runtime/` | Exact scientific cores and numerical normalization module |

Historical artwork is retained as provenance. The panels used in the paper are listed in
[figures/figure_index.csv](../../figures/figure_index.csv).

## Generate the simulation inputs

From the paper repository root, use the recorded benchmark environment:

```bash
python benchmarks/run_notebook.py benchmarks/ode_noise/notebooks/0001_simulation.ipynb --kernel chronoclone-paper
python benchmarks/run_notebook.py benchmarks/ode_noise/notebooks/0002_noise_generation.ipynb --kernel chronoclone-paper
```

The first notebook exposes the original rates, initial conditions, RK45 equations,
rounding and scaling, and regenerates the canonical profiles exactly. Its outputs
are separate in `results/reproduced_reference/`. The second reconstructs nine
representative noise conditions in `results/reproduced_tensors/` and checks the
fixed first timepoint, structural zeros, totals, eligibility and deterministic
random streams. If the complete repeat-level asset is present, it also compares
the generated tensor and eligible-ID hashes against the historical rows.

Reconstruct any configured condition without an estimator fit:

```bash
python benchmarks/ode_noise/scripts/generate_counts.py --gaussian 20 --uniform 50 --repeat 0 --output benchmarks/ode_noise/reproductions/counts_g20_u50_r0.npz
```

The complete aggregate stores seeds and tensor hashes, not every serialized
historical count tensor. The generator reconstructs those inputs from the exact
frozen cores. This addition does not rerun the full historical grid or replace
its results. Original manuscript panels and aggregate tables retain their bytes.

For the complete Cyclone applicability check, first retrieve the registered asset:

```bash
python scripts/fetch_assets.py --asset ode-repeat-evidence
python benchmarks/run_notebook.py benchmarks/ode_noise/notebooks/0004_cyclone_applicability_diagnostic.ipynb --kernel chronoclone-paper
```

The maintained package includes the same synthetic workflows under
[in_silico](https://github.com/hiromitsuhari/ChronoClone/tree/main/in_silico).
