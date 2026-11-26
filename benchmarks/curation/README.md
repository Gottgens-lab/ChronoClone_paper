# Benchmark curation records

The two earlier curation notebooks preserve the preceding assembly and figure
checks. Their author paths are now neutral provenance labels; they require the
author archives and are not clean-checkout reproduction entry points. Their
saved numerical results remain intact. The historical builder likewise records
how those notebooks were created.

Run `0003_publication_preservation.ipynb` for current portable preservation
checks. Run `../ode_noise/notebooks/0006_benchmark_evidence_and_smoke.ipynb` for
summary verification and the bounded clean-condition fit.

`copy_manifest.json` separates `source_sha256` (registered historical input),
`pre_publication_execution_sha256` (earlier generated output, where recorded),
and `publication_sha256` (current shipped bytes). Neutral source labels do not
claim that the original author directories were renamed. Source hashes inside
the aggregate remain the original experiment identifiers.

`publication_scientific_baseline.json` registers byte-identical scientific
artifacts and unchanged scientific notebook cells. Embedded image bytes remain
exact; path-only adaptations and display metadata are recorded separately.
