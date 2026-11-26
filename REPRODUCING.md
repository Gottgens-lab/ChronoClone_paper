# Reproducing the manuscript evidence

Run all commands from the repository root. These materials support three different operations: reading the saved results, checking the published evidence, and rerunning scientific workflows from their registered inputs.

## Inspect and verify

All figure panels (`figures/`), the supplementary workbooks (`supplementary_tables/`) and the executed notebooks are included; the manuscript text is not. They can be read without running code.

```bash
python scripts/verify_repository.py
```

This standard-library check verifies the shipped file inventory, current checksums, figure coverage, table links, notebook state, publication labels, and separation from the maintained package. Label checks include current path names, file contents, JSON strings, and decompressed archive members. CI runs the same check. It is an integrity check, not a rerun of the scientific analyses.

## Run the portable evidence notebooks

The review dependencies, including the historical benchmark smoke requirements, are pinned in `requirements-review.txt`. Python 3.11.15 is required for the exact benchmark environment guard. The benchmark-only equivalent is `benchmarks/environment.yml`. To create a new environment with micromamba:

```bash
micromamba create -f environment.yml
micromamba run -n chronoclone-paper-review python -m ipykernel install --user --name chronoclone-paper-review
```

Choose that kernel for portable review notebooks in the benchmark, Weinreb, and Singh entry points. Their checks operate on the supplied tables and compact checkpoints. Generated outputs belong in the ignored `generated/` directory.

The saved notebooks were run with Python 3.11.15; `environment.yml` describes that environment for reuse on another machine.

## Scientific reruns

Each study README identifies its current analyses and inputs. `source_notebooks/` and `current_source/` hold scientific notebook snapshots with retained results and normalized publication paths and metadata. Original source hashes are historical identifiers; current publication hashes describe the adapted files. Image payload adaptations preserve their decoded bytes. Retained historical outputs do not establish a fresh full rerun in this repository.

Where provided, the study reproduction preparer reconstructs the required source layout from checksum-registered publication files and external inputs. It checks missing dependencies before launching work. Inspect the generated notebook and its environment settings before a full run. Original source versions remain identified by their historical digests.

The fixed-reference downstream analyses can be reconstructed from the registered checkpoints. Recreating the original Singh 17-metaclone fit from raw sequencing remains outside the verified provenance. The supplied 20-metaclone alternative must not be substituted. Similarly, reproducing the exact upstream Weinreb landscape requires the registered source, not an assumed equivalence to a similarly named public file.

The complete ODE benchmark contains 260,100 noise realizations per method. Its repeat-level archive is a versioned release asset. A bounded runtime smoke check establishes launcher behavior; it does not repeat the entire benchmark or replace its original execution record.
