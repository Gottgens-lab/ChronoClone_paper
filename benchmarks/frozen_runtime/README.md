# Benchmark numerical implementation

The numerical cores, configurations and reference profiles retain their exact
historical bytes. `normalization_runtime/chronoclone/normalization.py` is extracted
byte-for-byte from the recorded distribution and is the only package function
called by the benchmark. A minimal import entrypoint and the original license
are included. This is not an installable package distribution.

The original wheel is excluded from this publication tree. Its identifier and
the exact extracted module hashes are recorded in
[`provenance/numerical_runtime.json`](provenance/numerical_runtime.json).
Historical source manifests continue to identify the original experiment;
current file hashes are in `../curation/publication_frozen_source_manifest.sha256`.
The benchmark's recorded aggregate and hash contract are unchanged.

The portable smoke wrapper verifies and loads this numerical module alongside
the exact scientific cores. Historical production launchers retain their original
topology and strict runtime guards; they are provenance, not the portable entrypoint.
No new shards are merged with the historical result grid.

The CLADES comparator implements mouse preprocessing from CLADES/clonaltrans
v1.2.0, commit `a0442270b48614cad86a582f48695cab54b480cf`, rather than the
downstream NeuralODE/Gillespie model. The Cyclone adapter is its documented
abundance core and remains an applicability diagnostic.
