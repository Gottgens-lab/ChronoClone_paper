#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/ChronoClone/analysis_workspace/analyses/chronoclone_method_noise_benchmark_day0_locked_20260812"
PYTHON="/path/to/ChronoClone/.runtime/chronoclone/bin/python"
SOURCE_ROOT="/path/to/ChronoClone/analysis_workspace/analyses/native_method_noise_benchmark_local_20260803"
LOG="$ROOT/logs/controller.out"

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR="$ROOT/cache/numba"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export IPYTHONDIR="$ROOT/cache/ipython"
export JUPYTER_CONFIG_DIR="$ROOT/cache/jupyter"
export PYTHONPATH="$ROOT:$ROOT/scripts:$SOURCE_ROOT/runtime/site-packages:$SOURCE_ROOT/benchmark_package/scripts"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Refuse a mixed-provenance run: changing any scientific/controller source
# while production is active changes this manifest and aborts before new work.
SOURCE_MANIFEST="$ROOT/results/provenance/production_source_manifest.sha256"
CURRENT_MANIFEST="$ROOT/results/provenance/.production_source_manifest.current"
mkdir -p "$ROOT/results/provenance"
/usr/bin/shasum -a 256 \
  "$ROOT/config/benchmark_config.json" \
  "$ROOT/corrected_core.py" \
  "$ROOT/scripts/source_runtime.py" \
  "$ROOT/scripts/cyclone_adapter.py" \
  "$ROOT/scripts/result_schema.py" \
  "$ROOT/scripts/run_chunk.py" \
  "$ROOT/scripts/run_local_benchmark.py" \
  "$ROOT/scripts/local_status.py" \
  > "$CURRENT_MANIFEST"
if [[ -f "$SOURCE_MANIFEST" ]]; then
  /usr/bin/cmp -s "$CURRENT_MANIFEST" "$SOURCE_MANIFEST" || {
    echo "Production source manifest changed; refusing mixed-provenance resume." >&2
    exit 2
  }
else
  /bin/mv "$CURRENT_MANIFEST" "$SOURCE_MANIFEST"
fi
/bin/rm -f "$CURRENT_MANIFEST"

mkdir -p "$ROOT/logs" "$ROOT/results" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR"
cd "$ROOT"

exec /usr/bin/caffeinate -i -m \
  "$PYTHON" "$ROOT/scripts/run_local_benchmark.py" --workers 8 \
  >>"$LOG" 2>&1
