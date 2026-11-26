#!/usr/bin/env bash
set -euo pipefail

ROOT="/path/to/ChronoClone/analysis_workspace/analyses/chronoclone_method_noise_benchmark_day0_locked_20260812"
PYTHON="/path/to/ChronoClone/.runtime/chronoclone/bin/python"
PID_FILE="$ROOT/results/controller.pid"
STATE_FILE="$ROOT/results/controller_state.json"
SESSION="cc_day0_locked_20260812"

mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/cache/numba" "$ROOT/cache/matplotlib"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Corrected benchmark controller is already running as PID $existing_pid"
    echo "Status: $PYTHON $ROOT/scripts/local_status.py"
    exit 0
  fi
fi

if /usr/bin/screen -list 2>/dev/null | /usr/bin/grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Detached screen session $SESSION already exists."
  echo "Status: $PYTHON $ROOT/scripts/local_status.py"
  exit 0
fi

/usr/bin/screen -DmS "$SESSION" "$ROOT/scripts/run_persistent_controller.sh"

for _ in {1..20}; do
  if [[ -f "$PID_FILE" ]]; then
    controller_pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$controller_pid" =~ ^[0-9]+$ ]] && kill -0 "$controller_pid" 2>/dev/null; then
      echo "Started corrected benchmark in screen session $SESSION (controller PID $controller_pid)."
      echo "Status: $PYTHON $ROOT/scripts/local_status.py"
      echo "Watch:  $PYTHON $ROOT/scripts/local_status.py --watch 30"
      exit 0
    fi
  fi
  /bin/sleep 0.25
done

echo "The screen session was launched but no live controller PID appeared." >&2
echo "Inspect $ROOT/logs/controller.out and $STATE_FILE" >&2
exit 1
