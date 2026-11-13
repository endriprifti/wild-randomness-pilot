#!/usr/bin/env bash
# Runs the unit tests and every --smoke job, cheapest first, and writes results/smoke_summary.csv.
# Usage (from the pilot folder, inside the venv):  bash run_smoke.sh [--device cuda|cpu]
# Each job must finish in under 300 s; a slower job is reported as SLOW, a failing one as FAIL.
set -u
cd "$(dirname "$0")"
EXTRA="$*"
PY=${PYTHON:-python}
mkdir -p results
SUMMARY=results/smoke_summary.csv
echo "job,status,seconds" > "$SUMMARY"

# Seconds since boot from /proc/uptime: monotonic, so a wall-clock jump (WSL2 resyncing its clock after the host
# sleeps) cannot make a duration negative or inflate it. Falls back to `date` where /proc/uptime is missing.
now () {
  if [ -r /proc/uptime ]; then cut -d' ' -f1 /proc/uptime | cut -d. -f1; else date +%s; fi
}

run_job () {
  local name="$1"; shift
  local t0=$(now)
  echo "=== $name: $*"
  if "$@" > "results/smoke_${name}.log" 2>&1; then status=PASS; else status=FAIL; fi
  local dt=$(( $(now) - t0 ))
  if [ "$status" = PASS ] && [ "$dt" -gt 300 ]; then status=SLOW; fi
  echo "$name,$status,$dt" >> "$SUMMARY"
  echo "    $status in ${dt}s (log: results/smoke_${name}.log)"
}

run_job unit_tests      $PY -m pytest -q
run_job p0_calibration  $PY p0_calibration.py --smoke $EXTRA
run_job emmy_check      $PY emmy_check.py --smoke $EXTRA
run_job p5_escape       $PY p5_escape.py --smoke $EXTRA
run_job p5_escape_h31   $PY p5_escape.py --smoke --part h31 $EXTRA
run_job p2_w4a4         $PY p2_w4a4.py --smoke $EXTRA
run_job p4_clipping     $PY p4_clipping.py --smoke $EXTRA
run_job p1_checkpoints  $PY p1_checkpoints.py --smoke $EXTRA
run_job p3_fractional   $PY p3_fractional_memory.py --smoke $EXTRA

echo
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
! grep -q ",FAIL," "$SUMMARY"
