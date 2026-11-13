#!/usr/bin/env bash
# Full stage-1 run on one GPU: the GPU jobs one after another, in the README's order, and P0 (CPU only) in the
# background next to them from the start, on P0_THREADS CPUs (default 4; the GPU jobs get the other CPUs while P0
# runs). The Emmy check is not part of it (it needs nvcc and its own environment; run it separately on a 5090).
#
# Usage, from the pilot folder inside the venv, in tmux or with nohup so that a dropped SSH session does not stop it:
#   nohup bash run_all.sh > logs/run_all.out 2>&1 &
#   bash run_all.sh --dry-run           # list the jobs and what would be skipped, run nothing
#   bash run_all.sh --only p4_resnet18  # run one job (comma-separated list for several)
#   bash run_all.sh --skip p0_calibration   # leave jobs out (for example P0, which needs only a CPU, on the laptop)
# Any other argument is passed to every job (for example --device cuda:0).
#
# What it writes:
#   logs/<job>.log         the job's output (appended on a rerun, with a header per attempt)
#   logs/progress.txt      one line per job start and end: UTC time, exit code, elapsed hours, running total
#                          (the total adds up job hours; P0 runs alongside the GPU jobs, so it is not wall time)
#   logs/<job>.done        written only when the job exits with 0; a rerun of run_all.sh skips that job
#                          (delete the file to run the job again). A failed job is retried on the next rerun.
#   results/<job>/         the job's CSV files and run.json (--out results/<job>)
#   results_archive/<job>.tar.gz   results/<job>/ plus logs/<job>.log, packed after the job (also after a failure)
#   results_archive/pins.json      a copy of pins.json after each job
# A failed job does not stop the rest. New Hub names are pinned on first use and written to pins.json (the jobs
# run without --require-pins); at the end the script says whether pins.json gained names to upload.
#
# Copy command: set PILOT_SSH (user@host) and PILOT_SSH_PORT on this machine, so the script prints a ready
# scp line for your laptop; the host is not written into the repository.
set -u
cd "$(dirname "$0")"
ROOT=$(pwd)
PY=${PYTHON:-python}
export PYTHONUNBUFFERED=1
mkdir -p logs results results_archive
PROGRESS=logs/progress.txt

DRY=0
ONLY=""
SKIPJ=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    --skip) SKIPJ="$2"; shift ;;
    --require-pins) echo "run_all.sh: --require-pins would fail jobs on names not yet pinned; not passed on" >&2 ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done

# one run_all.sh at a time (a second one would run the same jobs twice and race on pins.json)
if [ "$DRY" = 0 ] && command -v flock > /dev/null; then
  exec 9> logs/.run_all.lock
  if ! flock -n 9; then echo "run_all.sh is already running (logs/.run_all.lock); not starting a second one" >&2; exit 1; fi
fi

# seconds since boot (monotonic: a wall-clock jump cannot distort elapsed hours)
now () { if [ -r /proc/uptime ]; then cut -d' ' -f1 /proc/uptime; else date +%s; fi; }
stamp () { date -u +%Y-%m-%dT%H:%M:%SZ; }
hours () { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", (b - a) / 3600 }'; }
# running total: elapsed hours of every END line so far, over all runs of run_all.sh
total () { [ -f "$PROGRESS" ] && awk '$2 == "END" { for (i = 1; i <= NF; i++) if ($i == "elapsed") s += $(i + 1) }
                                      END { printf "%.2f", s + 0 }' "$PROGRESS" || printf "0.00"; }
plus () { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", a + b }'; }
log () { if [ "$DRY" = 1 ]; then printf '%s\n' "$*"; else printf '%s\n' "$*" | tee -a "$PROGRESS"; fi; }

PINS_BEFORE=$(sha256sum pins.json 2>/dev/null | cut -d' ' -f1)
CURRENT=""
CHILD=""
BG_PID=""
LAUNCH=()
on_signal () {
  # the job runs in the background (so this trap fires at once); stop it too, then record the interruption
  if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2> /dev/null; wait "$CHILD" 2> /dev/null; fi
  if [ -n "$BG_PID" ]; then kill -TERM "$BG_PID" 2> /dev/null; wait "$BG_PID" 2> /dev/null; fi   # P0 as well
  if [ -n "$CURRENT" ]; then
    local el; el=$(hours "$T0" "$(now)")
    log "$(stamp)  END    $CURRENT  exit 130 (interrupted)  elapsed $el h  total $(plus "$(total)" "$el") h"
  fi
  exit 130
}
trap on_signal INT TERM HUP

run_job () {
  local job="$1"; shift
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$job,"* ]]; then return 0; fi
  if [[ ",$SKIPJ," == *",$job,"* ]]; then return 0; fi
  if [ -f "logs/$job.done" ]; then
    log "$(stamp)  SKIP   $job  (finished OK: $(cat "logs/$job.done"); delete logs/$job.done to rerun)"
    return 0
  fi
  if [ "$DRY" = 1 ]; then echo "would run  $job:  ${LAUNCH[*]:-}${LAUNCH[*]:+ }$* --out results/$job ${EXTRA[*]:-}"; return 0; fi
  CURRENT="$job"
  T0=$(now)
  log "$(stamp)  START  $job  total $(total) h"
  {
    echo "### $(stamp)  ${LAUNCH[*]:-}${LAUNCH[*]:+ }$*  --out results/$job ${EXTRA[*]:-}"
    echo "### git $(git rev-parse --short HEAD 2>/dev/null || echo '?'); free disk: $(df -h . | awk 'NR==2 {print $4}')"
  } >> "logs/$job.log"
  "${LAUNCH[@]}" "$@" --out "results/$job" "${EXTRA[@]}" >> "logs/$job.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"
  local rc=$?
  CHILD=""
  local el; el=$(hours "$T0" "$(now)")
  log "$(stamp)  END    $job  exit $rc  elapsed $el h  total $(plus "$(total)" "$el") h"
  CURRENT=""
  [ "$rc" = 0 ] && stamp > "logs/$job.done"
  # pack whatever the job wrote, also after a failure (atomic: the archive appears only when complete)
  if [ -d "results/$job" ]; then
    tar -czf "results_archive/$job.tar.gz.tmp" "results/$job" "logs/$job.log" \
      && mv "results_archive/$job.tar.gz.tmp" "results_archive/$job.tar.gz"
  fi
  [ -f pins.json ] && cp pins.json results_archive/pins.json
  return 0
}

JOBS="p0_calibration p5_escape p5_escape_h31 p2_w4a4 p4_transformer p4_resnet18 p1_pythia p1_polypythia p1_olmo p3_fractional p1_olmo7b"
for j in ${ONLY//,/ } ${SKIPJ//,/ }; do
  [[ " $JOBS " == *" $j "* ]] || { echo "run_all.sh: unknown job '$j'; jobs: $JOBS" >&2; exit 2; }
done

# CPUs: the last P0_THREADS CPUs of this process's set go to P0, the others to the GPU jobs while P0 runs
P0_THREADS=${P0_THREADS:-4}
read -r P0_CPUS GPU_CPUS < <(python3 - "$P0_THREADS" 2> /dev/null <<'PYEOF' || echo "- -"
import os, sys
k, cpus = int(sys.argv[1]), sorted(os.sched_getaffinity(0))
print("- -" if len(cpus) <= k else ",".join(map(str, cpus[-k:])) + " " + ",".join(map(str, cpus[:-k])))
PYEOF
)
PIN=0
command -v taskset > /dev/null && [ "${P0_CPUS:--}" != "-" ] && PIN=1
[ "$PIN" = 0 ] && echo "run_all.sh: not pinning CPUs (no taskset, or no more than $P0_THREADS CPUs); P0 is limited by thread counts only" >&2

# P0 needs only a CPU: start it in the background now, with every thread pool capped at P0_THREADS
THREADS=(env OMP_NUM_THREADS="$P0_THREADS" MKL_NUM_THREADS="$P0_THREADS" OPENBLAS_NUM_THREADS="$P0_THREADS"
         NUMEXPR_NUM_THREADS="$P0_THREADS" LOKY_MAX_CPU_COUNT="$P0_THREADS")
if [ "$PIN" = 1 ]; then LAUNCH=("${THREADS[@]}" taskset -c "$P0_CPUS"); else LAUNCH=("${THREADS[@]}"); fi
if [ "$DRY" = 0 ] && [ ! -f logs/p0_calibration.done ] && { [ -z "$ONLY" ] || [[ ",$ONLY," == *",p0_calibration,"* ]]; } \
   && [[ ",$SKIPJ," != *",p0_calibration,"* ]]; then
  ( BG_PID=""; trap on_signal INT TERM HUP; run_job p0_calibration $PY p0_calibration.py ) &
  BG_PID=$!
else
  run_job p0_calibration $PY p0_calibration.py     # SKIP line, or the dry-run listing
fi

# GPU jobs, one after another; while P0 runs they are kept off its CPUs
gpu_job () {
  if [ "$PIN" = 1 ] && [ -n "$BG_PID" ] && kill -0 "$BG_PID" 2> /dev/null; then LAUNCH=(taskset -c "$GPU_CPUS")
  elif [ "$PIN" = 1 ] && [ "$DRY" = 1 ] && [ ! -f logs/p0_calibration.done ]; then LAUNCH=(taskset -c "$GPU_CPUS")
  else LAUNCH=(); fi
  run_job "$@"
}

# README order: P5, P2, P4, P1, P3, then OLMo 2 7B last (P0 is already running)
gpu_job p5_escape        $PY p5_escape.py
gpu_job p5_escape_h31    $PY p5_escape.py --part h31
gpu_job p2_w4a4          $PY p2_w4a4.py
gpu_job p4_transformer   $PY p4_clipping.py --model transformer
gpu_job p4_resnet18      $PY p4_clipping.py --model resnet18
gpu_job p1_pythia        $PY p1_checkpoints.py --suite pythia
gpu_job p1_polypythia    $PY p1_checkpoints.py --suite polypythia,polypythia_data,polypythia_weight
gpu_job p1_olmo          $PY p1_checkpoints.py --suite olmo
gpu_job p3_fractional    $PY p3_fractional_memory.py
gpu_job p1_olmo7b        $PY p1_checkpoints.py --suite olmo7b

[ "$DRY" = 1 ] && exit 0
if [ -n "$BG_PID" ] && kill -0 "$BG_PID" 2> /dev/null; then
  echo "GPU jobs finished; waiting for P0 (logs/p0_calibration.log)"
fi
[ -n "$BG_PID" ] && wait "$BG_PID"
BG_PID=""
echo
echo "Done. Total so far: $(total) h. Status per job:"
grep -E '  (END|SKIP) ' "$PROGRESS" | awk '{ last[$3] = $0 } END { for (j in last) print "  " last[j] }' | sort -k3
failed=$(grep -E '  END ' "$PROGRESS" | awk '{ last[$3] = $5 } END { for (j in last) if (last[j] != 0) print j }' | sort)
[ -n "$failed" ] && echo "Failed (rerun run_all.sh to retry them; see logs/<job>.log): $(echo $failed)"

PINS_AFTER=$(sha256sum pins.json 2>/dev/null | cut -d' ' -f1)
if [ "$PINS_BEFORE" != "$PINS_AFTER" ] || ! git diff --quiet -- pins.json 2>/dev/null; then
  added=$(git show HEAD:pins.json 2>/dev/null | python3 -c 'import json, sys
old = json.load(sys.stdin); new = json.load(open("pins.json")); print(len(set(new) - set(old)))' 2>/dev/null || echo "?")
  echo "pins.json gained $added name(s) since the last commit: upload results_archive/pins.json to be committed."
fi

SSH=${PILOT_SSH:-USER@HOST}
PORT=${PILOT_SSH_PORT:-22}
echo
echo "On your laptop, copy the archives and pins.json with:"
echo "  scp -P $PORT '$SSH:$ROOT/results_archive/*' ."
