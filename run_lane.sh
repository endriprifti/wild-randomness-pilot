#!/usr/bin/env bash
# A second lane for P1 jobs, next to a running run_all.sh, with run_all.sh's bookkeeping, so that run_all.sh's
# resume skips them later. Jobs run one after another, each with the same command and --out folder as in
# run_all.sh, and for each one this script writes what run_job writes:
#   logs/progress.txt      START and END lines in run_all.sh's format, tagged "(lane)" at the end
#   logs/<job>.log         the job's output, appended, with the same header
#   logs/<job>.done        only when the job exits with 0 (run_all.sh then skips the job)
#   results_archive/<job>.tar.gz and results_archive/pins.json, written atomically (temporary file, then mv)
#
# Usage, in the pilot folder inside the venv (CPUs 7-10, four threads per pool):
#   nohup taskset -c 7-10 bash run_lane.sh p1_pythia p1_polypythia p1_olmo > logs/run_lane.out 2>&1 &
#   bash run_lane.sh --dry-run p1_pythia p1_polypythia p1_olmo       # checks only, runs nothing
#
# Safety checks before each job:
#   - only P1 jobs: p1_checkpoints.py holds an exclusive GPU lock (logs/.p1_gpu.lock) for its whole run, so
#     at most one P1 job uses the GPU at a time, whether it runs here or in run_all.sh (p1_olmo7b waits);
#   - a job with logs/<job>.done is skipped; a job that progress.txt shows as running is refused;
#   - while run_all.sh runs (it holds logs/.run_all.lock), only jobs that come before its current job in its
#     order are accepted, since run_all.sh will not run those again;
#   - the GPU must have the job's peak (python memory_check.py p1) plus LANE_HEADROOM_GIB free (default 4);
#     otherwise the lane waits, checking every minute.
# Until the lane has finished, start any new run_all.sh with --skip for the lane's jobs.
#
# P1's CPU statistics run in a pool of processes next to the GPU pass, one per CPU of the lane (taskset), and on
# every CPU once run_all.sh's progress.txt shows p3_fractional ended (P1_CPUS_AFTER, passed in the environment so
# the commands stay run_all.sh's; P1_CPUS_AFTER= turns it off). A file logs/p1_cpus (e.g. 0-10) sets the CPUs
# at any time; the job reads it every minute.
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
THREADS=${LANE_THREADS:-4}
HEADROOM=${LANE_HEADROOM_GIB:-4}
export P1_CPUS_AFTER="${P1_CPUS_AFTER-p3_fractional=all}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS" LOKY_MAX_CPU_COUNT="$THREADS"
mkdir -p logs results results_archive
PROGRESS=logs/progress.txt

DRY=0
JOBS_HERE=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) JOBS_HERE+=("$a") ;;
  esac
done
[ "${#JOBS_HERE[@]}" -gt 0 ] || { echo "usage: bash run_lane.sh [--dry-run] p1_pythia p1_polypythia p1_olmo" >&2; exit 2; }

# run_all.sh's job order and the P1 commands, exactly as there (tests/test_run_lane.py compares them)
ALL_JOBS="p0_calibration p5_escape p5_escape_h31 p2_w4a4 p4_transformer p4_resnet18 p1_pythia p1_polypythia p1_olmo p3_fractional p1_olmo7b"
job_args () {
  case "$1" in
    p1_pythia)     echo "p1_checkpoints.py --suite pythia" ;;
    p1_polypythia) echo "p1_checkpoints.py --suite polypythia,polypythia_data,polypythia_weight" ;;
    p1_olmo)       echo "p1_checkpoints.py --suite olmo" ;;
    p1_olmo7b)     echo "p1_checkpoints.py --suite olmo7b" ;;
    *) return 1 ;;
  esac
}
# peak GPU memory in GiB by python memory_check.py p1 (2026-11-16; tensors only, so the headroom adds the CUDA
# context and allocator rounding)
job_gib () {
  case "$1" in
    p1_pythia) echo 16.3 ;; p1_polypythia) echo 7.7 ;; p1_olmo) echo 17.3 ;; p1_olmo7b) echo 31.5 ;;
  esac
}
index_of () { local i=0; for j in $ALL_JOBS; do [ "$j" = "$1" ] && { echo $i; return; }; i=$((i + 1)); done; echo -1; }

if [ "$DRY" = 0 ] && command -v flock > /dev/null; then
  exec 8> logs/.run_lane.lock
  flock -n 8 || { echo "run_lane.sh is already running (logs/.run_lane.lock)" >&2; exit 1; }
fi

stamp () { date -u +%Y-%m-%dT%H:%M:%SZ; }
now () { if [ -r /proc/uptime ]; then cut -d' ' -f1 /proc/uptime; else date +%s; fi; }
hours () { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", (b - a) / 3600 }'; }
total () { [ -f "$PROGRESS" ] && awk '$2 == "END" { for (i = 1; i <= NF; i++) if ($i == "elapsed") s += $(i + 1) }
                                      END { printf "%.2f", s + 0 }' "$PROGRESS" || printf "0.00"; }
plus () { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", a + b }'; }
# one write per line with O_APPEND: lines from run_all.sh and from this lane never mix within a line
log () { printf '%s\n' "$*"; [ "$DRY" = 1 ] || printf '%s\n' "$*" >> "$PROGRESS"; }

# jobs that progress.txt shows as started and not ended (lane lines included)
running_jobs () {
  [ -f "$PROGRESS" ] || return 0
  awk '$2 == "START" { s[$3] = 1 } $2 == "END" { delete s[$3] } END { for (j in s) print j }' "$PROGRESS"
}
run_all_active () {
  [ -f logs/.run_all.lock ] || return 1
  command -v flock > /dev/null || return 0                        # cannot tell: assume it runs
  ! flock -n logs/.run_all.lock true 2> /dev/null
}
gpu_free_gib () {
  command -v nvidia-smi > /dev/null || { echo ""; return; }
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | awk '{ printf "%.1f", $1 / 1024 }'
}

check_job () {   # prints why a job cannot run here (nothing if it can)
  local job="$1"
  job_args "$job" > /dev/null || { echo "not a P1 job (only P1 jobs hold the GPU lock that keeps lanes apart)"; return; }
  if running_jobs | grep -qx "$job"; then echo "progress.txt shows it running"; return; fi
  if run_all_active; then
    local cur; cur=$(running_jobs | grep -vx p0_calibration | while read -r j; do echo "$(index_of "$j") $j"; done \
                     | sort -n | tail -1 | cut -d' ' -f2)
    if [ -z "$cur" ]; then echo "run_all.sh is running but progress.txt shows no current job"; return; fi
    if [ "$(index_of "$job")" -ge "$(index_of "$cur")" ]; then
      echo "run_all.sh (now at $cur) will still run it; use run_all.sh for it, or stop run_all.sh first"; return
    fi
  fi
}

CHILD=""
CURRENT=""
on_signal () {
  if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2> /dev/null; wait "$CHILD" 2> /dev/null; fi
  if [ -n "$CURRENT" ]; then
    local el; el=$(hours "$T0" "$(now)")
    log "$(stamp)  END    $CURRENT  exit 130 (interrupted)  elapsed $el h  total $(plus "$(total)" "$el") h  (lane)"
  fi
  exit 130
}
trap on_signal INT TERM HUP

for job in "${JOBS_HERE[@]}"; do
  if [ -f "logs/$job.done" ]; then
    log "$(stamp)  SKIP   $job  (finished OK: $(cat "logs/$job.done"); delete logs/$job.done to rerun)  (lane)"
    continue
  fi
  why=$(check_job "$job")
  if [ -n "$why" ]; then
    echo "run_lane.sh: not running $job: $why" >&2
    [ "$DRY" = 1 ] && continue || exit 1
  fi
  need=$(awk -v a="$(job_gib "$job")" -v b="$HEADROOM" 'BEGIN { printf "%.1f", a + b }')
  read -r -a ARGS <<< "$(job_args "$job")"
  if [ "$DRY" = 1 ]; then
    echo "would run  $job:  $PY ${ARGS[*]} --out results/$job   (needs $need GiB free; free now: $(gpu_free_gib) GiB;" \
         "CPUs $(python3 -c 'import os; print(sorted(os.sched_getaffinity(0)))' 2> /dev/null))"
    continue
  fi
  while :; do
    free=$(gpu_free_gib)
    if [ -z "$free" ]; then echo "run_lane.sh: nvidia-smi not found; not checking free GPU memory" >&2; break; fi
    awk -v f="$free" -v n="$need" 'BEGIN { exit !(f >= n) }' && break
    echo "$(stamp) run_lane.sh: $job needs $need GiB free, $free GiB free; checking again in 60 s" >&2
    sleep 60
  done
  CURRENT="$job"
  T0=$(now)
  log "$(stamp)  START  $job  total $(total) h  (lane)"
  {
    echo "### $(stamp)  run_lane.sh: $PY ${ARGS[*]}  --out results/$job  (CPUs: $(python3 -c 'import os; print(sorted(os.sched_getaffinity(0)))' 2> /dev/null))"
    echo "### git $(git rev-parse --short HEAD 2>/dev/null || echo '?'); free disk: $(df -h . | awk 'NR==2 {print $4}')"
  } >> "logs/$job.log"
  "$PY" "${ARGS[@]}" --out "results/$job" >> "logs/$job.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"
  rc=$?
  CHILD=""
  el=$(hours "$T0" "$(now)")
  log "$(stamp)  END    $job  exit $rc  elapsed $el h  total $(plus "$(total)" "$el") h  (lane)"
  CURRENT=""
  [ "$rc" = 0 ] && stamp > "logs/$job.done"
  if [ -d "results/$job" ]; then
    tar -czf "results_archive/$job.tar.gz.lane.tmp" "results/$job" "logs/$job.log" \
      && mv "results_archive/$job.tar.gz.lane.tmp" "results_archive/$job.tar.gz"
  fi
  [ -f pins.json ] && cp pins.json results_archive/pins.json.lane.tmp && mv results_archive/pins.json.lane.tmp results_archive/pins.json
done
[ "$DRY" = 1 ] || echo "run_lane.sh: done. Status: $(grep -E '  END .*\(lane\)' "$PROGRESS" | awk '{ print $3 " exit " $5 }' | tr '\n' ';')"
