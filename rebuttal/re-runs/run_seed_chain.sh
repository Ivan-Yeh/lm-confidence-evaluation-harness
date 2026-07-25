#!/usr/bin/env bash
# Rebuttal rerun: TruthfulQA direct-QA sampling (lc/tp/su) -> in-domain calibration
# (100% lexicon, no beta-guided variant), for one model on one GPU, across 5
# independent seeds.
#
# Usage: run_seed_chain.sh <GPU> <MODEL> <REASONING_EFFORT>
#   REASONING_EFFORT is "low" or "null" (literal string "null" disables it, matching
#   scripts/local_direct_truthful_qa.sh's per-model convention).
#
# Each seed does, sequentially on the same GPU:
#   1. direct_qa_unified_lc  - fresh generation (repeat=20, temp=1, seeded), the
#      only expensive stage; also runs grading + majority-cluster building.
#   2. direct_qa_unified_tp  - cache_path=<lc dir>; reuses cached generations/
#      clusters/grading, only recomputes the tp confidence signal. Cheap.
#   3. direct_qa_unified_su  - same, cheap.
#   4. calibration.in_domain_calibration --seed <seed> --lc_path/--tp_path/--su_path
#      pointed explicitly at this seed's own dirs (never "latest", to avoid
#      races with other users' concurrent runs of the same model name).
#
# Every stage is cache-aware (see lm_conf/__main__.py and
# calibration/in_domain_calibration.py), so re-running this script after a
# crash just resumes/skips completed work.
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <GPU> <MODEL> <REASONING_EFFORT>" >&2
  exit 1
fi

GPU="$1"
MODEL="$2"
REASONING_EFFORT="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/ivan/miniconda3/envs/llm/bin/python"

# SCRIPT_DIR first so `lm_conf` / `calibration` resolve to the patched copies
# here; PROJECT_ROOT as fallback so untouched packages (e.g.
# linguistic_confidence_lexicon, used by calibration/utils.py) resolve to the
# real repo without needing to be duplicated.
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="$GPU"

SEEDS=(1 2 3 4 5)
DATASET=truthful_qa
LIMIT=null
ROUNDS=1
MAX_TOKENS=256
MAX_MODEL_LEN=2048
MIN_FREE_GB=10
# Mirrors the common_dir logic in calibration/in_domain_calibration.py.
if [[ -d /hdd ]]; then
  COMMON_DIR=/hdd/ivny
else
  COMMON_DIR=ivny
fi

SAFE_MODEL="$(echo "$MODEL" | tr '/' '_')"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR"

check_disk_space() {
  local free_gb
  free_gb=$(df --output=avail -BG /hdd 2>/dev/null | tail -1 | tr -dc '0-9')
  if [[ -n "$free_gb" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    echo "!!! ABORTING: only ${free_gb}GB free on /hdd (< ${MIN_FREE_GB}GB threshold). model=$MODEL" >&2
    exit 2
  fi
}

extract_results_path() {
  # Pulls the path from the __main__.py log line: "All results saved to: <path>"
  local logfile="$1"
  grep -oP 'All results saved to: \K.*' "$logfile" | tail -1
}

# GPUs here are shared with other users; a neighbor's job can occupy one for
# hours. GPU_HEADROOM_MIB is a rough floor for loading our largest model
# (gpt-oss-20b: ~13.7GiB weights) plus KV cache/activations with margin.
GPU_HEADROOM_MIB=20000
GPU_POLL_INTERVAL=60
GPU_POLL_LOG_EVERY=10   # log a status line every Nth poll (~10min at 60s) to avoid log spam
MAX_ATTEMPTS_AFTER_HEADROOM=10  # bounds retries once memory *looked* fine, to catch real bugs

# Blocks indefinitely (no timeout) until $GPU reports >= GPU_HEADROOM_MIB free.
# This is the "always be ready to load" loop: as soon as the neighbor's job
# frees enough VRAM, we proceed immediately.
wait_for_gpu_headroom() {
  local poll_count=0
  local free_mib
  while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -dc '0-9')
    if [[ -n "$free_mib" && "$free_mib" -ge "$GPU_HEADROOM_MIB" ]]; then
      return 0
    fi
    if (( poll_count % GPU_POLL_LOG_EVERY == 0 )); then
      echo "  ... [$(date '+%F %T')] GPU$GPU has ${free_mib:-?}MiB free (< ${GPU_HEADROOM_MIB}MiB needed); waiting" >&2
    fi
    poll_count=$((poll_count + 1))
    sleep "$GPU_POLL_INTERVAL"
  done
}

# Waits for apparent GPU headroom, then attempts the command. Every stage we
# run is cache-aware (raw generation / graded-outputs / calibration pkl
# caches), so re-attempting after a transient failure (e.g. a neighbor's job
# grabbing VRAM in the race between our headroom check and the actual model
# load) just resumes cheaply instead of redoing completed work. If memory
# looks fine but the command still fails MAX_ATTEMPTS_AFTER_HEADROOM times in
# a row, we give up and surface it (likely a real bug, not contention).
retry_with_backoff() {
  local logfile="$1"; shift
  local attempt=1
  while true; do
    wait_for_gpu_headroom
    if "$@" >> "$logfile" 2>&1; then
      return 0
    fi
    if [[ "$attempt" -ge "$MAX_ATTEMPTS_AFTER_HEADROOM" ]]; then
      return 1
    fi
    local backoff=$((30 * attempt))
    echo "  ... attempt $attempt/$MAX_ATTEMPTS_AFTER_HEADROOM failed despite apparent headroom, retrying in ${backoff}s (see $logfile)" >&2
    sleep "$backoff"
    attempt=$((attempt + 1))
  done
}

run_sampling_stage() {
  local task="$1" seed="$2" logfile="$3" cache_path="${4:-}"
  local extra_args=()
  if [[ -n "$cache_path" ]]; then
    extra_args+=("cache_path=$cache_path")
  fi
  : > "$logfile"
  retry_with_backoff "$logfile" \
    "$PYTHON_BIN" -m lm_conf dataset=$DATASET \
    rounds=$ROUNDS \
    limit=$LIMIT \
    task=$task \
    qa_model.name="$MODEL" \
    qa_model.backend=vllm \
    qa_model.reasoning_effort=$REASONING_EFFORT \
    qa_model.max_tokens=$MAX_TOKENS \
    qa_model.max_model_len=$MAX_MODEL_LEN \
    qa_model.seed=$seed \
    seed=$seed \
    "${extra_args[@]}"
  return $?
}

MAX_STAGE_RETRIES=5

# lm_conf/__main__.py catches per-round exceptions internally, logs them, and
# still exits 0 / prints "All results saved to:" at the end — so a round can
# silently fail (e.g. a missing pip dependency blowing up grading) without
# the process ever returning a nonzero exit code. run_sampling_stage's retry
# logic alone can't see that. This wrapper additionally verifies
# graded_outputs_0.pkl actually exists before trusting the run, and redoes
# the whole stage (fresh timestamped dir) if it doesn't.
run_sampling_stage_until_valid() {
  local task="$1" seed="$2" logfile="$3" cache_path="${4:-}"
  local out_path=""
  for stage_attempt in $(seq 1 "$MAX_STAGE_RETRIES"); do
    if ! run_sampling_stage "$task" "$seed" "$logfile" "$cache_path"; then
      echo "  !!! [$task] run failed outright (exit code) on stage-attempt $stage_attempt/$MAX_STAGE_RETRIES" >&2
      continue
    fi
    out_path="$(extract_results_path "$logfile")"
    if [[ -z "$out_path" ]]; then
      echo "  !!! [$task] could not extract results_path on stage-attempt $stage_attempt/$MAX_STAGE_RETRIES" >&2
      continue
    fi
    if [[ -f "$out_path/graded_outputs_0.pkl" ]]; then
      echo "$out_path"
      return 0
    fi
    echo "  !!! [$task] exited 0 but $out_path/graded_outputs_0.pkl is missing (a round likely hit a silently-swallowed exception -- check $logfile); redoing stage-attempt $((stage_attempt + 1))/$MAX_STAGE_RETRIES" >&2
  done
  return 1
}

overall_status=0

for seed in "${SEEDS[@]}"; do
  echo "=== [$(date '+%F %T')] model=$MODEL gpu=$GPU seed=$seed: starting ==="
  check_disk_space

  # --- lc: fresh generation (expensive) ---
  lc_log="$LOG_DIR/${SAFE_MODEL}_seed${seed}_lc.log"
  echo "  [lc] -> $lc_log"
  if ! LC_PATH="$(run_sampling_stage_until_valid direct_qa_unified_lc "$seed" "$lc_log")"; then
    echo "  !!! FAILED: lc sampling model=$MODEL seed=$seed exhausted $MAX_STAGE_RETRIES stage-attempts (see $lc_log)" >&2
    overall_status=1
    continue
  fi
  echo "  [lc] results_path=$LC_PATH"

  # --- tp: cheap, reuses lc's cache_path ---
  tp_log="$LOG_DIR/${SAFE_MODEL}_seed${seed}_tp.log"
  echo "  [tp] -> $tp_log"
  if ! TP_PATH="$(run_sampling_stage_until_valid direct_qa_unified_tp "$seed" "$tp_log" "$LC_PATH")"; then
    echo "  !!! FAILED: tp sampling model=$MODEL seed=$seed exhausted $MAX_STAGE_RETRIES stage-attempts (see $tp_log)" >&2
    overall_status=1
    continue
  fi
  echo "  [tp] results_path=$TP_PATH"

  # --- su: cheap, reuses lc's cache_path ---
  su_log="$LOG_DIR/${SAFE_MODEL}_seed${seed}_su.log"
  echo "  [su] -> $su_log"
  if ! SU_PATH="$(run_sampling_stage_until_valid direct_qa_unified_su "$seed" "$su_log" "$LC_PATH")"; then
    echo "  !!! FAILED: su sampling model=$MODEL seed=$seed exhausted $MAX_STAGE_RETRIES stage-attempts (see $su_log)" >&2
    overall_status=1
    continue
  fi
  echo "  [su] results_path=$SU_PATH"

  # --- calibration (in-domain, 100% lexicon, no beta-guided) ---
  check_disk_space
  calib_log="$LOG_DIR/${SAFE_MODEL}_seed${seed}_calibration.log"
  calib_out_dir="$COMMON_DIR/rebuttal_reruns_in_domain_calibration/$DATASET/$MODEL/seed_$seed"
  calib_metrics_csv="$calib_out_dir/calibration_performance.csv"
  echo "  [calibration] -> $calib_log"
  calibration_ok=0
  for calib_attempt in $(seq 1 "$MAX_STAGE_RETRIES"); do
    : > "$calib_log"
    if retry_with_backoff "$calib_log" \
          "$PYTHON_BIN" -m calibration.in_domain_calibration \
          --dataset $DATASET \
          --model "$MODEL" \
          --breakpoint metrics \
          --seed "$seed" \
          --lc_path "$LC_PATH" \
          --tp_path "$TP_PATH" \
          --su_path "$SU_PATH" \
        && [[ -f "$calib_metrics_csv" ]]; then
      calibration_ok=1
      break
    fi
    echo "  !!! [calibration] stage-attempt $calib_attempt/$MAX_STAGE_RETRIES did not produce $calib_metrics_csv; retrying" >&2
  done
  if [[ "$calibration_ok" -ne 1 ]]; then
    echo "  !!! FAILED: calibration model=$MODEL seed=$seed exhausted $MAX_STAGE_RETRIES stage-attempts (see $calib_log)" >&2
    overall_status=1
    continue
  fi

  echo "=== [$(date '+%F %T')] model=$MODEL gpu=$GPU seed=$seed: done ==="
done

echo "=== [$(date '+%F %T')] model=$MODEL gpu=$GPU: all seeds finished, overall_status=$overall_status ==="
exit $overall_status
