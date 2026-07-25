#!/usr/bin/env bash
# Lexicon-size ablation for in-domain calibration on TruthfulQA.
#
# Runs calibration.in_domain_calibration (the copy under rebuttal/lex-size/calibration,
# which adds LEXICON_FRAC-driven lexicon sampling on top of the unmodified pipeline)
# for 4 models x 4 lexicon fractions (25/50/75/100%).
#
# Each lexicon fraction is run across all 4 models in parallel (one model per GPU),
# lexicon fractions themselves run sequentially. Signal caches (lc/tp/su) are read
# as usual from /hdd/ivny/results; only the hedging-word / rewrite / eval / metrics
# caches are written fresh, under /hdd/ivny/in_domain_calibration_lex_size/truthful_qa_<pct>/<model>/.
set -uo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="/home/ivan/miniconda3/envs/llm/bin/python"

export PYTHONPATH="$PROJECT_ROOT"

MODELS=(
  "openai/gpt-oss-20b"
  "meta-llama/Llama-3.1-8B-Instruct"
  "qwen/Qwen3-8B"
  "mistralai/Mistral-7B-Instruct-v0.3"
)
GPUS=(0 1 2 3)
FRACS=(0.25 0.5 0.75 1.0)

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR"

overall_status=0

for frac in "${FRACS[@]}"; do
  pct=$(awk -v f="$frac" 'BEGIN{printf "%d", (f*100)+0.5}')
  echo "=== [$(date '+%F %T')] Lexicon fraction ${frac} (${pct}%) ==="

  pids=()
  pid_models=()
  for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu="${GPUS[$i]}"
    safe_model="$(echo "$model" | tr '/' '_')"
    logfile="$LOG_DIR/truthful_qa_${pct}_${safe_model}.log"
    echo "  launching model=$model gpu=$gpu -> $logfile"
    CUDA_VISIBLE_DEVICES="$gpu" LEXICON_FRAC="$frac" \
      "$PYTHON_BIN" -m calibration.in_domain_calibration \
        --dataset truthful_qa \
        --model "$model" \
        --breakpoint metrics \
        > "$logfile" 2>&1 &
    pids+=($!)
    pid_models+=("$model")
  done

  for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then
      echo "  !!! FAILED: model=${pid_models[$j]} frac=${frac} (see log above)"
      overall_status=1
    fi
  done

  echo "=== [$(date '+%F %T')] Finished lexicon fraction ${frac} ==="
done

exit $overall_status
