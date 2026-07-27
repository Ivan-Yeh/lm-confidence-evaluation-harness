#!/usr/bin/env bash
# Driver: launches run_model_chain.sh for all 4 models in parallel, one GPU
# each, so all 4 GPUs are claimed together up front (avoids interleaving with
# other users trickling in between our stages).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# model -> (gpu, reasoning_effort)
MODELS=(
  "openai/gpt-oss-20b:0:low"
  "meta-llama/Llama-3.1-8B-Instruct:1:low"
  "qwen/Qwen3-8B:2:null"
  "mistralai/Mistral-7B-Instruct-v0.3:3:low"
)

pids=()
pid_models=()

for entry in "${MODELS[@]}"; do
  IFS=':' read -r model gpu reasoning <<< "$entry"
  safe_model="$(echo "$model" | tr '/' '_')"
  driver_log="$LOG_DIR/${safe_model}_driver.log"
  echo "Launching model=$model gpu=$gpu reasoning_effort=$reasoning -> $driver_log"
  "$SCRIPT_DIR/run_model_chain.sh" "$gpu" "$model" "$reasoning" > "$driver_log" 2>&1 &
  pids+=($!)
  pid_models+=("$model")
done

overall_status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "!!! FAILED: model=${pid_models[$i]} (see logs/*_driver.log)"
    overall_status=1
  else
    echo "=== model=${pid_models[$i]} finished OK ==="
  fi
done

exit $overall_status
