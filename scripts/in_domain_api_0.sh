# #!/usr/bin/env bash

MAX_RETRIES=10000

run_with_retry() {
    local attempt=1
    while true; do
        echo "[Attempt $attempt] $*"
        "$@"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "Success on attempt $attempt."
            break
        fi
        echo "Failed (exit $exit_code). Retrying in 10s..."
        if [ $attempt -ge $MAX_RETRIES ]; then
            echo "Reached max retries ($MAX_RETRIES). Giving up."
            exit 1
        fi
        attempt=$((attempt + 1))
        sleep 10
    done
}

GPU=0
prompt_type=direct_qa
breakpoint=hedge
breakpoint=metrics

CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.in_domain_calibration --prompt_type $prompt_type --dataset truthful_qa --model google/gemma-4-31B-it --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.in_domain_calibration --prompt_type $prompt_type --dataset truthful_qa --model openai/gpt-oss-120b --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.in_domain_calibration --prompt_type $prompt_type --dataset truthful_qa --model qwen/Qwen3-235B-A22B-Instruct-2507-tput --breakpoint $breakpoint 
