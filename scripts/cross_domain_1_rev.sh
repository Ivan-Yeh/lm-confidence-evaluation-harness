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

GPU=2
prompt_type=direct_qa
breakpoint=hedge

breakpoint=metrics


# train on truthful_qa
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.cross_domain_calibration --prompt_type $prompt_type --re_estimate_lc True --train truthful_qa --test mmlu --model google/gemma-4-31B-it --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.cross_domain_calibration --prompt_type $prompt_type --re_estimate_lc True --train truthful_qa --test mmlu --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.cross_domain_calibration --prompt_type $prompt_type --re_estimate_lc True --train truthful_qa --test mmlu --model qwen/Qwen3-8B --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.cross_domain_calibration --prompt_type $prompt_type --re_estimate_lc True --train truthful_qa --test mmlu --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.cross_domain_calibration --prompt_type $prompt_type --re_estimate_lc True --train truthful_qa --test mmlu --model openai/gpt-oss-20b --breakpoint $breakpoint

CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.cross_domain_calibration --prompt_type $prompt_type --re_estimate_lc True --train truthful_qa --test mmlu --model openai/gpt-oss-120b --breakpoint $breakpoint
CUDA_VISIBLE_DEVICES=$GPU run_with_retry python -m calibration.cross_domain_calibration --prompt_type $prompt_type --re_estimate_lc True --train truthful_qa --test mmlu --model qwen/Qwen3-235B-A22B-Instruct-2507-tput --breakpoint $breakpoint 
