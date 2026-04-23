#!/usr/bin/env bash
GPU=3
breakpoint=metrics
MAX_RETRIES=10000
attempt=1

while true; do
    echo "[Attempt $attempt] Running calibration..."
    CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration \
        --dataset squadv2 \
        --model qwen/Qwen3-8B \
        --breakpoint $breakpoint
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
