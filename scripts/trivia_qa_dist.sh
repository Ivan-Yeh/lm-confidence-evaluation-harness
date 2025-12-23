#!/usr/bin/env bash

GPU=3
limit=null
rounds=3

models=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "openai/gpt-oss-20b"
    "qwen/Qwen2.5-7B-Instruct"
    "qwen/Qwen3-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-3-12b-it"
)

tasks=(
    # "dist_lnll" # gpu 1
    # "dist_linguistic_confidence" # gpu 1
    # "dist_p_true_mc" # gpu 2
    "dist_semantic_uncertainty" # gpu 3
)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=$task qa_model.name=$model qa_model.reasoning_effort=low qa_model.max_tokens=512 
    done
done