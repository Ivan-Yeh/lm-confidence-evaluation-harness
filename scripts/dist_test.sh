#!/usr/bin/env bash

GPU=0
limit=200


models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "openai/gpt-oss-20b"
    "qwen/Qwen2.5-7B-Instruct"
    "qwen/Qwen3-8B"
)

tasks=(
    "lnll"
    "semantic_uncertainty"
    "linguistic_confidence"
    "dist_lnll"
    "dist_semantic_uncertainty"
    "dist_linguistic_confidence"
)

# CUDA_VISIBLE_DEVICES=0 python -m lm_conf limit=200 dataset=trivia_qa task=dist_lnll qa_model.name=openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=0 python -m lm_conf limit=200 dataset=trivia_qa task=dist_semantic_uncertainty qa_model.name=openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=0 python -m lm_conf limit=200 dataset=trivia_qa task=dist_semantic_uncertainty qa_model.name=qwen/Qwen2.5-7B-Instruct

for task in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=$task qa_model.name=qwen/Qwen3-8B qa_model.max_tokens=512
done