# #!/usr/bin/env bash

GPU=2
limit=50
rounds=1
max_model_len=2048
max_tokens=256
dataset=mmlu_pro
backend=vllm

# selected api models:
# openai/gpt-oss-120b
# meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
# Qwen/Qwen3-235B-A22B-Instruct-2507-tput
# mistralai/Mixtral-8x22B-Instruct-v0.1

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_lc \
    qa_model.name=openai/gpt-oss-120b \
    qa_model.backend=vllm \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    linguistic_confidence_judge_model_0.backend=$backend \
    linguistic_confidence_judge_model_1.backend=$backend \
    linguistic_confidence_judge_model_2.backend=$backend

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_lc \
    qa_model.name=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
    qa_model.backend=vllm \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    linguistic_confidence_judge_model_0.backend=$backend \
    linguistic_confidence_judge_model_1.backend=$backend \
    linguistic_confidence_judge_model_2.backend=$backend

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_lc \
    qa_model.name=Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    qa_model.backend=vllm \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    linguistic_confidence_judge_model_0.backend=$backend \
    linguistic_confidence_judge_model_1.backend=$backend \
    linguistic_confidence_judge_model_2.backend=$backend

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_lc \
    qa_model.name=mistralai/Mixtral-8x22B-Instruct-v0.1 \
    qa_model.backend=vllm \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    linguistic_confidence_judge_model_0.backend=$backend \
    linguistic_confidence_judge_model_1.backend=$backend \
    linguistic_confidence_judge_model_2.backend=$backend