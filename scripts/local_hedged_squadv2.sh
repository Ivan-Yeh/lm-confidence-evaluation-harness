# #!/usr/bin/env bash

GPU=1
limit=null
rounds=1
max_model_len=2048
max_tokens=256
dataset=squadv2
backend=vllm



# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_tp \
#     qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/mistralai/Mistral-7B-Instruct-v0.3/2026-03-30_02-44-06

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_su \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/mistralai/Mistral-7B-Instruct-v0.3/2026-03-30_02-44-06

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_lc \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/mistralai/Mistral-7B-Instruct-v0.3/2026-03-30_02-44-06


# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_su \
#     qa_model.name=openai/gpt-oss-20b \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/openai/gpt-oss-20b/2026-03-29_22-44-40

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_su \
#     qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/meta-llama/Llama-3.1-8B-Instruct/2026-03-29_23-54-58

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_su \
#     qa_model.name=qwen/Qwen3-8B \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=null \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/qwen/Qwen3-8B/2026-03-30_01-09-33


# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_lc \
#     qa_model.name=openai/gpt-oss-20b \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/openai/gpt-oss-20b/2026-03-29_22-44-40

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_lc \
#     qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/meta-llama/Llama-3.1-8B-Instruct/2026-03-29_23-54-58

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_lc \
#     qa_model.name=qwen/Qwen3-8B \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=null \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/squadv2/hedged_qa_unified_tp/qwen/Qwen3-8B/2026-03-30_01-09-33