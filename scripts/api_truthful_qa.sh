# #!/usr/bin/env bash

GPU=0
limit=null
rounds=1
max_model_len=2048
max_tokens=256
dataset=truthful_qa
backend=vllm
mode=lc

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=direct_qa_unified_$mode \
#     qa_model.name=openai/gpt-oss-120b \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/$dataset/direct_qa_unified_lc/openai/gpt-oss-120b/2026-04-25_19-39-07

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=direct_qa_unified_$mode \
#     qa_model.name=qwen/Qwen3-235B-A22B-Instruct-2507-tput \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/$dataset/direct_qa_unified_lc/qwen/Qwen3-235B-A22B-Instruct-2507-tput/retrieved

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=direct_qa_unified_$mode \
#     qa_model.name=google/gemma-4-31B-it \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/$dataset/direct_qa_unified_lc/google/gemma-4-31B-it/2026-04-25_19-51-56


# hedged 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_$mode \
#     qa_model.name=openai/gpt-oss-120b \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/$dataset/hedged_qa_unified_lc/openai/gpt-oss-120b/retrieved

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_$mode \
    qa_model.name=qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/$dataset/hedged_qa_unified_lc/qwen/Qwen3-235B-A22B-Instruct-2507-tput/retrieved

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=hedged_qa_unified_$mode \
#     qa_model.name=google/gemma-4-31B-it \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len \
#     cache_path=/hdd/ivny/results/$dataset/hedged_qa_unified_lc/google/gemma-4-31B-it/retrieved

