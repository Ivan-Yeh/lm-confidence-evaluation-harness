# #!/usr/bin/env bash

GPU=2
limit=null
rounds=1
max_model_len=2048
max_tokens=256
dataset=mmlu
backend=together_ai_batch
mode=lc

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=direct_qa_unified_$mode \
#     qa_model.name=openai/gpt-oss-120b \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len &

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=direct_qa_unified_$mode \
#     qa_model.name=Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=low \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len &

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
#     rounds=$rounds \
#     limit=$limit \
#     task=direct_qa_unified_$mode \
#     qa_model.name=meta-llama/Llama-3.3-70B-Instruct-Turbo \
#     qa_model.backend=$backend \
#     qa_model.reasoning_effort=null \
#     qa_model.max_tokens=$max_tokens \
#     qa_model.max_model_len=$max_model_len &

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_$mode \
    qa_model.name=google/gemma-4-31B-it \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len 

