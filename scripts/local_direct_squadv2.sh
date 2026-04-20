# #!/usr/bin/env bash

GPU=1
limit=null
rounds=1
max_model_len=2048
max_tokens=256
dataset=squadv2
backend=vllm

task=direct_qa_unified_tp

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=openai/gpt-oss-20b \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/openai/gpt-oss-20b/2026-04-20_00-52-55

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/meta-llama/Llama-3.1-8B-Instruct/2026-04-20_02-08-08

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=qwen/Qwen3-8B \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=null \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/qwen/Qwen3-8B/2026-04-20_11-04-22

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/mistralai/Mistral-7B-Instruct-v0.3/2026-04-20_03-13-29


task=direct_qa_unified_su

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=openai/gpt-oss-20b \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/openai/gpt-oss-20b/2026-04-20_00-52-55

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/meta-llama/Llama-3.1-8B-Instruct/2026-04-20_02-08-08

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=qwen/Qwen3-8B \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=null \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/qwen/Qwen3-8B/2026-04-20_11-04-22

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=$task \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/mistralai/Mistral-7B-Instruct-v0.3/2026-04-20_03-13-29