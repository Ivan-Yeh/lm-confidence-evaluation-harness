# #!/usr/bin/env bash

GPU=3
limit=null
rounds=1
max_model_len=2048
max_tokens=256
backend=vllm

dataset=mmlu

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/meta-llama/Llama-3.1-8B-Instruct/2026-04-02_00-28-41

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=qwen/Qwen3-8B \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=null \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/qwen/Qwen3-8B/2026-04-02_03-52-45

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=openai/gpt-oss-20b \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/openai/gpt-oss-20b/2026-03-30_23-36-49

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/mistralai/Mistral-7B-Instruct-v0.3/2026-04-02_05-25-21

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/meta-llama/Llama-3.1-8B-Instruct/2026-04-02_00-28-41

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=qwen/Qwen3-8B \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=null \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/qwen/Qwen3-8B/2026-04-02_03-52-45

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=openai/gpt-oss-20b \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/openai/gpt-oss-20b/2026-03-30_23-36-49

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/mmlu/direct_qa_unified_lc/mistralai/Mistral-7B-Instruct-v0.3/2026-04-02_05-25-21

# ---------------------------------------------------------------------

dataset=squadv2

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/meta-llama/Llama-3.1-8B-Instruct/2026-04-02_10-14-39

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=qwen/Qwen3-8B \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/qwen/Qwen3-8B/2026-04-02_11-39-26

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=openai/gpt-oss-20b \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/openai/gpt-oss-20b/2026-04-02_17-20-07

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_tp \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/mistralai/Mistral-7B-Instruct-v0.3/2026-04-02_08-24-18

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=meta-llama/Llama-3.1-8B-Instruct \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/meta-llama/Llama-3.1-8B-Instruct/2026-04-02_10-14-39

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=qwen/Qwen3-8B \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/qwen/Qwen3-8B/2026-04-02_11-39-26

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=openai/gpt-oss-20b \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/openai/gpt-oss-20b/2026-04-02_17-20-07

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=direct_qa_unified_su \
    qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    cache_path=/hdd/ivny/results/squadv2/direct_qa_unified_lc/mistralai/Mistral-7B-Instruct-v0.3/2026-04-02_08-24-18