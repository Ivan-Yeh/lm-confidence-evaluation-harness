#!/usr/bin/env bash

GPU=3
limit=null
rounds=2
max_model_len=2048
max_tokens=200

# meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=lnll_gen qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=linguistic_confidence qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=p_true_mc qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# meta-llama/Llama-3.1-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=lnll_gen qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=linguistic_confidence qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=p_true_mc qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=semantic_uncertainty qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=lnll_gen qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=linguistic_confidence qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=p_true_mc qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=semantic_uncertainty qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
# qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=lnll_gen qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=linguistic_confidence qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=p_true_mc qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=semantic_uncertainty qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
# qwen/Qwen3-8B
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=lnll_gen qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=p_true_mc qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=semantic_uncertainty qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
# mistralai/Mistral-7B-Instruct-v0.3
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=lnll_gen qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=linguistic_confidence qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=p_true_mc qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
sleep 10
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=semantic_uncertainty qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# google/gemma-3-12b-it
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=lnll_gen qa_model.name=google/gemma-3-12b-it qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=linguistic_confidence qa_model.name=google/gemma-3-12b-it qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=p_true_mc qa_model.name=google/gemma-3-12b-it qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=semantic_uncertainty qa_model.name=google/gemma-3-12b-it qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len