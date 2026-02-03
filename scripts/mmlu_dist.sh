#!/usr/bin/env bash

GPU=2
limit=null
rounds=2
max_model_len=2048
max_tokens=200

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_lnll qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_lnll qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_lnll qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_lnll qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_lnll qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_lnll qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_p_true_cont qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_p_true_cont qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_p_true_cont qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_p_true_cont qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_p_true_cont qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_p_true_cont qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len filtered_output_path="/hdd/ivny/results/mmlu/dist_semantic_uncertainty/meta-llama/Llama-3.1-8B-Instruct/2026-01-26_12-40-06"
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len filtered_output_path="/hdd/ivny/results/mmlu/dist_semantic_uncertainty/mistralai/Mistral-7B-Instruct-v0.3/2026-01-26_14-34-44"
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len filtered_output_path="/hdd/ivny/results/mmlu/dist_semantic_uncertainty/qwen/Qwen3-8B/2026-01-26_16-47-07"
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=mmlu rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

max_model_len=2048
max_tokens=512

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len