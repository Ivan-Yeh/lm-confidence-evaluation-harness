#!/usr/bin/env bash

GPU=1
limit=null
rounds=1
max_model_len=2048
max_tokens=512

# dist_lnll
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_lnll qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_lnll qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_lnll qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_lnll qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_lnll qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_lnll qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# dist_linguistic_confidence 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_linguistic_confidence qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# dist_semantic_uncertainty 
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len filtered_output_path="/hdd/ivny/results/trivia_qa/dist_semantic_uncertainty/meta-llama/Llama-3.1-8B-Instruct/2026-02-08_03-35-14"
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=null qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=dist_semantic_uncertainty qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len