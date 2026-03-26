#!/usr/bin/env bash

GPU=1
limit=null
rounds=1
max_model_len=2048
max_tokens=256

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_tp qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_tp qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_tp qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_tp qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_tp qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=null qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_tp qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_lc qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len filtered_output_path="/hdd/ivny/results/trivia_qa/hedged_qa_unified_lc/meta-llama/Meta-Llama-3-8B-Instruct/2026-03-26_02-26-02"
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_lc qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_lc qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_lc qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_lc qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=null qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_lc qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_su qa_model.name=meta-llama/Llama-3.1-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_su qa_model.name=mistralai/Mistral-7B-Instruct-v0.3 qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_su qa_model.name=qwen/Qwen3-8B qa_model.reasoning_effort=null qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_su qa_model.name=openai/gpt-oss-20b qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_su qa_model.name=qwen/Qwen2.5-7B-Instruct qa_model.reasoning_effort=null qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=trivia_qa rounds=$rounds limit=$limit task=hedged_qa_unified_su qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct qa_model.reasoning_effort=low qa_model.max_tokens=$max_tokens qa_model.max_model_len=$max_model_len 
