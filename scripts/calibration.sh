#!/usr/bin/env bash

GPU=3
MODEL="openai/gpt-oss-20b"

# MMLU
# mmlu - lnll
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/meta-llama/Llama-3.1-8B-Instruct/2026-01-12_03-43-10
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-12_01-55-40
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/qwen/Qwen2.5-7B-Instruct/2026-01-12_07-23-22
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/qwen/Qwen3-8B/2026-01-12_08-44-05
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/mistralai/Mistral-7B-Instruct-v0.3/2026-01-12_20-11-42
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/openai/gpt-oss-20b/2026-01-12_05-33-45

# mmlu - lc 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/meta-llama/Llama-3.1-8B-Instruct/2026-01-31_03-17-58
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-31_00-21-17
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/qwen/Qwen2.5-7B-Instruct/2026-01-31_08-40-42
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/qwen/Qwen2.5-7B-Instruct/2026-02-03_00-53-50
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/qwen/Qwen3-8B/2026-01-31_10-43-10
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/openai/gpt-oss-20b/2026-01-31_06-12-32

# mmlu - semantic uncertainty
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_semantic_uncertainty/meta-llama/Llama-3.1-8B-Instruct/2026-01-26_22-40-49
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_semantic_uncertainty/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-27_10-46-40
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_semantic_uncertainty/qwen/Qwen2.5-7B-Instruct/2026-01-27_09-02-39
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_semantic_uncertainty/qwen/Qwen3-8B/2026-01-27_00-51-25
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_semantic_uncertainty/mistralai/Mistral-7B-Instruct-v0.3/2026-01-26_23-53-54

# TRIVIA QA
# trivia_qa - lnll
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/meta-llama/Llama-3.1-8B-Instruct/2026-01-12_06-21-48
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-12_01-55-31
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/qwen/Qwen2.5-7B-Instruct/2026-01-12_17-52-11
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/qwen/Qwen3-8B/2026-01-13_01-09-42
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/mistralai/Mistral-7B-Instruct-v0.3/2026-01-14_08-49-59
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/openai/gpt-oss-20b/2026-01-12_12-22-09

# trivia_qa - lc 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/meta-llama/Llama-3.1-8B-Instruct/2026-01-31_03-46-23
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-31_00-21-19
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/qwen/Qwen2.5-7B-Instruct/2026-02-01_11-58-45
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/qwen/Qwen3-8B/2026-02-01_14-15-22
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/mistralai/Mistral-7B-Instruct-v0.3/2026-02-01_17-09-20
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/openai/gpt-oss-20b/2026-01-31_07-03-14

# trivia_qa - semantic uncertainty
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_semantic_uncertainty/meta-llama/Llama-3.1-8B-Instruct/2026-01-26_22-42-25
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_semantic_uncertainty/mistralai/Mistral-7B-Instruct-v0.3/2026-01-27_06-19-36
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_semantic_uncertainty/openai/gpt-oss-20b/2026-01-27_18-43-47


# SQUAD v2
# squadv2 - lnll
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_lnll/meta-llama/Llama-3.1-8B-Instruct/2026-01-26_00-55-13
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_lnll/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-29_06-47-37
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_lnll/qwen/Qwen2.5-7B-Instruct/2026-01-29_14-35-40
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_lnll/qwen/Qwen3-8B/2026-01-28_02-31-40
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_lnll/mistralai/Mistral-7B-Instruct-v0.3/2026-01-26_05-51-50
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_lnll/openai/gpt-oss-20b/2026-01-27_12-27-12

# squadv2 - lc
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_linguistic_confidence/meta-llama/Llama-3.1-8B-Instruct/2026-01-31_00-21-20
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_linguistic_confidence/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-31_08-55-30
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_linguistic_confidence/qwen/Qwen2.5-7B-Instruct/2026-01-31_11-07-35
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_linguistic_confidence/qwen/Qwen3-8B/2026-01-31_06-27-23
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_linguistic_confidence/mistralai/Mistral-7B-Instruct-v0.3/2026-01-31_02-27-23
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_linguistic_confidence/openai/gpt-oss-20b/2026-01-31_04-30-54

# squadv2 - semantic uncertainty
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_semantic_uncertainty/meta-llama/Llama-3.1-8B-Instruct/2026-01-26_22-44-00
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_semantic_uncertainty/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-29_09-59-33
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_semantic_uncertainty/qwen/Qwen2.5-7B-Instruct/2026-01-29_19-19-13
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_semantic_uncertainty/qwen/Qwen3-8B/2026-01-28_15-12-28
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_semantic_uncertainty/mistralai/Mistral-7B-Instruct-v0.3/2026-01-27_05-09-59
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/squadv2/dist_semantic_uncertainty/openai/gpt-oss-20b/2026-01-27_18-06-14

    