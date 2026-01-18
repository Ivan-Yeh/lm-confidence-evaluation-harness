#!/usr/bin/env bash

GPU=0
MODEL="openai/gpt-oss-20b"

# mmlu - lnll
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/meta-llama/Llama-3.1-8B-Instruct/2026-01-12_03-43-10
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-12_01-55-40
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/qwen/Qwen2.5-7B-Instruct/2026-01-12_07-23-22
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/qwen/Qwen3-8B/2026-01-12_08-44-05
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/mistralai/Mistral-7B-Instruct-v0.3/2026-01-12_20-11-42
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --answer-prepend "The answer is " --results_path /hdd/ivny/results/mmlu/dist_lnll/openai/gpt-oss-20b/2026-01-12_05-33-45

# trivia qa - lnll
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/meta-llama/Llama-3.1-8B-Instruct/2026-01-12_06-21-48
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-12_01-55-31
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/qwen/Qwen2.5-7B-Instruct/2026-01-12_17-52-11
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/qwen/Qwen3-8B/2026-01-13_01-09-42
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/mistralai/Mistral-7B-Instruct-v0.3/2026-01-14_08-49-59
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_lnll/openai/gpt-oss-20b/2026-01-12_12-22-09

# mmlu - lc
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/meta-llama/Llama-3.1-8B-Instruct/2026-01-12_23-56-50
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-12_22-07-53
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/qwen/Qwen2.5-7B-Instruct/2026-01-13_03-31-49
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/qwen/Qwen3-8B/2026-01-13_05-14-17
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/mistralai/Mistral-7B-Instruct-v0.3/2026-01-13_07-39-23
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/mmlu/dist_linguistic_confidence/openai/gpt-oss-20b/2026-01-13_01-59-48

# trivia qa - lc
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/meta-llama/Llama-3.1-8B-Instruct/2026-01-13_20-24-13
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/meta-llama/Meta-Llama-3-8B-Instruct/2026-01-13_19-01-11
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/qwen/Qwen2.5-7B-Instruct/2026-01-13_23-09-32
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/qwen/Qwen3-8B/2026-01-14_01-02-48
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/mistralai/Mistral-7B-Instruct-v0.3/2026-01-14_03-24-46
CUDA_VISIBLE_DEVICES=$GPU python -m calibration --post-hoc-method platt --model $MODEL --results_path /hdd/ivny/results/trivia_qa/dist_linguistic_confidence/openai/gpt-oss-20b/2026-01-13_21-48-52




    