#!/usr/bin/env bash


GPU=1
MODIFIER="openai/gpt-oss-20b"
BREAKPOINT="eval"

MODELS=(
    "openai/gpt-oss-20b"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "qwen/Qwen3-8B"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "qwen/Qwen2.5-7B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU python -m calibration.cross_domain --cache_parent_path /hdd/ivny/results/ --breakpoint $BREAKPOINT --train_dataset mmlu --test_dataset trivia_qa --modifier $MODIFIER --model $MODEL --estimation_method dist_semantic_uncertainty --post-hoc-method platt_uni 
    CUDA_VISIBLE_DEVICES=$GPU python -m calibration.cross_domain --cache_parent_path /hdd/ivny/results/ --breakpoint $BREAKPOINT --train_dataset mmlu --test_dataset squadv2 --modifier $MODIFIER --model $MODEL --estimation_method dist_semantic_uncertainty --post-hoc-method platt_uni
    CUDA_VISIBLE_DEVICES=$GPU python -m calibration.cross_domain --cache_parent_path /hdd/ivny/results/ --breakpoint $BREAKPOINT --train_dataset trivia_qa --test_dataset mmlu --modifier $MODIFIER --model $MODEL --estimation_method dist_semantic_uncertainty --post-hoc-method platt_uni --answer-prepend-test "The answer is "
    CUDA_VISIBLE_DEVICES=$GPU python -m calibration.cross_domain --cache_parent_path /hdd/ivny/results/ --breakpoint $BREAKPOINT --train_dataset trivia_qa --test_dataset squadv2 --modifier $MODIFIER --model $MODEL --estimation_method dist_semantic_uncertainty --post-hoc-method platt_uni
    CUDA_VISIBLE_DEVICES=$GPU python -m calibration.cross_domain --cache_parent_path /hdd/ivny/results/ --breakpoint $BREAKPOINT --train_dataset squadv2 --test_dataset trivia_qa --modifier $MODIFIER --model $MODEL --estimation_method dist_semantic_uncertainty --post-hoc-method platt_uni
    CUDA_VISIBLE_DEVICES=$GPU python -m calibration.cross_domain --cache_parent_path /hdd/ivny/results/ --breakpoint $BREAKPOINT --train_dataset squadv2 --test_dataset mmlu --modifier $MODIFIER --model $MODEL --estimation_method dist_semantic_uncertainty --post-hoc-method platt_uni --answer-prepend-test "The answer is "
done
