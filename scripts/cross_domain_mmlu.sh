# #!/usr/bin/env bash
GPU=0


breakpoint=hedge

CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model openai/gpt-oss-20b --breakpoint $breakpoint
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model qwen/Qwen3-8B --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 

CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model openai/gpt-oss-20b --breakpoint $breakpoint
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model qwen/Qwen3-8B --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 


breakpoint=eval
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model openai/gpt-oss-20b --breakpoint $breakpoint
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model qwen/Qwen3-8B --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 

CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model openai/gpt-oss-20b --breakpoint $breakpoint
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model qwen/Qwen3-8B --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 


breakpoint=metrics
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model openai/gpt-oss-20b --breakpoint $breakpoint
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model qwen/Qwen3-8B --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test truthful_qa --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 

CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model openai/gpt-oss-20b --breakpoint $breakpoint
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model qwen/Qwen3-8B --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --train mmlu --test squadv2 --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 
