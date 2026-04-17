# #!/usr/bin/env bash
GPU=2

breakpoint=metrics
# truthful_qa
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 5 --dataset truthful_qa --model openai/gpt-oss-20b --breakpoint $breakpoint
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 5 --dataset truthful_qa --model qwen/Qwen3-8B --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 5 --dataset truthful_qa --model mistralai/Mistral-7B-Instruct-v0.3 --breakpoint $breakpoint 

# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 1 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 3 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 5 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 10 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 15 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 20 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 25 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 30 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 