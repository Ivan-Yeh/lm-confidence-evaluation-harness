# #!/usr/bin/env bash
GPU=2

breakpoint=metrics 

# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 1 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 5 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 10 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 15 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 20 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 25 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 
# CUDA_VISIBLE_DEVICES=$GPU python -m calibration.in_domain_calibration --top_k 30 --dataset truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --breakpoint $breakpoint 