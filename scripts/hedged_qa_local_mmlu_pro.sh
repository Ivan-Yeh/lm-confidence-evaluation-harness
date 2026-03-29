# #!/usr/bin/env bash

GPU=0
limit=null
rounds=1
max_model_len=2048
max_tokens=256
dataset=squadv2
backend=vllm


CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf dataset=$dataset \
    rounds=$rounds \
    limit=$limit \
    task=hedged_qa_unified_tp \
    qa_model.name=openai/gpt-oss-20b \
    qa_model.backend=$backend \
    qa_model.reasoning_effort=low \
    qa_model.max_tokens=$max_tokens \
    qa_model.max_model_len=$max_model_len \
    linguistic_confidence_judge_model_0.backend=$backend \
    linguistic_confidence_judge_model_1.backend=$backend \
    linguistic_confidence_judge_model_2.backend=$backend 