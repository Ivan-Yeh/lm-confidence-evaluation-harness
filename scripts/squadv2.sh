GPU=0
limit=200

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=lnll qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=top_k_vol qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=semantic_uncertainty qa_model.name=openai/gpt-oss-20b

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=lnll qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=top_k_vol qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=semantic_uncertainty qa_model.name=Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=lnll qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=top_k_vol qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=squadv2 task=semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct