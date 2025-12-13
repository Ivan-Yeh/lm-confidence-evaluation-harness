GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=lnll qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=lnll qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=lnll qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=lnll qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=lnll qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=linguistic_confidence qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=linguistic_confidence qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=linguistic_confidence qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=linguistic_confidence qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=linguistic_confidence qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_cont qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_cont qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_cont qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_cont qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_cont qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_mc qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_mc qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_mc qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_mc qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=p_true_mc qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=semantic_uncertainty qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=semantic_uncertainty qa_model.name=Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=vnc_generative qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=vnc_generative qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm-conf dataset=truthful_qa task=vnc_generative qa_model.name=Qwen/Qwen2.5-7B-Instruct