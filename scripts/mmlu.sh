GPU=0

# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lnll qa_model.name=openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lnll qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lnll qa_model.name=Qwen/Qwen2.5-7B-Instruct

# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc qa_model.name=openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc qa_model.name=meta-llama/Meta-Llama-3-8B
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc qa_model.name=Qwen/Qwen2.5-7B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc qa_model.name=Qwen/Qwen2.5-7B

# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_p_true qa_model.name=openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_p_true qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_p_true qa_model.name=Qwen/Qwen2.5-7B-Instruct

# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_semantic_uncertainty qa_model.name=openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_semantic_uncertainty qa_model.name=Qwen/Qwen2.5-7B-Instruct

# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_vnc_generative qa_model.name=openai/gpt-oss-20b
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_vnc_generative qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_vnc_generative qa_model.name=Qwen/Qwen2.5-7B-Instruct

# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_vnc_continuation qa_model.name=openai/gpt-oss-20b 
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_vnc_continuation qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_vnc_continuation qa_model.name=Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc_cont qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc_cont qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc_cont qa_model.name=Qwen/Qwen2.5-7B
CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc_cont qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py  dataset=mmlu task=mmlu_lc_cont qa_model.name=meta-llama/Meta-Llama-3-8B