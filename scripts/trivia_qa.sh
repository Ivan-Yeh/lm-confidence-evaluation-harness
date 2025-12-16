GPU=0
limit=200

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=lnll qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=top_k_vol qa_model.name=openai/gpt-oss-20b

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=lnll qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=top_k_vol qa_model.name=Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=lnll qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=top_k_vol qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct


# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=linguistic_confidence qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=linguistic_confidence qa_model.name=Qwen/Qwen2.5-7B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=linguistic_confidence qa_model.name=openai/gpt-oss-20b qa_model.stop_sequences=[]
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=linguistic_confidence qa_model.name=meta-llama/Meta-Llama-3-8B
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=linguistic_confidence qa_model.name=Qwen/Qwen2.5-7B

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_cont qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_cont qa_model.name=Qwen/Qwen2.5-7B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_cont qa_model.name=openai/gpt-oss-20b qa_model.stop_sequences=[]
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_cont qa_model.name=meta-llama/Meta-Llama-3-8B
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_cont qa_model.name=Qwen/Qwen2.5-7B

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_mc qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_mc qa_model.name=Qwen/Qwen2.5-7B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_mc qa_model.name=openai/gpt-oss-20b qa_model.stop_sequences=[]
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_mc qa_model.name=meta-llama/Meta-Llama-3-8B
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=p_true_mc qa_model.name=Qwen/Qwen2.5-7B

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=semantic_uncertainty qa_model.name=openai/gpt-oss-20b qa_model.stop_sequences=[]
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=semantic_uncertainty qa_model.name=Qwen/Qwen2.5-7B-Instruct

# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=vnc_generative qa_model.name=openai/gpt-oss-20b qa_model.stop_sequences=[]
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=vnc_generative qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=vnc_generative qa_model.name=Qwen/Qwen2.5-7B-Instruct