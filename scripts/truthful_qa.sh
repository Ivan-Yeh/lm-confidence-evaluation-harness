GPU=0

CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lnll qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lnll qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lnll qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lnll qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lnll qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lc qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lc qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lc qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lc qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_lc qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_cont qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_cont qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_cont qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_cont qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_cont qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_mc qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_mc qa_model.name=Qwen/Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_mc qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_mc qa_model.name=meta-llama/Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_p_true_mc qa_model.name=Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_semantic_uncertainty qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_semantic_uncertainty qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_semantic_uncertainty qa_model.name=Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_vnc_generative qa_model.name=openai/gpt-oss-20b
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_vnc_generative qa_model.name=meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=$GPU python main.py dataset=truthful_qa task=truthful_qa_gen_vnc_generative qa_model.name=Qwen/Qwen2.5-7B-Instruct