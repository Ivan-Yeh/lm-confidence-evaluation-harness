GPU=0
limit=200

models=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "openai/gpt-oss-20b"
    "qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-8B"
)

tasks=(
    "dist_lnll"
    "dist_semantic_uncertainty"
    "dist_linguistic_confidence"
    "lnll"
    "semantic_uncertainty"
    "linguistic_confidence"
)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        CUDA_VISIBLE_DEVICES=$GPU python -m lm_conf limit=$limit dataset=trivia_qa task=$task qa_model.name=$model
    done
done