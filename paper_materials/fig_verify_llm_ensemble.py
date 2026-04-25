import sys
import os
import re
import gc
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.utils import build_linguistic_evaluator_prompts
from lm_conf.default_utils.custom_types import PromptCollection
from lm_conf.models.model_manager import ModelManager

LEXICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "linguistic_confidence_lexicon")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

eval_cfg = {
    "lc_eval_0": {
        "backend": "vllm",
        "name": "qwen/Qwen3-8B",
        "max_model_len": 10240,
        "temperature": 1.0,
        "max_tokens": 256,
        "repeat": 3,
        "reasoning_effort": None,
    },
    "lc_eval_1": {
        "backend": "vllm",
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "max_model_len": 10240,
        "temperature": 1.0,
        "max_tokens": 256,
        "repeat": 3,
        "reasoning_effort": "low",
    },
    "lc_eval_2": {
        "backend": "vllm",
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_model_len": 10240,
        "temperature": 1.0,
        "max_tokens": 256,
        "repeat": 3,
        "reasoning_effort": "low",
    },
}
evaluator_keys = ["lc_eval_0", "lc_eval_1", "lc_eval_2"]

lexicon_df = pd.read_csv(os.path.join(LEXICON_DIR, "distilled_hedging_lexicon.csv"))
responses = lexicon_df["uncertainty_expression"].tolist()
target_means = [None] * len(responses)

cache_file = os.path.join(SAVE_DIR, "verify_llm_ensemble_scores.pkl")

# --- LLM inference (cached) ---
llm_scores_collection = None
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        cached = pickle.load(f)
    if len(cached) == len(responses):
        llm_scores_collection = cached
        print(f"Loaded cached LLM scores from pkl: {len(llm_scores_collection)} entries")
    else:
        print(f"Cache length mismatch ({len(cached)} vs {len(responses)}). Recomputing.")

if llm_scores_collection is None:
    def _extract_score(text: str) -> float:
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return min(max(float(match.group(1)), 0.0), 100.0) / 100.0
        return np.nan

    prompts = build_linguistic_evaluator_prompts(responses, target_means)
    llm_scores_collection = [[] for _ in responses]

    for evaluator in evaluator_keys:
        model_manager = ModelManager(master_cfg=eval_cfg, model_config_type=evaluator)
        prompt_collection = PromptCollection(
            system_prompt="You are a careful assistant.",
            context_texts=prompts,
        )
        outputs = model_manager.run_generation(prompt_collection)
        for output in outputs:
            for i, text in enumerate(output.output_texts):
                score = _extract_score(text)
                if not np.isnan(score):
                    llm_scores_collection[i].append(score)
        del model_manager
        gc.collect()

    with open(cache_file, "wb") as f:
        pickle.dump(llm_scores_collection, f)
    print(f"Saved LLM scores to {cache_file}")

# --- aggregate both score sets by extracted_hedging_expression ---
lexicon_df["llm_scores"] = llm_scores_collection
lexicon_df["human_scores"] = lexicon_df["normalized_confidence_scores"].apply(ast.literal_eval)

grouped_llm = (
    lexicon_df
    .groupby("extracted_hedging_expression")["llm_scores"]
    .apply(lambda lists: [s for lst in lists for s in lst])
    .reset_index()
)
grouped_llm.columns = ["expression", "llm_scores"]

grouped_human = (
    lexicon_df
    .groupby("extracted_hedging_expression")["human_scores"]
    .apply(lambda lists: [s for lst in lists for s in lst])
    .reset_index()
)
grouped_human.columns = ["expression", "human_scores"]

grouped = grouped_llm.merge(grouped_human, on="expression")
grouped["llm_mean"] = grouped["llm_scores"].apply(lambda x: float(np.mean(x)) if x else np.nan)
grouped["llm_median"] = grouped["llm_scores"].apply(lambda x: float(np.median(x)) if x else np.nan)
grouped = grouped.dropna(subset=["llm_median"])

# select 20 expressions spanning low-to-high confidence, most aligned per level
def _iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if x else np.nan

grouped["llm_iqr"]    = grouped["llm_scores"].apply(_iqr)
grouped["human_median"] = grouped["human_scores"].apply(lambda x: float(np.median(x)) if x else np.nan)
grouped["human_iqr"]  = grouped["human_scores"].apply(_iqr)
grouped["median_diff"] = (grouped["llm_median"] - grouped["human_median"]).abs()
grouped["alignment_err"] = grouped["median_diff"] + (grouped["llm_iqr"] - grouped["human_iqr"]).abs()
grouped = grouped.dropna(subset=["llm_median", "human_median", "alignment_err"])
grouped_sorted = (
    grouped[(grouped["median_diff"] <= 0.05) & (grouped["llm_median"] >= 0)]
    .sort_values("llm_median", ascending=True)
    .reset_index(drop=True)
)

n_select = 15
n = len(grouped_sorted)
selected_rows = []
for i in range(n_select):
    lo = int(round(i * n / n_select))
    hi = int(round((i + 1) * n / n_select))
    chunk = grouped_sorted.iloc[lo:hi]
    if not chunk.empty:
        selected_rows.append(chunk.loc[chunk["alignment_err"].idxmin()])
selected_df = pd.DataFrame(selected_rows)
high_mask = selected_df["llm_median"] >= 0.90
if high_mask.sum() > 2:
    keep_high = selected_df[high_mask].nsmallest(2, "alignment_err")
    selected_df = pd.concat([selected_df[~high_mask], keep_high])
selected_df = selected_df.sort_values("llm_median", ascending=True).reset_index(drop=True)

# --- horizontal box plots ---
fig, ax = plt.subplots(figsize=(8, 6))
positions = np.arange(len(selected_df))
width = 0.35

llm_data   = [[s * 100 for s in row["llm_scores"]]   for _, row in selected_df.iterrows()]
human_data = [[s * 100 for s in row["human_scores"]] for _, row in selected_df.iterrows()]

bp1 = ax.boxplot(
    llm_data, positions=positions - width / 2, widths=width * 0.85, vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="#1f77b4", alpha=0.7),
    medianprops=dict(color="black", lw=2),
    whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
    flierprops=dict(marker="o", markersize=3, alpha=0.5),
    showfliers=False,  # show outliers for LLM scores
)
bp2 = ax.boxplot(
    human_data, positions=positions + width / 2, widths=width * 0.85, vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="#ff7f0e", alpha=0.7),
    medianprops=dict(color="black", lw=2),
    whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5),
    flierprops=dict(marker="o", markersize=3, alpha=0.5),
    showfliers=False,  # show outliers for Human scores
)

ax.set_yticks(positions)
ax.set_yticklabels(selected_df["expression"].tolist(), fontsize=9)
ax.set_xlabel("Confidence Score (%)", fontsize=12)
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["LLM Evaluator", "Human Annotated"], fontsize=11)
ax.set_xlim(0, 100)
ax.grid(True, axis="x", linestyle=":", alpha=0.6)

plt.tight_layout()
out_path = os.path.join(SAVE_DIR, "figs/annotated_vs_llm_scores_boxplots.pdf")
plt.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved figure → {out_path}")
