"""
Build the human-annotated hedging-cue hint block inserted into the
LINGUISTIC_EVALUATOR_PROMPT template, per the spec:

    human_annotated_cues = pd.read_csv(.../hedging_word_aggregated.csv)[["hedging_word", "mean", "std"]]
    human_annotated_cues["mean"] *= 100.0
    human_annotated_cues["std"] *= 100.0
    human_annotated_cues = human_annotated_cues.sort_values("mean").round(2).to_dict(orient="records")

Then randomly sample 15 of these (seed=5), keeping low-to-high order.
"""
import os

import numpy as np
import pandas as pd

LEXICON_DIR = "/home/ivan/lm-confidence-evaluation-harness/linguistic_confidence_lexicon"
HEDGE_CUE_SEED = 5
N_CUES = 15


def load_human_annotated_cues() -> list[dict]:
    human_annotated_cues = pd.read_csv(
        os.path.join(LEXICON_DIR, "hedging_word_aggregated.csv")
    )[["hedging_word", "mean", "std"]]
    human_annotated_cues["mean"] *= 100.0
    human_annotated_cues["std"] *= 100.0
    human_annotated_cues = human_annotated_cues.sort_values("mean").round(2).to_dict(orient="records")
    return human_annotated_cues


def sample_human_cues(seed: int = HEDGE_CUE_SEED, n: int = N_CUES) -> list[dict]:
    all_cues = load_human_annotated_cues()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_cues), size=n, replace=False)
    idx.sort()  # all_cues is already ascending by mean, so this keeps low-to-high order
    return [all_cues[i] for i in idx]


def format_cues_for_prompt(cues: list[dict]) -> str:
    lines = [
        f'- "{row["hedging_word"]}": mean confidence = {row["mean"]:.2f}, std = {row["std"]:.2f}'
        for row in cues
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    cues = sample_human_cues()
    print(format_cues_for_prompt(cues))
