"""
Sample 50 questions from the LLAMA truthful_qa in-domain calibration results,
subject to two constraints:
  1. Tone-alignment filter: the text's tone plausibly matches its own recorded
     "_lc" confidence for BOTH the original response and all 3 rewrites (see
     filter_reasonable_rows.py for the alignment criterion). This avoids picking
     statements where the pipeline's own confidence estimate looks like noise
     relative to the actual wording.
  2. Accuracy-matched: the sample's correctness rate is stratified to match the
     full 571-row dataset's accuracy rate, rather than whatever rate happens to
     fall out of a plain random draw from the qualifying pool. This avoids a
     base-rate mismatch that mechanically skews generalised-ECE comparisons
     (a judge whose mean confidence happens to sit closer to the *sample's*
     accuracy rate will look better calibrated for reasons unrelated to
     per-item discrimination).

Run with the `llm` conda env (has pandas + the lm_conf package importable
from the repo root):
    /home/ivan/miniconda3/envs/llm/bin/python sample_data.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/home/ivan/lm-confidence-evaluation-harness"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from lm_conf.confidence_metrics.distributionals import BetaDistribution  # noqa: E402,F401
from filter_reasonable_rows import analyze, RESPONSE_COLS, EXISTING_LC_COLS  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "cache")
SAMPLE_SEED = 42
N_SAMPLES = 50
MIN_REWRITES_ALIGNED = 3  # strict: all 3 rewrites must be aligned, plus the original


def clean_accuracy(value) -> float:
    if value == "" or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)


def main():
    df, analysis = analyze()
    acc = df["accuracy"].apply(clean_accuracy)

    full_acc_rate = acc.mean()
    print(f"Full dataset (N={len(df)}) accuracy rate: {full_acc_rate:.4f}")

    mask = analysis["original_aligned"] & (analysis["n_rewrites_aligned"] >= MIN_REWRITES_ALIGNED)
    qualifying_idx = analysis.loc[mask, "row_idx"].to_numpy()
    print(f"{len(qualifying_idx)} of {len(df)} rows pass the alignment filter "
          f"(original aligned AND >= {MIN_REWRITES_ALIGNED}/3 rewrites aligned)")

    qualifying_correct = np.array([i for i in qualifying_idx if acc.iloc[i] == 1.0])
    qualifying_incorrect = np.array([i for i in qualifying_idx if acc.iloc[i] == 0.0])
    print(f"Within qualifying pool: {len(qualifying_correct)} correct, {len(qualifying_incorrect)} incorrect")

    n_correct_target = round(N_SAMPLES * full_acc_rate)
    n_incorrect_target = N_SAMPLES - n_correct_target
    print(f"Target for {N_SAMPLES}-sample to match full-dataset rate: "
          f"{n_correct_target} correct, {n_incorrect_target} incorrect")

    rng = np.random.default_rng(SAMPLE_SEED)
    sampled_correct = rng.choice(qualifying_correct, size=n_correct_target, replace=False)
    sampled_incorrect = rng.choice(qualifying_incorrect, size=n_incorrect_target, replace=False)
    sample_idx = np.concatenate([sampled_correct, sampled_incorrect])
    sample_idx.sort()

    sampled = df.iloc[sample_idx].reset_index(drop=True)
    sampled.insert(0, "source_row_index", sample_idx)

    keep_cols = ["source_row_index", "accuracy"] + RESPONSE_COLS + list(EXISTING_LC_COLS.values())
    sampled = sampled[keep_cols]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "sampled_rows.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(sampled, f)

    sample_acc_rate = sampled["accuracy"].apply(clean_accuracy).mean()
    print(f"\nSampled {len(sampled)} rows -> {out_path}")
    print(f"Sample accuracy rate: {sample_acc_rate:.4f} (target was {full_acc_rate:.4f})")
    print(sampled[["source_row_index", "accuracy"]].to_string())


if __name__ == "__main__":
    main()
