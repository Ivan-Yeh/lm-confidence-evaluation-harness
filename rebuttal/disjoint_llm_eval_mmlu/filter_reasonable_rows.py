"""
Scan the full calibration_details.pkl and select responses where the text's
tone plausibly matches its own recorded "_lc" confidence (e.g. a confidently
worded / non-hedged sentence should have a high _lc mean; a heavily hedged
sentence should have a low one). This filters out rows where the pipeline's
own confidence estimate looks like noise relative to the actual wording,
before spending API/agent budget on them.

Alignment check per statement:
  - Search the statement text for any hedging-lexicon phrase (case-insensitive).
  - If found, expected confidence = mean of matched phrases' lexicon means (0-1).
  - If none found, treat as a decisive/non-hedged statement -> expected high (0.85),
    per the same rule used in the LINGUISTIC_EVALUATOR_PROMPT itself.
  - "Aligned" = expected and the actual recorded "_lc" mean fall on the same side
    of 0.5 (both indicate high confidence, or both indicate low confidence).

A row is kept if original_response is aligned AND at least `min_rewrites_aligned`
of the 3 rewritten responses are aligned with their own recorded "_lc" estimate.

Run with the `llm` conda env python.
"""
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/ivan/lm-confidence-evaluation-harness")

SOURCE_PKL = (
    "/hdd/ivny/direct_qa_in_domain_calibration/mmlu/"
    "meta-llama/Llama-3.1-8B-Instruct/calibration_details.pkl"
)
LEXICON_PATH = "/home/ivan/lm-confidence-evaluation-harness/linguistic_confidence_lexicon/hedging_word_aggregated.csv"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

RESPONSE_COLS = [
    "original_response",
    "calibrated_lc_rewritten_response",
    "calibrated_tp_rewritten_response",
    "calibrated_su_rewritten_response",
]
EXISTING_LC_COLS = {
    "original_response": "original_lc",
    "calibrated_lc_rewritten_response": "calibrated_lc_rewritten_lc",
    "calibrated_tp_rewritten_response": "calibrated_tp_rewritten_lc",
    "calibrated_su_rewritten_response": "calibrated_su_rewritten_lc",
}
NO_HEDGE_EXPECTED_MU = 0.85
N_SAMPLES = 50
SAMPLE_SEED = 42


def load_lexicon() -> list[tuple[str, float]]:
    df = pd.read_csv(LEXICON_PATH)[["hedging_word", "mean"]]
    return list(df.itertuples(index=False, name=None))


def expected_mu_for_text(text: str, lexicon: list[tuple[str, float]]) -> tuple[float, list[str]]:
    text_lower = text.lower()
    matched = []
    for phrase, mean in lexicon:
        phrase_lower = phrase.lower()
        if " " in phrase_lower:
            hit = phrase_lower in text_lower
        else:
            hit = re.search(rf"\b{re.escape(phrase_lower)}\b", text_lower) is not None
        if hit:
            matched.append((phrase, mean))
    if not matched:
        return NO_HEDGE_EXPECTED_MU, []
    return float(np.mean([m for _, m in matched])), [p for p, _ in matched]


def aligned(expected_mu: float, actual_mu: float) -> bool:
    return (expected_mu >= 0.5) == (actual_mu >= 0.5)


def analyze():
    with open(SOURCE_PKL, "rb") as f:
        df: pd.DataFrame = pickle.load(f)
    df = df.reset_index(drop=True)  # source index is non-contiguous; downstream code assumes 0-based positions
    lexicon = load_lexicon()

    records = []
    for row_idx, row in df.iterrows():
        rec = {"row_idx": row_idx}
        n_aligned_rewrites = 0
        for col in RESPONSE_COLS:
            text = row[col]
            actual_mu = float(row[EXISTING_LC_COLS[col]].mu)
            expected_mu, matched = expected_mu_for_text(text, lexicon)
            is_aligned = aligned(expected_mu, actual_mu)
            rec[f"{col}__expected_mu"] = expected_mu
            rec[f"{col}__actual_mu"] = actual_mu
            rec[f"{col}__aligned"] = is_aligned
            rec[f"{col}__matched"] = matched
            if col != "original_response" and is_aligned:
                n_aligned_rewrites += 1
        rec["original_aligned"] = rec["original_response__aligned"]
        rec["n_rewrites_aligned"] = n_aligned_rewrites
        records.append(rec)

    result_df = pd.DataFrame(records)

    for min_rewrites in [3, 2, 1, 0]:
        mask = result_df["original_aligned"] & (result_df["n_rewrites_aligned"] >= min_rewrites)
        print(f"original aligned AND >= {min_rewrites}/3 rewrites aligned: {mask.sum()} rows qualify")

    return df, result_df


if __name__ == "__main__":
    df, result_df = analyze()
    with open(os.path.join(CACHE_DIR, "_filter_analysis.pkl"), "wb") as f:
        pickle.dump({"source_df": df, "analysis": result_df}, f)
