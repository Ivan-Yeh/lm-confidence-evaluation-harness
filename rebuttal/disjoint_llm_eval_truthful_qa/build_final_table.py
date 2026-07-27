"""
Combine:
  - the 50 sampled rows (original_response, calibrated_*_rewritten_response,
    correctness, and the pipeline's own "_lc" Beta estimates for each),
  - the disjoint Together (MiniMaxAI/MiniMax-M3) judge's 9-sample scores, and
  - the 10-persona agent annotator scores,
into one detailed table with a fitted BetaDistribution per response x judge,
then compute generalised ECE / Faithfulness Divergence for every confidence
column against `correctness`.

Run with the `llm` conda env python from this directory.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/home/ivan/lm-confidence-evaluation-harness"
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from lm_conf.confidence_metrics.distributionals import BetaDistribution  # noqa: E402
from lm_conf.default_utils.custom_types import OrganisedOutputs  # noqa: E402
from lm_conf.post_processing.metrics import generalised_ece, faithfulness_divergence  # noqa: E402
from prompts import hint_range  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SAMPLED_ROWS_PATH = os.path.join(CACHE_DIR, "sampled_rows.pkl")
TOGETHER_SCORES_PATH = os.path.join(CACHE_DIR, "together_scores.json")
AGENT_SCORES_DIR = os.path.join(CACHE_DIR, "agent_scores")
MANIFEST_PATH = os.path.join(CACHE_DIR, "agent_statement_manifest.json")

RESPONSE_COLS = [
    "original_response",
    "calibrated_lc_rewritten_response",
    "calibrated_tp_rewritten_response",
    "calibrated_su_rewritten_response",
]

# Existing-CSV "_lc" column (pipeline's own lc estimate) for each response column.
EXISTING_LC_COLS = {
    "original_response": "original_lc",
    "calibrated_lc_rewritten_response": "calibrated_lc_rewritten_lc",
    "calibrated_tp_rewritten_response": "calibrated_tp_rewritten_lc",
    "calibrated_su_rewritten_response": "calibrated_su_rewritten_lc",
}

N_PERSONAS = 10


def fit_beta_from_scores(
    scores_0_100: list[float],
    range_low: float = 0.0,
    range_high: float = 100.0,
) -> BetaDistribution:
    # Defensive clip to the hint range in case a judge (model or agent) didn't comply
    # with the "score must fall within the reference range" instruction.
    valid = [
        min(max(s, range_low), range_high) / 100.0
        for s in scores_0_100
        if s is not None and not (isinstance(s, float) and np.isnan(s))
    ]
    if not valid:
        return BetaDistribution(mu=0.5, sigma=1e-6)
    mu = float(np.mean(valid))
    sigma = float(np.std(valid)) if len(valid) > 1 else 1e-6
    return BetaDistribution(mu=mu, sigma=sigma)


def load_together_scores() -> dict[str, list[float]]:
    with open(TOGETHER_SCORES_PATH) as f:
        cache = json.load(f)
    out = {}
    for statement_id, entry in cache.items():
        scores = list(entry["scores"].values())
        out[statement_id] = [s for s in scores if s is not None]
    return out


def load_agent_scores() -> dict[str, list[float]]:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    statement_ids = manifest["statement_ids"]
    personas = manifest["personas"]

    per_statement: dict[str, list[float]] = {sid: [] for sid in statement_ids}
    for persona in personas:
        path = os.path.join(AGENT_SCORES_DIR, f"{persona['id']}.json")
        if not os.path.exists(path):
            print(f"WARNING: missing agent score file {path}, skipping persona {persona['id']}")
            continue
        with open(path) as f:
            scores = json.load(f)
        missing = [sid for sid in statement_ids if sid not in scores]
        if missing:
            print(f"WARNING: persona {persona['id']} missing {len(missing)} ids, e.g. {missing[:5]}")
        for sid in statement_ids:
            if sid in scores and scores[sid] is not None:
                per_statement[sid].append(float(scores[sid]))
    return per_statement


def build_detailed_table() -> pd.DataFrame:
    with open(SAMPLED_ROWS_PATH, "rb") as f:
        sampled = pickle.load(f)

    together_scores = load_together_scores()
    agent_scores = load_agent_scores()

    rows = []
    for row_idx, row in sampled.iterrows():
        record = {
            "source_row_index": row["source_row_index"],
            "correctness": row["accuracy"],
        }
        for col in RESPONSE_COLS:
            record[col] = row[col]

        for col in RESPONSE_COLS:
            statement_id = f"{row_idx}__{col}"
            existing_lc_col = EXISTING_LC_COLS[col]
            existing_lc = row[existing_lc_col]
            range_low, range_high = hint_range(float(existing_lc.mu) * 100.0, float(existing_lc.sigma) * 100.0)
            new_model_beta = fit_beta_from_scores(together_scores.get(statement_id, []), range_low, range_high)
            annotators_beta = fit_beta_from_scores(agent_scores.get(statement_id, []), range_low, range_high)

            if col == "original_response":
                record["original_response_lc"] = row[existing_lc_col]
                record["original_response_lc_new_model"] = new_model_beta
                record["original_response_lc_annotators"] = annotators_beta
            else:
                record[f"{col}_lc"] = row[existing_lc_col]
                record[f"{col}_lc_new_model"] = new_model_beta
                record[f"{col}_lc_annotators"] = annotators_beta

        rows.append(record)

    ordered_cols = [
        "source_row_index",
        "original_response",
        "correctness",
        "calibrated_lc_rewritten_response",
        "calibrated_tp_rewritten_response",
        "calibrated_su_rewritten_response",
        "original_response_lc",
        "original_response_lc_new_model",
        "original_response_lc_annotators",
        "calibrated_lc_rewritten_response_lc",
        "calibrated_tp_rewritten_response_lc",
        "calibrated_su_rewritten_response_lc",
        "calibrated_lc_rewritten_response_lc_new_model",
        "calibrated_tp_rewritten_response_lc_new_model",
        "calibrated_su_rewritten_response_lc_new_model",
        "calibrated_lc_rewritten_response_lc_annotators",
        "calibrated_tp_rewritten_response_lc_annotators",
        "calibrated_su_rewritten_response_lc_annotators",
    ]
    return pd.DataFrame(rows)[ordered_cols]


CONFIDENCE_COLS = [
    "original_response_lc",
    "original_response_lc_new_model",
    "original_response_lc_annotators",
    "calibrated_lc_rewritten_response_lc",
    "calibrated_lc_rewritten_response_lc_new_model",
    "calibrated_lc_rewritten_response_lc_annotators",
    "calibrated_tp_rewritten_response_lc",
    "calibrated_tp_rewritten_response_lc_new_model",
    "calibrated_tp_rewritten_response_lc_annotators",
    "calibrated_su_rewritten_response_lc",
    "calibrated_su_rewritten_response_lc_new_model",
    "calibrated_su_rewritten_response_lc_annotators",
]


def compute_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    accuracies = df["correctness"].astype(float).tolist()
    rows = []
    for col in CONFIDENCE_COLS:
        confidences = df[col].tolist()
        organised = OrganisedOutputs(
            extracted_answers=[[""] * len(confidences)],
            extracted_confidences=[confidences],
            accuracy_scores=[accuracies],
        )
        gece = generalised_ece({}, organised)[0]
        fd = faithfulness_divergence({}, organised)[0]
        rows.append({"column": col, "generalised_ECE": gece, "faithfulness_divergence": fd})
    return pd.DataFrame(rows)


def betas_to_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CONFIDENCE_COLS:
        out[col] = out[col].apply(repr)
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    detailed = build_detailed_table()

    detailed_pkl_path = os.path.join(RESULTS_DIR, "detailed_table.pkl")
    with open(detailed_pkl_path, "wb") as f:
        pickle.dump(detailed, f)

    detailed_csv_path = os.path.join(RESULTS_DIR, "detailed_table.csv")
    betas_to_str(detailed).to_csv(detailed_csv_path, index=False)

    print(f"Detailed table: {detailed_pkl_path} / {detailed_csv_path} ({detailed.shape})")

    summary = compute_summary_metrics(detailed)
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary metrics -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
