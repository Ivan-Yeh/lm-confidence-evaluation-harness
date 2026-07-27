"""
Score all (row, response_column) statements with the disjoint Together judge
(MiniMaxAI/MiniMax-M3), 9 independent samples each, with a per-statement
"original confidence" hint (mean +/- 2 std reference range).

Resumable: results are cached per (statement_id, attempt) in a single JSON
file, so re-running only fills in missing entries. Uses a small thread pool;
each request goes through together_client.query_together, which already
retries with backoff on rate limits.

Requires TOGETHER_API_KEY to be exported in the same shell invocation that
runs this script -- never stored in a file.
"""
import json
import logging
import os
import pickle
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/ivan/lm-confidence-evaluation-harness")

from prompts import build_evaluator_prompt, hint_range  # noqa: E402
from human_cues import sample_human_cues, format_cues_for_prompt  # noqa: E402
from together_client import query_together  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
RESULTS_CACHE_PATH = os.path.join(CACHE_DIR, "together_scores.json")
SAMPLED_ROWS_PATH = os.path.join(CACHE_DIR, "sampled_rows.pkl")

N_SAMPLES_PER_STATEMENT = 9
MAX_WORKERS = 6

RESPONSE_COLS = [
    "original_response",
    "calibrated_lc_rewritten_response",
    "calibrated_tp_rewritten_response",
    "calibrated_su_rewritten_response",
]

# Existing-CSV "_lc" column (pipeline's own lc estimate) for each response column;
# used as the per-statement "original confidence" hint in the evaluator prompt.
EXISTING_LC_COLS = {
    "original_response": "original_lc",
    "calibrated_lc_rewritten_response": "calibrated_lc_rewritten_lc",
    "calibrated_tp_rewritten_response": "calibrated_tp_rewritten_lc",
    "calibrated_su_rewritten_response": "calibrated_su_rewritten_lc",
}

_lock = threading.Lock()


def extract_score(text: str, range_low: float = 0.0, range_high: float = 100.0) -> float | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    score = float(match.group(1))
    return min(max(score, range_low), range_high)


def load_cache() -> dict:
    if os.path.exists(RESULTS_CACHE_PATH):
        with open(RESULTS_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    tmp_path = RESULTS_CACHE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp_path, RESULTS_CACHE_PATH)


def build_statements() -> dict[str, dict]:
    with open(SAMPLED_ROWS_PATH, "rb") as f:
        sampled = pickle.load(f)

    statements = {}
    for row_idx, row in sampled.iterrows():
        for col in RESPONSE_COLS:
            statement_id = f"{row_idx}__{col}"
            existing_lc = row[EXISTING_LC_COLS[col]]
            statements[statement_id] = {
                "sentence": row[col],
                "orig_conf_mean": float(existing_lc.mu) * 100.0,
                "orig_conf_std": float(existing_lc.sigma) * 100.0,
            }
    return statements


def run(limit_statements: int | None = None, n_samples: int = N_SAMPLES_PER_STATEMENT) -> None:
    cues_text = format_cues_for_prompt(sample_human_cues())
    statements = build_statements()

    if limit_statements is not None:
        statements = dict(list(statements.items())[:limit_statements])

    cache = load_cache()

    jobs = []
    for statement_id, info in statements.items():
        entry = cache.setdefault(statement_id, {"sentence": info["sentence"], "raw": {}, "scores": {}})
        entry["sentence"] = info["sentence"]
        for attempt in range(n_samples):
            attempt_key = str(attempt)
            if attempt_key in entry["scores"] and entry["scores"][attempt_key] is not None:
                continue
            jobs.append((statement_id, info, attempt_key))

    logging.info(
        "Statements: %d, total samples needed: %d, jobs to run: %d",
        len(statements), len(statements) * n_samples, len(jobs),
    )

    if not jobs:
        logging.info("Nothing to do; cache already complete.")
        return

    def worker(job):
        statement_id, info, attempt_key = job
        prompt = build_evaluator_prompt(
            info["sentence"],
            orig_conf_mean=info["orig_conf_mean"],
            orig_conf_std=info["orig_conf_std"],
            cues_text=cues_text,
        )
        try:
            raw = query_together(prompt)
        except Exception as e:
            logging.error("Failed statement=%s attempt=%s: %s", statement_id, attempt_key, e)
            return statement_id, attempt_key, None, str(e)
        range_low, range_high = hint_range(info["orig_conf_mean"], info["orig_conf_std"])
        score = extract_score(raw, range_low=range_low, range_high=range_high)
        return statement_id, attempt_key, raw, score

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for fut in as_completed(futures):
            statement_id, attempt_key, raw, score = fut.result()
            with _lock:
                entry = cache.setdefault(statement_id, {"sentence": statements[statement_id]["sentence"], "raw": {}, "scores": {}})
                entry["raw"][attempt_key] = raw
                entry["scores"][attempt_key] = score
                completed += 1
                if completed % 25 == 0:
                    save_cache(cache)
                    logging.info("Progress: %d/%d", completed, len(jobs))

    save_cache(cache)
    logging.info("Done. Cache saved to %s", RESULTS_CACHE_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-statements", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_STATEMENT)
    args = parser.parse_args()
    run(limit_statements=args.limit_statements, n_samples=args.n_samples)
