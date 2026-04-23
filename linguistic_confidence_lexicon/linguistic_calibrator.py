import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.stats import beta

def _compute_distance(record, target_samples, sample_size: int, seed: int):
    hedging_word, alpha, beta_param, mean = record
    rng = np.random.default_rng(seed)
    word_samples = beta.rvs(alpha, beta_param, size=sample_size, random_state=rng)
    w_dist = wasserstein_distance(target_samples, word_samples)
    return {
        "hedging_word": hedging_word,
        "alpha": alpha,
        "beta": beta_param,
        "mean": mean,
        "wasserstein_distance": w_dist,
    }


def find_closest_hedging_words(
    target_alpha,
    target_beta,
    lexicon_df,
    top_k=10,
    shortlist_k: int = 30,
    sample_size: int = 300,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Find the top K closest hedging words to a target beta distribution.

    Two-stage retrieval:
      1. Shortlist `shortlist_k` candidates by |target_mean - lexicon_mean|.
      2. Re-rank the shortlist with Wasserstein-2 distance and return top_k.

    Args:
        target_alpha:  alpha parameter of target Beta distribution
        target_beta:   beta parameter of target Beta distribution
        lexicon_df:    DataFrame with hedging words and their beta parameters
        top_k:         number of results to return
        shortlist_k:   size of mean-distance shortlist fed into W2 ranking
        sample_size:   MC samples per candidate for W2 estimation
        n_jobs:        worker processes for W2 computation (1 = sequential)

    Returns:
        DataFrame with top_k rows sorted by wasserstein_distance.
    """
    empty = pd.DataFrame(columns=["hedging_word", "alpha", "beta", "mean", "wasserstein_distance"])

    if not (np.isfinite(target_alpha) and np.isfinite(target_beta)
            and target_alpha > 0 and target_beta > 0):
        return empty

    target_mean = target_alpha / (target_alpha + target_beta)

    # --- Stage 1: shortlist by |mean - target_mean| ---
    records = []
    for row in lexicon_df.itertuples(index=False):
        alpha = getattr(row, "alpha_param", np.nan)
        beta_param = getattr(row, "beta_param", np.nan)
        if pd.isna(alpha) or pd.isna(beta_param) or alpha <= 0 or beta_param <= 0:
            continue
        records.append((row.hedging_word, alpha, beta_param, row.mean))

    if not records:
        return empty

    shortlist_k = max(top_k, min(shortlist_k, len(records)))
    records.sort(key=lambda r: abs(r[3] - target_mean))
    shortlist = records[:shortlist_k]

    # --- Stage 2: re-rank shortlist by Wasserstein-2 distance ---
    rng = np.random.default_rng()
    target_samples = beta.rvs(target_alpha, target_beta, size=sample_size, random_state=rng)

    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() or 1)

    if n_jobs <= 1:
        distances = [
            _compute_distance(record, target_samples, sample_size, int(rng.integers(0, 2**32 - 1)))
            for record in shortlist
        ]
    else:
        seeds = rng.integers(0, 2**32 - 1, size=len(shortlist), dtype=np.uint32)
        max_workers = min(n_jobs, len(shortlist))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            distances = list(
                executor.map(
                    _compute_distance,
                    shortlist,
                    [target_samples] * len(shortlist),
                    [sample_size] * len(shortlist),
                    seeds.tolist(),
                )
            )

    result_df = pd.DataFrame(distances).sort_values("wasserstein_distance").head(top_k)
    return result_df.reset_index(drop=True)