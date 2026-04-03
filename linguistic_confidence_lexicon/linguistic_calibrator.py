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
    sample_size: int = 300,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Find the top K closest hedging words to a target beta distribution.
    
    Args:
        target_alpha: alpha parameter of target beta distribution
        target_beta: beta parameter of target beta distribution
        lexicon_df: DataFrame with hedging words and their beta parameters
        top_k: number of closest matches to return
    
    Returns:
        DataFrame with top K closest hedging words and their distances
    """
    rng = np.random.default_rng()
    target_samples = beta.rvs(target_alpha, target_beta, size=sample_size, random_state=rng)

    records = []
    for row in lexicon_df.itertuples(index=False):
        alpha = getattr(row, "alpha_param", np.nan)
        beta_param = getattr(row, "beta_param", np.nan)
        if pd.isna(alpha) or pd.isna(beta_param):
            continue
        records.append((row.hedging_word, alpha, beta_param, row.mean))

    if not records:
        return pd.DataFrame(columns=["hedging_word", "alpha", "beta", "mean", "wasserstein_distance"])

    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() or 1)

    if n_jobs <= 1:
        distances = [
            _compute_distance(record, target_samples, sample_size, int(rng.integers(0, 2**32 - 1)))
            for record in records
        ]
    else:
        # Per-task seeds keep sampling independent across worker processes.
        seeds = rng.integers(0, 2**32 - 1, size=len(records), dtype=np.uint32)
        max_workers = min(n_jobs, len(records))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            distances = list(
                executor.map(
                    _compute_distance,
                    records,
                    [target_samples] * len(records),
                    [sample_size] * len(records),
                    seeds.tolist(),
                )
            )
    
    # Sort by distance and return top K
    result_df = pd.DataFrame(distances).sort_values('wasserstein_distance').head(top_k)
    return result_df.reset_index(drop=True)