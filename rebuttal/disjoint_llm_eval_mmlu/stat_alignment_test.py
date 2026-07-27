"""
Statistical alignment tests between the three confidence judges' means
(pipeline/"ensemble" _lc, disjoint new_model, and the 10-agent annotators)
on the detailed_table.csv output.

For each response type (original + 3 rewrites) and pooled across all of
them, computes for every judge pair:
  - Pearson r (linear correlation) and Spearman rho (rank correlation)
  - Lin's concordance correlation coefficient (CCC) -- agreement, not just
    correlation: penalizes systematic shift/scale differences that
    correlation alone would miss
  - Paired t-test and Wilcoxon signed-rank test on the mean differences
    (H0: no systematic difference between the two judges' means)

Run with the `llm` conda env python.
"""
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DETAILED_CSV = os.path.join(RESULTS_DIR, "detailed_table.csv")

RESPONSE_TYPES = {
    "original_response": "original_response",
    "calibrated_lc_rewritten": "calibrated_lc_rewritten_response",
    "calibrated_tp_rewritten": "calibrated_tp_rewritten_response",
    "calibrated_su_rewritten": "calibrated_su_rewritten_response",
}

JUDGES = ["pipeline", "new_model", "annotators"]


def col_name(response_prefix: str, judge: str) -> str:
    base = response_prefix if response_prefix != "original_response" else "original_response"
    suffix = {"pipeline": "_lc", "new_model": "_lc_new_model", "annotators": "_lc_annotators"}[judge]
    return f"{base}{suffix}"


def extract_mu(beta_repr: str) -> float:
    m = re.search(r"mu=([0-9.eE+-]+)", beta_repr)
    return float(m.group(1))


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    sxy = np.mean((x - mx) * (y - my))
    return (2 * sxy) / (vx + vy + (mx - my) ** 2)


def pair_stats(x: np.ndarray, y: np.ndarray) -> dict:
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    ccc = lin_ccc(x, y)
    t_stat, t_p = stats.ttest_rel(x, y)
    try:
        w_stat, w_p = stats.wilcoxon(x, y)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {
        "n": len(x),
        "mean_diff": float(np.mean(x - y)),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "ccc": ccc,
        "paired_t": t_stat, "paired_t_p": t_p,
        "wilcoxon_p": w_p,
    }


def main():
    df = pd.read_csv(DETAILED_CSV)

    mu = {}
    for key, response_prefix in RESPONSE_TYPES.items():
        for judge in JUDGES:
            col = col_name(response_prefix, judge)
            mu[(key, judge)] = df[col].apply(extract_mu).to_numpy()

    pairs = [("pipeline", "new_model"), ("pipeline", "annotators"), ("new_model", "annotators")]

    print("=== Per response-type alignment (N=50 each) ===\n")
    per_type_rows = []
    for key in RESPONSE_TYPES:
        print(f"--- {key} ---")
        for a, b in pairs:
            s = pair_stats(mu[(key, a)], mu[(key, b)])
            print(f"  {a:11s} vs {b:11s}: r={s['pearson_r']:.3f} (p={s['pearson_p']:.4f})  "
                  f"rho={s['spearman_r']:.3f} (p={s['spearman_p']:.4f})  CCC={s['ccc']:.3f}  "
                  f"mean_diff={s['mean_diff']:+.4f}  paired-t p={s['paired_t_p']:.4f}  wilcoxon p={s['wilcoxon_p']:.4f}")
            row = {"response_type": key, "pair": f"{a}_vs_{b}", **s}
            per_type_rows.append(row)
        print()

    print("=== Pooled across all 4 response types (N=200) ===\n")
    pooled_rows = []
    for a, b in pairs:
        x = np.concatenate([mu[(key, a)] for key in RESPONSE_TYPES])
        y = np.concatenate([mu[(key, b)] for key in RESPONSE_TYPES])
        s = pair_stats(x, y)
        print(f"  {a:11s} vs {b:11s}: r={s['pearson_r']:.3f} (p={s['pearson_p']:.2e})  "
              f"rho={s['spearman_r']:.3f} (p={s['spearman_p']:.2e})  CCC={s['ccc']:.3f}  "
              f"mean_diff={s['mean_diff']:+.4f}  paired-t p={s['paired_t_p']:.4f}  wilcoxon p={s['wilcoxon_p']:.4f}")
        pooled_rows.append({"pair": f"{a}_vs_{b}", **s})

    per_type_df = pd.DataFrame(per_type_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    per_type_df.to_csv(os.path.join(RESULTS_DIR, "alignment_stats_per_response_type.csv"), index=False)
    pooled_df.to_csv(os.path.join(RESULTS_DIR, "alignment_stats_pooled.csv"), index=False)
    print(f"\nSaved: results/alignment_stats_per_response_type.csv, results/alignment_stats_pooled.csv")


if __name__ == "__main__":
    main()
