"""
Per-row test: does the MEAN and does the VARIANCE of the calibrated signal
distribution propagate to the rewritten linguistic-confidence space?

Complements fd_test.py (which only tested kappa/concentration transfer with a
naive raw pool across all 21 model/dataset combos -- and got weak/mixed
results, including a negative correlation for mmlu/lc, because pooling
absolute kappa across models with very different baseline scales is
contaminated by Simpson's paradox) and ensemble_concentration_test/ (which
fixed that for kappa specifically with a fixed-effects design).

Here we run the same two-sided test (naive raw pool AND within-group
fixed-effects pool) on mu (mean) and var (variance) separately, per row:

  mu_target,  var_target  = mean/variance of calibrated_{signal}            (pre-rewrite target)
  mu_output,  var_output  = mean/variance of calibrated_{signal}_rewritten_lc (post-rewrite, ensemble re-measured)

across all 21 (dataset, model) combos in /hdd/ivny/direct_qa_in_domain_calibration,
all rows, 3 signals (lc/tp/su) -- the `rewritten` pathway (LLM rewrite
conditioned on retrieved lexicon hedge words), same mechanism the reviewer
named in concern #3.

Run with the `llm` conda env python.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

DATA_ROOT = "/hdd/ivny/direct_qa_in_domain_calibration"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS = ["lc", "tp", "su"]

BETA_RE = re.compile(r"alpha=([-\d.eE]+),\s*beta=([-\d.eE]+),\s*mu=([-\d.eE]+),\s*sigma=([-\d.eE]+)")


def parse_beta(s):
    m = BETA_RE.search(str(s))
    if m is None:
        return None
    alpha, beta, mu, sigma = (float(x) for x in m.groups())
    return dict(mu=mu, var=sigma ** 2, kappa=alpha + beta)


def find_files():
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*", "*", "calibration_details.csv")))
    records = []
    for p in paths:
        rel = os.path.relpath(p, DATA_ROOT)
        parts = rel.split(os.sep)
        dataset = parts[0]
        model = "/".join(parts[1:-1])
        records.append((dataset, model, p))
    return records


def build_long_table() -> pd.DataFrame:
    rows = []
    for dataset, model, path in find_files():
        df = pd.read_csv(path)
        for signal in SIGNALS:
            tgt_col = f"calibrated_{signal}"
            out_col = f"calibrated_{signal}_rewritten_lc"
            if tgt_col not in df.columns or out_col not in df.columns:
                continue
            tgt_parsed = df[tgt_col].map(parse_beta)
            out_parsed = df[out_col].map(parse_beta)
            for idx, (t, o) in enumerate(zip(tgt_parsed, out_parsed)):
                if t is None or o is None:
                    continue
                rows.append(dict(
                    dataset=dataset, model=model, signal=signal, row=idx,
                    mu_target=t["mu"], var_target=t["var"], kappa_target=t["kappa"],
                    mu_output=o["mu"], var_output=o["var"], kappa_output=o["kappa"],
                ))
    return pd.DataFrame(rows)


def corr_block(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return dict(n=n, spearman_r=np.nan, spearman_p=np.nan, pearson_r=np.nan, pearson_p=np.nan)
    spear_r, spear_p = stats.spearmanr(x, y)
    pear_r, pear_p = stats.pearsonr(x, y)
    return dict(n=n, spearman_r=spear_r, spearman_p=spear_p, pearson_r=pear_r, pearson_p=pear_p)


def run_group_correlations(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, model, signal), g in df.groupby(["dataset", "model", "signal"]):
        mu_res = corr_block(g["mu_target"].to_numpy(), g["mu_output"].to_numpy())
        var_res = corr_block(g["var_target"].to_numpy(), g["var_output"].to_numpy())
        out.append(dict(dataset=dataset, model=model, signal=signal,
                         **{f"mu_{k}": v for k, v in mu_res.items()},
                         **{f"var_{k}": v for k, v in var_res.items()}))
    return pd.DataFrame(out).sort_values(["signal", "dataset", "model"])


def summarize_scoreboard(group_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for quantity in ["mu", "var"]:
        r_col, p_col = f"{quantity}_spearman_r", f"{quantity}_spearman_p"
        for signal, g in group_df.groupby("signal"):
            valid = g.dropna(subset=[r_col])
            n_groups = len(valid)
            n_pos = int((valid[r_col] > 0).sum())
            n_sig_pos = int(((valid[r_col] > 0) & (valid[p_col] < 0.05)).sum())
            n_sig_neg = int(((valid[r_col] < 0) & (valid[p_col] < 0.05)).sum())
            out.append(dict(
                quantity=quantity, signal=signal, n_groups=n_groups, n_positive=n_pos,
                frac_positive=n_pos / n_groups if n_groups else np.nan,
                n_significant_positive=n_sig_pos, n_significant_negative=n_sig_neg,
                median_spearman_r=valid[r_col].median(),
            ))
    return pd.DataFrame(out)


def add_demeaned(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby(["dataset", "model", "signal"])
    for col in ["mu_target", "mu_output", "var_target", "var_output"]:
        df[f"{col}_dm"] = df[col] - group[col].transform("mean")
    return df


def run_fe_pooled_correlation(df: pd.DataFrame, df_dm: pd.DataFrame) -> pd.DataFrame:
    """Fixed-effects (within-group demeaned) pooled correlation: removes each group's own
    mean level before pooling, so heterogeneous baseline mu/var scales across models/datasets
    can't dominate or hide the pooled estimate (the Simpson's-paradox fix)."""
    out = []
    for quantity, tgt_col, out_col in [("mu", "mu_target_dm", "mu_output_dm"),
                                         ("var", "var_target_dm", "var_output_dm")]:
        for signal, g in df_dm.groupby("signal"):
            pr, pp = stats.pearsonr(g[tgt_col], g[out_col])
            sr, sp = stats.spearmanr(g[tgt_col], g[out_col])
            out.append(dict(quantity=quantity, signal=signal, level="within_group_pooled", n=len(g),
                             pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp))
        pr, pp = stats.pearsonr(df_dm[tgt_col], df_dm[out_col])
        sr, sp = stats.spearmanr(df_dm[tgt_col], df_dm[out_col])
        out.append(dict(quantity=quantity, signal="ALL", level="within_group_pooled", n=len(df_dm),
                         pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp))

        # naive raw pool for contrast (no fixed effects -- what fd_test.py originally did)
        for signal, g in df.groupby("signal"):
            sr, sp = stats.spearmanr(g[f"{quantity}_target"], g[f"{quantity}_output"])
            out.append(dict(quantity=quantity, signal=signal, level="raw_pooled_no_fe", n=len(g),
                             pearson_r=np.nan, pearson_p=np.nan, spearman_r=sr, spearman_p=sp))
    return pd.DataFrame(out)


def main():
    df = build_long_table()
    print(f"Built long table: {df.shape}")
    df.to_csv(os.path.join(OUT_DIR, "fd_test_mean_variance_data.csv"), index=False)

    group_df = run_group_correlations(df)
    group_df.to_csv(os.path.join(OUT_DIR, "fd_test_mean_variance_by_group.csv"), index=False)
    print("\n=== Per-group (dataset, model, signal) correlation: mu and var, target vs output ===")
    print(group_df.to_string(index=False))

    scoreboard_df = summarize_scoreboard(group_df)
    scoreboard_df.to_csv(os.path.join(OUT_DIR, "fd_test_mean_variance_scoreboard.csv"), index=False)
    print("\n=== Scoreboard across all 21 (dataset, model) groups, per signal ===")
    print(scoreboard_df.to_string(index=False))

    df_dm = add_demeaned(df)
    fe_df = run_fe_pooled_correlation(df, df_dm)
    fe_df.to_csv(os.path.join(OUT_DIR, "fd_test_mean_variance_fe_pooled.csv"), index=False)
    print("\n=== Fixed-effects pooled correlation (mu, var) vs raw pooled (no FE) ===")
    print(fe_df.to_string(index=False))


if __name__ == "__main__":
    main()
