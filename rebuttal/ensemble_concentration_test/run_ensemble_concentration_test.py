"""
Reviewer concern #3: "Concentration is never calibrated, yet FD depends on it
... the output's concentration is simply whatever the retrieved lexicon
expressions happen to impose. The faithfulness claim therefore rests on a
quantity the pipeline does not control."

Unlike `rebuttal/fd_test/` (naive pooled Pearson/Spearman on raw kappa,
mixed with all models/datasets/signals thrown together -- which is
contaminated by Simpson's paradox because different models/datasets sit at
very different absolute kappa scales) and `rebuttal/concentration_control_test/`
(rigorous stats, but only n=50 statements x 2 datasets, judged by an external
disjoint model (MiniMax-M3) and Claude agent personas), this script runs the
same rigorous battery at full scale using the pipeline's OWN ensemble
linguistic-confidence extractor -- the actual read-out RALC's FD metric is
computed on -- rather than an external judge.

Mechanism under test:
  1. `calibrated_{signal}` (signal in lc/tp/su) is the calibration TARGET: a
     Beta distribution the isotonic/Platt-mapped confidence should express.
  2. To realize that target in text, the pipeline retrieves hedging words
     near the target from the lexicon and asks an LLM to rewrite the
     original response using them -> `calibrated_{signal}_rewritten_response`.
     This is exactly the "retrieved lexicon expressions" step the reviewer
     is worried is uncontrolled. (`_beta_guided_response` is an alternate
     pathway that conditions directly on the Beta(alpha,beta) numbers instead
     of discrete lexicon words -- included as a secondary robustness check,
     not the reviewer's primary target.)
  3. That generated text is then re-scored from scratch by the SAME 3-model
     ensemble LC extractor (Qwen3-8B, Llama-3.1-8B-Instruct, Mistral-7B-
     Instruct-v0.3; 3 samples each = 9 calls, Beta-fit) used to score
     original_lc -> `calibrated_{signal}_rewritten_lc` / `_beta_guided_lc`.
     This is "our ensemble method," the actual quantity FD is computed on
     (see calibration_performance.csv's `..._rewritten_lc_faithfulness_
     divergence` rows) -- not an external judge's opinion.

Question: does kappa_target (step 1, chosen before any lexicon retrieval)
control kappa_output (step 3, measured after lexicon retrieval + generation),
or is kappa_output an arbitrary byproduct of whichever lexicon entry got
retrieved?

Data: every (dataset, model) in /hdd/ivny/direct_qa_in_domain_calibration
(3 datasets x 7 models = 21 combos), all rows (not a 50-row subsample) x 3
signals x 2 rewrite variants (rewritten, beta_guided) = 84 (dataset, model,
signal, variant) groups, ~9.8k-14k rows/group depending on dataset.

Tests run:
  1. Per-group correlation: kappa_target vs kappa_output, Spearman +
     Pearson-on-log(kappa). Reported per (dataset, model, signal, variant)
     plus a sign/significance scoreboard across all 84 groups.
  2. Fixed-effects (within-group demeaned) pooled correlation: removes each
     group's own mean level from kappa_target/kappa_output/var_output before
     pooling, so heterogeneous absolute kappa scales across models/datasets
     cannot manufacture or hide a relationship (the Simpson's-paradox fix for
     what likely sank the naive pooling in `fd_test/`).
  3. Within-group OLS: demeaned var_output ~ demeaned log(kappa_target) +
     demeaned mu_target + demeaned mu_target*(1-mu_target). The mu terms
     absorb the mechanical "extreme means have bounded variance" confound so
     we isolate kappa_target's effect on output variance beyond it.
  4. Tertile split of (demeaned) kappa_target + Levene's test for equal
     variance of (demeaned) var_output across Low/Med/High bins.
  5. Permutation test: shuffle kappa_target WITHIN each group (preserving
     group structure / marginal scale) 2,000x, rebuild the null distribution
     of the pooled Spearman correlation between kappa_target and kappa_output.
  6. Link table: merge each group's transfer correlation with the model's
     actual reported `_faithfulness_divergence` from calibration_performance.csv,
     to show the FD numbers RALC reports are not decoupled from how well
     concentration transfers.

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
VARIANTS = ["rewritten", "beta_guided"]
RNG_SEED = 42
N_PERM = 2000

BETA_RE = re.compile(r"alpha=([-\d.eE]+),\s*beta=([-\d.eE]+),\s*mu=([-\d.eE]+),\s*sigma=([-\d.eE]+)")


def parse_beta(s: str):
    m = BETA_RE.search(str(s))
    if m is None:
        return None
    alpha, beta, mu, sigma = (float(x) for x in m.groups())
    return dict(kappa=alpha + beta, mu=mu, var=sigma ** 2)


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
            if tgt_col not in df.columns:
                continue
            tgt_parsed = df[tgt_col].map(parse_beta)
            for variant in VARIANTS:
                out_col = f"calibrated_{signal}_{variant}_lc"
                if out_col not in df.columns:
                    continue
                out_parsed = df[out_col].map(parse_beta)
                for idx, (t, o) in enumerate(zip(tgt_parsed, out_parsed)):
                    if t is None or o is None:
                        continue
                    rows.append(dict(
                        dataset=dataset, model=model, signal=signal, variant=variant, row=idx,
                        kappa_target=t["kappa"], mu_target=t["mu"], var_target=t["var"],
                        kappa_output=o["kappa"], mu_output=o["mu"], var_output=o["var"],
                    ))
    return pd.DataFrame(rows)


def corr_block(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return dict(n=n, spearman_r=np.nan, spearman_p=np.nan, pearson_log_r=np.nan, pearson_log_p=np.nan)
    spear_r, spear_p = stats.spearmanr(x, y)
    plog_r, plog_p = stats.pearsonr(np.log(x), np.log(y))
    return dict(n=n, spearman_r=spear_r, spearman_p=spear_p, pearson_log_r=plog_r, pearson_log_p=plog_p)


def run_group_correlations(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, model, signal, variant), g in df.groupby(["dataset", "model", "signal", "variant"]):
        res = corr_block(g["kappa_target"].to_numpy(), g["kappa_output"].to_numpy())
        out.append(dict(dataset=dataset, model=model, signal=signal, variant=variant, **res))
    return pd.DataFrame(out).sort_values(["variant", "signal", "dataset", "model"])


def summarize_scoreboard(group_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for variant, g in group_df.groupby("variant"):
        valid = g.dropna(subset=["spearman_r"])
        n_groups = len(valid)
        n_pos = int((valid["spearman_r"] > 0).sum())
        n_sig_pos = int(((valid["spearman_r"] > 0) & (valid["spearman_p"] < 0.05)).sum())
        n_sig_neg = int(((valid["spearman_r"] < 0) & (valid["spearman_p"] < 0.05)).sum())
        out.append(dict(
            variant=variant, n_groups=n_groups, n_positive=n_pos,
            frac_positive=n_pos / n_groups if n_groups else np.nan,
            n_significant_positive=n_sig_pos, n_significant_negative=n_sig_neg,
            median_spearman_r=valid["spearman_r"].median(),
        ))
    return pd.DataFrame(out)


def add_demeaned(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_kappa_target"] = np.log(df["kappa_target"])
    df["log_kappa_output"] = np.log(df["kappa_output"])
    group = df.groupby(["dataset", "model", "signal", "variant"])
    for col in ["log_kappa_target", "log_kappa_output", "var_output", "mu_target"]:
        df[f"{col}_dm"] = df[col] - group[col].transform("mean")
    df["boundary_target"] = df["mu_target"] * (1 - df["mu_target"])
    df["boundary_target_dm"] = df["boundary_target"] - group["boundary_target"].transform("mean")
    df["group_id"] = (df["dataset"] + "|" + df["model"] + "|" + df["signal"] + "|" + df["variant"])
    return df


def run_fe_pooled_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Within-group (fixed-effects) pooled correlation: kappa_target vs kappa_output
    after removing each group's own mean, so heterogeneous baseline kappa across
    models/datasets/signals cannot dominate the pooled estimate."""
    out = []
    for variant, g in df.groupby("variant"):
        r, p = stats.pearsonr(g["log_kappa_target_dm"], g["log_kappa_output_dm"])
        rs, ps = stats.spearmanr(g["log_kappa_target_dm"], g["log_kappa_output_dm"])
        out.append(dict(variant=variant, level="within_group_pooled", n=len(g),
                         pearson_r_on_demeaned_log_kappa=r, pearson_p=p,
                         spearman_r=rs, spearman_p=ps))
    # also raw (non-demeaned) pooled, for contrast with fd_test/'s naive approach
    for variant, g in df.groupby("variant"):
        rs, ps = stats.spearmanr(g["kappa_target"], g["kappa_output"])
        out.append(dict(variant=variant, level="raw_pooled_no_fe", n=len(g),
                         pearson_r_on_demeaned_log_kappa=np.nan, pearson_p=np.nan,
                         spearman_r=rs, spearman_p=ps))
    return pd.DataFrame(out)


def run_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Within-group OLS: var_output_dm ~ log_kappa_target_dm + mu_target_dm + boundary_target_dm
    (no intercept needed -- demeaning already removes group-level means)."""
    out = []
    for variant, g in df.groupby("variant"):
        y = g["var_output_dm"].to_numpy()
        X = np.column_stack([
            g["log_kappa_target_dm"].to_numpy(),
            g["mu_target_dm"].to_numpy(),
            g["boundary_target_dm"].to_numpy(),
        ])
        beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta_hat
        resid = y - y_hat
        n, k = X.shape
        dof = n - k
        sigma2 = np.sum(resid ** 2) / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_stats = beta_hat / se
        p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot
        names = ["log_kappa_target_dm", "mu_target_dm", "boundary_target_dm"]
        for name, coef, se_i, t_i, p_i in zip(names, beta_hat, se, t_stats, p_vals):
            out.append(dict(variant=variant, term=name, coef=coef, se=se_i, t=t_i, p=p_i, n=n, r2=r2))
    return pd.DataFrame(out)


def run_levene(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for variant, g in df.groupby("variant"):
        tertiles = pd.qcut(g["log_kappa_target_dm"], 3, labels=["low", "mid", "high"])
        g = g.assign(bin=tertiles)
        groups = [gg["var_output_dm"].to_numpy() for _, gg in g.groupby("bin", observed=True)]
        labels = [lbl for lbl, _ in g.groupby("bin", observed=True)]
        stat, p = stats.levene(*groups, center="median")
        means = {f"mean_var_output_dm_{lbl}": np.mean(gg) for lbl, gg in zip(labels, groups)}
        ns = {f"n_{lbl}": len(gg) for lbl, gg in zip(labels, groups)}
        out.append(dict(variant=variant, levene_stat=stat, levene_p=p, **means, **ns))
    return pd.DataFrame(out)


def run_permutation(df: pd.DataFrame, n_perm: int = N_PERM, seed: int = RNG_SEED) -> pd.DataFrame:
    """Shuffle kappa_target WITHIN each (dataset, model, signal, variant) group,
    recompute the pooled Spearman correlation vs kappa_output, and see how often
    a within-group-arbitrary assignment of concentration produces a correlation
    at least as strong as observed."""
    rng = np.random.default_rng(seed)
    out = []
    for variant, g in df.groupby("variant"):
        x = g["kappa_target"].to_numpy().copy()
        y = g["kappa_output"].to_numpy()
        group_ids = g["group_id"].to_numpy()
        group_idx = {}
        for gid in np.unique(group_ids):
            group_idx[gid] = np.where(group_ids == gid)[0]

        observed_r, _ = stats.spearmanr(x, y)
        null_r = np.empty(n_perm)
        x_perm = x.copy()
        for i in range(n_perm):
            for idx in group_idx.values():
                x_perm[idx] = x[rng.permutation(idx)]
            null_r[i], _ = stats.spearmanr(x_perm, y)
        p_perm = (np.sum(np.abs(null_r) >= np.abs(observed_r)) + 1) / (n_perm + 1)
        out.append(dict(
            variant=variant, n=len(x), n_groups=len(group_idx), n_perm=n_perm,
            observed_spearman_r=observed_r, null_mean=null_r.mean(), null_std=null_r.std(),
            perm_p_two_sided=p_perm,
        ))
    return pd.DataFrame(out)


def load_performance_link(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, model, _ in find_files():
        perf_path = os.path.join(DATA_ROOT, dataset, model, "calibration_performance.csv")
        if not os.path.exists(perf_path):
            continue
        perf = pd.read_csv(perf_path).set_index("metric")["value"]
        for signal in SIGNALS:
            for variant, out_prefix in [("rewritten", f"calibrated_{signal}_rewritten_lc"),
                                          ("beta_guided", f"calibrated_{signal}_beta_guided_lc")]:
                fd_key = f"{out_prefix}_faithfulness_divergence"
                ece_key = f"{out_prefix}_generalised_ECE"
                if fd_key not in perf.index:
                    continue
                match = group_df[(group_df.dataset == dataset) & (group_df.model == model)
                                  & (group_df.signal == signal) & (group_df.variant == variant)]
                r = match["spearman_r"].iloc[0] if len(match) else np.nan
                p = match["spearman_p"].iloc[0] if len(match) else np.nan
                rows.append(dict(
                    dataset=dataset, model=model, signal=signal, variant=variant,
                    kappa_transfer_spearman_r=r, kappa_transfer_spearman_p=p,
                    faithfulness_divergence=perf.get(fd_key, np.nan),
                    generalised_ECE=perf.get(ece_key, np.nan),
                ))
    return pd.DataFrame(rows)


def main():
    df = build_long_table()
    print(f"Built long table: {df.shape}")
    df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_data.csv"), index=False)

    group_df = run_group_correlations(df)
    group_df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_by_group.csv"), index=False)
    print("\n=== Per-group (dataset, model, signal, variant) correlation ===")
    print(group_df.to_string(index=False))

    scoreboard_df = summarize_scoreboard(group_df)
    scoreboard_df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_scoreboard.csv"), index=False)
    print("\n=== Scoreboard across all groups ===")
    print(scoreboard_df.to_string(index=False))

    df_dm = add_demeaned(df)

    fe_df = run_fe_pooled_correlation(df_dm)
    fe_df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_fe_pooled.csv"), index=False)
    print("\n=== Fixed-effects pooled correlation (vs raw pooled, no FE) ===")
    print(fe_df.to_string(index=False))

    reg_df = run_regression(df_dm)
    reg_df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_regression.csv"), index=False)
    print("\n=== Within-group OLS: var_output ~ log(kappa_target) + mu confound ===")
    print(reg_df.to_string(index=False))

    levene_df = run_levene(df_dm)
    levene_df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_levene.csv"), index=False)
    print("\n=== Levene's test across within-group kappa_target tertiles ===")
    print(levene_df.to_string(index=False))

    perm_df = run_permutation(df_dm)
    perm_df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_permutation.csv"), index=False)
    print(f"\n=== Permutation test ({N_PERM} within-group shuffles) ===")
    print(perm_df.to_string(index=False))

    link_df = load_performance_link(group_df)
    link_df.to_csv(os.path.join(OUT_DIR, "ensemble_concentration_fd_link.csv"), index=False)
    print("\n=== Link to reported FD/ECE per (dataset, model, signal, variant) ===")
    print(link_df.to_string(index=False))


if __name__ == "__main__":
    main()
