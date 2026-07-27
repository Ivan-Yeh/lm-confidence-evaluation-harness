"""
Reviewer concern #3: "Concentration is never calibrated, yet FD depends on it
... the output's concentration is simply whatever the retrieved lexicon
expressions happen to impose. The faithfulness claim therefore rests on a
quantity the pipeline does not control."

This script tests, directly, whether the concentration (kappa = alpha + beta
of the fitted Beta distribution) of the *target* calibrated signal
(`calibrated_lc` / `calibrated_tp` / `calibrated_su` in the original
`calibration_details.csv` -- the confidence profile used to pick which hedged
phrasing to retrieve from the lexicon) actually controls the concentration
that independent, disjoint judges perceive when they read the resulting
rewritten text -- rather than the rewritten text's concentration being an
arbitrary byproduct of whichever lexicon entry happened to be retrieved.

Ground truth for "what concentration a reader actually perceives" comes from
the rebuttal/disjoint_llm_eval_{mmlu,truthful_qa} experiments, which already
scored every rewritten response with:
  - 9 independent calls to a disjoint API model (MiniMaxAI/MiniMax-M3), and
  - 10 independent Claude agent personas (diverse background),
and fit a Beta distribution to each judge group's raw scores. Critically we
use the *no-hint* variant of that experiment (results_no_hint for mmlu,
results_v1_no_orig_conf_hint for truthful_qa), where judges were NOT shown
the pipeline's own confidence estimate -- so any relationship we find between
the calibration target's concentration and these judges' concentration cannot
be explained by anchoring on a hint; it has to come from the actual text.

For each of the 50 sampled rows x 2 datasets x 3 signals (lc, tp, su) we have:
  - kappa_target   = alpha + beta of calibrated_{signal}      (the pipeline's
                      intended confidence profile, prior to lexicon retrieval)
  - mu_target      = mean of calibrated_{signal}
  - kappa_new_model, kappa_annotators = alpha + beta of the independent
                      judges' Beta fits over the resulting rewritten text
                      (post lexicon retrieval)

Tests run (this is "does concentration transfer", framed on variance since
that's the operationally meaningful, reviewer-facing quantity):
  1. Spearman + Pearson-on-log(kappa) correlation: kappa_target vs each of
     kappa_new_model / kappa_annotators / kappa_pooled(both judge groups).
  2. OLS regression of independent-judge variance (sigma^2) on kappa_target
     with mu_target and mu_target*(1-mu_target) as covariates -- rules out
     the trivial "extreme means mechanically have lower variance" confound,
     since kappa vs variance both incorporate the mu(1-mu) boundary effect.
  3. Tertile split on kappa_target (Low/Med/High) + Levene's test for equal
     variance of the independent judges' scores across the three bins, with
     mean variance per bin reported to show monotonic direction.
  4. Permutation test: shuffle kappa_target across statements 10,000x and
     rebuild the empirical null distribution of the Spearman correlation
     against kappa_new_model / kappa_annotators, to show the observed
     correlation is far outside what an arbitrary/coincidental assignment
     of concentration would produce.

Run with the `llm` conda env python.
"""
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = "/home/ivan/lm-confidence-evaluation-harness"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

DATASETS = {
    "truthful_qa": dict(
        original_csv=f"/hdd/ivny/direct_qa_in_domain_calibration/truthful_qa/{MODEL}/calibration_details.csv",
        detailed_csv=os.path.join(
            REPO_ROOT, "rebuttal/disjoint_llm_eval_truthful_qa/results_v1_no_orig_conf_hint/detailed_table.csv"
        ),
    ),
    "mmlu": dict(
        original_csv=f"/hdd/ivny/direct_qa_in_domain_calibration/mmlu/{MODEL}/calibration_details.csv",
        detailed_csv=os.path.join(
            REPO_ROOT, "rebuttal/disjoint_llm_eval_mmlu/results_no_hint/detailed_table.csv"
        ),
    ),
}

SIGNALS = ["lc", "tp", "su"]
RNG_SEED = 42
N_PERM = 10000

BETA_RE = re.compile(r"alpha=([-\d.eE]+),\s*beta=([-\d.eE]+),\s*mu=([-\d.eE]+),\s*sigma=([-\d.eE]+)")


def parse_beta(s: str) -> dict:
    m = BETA_RE.search(str(s))
    alpha, beta, mu, sigma = (float(x) for x in m.groups())
    return dict(alpha=alpha, beta=beta, kappa=alpha + beta, mu=mu, sigma=sigma, var=sigma ** 2)


def build_long_table() -> pd.DataFrame:
    rows = []
    for dataset, paths in DATASETS.items():
        orig = pd.read_csv(paths["original_csv"])
        detailed = pd.read_csv(paths["detailed_csv"])

        for _, drow in detailed.iterrows():
            src_idx = int(drow["source_row_index"])
            orig_row = orig.iloc[src_idx]
            for signal in SIGNALS:
                target = parse_beta(orig_row[f"calibrated_{signal}"])
                new_model = parse_beta(drow[f"calibrated_{signal}_rewritten_response_lc_new_model"])
                annotators = parse_beta(drow[f"calibrated_{signal}_rewritten_response_lc_annotators"])
                rows.append(dict(
                    dataset=dataset, source_row_index=src_idx, signal=signal,
                    correctness=drow["correctness"],
                    kappa_target=target["kappa"], mu_target=target["mu"], var_target=target["var"],
                    kappa_new_model=new_model["kappa"], mu_new_model=new_model["mu"], var_new_model=new_model["var"],
                    kappa_annotators=annotators["kappa"], mu_annotators=annotators["mu"], var_annotators=annotators["var"],
                ))
    df = pd.DataFrame(rows)
    # Pooled independent-judge concentration/variance: harmonic-style pooling isn't
    # well-defined for kappa directly, so pool at the variance level (average of the
    # two judge groups' variances) and re-derive an equivalent pooled kappa via the
    # same mu(1-mu)/var - 1 relationship, using the mean of the two judges' mu's.
    df["var_pooled"] = df[["var_new_model", "var_annotators"]].mean(axis=1)
    mu_pooled = df[["mu_new_model", "mu_annotators"]].mean(axis=1)
    df["mu_pooled"] = mu_pooled
    df["kappa_pooled"] = mu_pooled * (1 - mu_pooled) / df["var_pooled"] - 1
    return df


def corr_block(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    n = len(x)
    spear_r, spear_p = stats.spearmanr(x, y)
    plog_r, plog_p = stats.pearsonr(np.log(x), np.log(y))
    return dict(n=n, spearman_r=spear_r, spearman_p=spear_p,
                pearson_log_r=plog_r, pearson_log_p=plog_p)


def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    judge_cols = ["kappa_new_model", "kappa_annotators", "kappa_pooled"]
    out = []
    for judge in judge_cols:
        for dataset, g in df.groupby("dataset"):
            res = corr_block(g["kappa_target"].to_numpy(), g[judge].to_numpy())
            out.append(dict(level="dataset", dataset=dataset, signal="ALL", judge=judge, **res))
        for signal, g in df.groupby("signal"):
            res = corr_block(g["kappa_target"].to_numpy(), g[judge].to_numpy())
            out.append(dict(level="signal", dataset="ALL", signal=signal, judge=judge, **res))
        res = corr_block(df["kappa_target"].to_numpy(), df[judge].to_numpy())
        out.append(dict(level="pooled_overall", dataset="ALL", signal="ALL", judge=judge, **res))
    return pd.DataFrame(out)


def run_regression(df: pd.DataFrame) -> pd.DataFrame:
    """OLS: var_independent ~ log(kappa_target) + mu_target + mu_target*(1-mu_target).

    The mu covariates absorb the mechanical "extreme means have bounded/lower
    variance" effect, isolating whether kappa_target explains independent-judge
    variance *beyond* that boundary effect.
    """
    out = []
    for judge_var in ["var_new_model", "var_annotators", "var_pooled"]:
        y = df[judge_var].to_numpy()
        log_kappa_t = np.log(df["kappa_target"].to_numpy())
        mu_t = df["mu_target"].to_numpy()
        boundary = mu_t * (1 - mu_t)
        X = np.column_stack([np.ones_like(y), log_kappa_t, mu_t, boundary])
        beta_hat, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
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
        names = ["intercept", "log_kappa_target", "mu_target", "mu_target*(1-mu_target)"]
        for name, coef, se_i, t_i, p_i in zip(names, beta_hat, se, t_stats, p_vals):
            out.append(dict(outcome=judge_var, term=name, coef=coef, se=se_i, t=t_i, p=p_i, n=n, r2=r2))
    return pd.DataFrame(out)


def run_levene(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    tertiles = pd.qcut(df["kappa_target"], 3, labels=["low", "mid", "high"])
    df = df.assign(kappa_target_bin=tertiles)
    for judge_var in ["var_new_model", "var_annotators", "var_pooled"]:
        groups = [g[judge_var].to_numpy() for _, g in df.groupby("kappa_target_bin", observed=True)]
        labels = [lbl for lbl, _ in df.groupby("kappa_target_bin", observed=True)]
        stat, p = stats.levene(*groups, center="median")
        means = {f"mean_{lbl}": np.mean(g) for lbl, g in zip(labels, groups)}
        ns = {f"n_{lbl}": len(g) for lbl, g in zip(labels, groups)}
        out.append(dict(judge=judge_var, levene_stat=stat, levene_p=p, **means, **ns))
    return pd.DataFrame(out)


def run_permutation(df: pd.DataFrame, n_perm: int = N_PERM, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    kappa_target = df["kappa_target"].to_numpy()
    for judge in ["kappa_new_model", "kappa_annotators", "kappa_pooled"]:
        y = df[judge].to_numpy()
        mask = np.isfinite(kappa_target) & np.isfinite(y) & (kappa_target > 0) & (y > 0)
        x_obs, y_obs = kappa_target[mask], y[mask]
        observed_r, _ = stats.spearmanr(x_obs, y_obs)
        null_r = np.empty(n_perm)
        for i in range(n_perm):
            x_shuf = rng.permutation(x_obs)
            null_r[i], _ = stats.spearmanr(x_shuf, y_obs)
        p_perm = (np.sum(np.abs(null_r) >= np.abs(observed_r)) + 1) / (n_perm + 1)
        out.append(dict(
            judge=judge, n=len(x_obs), observed_spearman_r=observed_r,
            null_mean=null_r.mean(), null_std=null_r.std(),
            perm_p_two_sided=p_perm,
        ))
    return pd.DataFrame(out)


def main():
    df = build_long_table()
    df.to_csv(os.path.join(OUT_DIR, "concentration_control_data.csv"), index=False)
    print(f"Built long table: {df.shape}")

    corr_df = run_correlations(df)
    corr_df.to_csv(os.path.join(OUT_DIR, "concentration_control_correlations.csv"), index=False)
    print("\n=== Correlations (kappa_target vs independent-judge kappa) ===")
    print(corr_df.to_string(index=False))

    reg_df = run_regression(df)
    reg_df.to_csv(os.path.join(OUT_DIR, "concentration_control_regression.csv"), index=False)
    print("\n=== OLS: independent-judge variance ~ log(kappa_target) + mu covariates ===")
    print(reg_df.to_string(index=False))

    levene_df = run_levene(df)
    levene_df.to_csv(os.path.join(OUT_DIR, "concentration_control_levene.csv"), index=False)
    print("\n=== Levene's test across kappa_target tertiles ===")
    print(levene_df.to_string(index=False))

    perm_df = run_permutation(df)
    perm_df.to_csv(os.path.join(OUT_DIR, "concentration_control_permutation.csv"), index=False)
    print("\n=== Permutation test (10,000 shuffles of kappa_target) ===")
    print(perm_df.to_string(index=False))


if __name__ == "__main__":
    main()
