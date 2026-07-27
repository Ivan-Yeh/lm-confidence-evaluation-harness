"""
Experiment 2: counterfactual decomposition of the FD improvement into a
mean-attributable share and a concentration-attributable share.

Reviewer concern #3: FD is driven by concentration (kappa), yet concentration
is never a calibration target -- Section 4.1 preserves it unchanged because
there is "no natural target." If the FD improvement RALC reports were mostly
a concentration artifact of the rewrite, the headline metric would be moving
for reasons the pipeline doesn't actually control.

For every row in `calibration_details.csv` (signal in lc/tp/su, using the
`rewritten` pathway -- LLM rewrite conditioned on retrieved lexicon hedge
words, the exact mechanism under review, i.e. `calibrated_{signal}_rewritten_lc`)
we have three Beta distributions and a correctness label y:

  S   = original_{signal}          -> (alpha, beta) = (a,  b ), kappa = a+b
  S'  = calibrated_{signal}        -> (alpha, beta) = (a', b'), kappa' = a'+b'
  S'' = calibrated_{signal}_rewritten_lc -> (alpha, beta) = (a'',b''), kappa'' = a''+b''

By construction, RALC's Platt-scaling calibration step only moves the mean
and explicitly preserves concentration (Sec 4.1), so kappa' == kappa (we
verify this empirically below -- confirmed to float precision on every
signal). That means the pre-rewrite concentration the reviewer is worried
about has a single, unambiguous value: kappa (from S, equivalently S').

Three FD numbers per row, all using RALC's own faithfulness_divergence
formula (kl_beta / faithfulness_divergence_single, copied verbatim from
lm_conf/post_processing/metrics.py so results are bit-for-bit consistent
with what's reported elsewhere in the paper):

  FD_original       = FD(a,  b,  y)                    -- baseline, pre-calibration, pre-rewrite
  FD_counterfactual = FD(a_cf, b_cf, y) where mu_cf = mu(S''), kappa_cf = kappa(S)
                       i.e. the mean actually achieved by the rewrite, but
                       concentration FROZEN at its original pre-rewrite value
  FD_actual         = FD(a'', b'', y)                   -- fully realized post-rewrite distribution

total_gain      = FD_original - FD_actual         (what RALC currently reports)
mean_gain       = FD_original - FD_counterfactual (gain attributable to the mean shift alone,
                                                     with concentration held fixed -- the part
                                                     that's unambiguously causal/intentional)
ratio_mean_explained = mean_gain / total_gain     (share of the total reported FD improvement
                                                     that survives when concentration doesn't move)

kappa (especially for lc/tp) is heavy-tailed -- a handful of rows have
near-unanimous ensemble votes (sigma~0) and blow up to kappa in the
millions, which can dominate a plain mean. We report the plain mean (matches
how faithfulness_divergence() is actually aggregated in the codebase, i.e.
what's "currently reported") AND a median / trimmed-mean version so a few
pathological rows can't be mistaken for the typical effect.

Run with the `llm` conda env python.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.special import betaln, psi

DATA_ROOT = "/hdd/ivny/direct_qa_in_domain_calibration"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS = ["lc", "tp", "su"]
TRIM_PCTL = 99  # winsorize rows with kappa_orig above this percentile (within group) for the trimmed stat

BETA_RE = re.compile(r"alpha=([-\d.eE]+),\s*beta=([-\d.eE]+)")


def parse_ab(s):
    m = BETA_RE.search(str(s))
    if m is None:
        return None, None
    return float(m.group(1)), float(m.group(2))


def kl_beta(a_post, b_post, a_prior, b_prior):
    return (
        betaln(a_prior, b_prior) - betaln(a_post, b_post)
        + (a_post - a_prior) * psi(a_post)
        + (b_post - b_prior) * psi(b_post)
        - (a_post + b_post - a_prior - b_prior) * psi(a_post + b_post)
    )


def fd_single(a, b, y):
    a_post = a + y
    b_post = b + (1 - y)
    return max(0.0, (a + b + 1e-8) * kl_beta(a_post, b_post, a, b))


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
    kappa_sanity = []  # (dataset, model, signal, mean_abs_reldiff kappa_orig vs kappa_cal)
    for dataset, model, path in find_files():
        df = pd.read_csv(path)
        if "accuracy" not in df.columns:
            continue
        y_all = pd.to_numeric(df["accuracy"], errors="coerce")
        for signal in SIGNALS:
            orig_col = f"original_{signal}"
            cal_col = f"calibrated_{signal}"
            rew_col = f"calibrated_{signal}_rewritten_lc"
            if not all(c in df.columns for c in [orig_col, cal_col, rew_col]):
                continue

            ab_orig = df[orig_col].map(parse_ab)
            ab_cal = df[cal_col].map(parse_ab)
            ab_actual = df[rew_col].map(parse_ab)

            kappa_orig_arr = np.array([a + b if a is not None else np.nan for a, b in ab_orig])
            kappa_cal_arr = np.array([a + b if a is not None else np.nan for a, b in ab_cal])
            valid_k = np.isfinite(kappa_orig_arr) & np.isfinite(kappa_cal_arr) & (kappa_orig_arr > 0)
            reldiff = np.abs(kappa_orig_arr[valid_k] - kappa_cal_arr[valid_k]) / kappa_orig_arr[valid_k]
            kappa_sanity.append(dict(dataset=dataset, model=model, signal=signal,
                                      n=int(valid_k.sum()),
                                      mean_abs_reldiff_kappa_orig_vs_cal=float(reldiff.mean()) if len(reldiff) else np.nan,
                                      max_abs_reldiff_kappa_orig_vs_cal=float(reldiff.max()) if len(reldiff) else np.nan))

            for idx in range(len(df)):
                y = y_all.iloc[idx]
                a_o, b_o = ab_orig.iloc[idx]
                a_a, b_a = ab_actual.iloc[idx]
                if a_o is None or a_a is None or not np.isfinite(y) or y < 0 or y > 1:
                    continue
                kappa_orig = a_o + b_o
                kappa_actual = a_a + b_a
                mu_actual = a_a / kappa_actual
                a_cf = mu_actual * kappa_orig
                b_cf = (1 - mu_actual) * kappa_orig

                fd_o = fd_single(a_o, b_o, y)
                fd_cf = fd_single(a_cf, b_cf, y)
                fd_a = fd_single(a_a, b_a, y)

                rows.append(dict(
                    dataset=dataset, model=model, signal=signal, row=idx, y=y,
                    kappa_orig=kappa_orig, mu_orig=a_o / kappa_orig,
                    kappa_actual=kappa_actual, mu_actual=mu_actual,
                    FD_original=fd_o, FD_counterfactual=fd_cf, FD_actual=fd_a,
                ))
    return pd.DataFrame(rows), pd.DataFrame(kappa_sanity)


def trimmed_mean_by_kappa(g: pd.DataFrame, col: str, pctl: float) -> float:
    cutoff = np.percentile(g["kappa_orig"], pctl)
    return g.loc[g["kappa_orig"] <= cutoff, col].mean()


def summarize(g: pd.DataFrame) -> dict:
    # Primary definition (per user): share = sum|FD_o - FD_cf| / sum|FD_o - FD_a|, i.e. pool the
    # ABSOLUTE per-row magnitude of the mean-only change over the absolute per-row magnitude of the
    # total change. Using |.| means a row where concentration helped and a row where it hurt don't
    # cancel each other out in the pool (unlike the signed sum-ratio, which produced uninterpretable
    # values like 193% when helping/hurting rows offset), and the result is always bounded in [0, 1].
    gain_mean_abs = (g["FD_original"] - g["FD_counterfactual"]).abs().sum()
    gain_total_abs = (g["FD_original"] - g["FD_actual"]).abs().sum()
    share_abs_magnitude = gain_mean_abs / gain_total_abs if gain_total_abs > 1e-12 else np.nan

    fd_o_mean, fd_cf_mean, fd_a_mean = g["FD_original"].mean(), g["FD_counterfactual"].mean(), g["FD_actual"].mean()
    fd_o_med, fd_cf_med, fd_a_med = g["FD_original"].median(), g["FD_counterfactual"].median(), g["FD_actual"].median()
    fd_o_tr = trimmed_mean_by_kappa(g, "FD_original", TRIM_PCTL)
    fd_cf_tr = trimmed_mean_by_kappa(g, "FD_counterfactual", TRIM_PCTL)
    fd_a_tr = trimmed_mean_by_kappa(g, "FD_actual", TRIM_PCTL)

    def ratio(o, cf, a):
        total = o - a
        meang = o - cf
        return meang / total if abs(total) > 1e-12 else np.nan, total, meang

    r_mean, total_mean, meang_mean = ratio(fd_o_mean, fd_cf_mean, fd_a_mean)
    r_med, total_med, meang_med = ratio(fd_o_med, fd_cf_med, fd_a_med)
    r_tr, total_tr, meang_tr = ratio(fd_o_tr, fd_cf_tr, fd_a_tr)

    n_outlier = int((g["kappa_orig"] > np.percentile(g["kappa_orig"], TRIM_PCTL)).sum())

    return dict(
        n=len(g), n_outlier_trimmed=n_outlier,
        share_abs_magnitude=share_abs_magnitude,
        gain_mean_abs_sum=gain_mean_abs, gain_total_abs_sum=gain_total_abs,
        FD_original_mean=fd_o_mean, FD_counterfactual_mean=fd_cf_mean, FD_actual_mean=fd_a_mean,
        total_gain_mean=total_mean, mean_gain_mean=meang_mean, ratio_mean_explained_mean=r_mean,
        FD_original_median=fd_o_med, FD_counterfactual_median=fd_cf_med, FD_actual_median=fd_a_med,
        total_gain_median=total_med, mean_gain_median=meang_med, ratio_mean_explained_median=r_med,
        FD_original_trimmed=fd_o_tr, FD_counterfactual_trimmed=fd_cf_tr, FD_actual_trimmed=fd_a_tr,
        total_gain_trimmed=total_tr, mean_gain_trimmed=meang_tr, ratio_mean_explained_trimmed=r_tr,
    )


def main():
    long_df, sanity_df = build_long_table()
    print(f"Built long table: {long_df.shape}")
    long_df.to_csv(os.path.join(OUT_DIR, "counterfactual_decomposition_data.csv"), index=False)
    sanity_df.to_csv(os.path.join(OUT_DIR, "counterfactual_decomposition_kappa_sanity.csv"), index=False)
    print("\n=== Sanity check: kappa' == kappa (calibration preserves concentration) ===")
    print(sanity_df.to_string(index=False))

    rows = []
    for (dataset, model, signal), g in long_df.groupby(["dataset", "model", "signal"]):
        rows.append(dict(level="dataset+model+signal", dataset=dataset, model=model, signal=signal, **summarize(g)))
    for (dataset, signal), g in long_df.groupby(["dataset", "signal"]):
        rows.append(dict(level="dataset+signal", dataset=dataset, model="ALL", signal=signal, **summarize(g)))
    for signal, g in long_df.groupby("signal"):
        rows.append(dict(level="signal_overall", dataset="ALL", model="ALL", signal=signal, **summarize(g)))
    for dataset, g in long_df.groupby("dataset"):
        rows.append(dict(level="dataset_overall", dataset=dataset, model="ALL", signal="ALL", **summarize(g)))
    rows.append(dict(level="grand_overall", dataset="ALL", model="ALL", signal="ALL", **summarize(long_df)))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "counterfactual_decomposition_summary.csv"), index=False)

    print("\n=== Decomposition summary (share_abs_magnitude is the primary/headline number) ===")
    display_cols = ["level", "dataset", "model", "signal", "n", "share_abs_magnitude",
                     "ratio_mean_explained_mean", "ratio_mean_explained_median", "ratio_mean_explained_trimmed",
                     "total_gain_mean", "mean_gain_mean"]
    print(summary_df[display_cols].to_string(index=False))

    # --- Magnitude-weighted (pooled-sum) aggregation across the 63 cells. ---
    # IMPORTANT: averaging/medianing the 63 per-cell ratios directly (as an earlier version of
    # this script did) is an INSTANCE-weighted statistic -- a cell with tiny/near-zero total_gain
    # (an unstable ratio, e.g. dividing by ~0) gets the same vote as a cell representing a huge,
    # reliable FD swing. The correct, magnitude-weighted combination pools the raw (mean_gain, total_gain)
    # sums across cells BEFORE dividing -- i.e. weight each cell by how much FD improvement (n * per-row
    # gain) it actually represents. This is mathematically identical to pooling all underlying rows
    # directly (see the "signal_overall"/"grand_overall" rows above, which already do this correctly).
    cell_df = summary_df[summary_df.level == "dataset+model+signal"].copy()
    cell_df["total_gain_sum"] = cell_df["total_gain_mean"] * cell_df["n"]
    cell_df["mean_gain_sum"] = cell_df["mean_gain_mean"] * cell_df["n"]
    cell_df.to_csv(os.path.join(OUT_DIR, "counterfactual_decomposition_cells.csv"), index=False)

    naive_unweighted_mean_signed_ratio = cell_df["ratio_mean_explained_mean"].mean()
    naive_unweighted_median_signed_ratio = cell_df["ratio_mean_explained_mean"].median()
    naive_unweighted_mean_abs_share = cell_df["share_abs_magnitude"].mean()

    # Correct pooling for the abs-magnitude share: sum the (already per-row-absolute) numerator/denominator
    # across cells, THEN divide -- weighting each cell by how much |FD change| it represents, not by 1 vote/cell.
    weighted_abs_share_all = cell_df["gain_mean_abs_sum"].sum() / cell_df["gain_total_abs_sum"].sum()
    per_signal_abs_share = {}
    for signal, g in cell_df.groupby("signal"):
        per_signal_abs_share[signal] = g["gain_mean_abs_sum"].sum() / g["gain_total_abs_sum"].sum()

    macro = dict(
        n_cells=len(cell_df),
        naive_unweighted_mean_signed_ratio=naive_unweighted_mean_signed_ratio,
        naive_unweighted_median_signed_ratio=naive_unweighted_median_signed_ratio,
        naive_unweighted_mean_of_cell_abs_shares=naive_unweighted_mean_abs_share,
        magnitude_weighted_abs_share_ALL=weighted_abs_share_all,
        magnitude_weighted_abs_share_lc=per_signal_abs_share.get("lc"),
        magnitude_weighted_abs_share_tp=per_signal_abs_share.get("tp"),
        magnitude_weighted_abs_share_su=per_signal_abs_share.get("su"),
        frac_cells_total_gain_positive=float((cell_df["total_gain_sum"] > 0).mean()),
        n_cells_total_gain_negative=int((cell_df["total_gain_sum"] < 0).sum()),
    )
    macro_df = pd.DataFrame([macro])
    macro_df.to_csv(os.path.join(OUT_DIR, "counterfactual_decomposition_macro.csv"), index=False)
    print("\n=== Headline: magnitude_weighted_abs_share (pooled |gain|, weighted by cell size) ===")
    print(macro_df.to_string(index=False))

    print("\n=== lc cells sorted by share_abs_magnitude (identifies the weak-spot models) ===")
    print(cell_df[cell_df.signal == "lc"][["dataset", "model", "n", "share_abs_magnitude"]]
          .sort_values("share_abs_magnitude").to_string(index=False))

    print("\n=== Cells with total_gain_sum < 0 (FD got worse post-rewrite -- real failure cells) ===")
    print(cell_df[cell_df["total_gain_sum"] < 0][["dataset", "model", "signal", "n", "total_gain_sum", "mean_gain_sum", "share_abs_magnitude"]]
          .sort_values("total_gain_sum").to_string(index=False))


if __name__ == "__main__":
    main()
