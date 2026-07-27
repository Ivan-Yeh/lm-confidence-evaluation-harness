"""
Fidelity (fd) test: does the concentration (kappa = alpha + beta of the fitted
Beta distribution) of a calibrated confidence signal (lc, tp, su) survive the
rewrite-to-text-then-re-extract-lc round trip?

For every entry in each model/dataset's `calibration_details.csv`, each of the
three confidence signals (lc, tp, su) is stored as a Beta distribution string
(e.g. "BetaDistribution(alpha=..., beta=..., mu=..., sigma=...)") both in its
calibrated form (`calibrated_{signal}`) and after the calibrated response was
rewritten to hedge accordingly and re-scored with the linguistic-confidence
extractor (`calibrated_{signal}_rewritten_lc`).

kappa = alpha + beta is the standard Beta concentration parameter (higher
kappa => tighter/more peaked distribution => the model is more "certain").
This script extracts kappa for both spaces and tests whether they correlate,
per (dataset, model, signal), pooled per (dataset, signal), pooled per
(signal) across all models/datasets, and pooled overall.

Outputs (all written next to this script):
  - fd_test_kappa_long.csv       raw per-entry kappa pairs (long format)
  - fd_test_results_per_model.csv correlation test results per dataset/model/signal
  - fd_test_results_pooled.csv    correlation test results pooled at various levels
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

BETA_RE = re.compile(r"alpha=([-\d.eE]+),\s*beta=([-\d.eE]+)")


def kappa_from_beta_str(series: pd.Series) -> np.ndarray:
    """Parse 'BetaDistribution(alpha=..., beta=..., ...)' strings and return alpha+beta."""
    out = np.full(len(series), np.nan)
    for i, v in enumerate(series):
        m = BETA_RE.search(str(v))
        if m:
            alpha, beta = float(m.group(1)), float(m.group(2))
            out[i] = alpha + beta
    return out


def find_files():
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, "*", "*", "*", "calibration_details.csv")))
    records = []
    for p in paths:
        rel = os.path.relpath(p, DATA_ROOT)
        parts = rel.split(os.sep)
        dataset = parts[0]
        model = "/".join(parts[1:-1])  # org/model-name
        records.append((dataset, model, p))
    return records


def corr_test(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return dict(n=n, pearson_r=np.nan, pearson_p=np.nan,
                     spearman_r=np.nan, spearman_p=np.nan,
                     pearson_log_r=np.nan, pearson_log_p=np.nan)
    pear_r, pear_p = stats.pearsonr(x, y)
    spear_r, spear_p = stats.spearmanr(x, y)
    # kappa is a concentration parameter and typically heavy-tailed / log-scaled,
    # so also report Pearson correlation in log-space as a robustness check.
    xl, yl = np.log(x), np.log(y)
    logmask = np.isfinite(xl) & np.isfinite(yl)
    if logmask.sum() >= 3:
        plog_r, plog_p = stats.pearsonr(xl[logmask], yl[logmask])
    else:
        plog_r, plog_p = np.nan, np.nan
    return dict(n=n, pearson_r=pear_r, pearson_p=pear_p,
                spearman_r=spear_r, spearman_p=spear_p,
                pearson_log_r=plog_r, pearson_log_p=plog_p)


def main():
    files = find_files()
    long_rows = []

    for dataset, model, path in files:
        df = pd.read_csv(path)
        for signal in SIGNALS:
            cal_col = f"calibrated_{signal}"
            rew_col = f"calibrated_{signal}_rewritten_lc"
            kappa_cal = kappa_from_beta_str(df[cal_col])
            kappa_rew = kappa_from_beta_str(df[rew_col])
            for idx, (kc, kr) in enumerate(zip(kappa_cal, kappa_rew)):
                long_rows.append(dict(
                    dataset=dataset, model=model, signal=signal, row=idx,
                    kappa_calibrated=kc, kappa_rewritten_lc=kr,
                ))

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(os.path.join(OUT_DIR, "fd_test_kappa_long.csv"), index=False)

    # --- per (dataset, model, signal) ---
    per_model_rows = []
    for (dataset, model, signal), g in long_df.groupby(["dataset", "model", "signal"]):
        res = corr_test(g["kappa_calibrated"].to_numpy(), g["kappa_rewritten_lc"].to_numpy())
        per_model_rows.append(dict(dataset=dataset, model=model, signal=signal, **res))
    per_model_df = pd.DataFrame(per_model_rows).sort_values(["dataset", "signal", "model"])
    per_model_df.to_csv(os.path.join(OUT_DIR, "fd_test_results_per_model.csv"), index=False)

    # --- pooled ---
    pooled_rows = []

    # pooled per (dataset, signal) across models
    for (dataset, signal), g in long_df.groupby(["dataset", "signal"]):
        res = corr_test(g["kappa_calibrated"].to_numpy(), g["kappa_rewritten_lc"].to_numpy())
        pooled_rows.append(dict(level="dataset+signal", dataset=dataset, model="ALL",
                                 signal=signal, **res))

    # pooled per signal across everything
    for signal, g in long_df.groupby("signal"):
        res = corr_test(g["kappa_calibrated"].to_numpy(), g["kappa_rewritten_lc"].to_numpy())
        pooled_rows.append(dict(level="signal_overall", dataset="ALL", model="ALL",
                                 signal=signal, **res))

    # pooled per dataset across signals/models
    for dataset, g in long_df.groupby("dataset"):
        res = corr_test(g["kappa_calibrated"].to_numpy(), g["kappa_rewritten_lc"].to_numpy())
        pooled_rows.append(dict(level="dataset_overall", dataset=dataset, model="ALL",
                                 signal="ALL", **res))

    # grand overall
    res = corr_test(long_df["kappa_calibrated"].to_numpy(), long_df["kappa_rewritten_lc"].to_numpy())
    pooled_rows.append(dict(level="grand_overall", dataset="ALL", model="ALL", signal="ALL", **res))

    pooled_df = pd.DataFrame(pooled_rows)
    cols = ["level", "dataset", "model", "signal", "n", "pearson_r", "pearson_p",
            "spearman_r", "spearman_p", "pearson_log_r", "pearson_log_p"]
    pooled_df = pooled_df[cols]
    pooled_df.to_csv(os.path.join(OUT_DIR, "fd_test_results_pooled.csv"), index=False)

    print("Per-model results:")
    print(per_model_df.to_string(index=False))
    print("\nPooled results:")
    print(pooled_df.to_string(index=False))


if __name__ == "__main__":
    main()
