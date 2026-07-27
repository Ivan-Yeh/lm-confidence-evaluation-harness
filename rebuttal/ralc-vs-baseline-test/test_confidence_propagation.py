"""
Significance test for the "Confidence Cross-Space Correlation" panel of
fig_calibration_performance_baseline_comparison (RALC rho vs Direct
Beta-Guided rho, per signal lc/tp/su), addressing the reviewer's request for
a significance test on RALC-vs-baseline deltas.

Reproduces the exact x/y pairing used by
paper_materials/fig_signal_linguistic_confidence_propagation.ipynb (RALC) and
fig_signal_beta_linguistic_confidence_propagation.ipynb (Direct Beta-Guided):
for each signal, x = mean of the `calibrated_{signal}` Beta distribution
(the target confidence signal), y = mean of the re-extracted linguistic
confidence after rewriting the response to hedge accordingly -- either via
RALC (`calibrated_{signal}_rewritten_lc`) or Direct Beta-Guided
(`calibrated_{signal}_beta_guided_lc`). x is identical between the two
notebooks (same source column, same row order), so for every example RALC's
y and Beta-Guided's y are directly paired against the same x -- i.e. RALC and
Beta-Guided rho's are *dependent* (overlapping) correlations computed on the
same sample, which is exactly what a paired bootstrap test needs (no
independence assumption required, unlike an unpaired test).

No model exclusion here, matching the two source notebooks (they do not
filter out gpt-oss-120b / Qwen3-235B).

Two levels of evidence are reported:
  1. Cell-level: one Spearman rho per (dataset, model, signal) (n=21 cells
     per signal / 63 pooled) -- paired t-test + Wilcoxon signed-rank test on
     rho_RALC - rho_BetaGuided across cells.
  2. Pooled-per-signal: a single Spearman rho computed on all rows for that
     signal (huge N, matching the number the figure's bar is drawn from) --
     paired bootstrap (10,000 resamples of row indices, applied jointly to
     x/y_RALC/y_beta to preserve pairing) giving a 95% CI and one-sided
     p-value for rho_RALC - rho_BetaGuided > 0.

Run with the `llm` conda env python.
"""
import os
import glob
import re

import numpy as np
import pandas as pd
from scipy import stats

DIRECT_ROOT = "/hdd/ivny/direct_qa_in_domain_calibration"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNALS = ["lc", "tp", "su"]
N_BOOT = 2000
RNG = np.random.default_rng(42)

MU_RE = re.compile(r"mu=([0-9.eE+-]+)")


def extract_mu(s):
    m = MU_RE.search(str(s))
    return float(m.group(1)) if m else np.nan


def find_leaves():
    records = []
    for path in sorted(glob.glob(os.path.join(DIRECT_ROOT, "*", "*", "*", "calibration_details.csv"))):
        rel = os.path.relpath(path, DIRECT_ROOT)
        dataset, org, model = rel.split(os.sep)[:3]
        records.append((dataset, f"{org}/{model}", path))
    return records


def load_all():
    """Returns dict: signal -> DataFrame[dataset, model, x (signal mu), y_ralc, y_beta]."""
    leaves = find_leaves()
    per_signal = {sig: [] for sig in SIGNALS}
    for dataset, model, path in leaves:
        df = pd.read_csv(path)
        for sig in SIGNALS:
            x = df[f"calibrated_{sig}"].apply(extract_mu).to_numpy()
            y_ralc = df[f"calibrated_{sig}_rewritten_lc"].apply(extract_mu).to_numpy()
            y_beta = df[f"calibrated_{sig}_beta_guided_lc"].apply(extract_mu).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y_ralc) & np.isfinite(y_beta)
            per_signal[sig].append(pd.DataFrame({
                "dataset": dataset, "model": model,
                "x": x[mask], "y_ralc": y_ralc[mask], "y_beta": y_beta[mask],
            }))
    return {sig: pd.concat(v, ignore_index=True) for sig, v in per_signal.items()}


def paired_bootstrap_rho_delta(x, y_ralc, y_beta, n_boot=N_BOOT):
    n = len(x)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        xs, y1s, y2s = x[idx], y_ralc[idx], y_beta[idx]
        rho_ralc = stats.spearmanr(xs, y1s).correlation
        rho_beta = stats.spearmanr(xs, y2s).correlation
        deltas[b] = rho_ralc - rho_beta
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_one = float(np.mean(deltas <= 0))  # H1: rho_RALC > rho_BetaGuided
    return float(lo), float(hi), p_one, deltas


def main():
    data = load_all()

    # --- Cell-level: one rho per (dataset, model, signal) ---
    cell_rows = []
    for sig, df in data.items():
        for (dataset, model), g in df.groupby(["dataset", "model"]):
            rho_ralc = stats.spearmanr(g["x"], g["y_ralc"]).correlation
            rho_beta = stats.spearmanr(g["x"], g["y_beta"]).correlation
            cell_rows.append(dict(signal=sig, dataset=dataset, model=model,
                                   n=len(g), rho_ralc=rho_ralc, rho_beta_guided=rho_beta,
                                   delta=rho_ralc - rho_beta))
    cell_df = pd.DataFrame(cell_rows)
    cell_df.to_csv(os.path.join(OUT_DIR, "confidence_propagation_cell_level.csv"), index=False)

    print("=== Cell-level (per dataset x model) rho, RALC vs Direct Beta-Guided ===")
    cell_results = []
    for sig in SIGNALS + ["POOLED"]:
        sub = cell_df if sig == "POOLED" else cell_df[cell_df["signal"] == sig]
        t_stat, t_p_two = stats.ttest_rel(sub["rho_ralc"], sub["rho_beta_guided"])
        t_p_one = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
        try:
            w_stat, w_p_one = stats.wilcoxon(sub["rho_ralc"], sub["rho_beta_guided"], alternative="greater")
        except ValueError:
            w_stat, w_p_one = np.nan, np.nan
        row = dict(signal=sig, n_cells=len(sub),
                   mean_rho_ralc=sub["rho_ralc"].mean(), mean_rho_beta=sub["rho_beta_guided"].mean(),
                   mean_delta=sub["delta"].mean(),
                   pct_cells_ralc_higher=float(np.mean(sub["delta"] > 0)) * 100,
                   paired_t_stat=t_stat, paired_t_p_onesided=t_p_one,
                   wilcoxon_stat=w_stat, wilcoxon_p_onesided=w_p_one)
        cell_results.append(row)
        print(f"  signal={sig:8s} n_cells={row['n_cells']:3d} "
              f"rho_RALC={row['mean_rho_ralc']:.3f} rho_Beta={row['mean_rho_beta']:.3f} "
              f"delta={row['mean_delta']:+.3f} ({row['pct_cells_ralc_higher']:.0f}% cells favor RALC) "
              f"paired-t p={t_p_one:.2e} wilcoxon p={w_p_one:.2e}")
    cell_results_df = pd.DataFrame(cell_results)
    cell_results_df.to_csv(os.path.join(OUT_DIR, "confidence_propagation_cell_significance.csv"), index=False)

    # --- Pooled-per-signal: full-N rho + paired bootstrap ---
    print("\n=== Pooled-per-signal rho (full N) + paired bootstrap on rho_RALC - rho_Beta ===")
    pooled_results = []
    for sig, df in data.items():
        x = df["x"].to_numpy()
        y_ralc = df["y_ralc"].to_numpy()
        y_beta = df["y_beta"].to_numpy()
        rho_ralc = stats.spearmanr(x, y_ralc).correlation
        rho_beta = stats.spearmanr(x, y_beta).correlation
        lo, hi, p_one, _ = paired_bootstrap_rho_delta(x, y_ralc, y_beta)
        row = dict(signal=sig, n=len(x), rho_ralc=rho_ralc, rho_beta_guided=rho_beta,
                   delta=rho_ralc - rho_beta, boot_ci_low=lo, boot_ci_high=hi,
                   boot_p_onesided=p_one)
        pooled_results.append(row)
        print(f"  signal={sig:4s} n={row['n']:7d} rho_RALC={rho_ralc:.4f} rho_Beta={rho_beta:.4f} "
              f"delta={row['delta']:+.4f}  bootstrap 95% CI=[{lo:.4f}, {hi:.4f}]  p={p_one:.4f}")
    pooled_df = pd.DataFrame(pooled_results)
    pooled_df.to_csv(os.path.join(OUT_DIR, "confidence_propagation_pooled_significance.csv"), index=False)

    print("\nSaved: confidence_propagation_cell_level.csv, confidence_propagation_cell_significance.csv, "
          "confidence_propagation_pooled_significance.csv")


if __name__ == "__main__":
    main()
