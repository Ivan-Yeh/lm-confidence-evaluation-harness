"""
Significance tests for the reviewer question: "no significance test is reported
on the RALC-vs-baseline deltas" (Fig: fig_calibration_performance_baseline_comparison).

Tests whether RALC (calibrated_{lc,tp,su}_rewritten_lc) has significantly lower
generalised_ECE / faithfulness_divergence than:
  (a) Original Direct QA (original_lc)
  (b) Direct Beta-Guided  (calibrated_{lc,tp,su}_beta_guided_lc)
  (c) Hedged QA           (results/<dataset>/hedged_qa_unified_lc/.../eval_metrics.csv)

Unit of analysis for (a)/(b): one row per (dataset, model, signal in {lc,tp,su}),
using the exact scalar metrics already written to each leaf's
calibration_performance.csv by calibration/in_domain_calibration.py -- no
metric re-implementation, so these are exactly the numbers the figure's bars
are averages of.

Unit of analysis for (c): one row per (dataset, model), RALC value = mean over
the 3 signals, since Hedged QA is a single pipeline (not signal-specific).

Model/dataset filter matches the figure notebook exactly: gpt-oss-120b and
Qwen3-235B-A22B-Instruct-2507-tput are excluded from the direct_qa /
hedged_qa panels (they only appear in the separate "Beta" section of the
notebook, not in the main comparison figure).

For each comparison we report:
  - paired t-test (one-sided: RALC < baseline)
  - Wilcoxon signed-rank test (one-sided: RALC < baseline)
  - mean/median delta, paired Cohen's d
  - 10,000-resample paired bootstrap 95% CI on the mean delta (resampling the
    (dataset, model[, signal]) cells with replacement)

Run with the `llm` conda env python.
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy import stats

DIRECT_ROOT = "/hdd/ivny/direct_qa_in_domain_calibration"
RESULTS_ROOT = "/hdd/ivny/results"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_MODELS = {"gpt-oss-120b", "qwen3-235b-a22b-instruct-2507-tput"}
SIGNALS = ["lc", "tp", "su"]
METRICS = ["generalised_ECE", "faithfulness_divergence"]
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def load_calibration_performance():
    rows = []
    for path in sorted(glob.glob(os.path.join(DIRECT_ROOT, "*", "*", "*", "calibration_performance.csv"))):
        rel = os.path.relpath(path, DIRECT_ROOT)
        dataset, org, model = rel.split(os.sep)[:3]
        if model.lower() in EXCLUDE_MODELS:
            continue
        perf = pd.read_csv(path, index_col=0)["value"].to_dict()
        rows.append(dict(dataset=dataset, model=f"{org}/{model}", **perf))
    return pd.DataFrame(rows)


def get_latest_subdir(path):
    subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


def load_hedged_qa():
    rows = []
    for dataset in ["mmlu", "squadv2", "truthful_qa"]:
        base = os.path.join(RESULTS_ROOT, dataset, "hedged_qa_unified_lc")
        if not os.path.isdir(base):
            continue
        for org in os.listdir(base):
            org_path = os.path.join(base, org)
            if not os.path.isdir(org_path):
                continue
            for model in os.listdir(org_path):
                if model.lower() in EXCLUDE_MODELS:
                    continue
                model_path = os.path.join(org_path, model)
                latest = get_latest_subdir(model_path)
                if latest is None:
                    continue
                csv_path = os.path.join(latest, "eval_metrics.csv")
                if not os.path.exists(csv_path):
                    continue
                row = pd.read_csv(csv_path).iloc[0]
                rows.append(dict(
                    dataset=dataset, model=f"{org}/{model}",
                    generalised_ECE=row["generalised_ece"],
                    faithfulness_divergence=row["faithfulness_divergence"],
                ))
    return pd.DataFrame(rows)


def build_cell_table(perf_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, signal): RALC vs Original vs Beta-Guided."""
    rows = []
    for _, r in perf_df.iterrows():
        for sig in SIGNALS:
            row = dict(dataset=r["dataset"], model=r["model"], signal=sig)
            for metric in METRICS:
                row[f"original_{metric}"] = r.get(f"original_lc_{metric}")
                row[f"ralc_{metric}"] = r.get(f"calibrated_{sig}_rewritten_lc_{metric}")
                row[f"beta_guided_{metric}"] = r.get(f"calibrated_{sig}_beta_guided_lc_{metric}")
            rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_ci(delta: np.ndarray, n_boot=N_BOOT):
    n = len(delta)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    # one-sided bootstrap p-value for H1: mean delta < 0 (RALC lower/better)
    p_boot = float(np.mean(boot_means >= 0))
    return float(lo), float(hi), p_boot


def run_comparison(ralc: np.ndarray, baseline: np.ndarray, label: str) -> dict:
    mask = np.isfinite(ralc) & np.isfinite(baseline)
    ralc, baseline = ralc[mask], baseline[mask]
    n = len(ralc)
    delta = ralc - baseline  # negative => RALC better (lower ECE/FD)

    t_stat, t_p_two = stats.ttest_rel(ralc, baseline)
    t_p_one = t_p_two / 2 if t_stat < 0 else 1 - t_p_two / 2

    try:
        w_stat, w_p_one = stats.wilcoxon(ralc, baseline, alternative="less")
    except ValueError:
        w_stat, w_p_one = np.nan, np.nan

    d = delta.mean() / delta.std(ddof=1) if delta.std(ddof=1) > 0 else np.nan
    lo, hi, p_boot = paired_bootstrap_ci(delta)

    return dict(
        comparison=label, n=n,
        mean_ralc=ralc.mean(), mean_baseline=baseline.mean(),
        mean_delta=delta.mean(), median_delta=np.median(delta),
        pct_cells_ralc_better=float(np.mean(delta < 0)) * 100,
        cohens_d=d,
        paired_t_stat=t_stat, paired_t_p_onesided=t_p_one,
        wilcoxon_stat=w_stat, wilcoxon_p_onesided=w_p_one,
        boot_ci_low=lo, boot_ci_high=hi, boot_p_onesided=p_boot,
    )


def main():
    perf_df = load_calibration_performance()
    print(f"Loaded calibration_performance.csv for {len(perf_df)} (dataset, model) leaves "
          f"(after excluding {sorted(EXCLUDE_MODELS)}):")
    print(perf_df[["dataset", "model"]].to_string(index=False))

    cell_df = build_cell_table(perf_df)
    cell_df.to_csv(os.path.join(OUT_DIR, "ece_fd_cell_level_long.csv"), index=False)

    hedged_df = load_hedged_qa()
    hedged_df.to_csv(os.path.join(OUT_DIR, "hedged_qa_cell_level.csv"), index=False)
    print(f"\nLoaded Hedged QA eval_metrics.csv for {len(hedged_df)} (dataset, model) leaves.")

    results = []

    # --- RALC vs Original / Beta-Guided, per signal and pooled across signals ---
    for metric in METRICS:
        for sig in SIGNALS:
            sub = cell_df[cell_df["signal"] == sig]
            results.append(run_comparison(
                sub[f"ralc_{metric}"].to_numpy(), sub[f"original_{metric}"].to_numpy(),
                f"{metric} | RALC vs Original Direct QA | signal={sig}"))
            results.append(run_comparison(
                sub[f"ralc_{metric}"].to_numpy(), sub[f"beta_guided_{metric}"].to_numpy(),
                f"{metric} | RALC vs Direct Beta-Guided | signal={sig}"))
        # pooled across signals (n = datasets x models x 3 signals)
        results.append(run_comparison(
            cell_df[f"ralc_{metric}"].to_numpy(), cell_df[f"original_{metric}"].to_numpy(),
            f"{metric} | RALC vs Original Direct QA | signal=POOLED"))
        results.append(run_comparison(
            cell_df[f"ralc_{metric}"].to_numpy(), cell_df[f"beta_guided_{metric}"].to_numpy(),
            f"{metric} | RALC vs Direct Beta-Guided | signal=POOLED"))

    # --- RALC (mean over 3 signals per dataset/model) vs Hedged QA ---
    ralc_mean = (
        cell_df.groupby(["dataset", "model"])[[f"ralc_{m}" for m in METRICS]]
        .mean()
        .reset_index()
    )
    merged = ralc_mean.merge(hedged_df, on=["dataset", "model"], how="inner",
                              suffixes=("", "_hedged"))
    print(f"\nMatched {len(merged)} (dataset, model) cells between RALC and Hedged QA.")
    for metric in METRICS:
        results.append(run_comparison(
            merged[f"ralc_{metric}"].to_numpy(), merged[metric].to_numpy(),
            f"{metric} | RALC vs Hedged QA | signal=MEAN-OF-3"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, "ece_fd_significance_results.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Significance test results (one-sided: RALC < baseline) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved: ece_fd_cell_level_long.csv, hedged_qa_cell_level.csv, ece_fd_significance_results.csv")


if __name__ == "__main__":
    main()
