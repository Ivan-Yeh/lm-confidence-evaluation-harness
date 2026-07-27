"""
Significance test for the "Mean Entailment Score" panel of
fig_calibration_performance_baseline_comparison, addressing the reviewer's
request for a significance test on RALC-vs-baseline deltas.

Reuses the exact NLI scoring recipe from the source notebook (cross-encoder/
nli-deberta-v3-base, soft entailment = P(entail) + 0.5*P(neutral), direction
rewritten -> original):

  1. RALC vs Direct Beta-Guided (PAIRED, same underlying examples from
     calibration_details.csv -- calibrated_{lc,tp,su}_rewritten_response vs
     calibrated_{lc,tp,su}_beta_guided_response, both scored against the same
     original_response). Model exclusion matches the figure's direct_qa
     panel (gpt-oss-120b, Qwen3-235B excluded).
     -> Wilcoxon signed-rank test + paired t-test, per signal and pooled.

  2. RALC vs Hedged QA (UNPAIRED -- different pipeline/example set: hedged_qa
     responses vs direct_qa original responses, results/<dataset>/
     {direct,hedged}_qa_unified_lc; no model exclusion, matching the source
     notebook's direct_vs_hedged computation).
     -> Mann-Whitney U test (independent samples) between the pooled RALC
     score distribution and the pooled Hedged QA score distribution.

Run with the `llm` conda env python (needs sentence-transformers + a GPU).
"""
import os
import glob

import numpy as np
import pandas as pd
import scipy.special as sp
from scipy import stats

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DIRECT_CAL_ROOT = "/hdd/ivny/direct_qa_in_domain_calibration"
RESULTS_ROOT = "/hdd/ivny/results"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_MODELS = {"gpt-oss-120b", "qwen3-235b-a22b-instruct-2507-tput"}
SIGNALS = ["lc", "tp", "su"]
BATCH_SIZE = 64


def load_calibration_details():
    rows = []
    for path in sorted(glob.glob(os.path.join(DIRECT_CAL_ROOT, "*", "*", "*", "calibration_details.csv"))):
        rel = os.path.relpath(path, DIRECT_CAL_ROOT)
        dataset, org, model = rel.split(os.sep)[:3]
        if model.lower() in EXCLUDE_MODELS:
            continue
        df = pd.read_csv(path)
        df["dataset"] = dataset
        df["model"] = f"{org}/{model}"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def latest_valid_eval_details(path):
    """Newest-mtime-first subdir whose eval_details.csv actually has a response_0
    column with data -- some newer rerun dirs (2026-07-25) are broken/partial and
    would otherwise shadow the good original runs under naive max(mtime)."""
    subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for d in sorted(subdirs, key=os.path.getmtime, reverse=True):
        csv_path = os.path.join(d, "eval_details.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            cols = pd.read_csv(csv_path, nrows=1).columns.tolist()
        except Exception:
            continue
        if "response_0" in cols:
            return csv_path
    return None


def load_direct_hedged_responses():
    """Pooled direct-QA original responses and hedged-QA responses (no model exclusion),
    matching the source notebook's direct_vs_hedged_entailment computation."""
    direct_resp, hedged_resp = [], []
    for dataset in ["mmlu", "squadv2", "truthful_qa"]:
        for prompt_type, sink in [("direct_qa_unified_lc", direct_resp), ("hedged_qa_unified_lc", hedged_resp)]:
            base = os.path.join(RESULTS_ROOT, dataset, prompt_type)
            if not os.path.isdir(base):
                continue
            for org in sorted(os.listdir(base)):
                org_path = os.path.join(base, org)
                if not os.path.isdir(org_path):
                    continue
                for model in sorted(os.listdir(org_path)):
                    model_path = os.path.join(org_path, model)
                    csv_path = latest_valid_eval_details(model_path)
                    if csv_path is None:
                        print(f"  no valid eval_details.csv under {model_path}")
                        continue
                    sink.append(pd.read_csv(csv_path)["response_0"])
    return pd.concat(direct_resp, ignore_index=True), pd.concat(hedged_resp, ignore_index=True)


def entailment_scores(nli_model, rewritten, original, batch_size=BATCH_SIZE):
    pairs = [(str(r), str(o)) for r, o in zip(rewritten, original)]
    logits = nli_model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    probs = sp.softmax(logits, axis=1)
    # NLI label order: [contradiction, entailment, neutral]
    return probs[:, 1] + 0.5 * probs[:, 2]


def paired_bootstrap_ci(delta, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    p_one = float(np.mean(boot_means <= 0))  # H1: RALC - baseline > 0
    return float(lo), float(hi), p_one


def main():
    from sentence_transformers import CrossEncoder
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    print("=== Part 1: RALC vs Direct Beta-Guided (paired, per-example) ===")
    df = load_calibration_details()
    print(f"Loaded {len(df)} calibration rows across {df['model'].nunique()} models x "
          f"{df['dataset'].nunique()} datasets (excluding {sorted(EXCLUDE_MODELS)}).")

    original = df["original_response"].tolist()
    per_example_rows = []
    part1_results = []
    for sig in SIGNALS:
        print(f"\n-- signal={sig} --")
        ralc_resp = df[f"calibrated_{sig}_rewritten_response"].tolist()
        beta_resp = df[f"calibrated_{sig}_beta_guided_response"].tolist()

        s_ralc = entailment_scores(nli_model, ralc_resp, original)
        s_beta = entailment_scores(nli_model, beta_resp, original)

        per_example_rows.append(pd.DataFrame({
            "signal": sig, "dataset": df["dataset"], "model": df["model"],
            "entail_ralc": s_ralc, "entail_beta_guided": s_beta,
        }))

        delta = s_ralc - s_beta
        t_stat, t_p_two = stats.ttest_rel(s_ralc, s_beta)
        t_p_one = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
        w_stat, w_p_one = stats.wilcoxon(s_ralc, s_beta, alternative="greater")
        lo, hi, boot_p = paired_bootstrap_ci(delta)
        row = dict(signal=sig, n=len(s_ralc), mean_ralc=s_ralc.mean(), mean_beta=s_beta.mean(),
                   mean_delta=delta.mean(), pct_examples_ralc_higher=float(np.mean(delta > 0)) * 100,
                   paired_t_stat=t_stat, paired_t_p_onesided=t_p_one,
                   wilcoxon_stat=w_stat, wilcoxon_p_onesided=w_p_one,
                   boot_ci_low=lo, boot_ci_high=hi, boot_p_onesided=boot_p)
        part1_results.append(row)
        print(f"  mean entail: RALC={row['mean_ralc']:.4f} Beta-Guided={row['mean_beta']:.4f} "
              f"delta={row['mean_delta']:+.4f} paired-t p={t_p_one:.2e} wilcoxon p={w_p_one:.2e} "
              f"boot 95% CI=[{lo:.4f},{hi:.4f}] p={boot_p:.4f}")

    # Pooled across signals
    all_ralc = np.concatenate([r["entail_ralc"].to_numpy() for r in per_example_rows])
    all_beta = np.concatenate([r["entail_beta_guided"].to_numpy() for r in per_example_rows])
    delta = all_ralc - all_beta
    t_stat, t_p_two = stats.ttest_rel(all_ralc, all_beta)
    t_p_one = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2
    w_stat, w_p_one = stats.wilcoxon(all_ralc, all_beta, alternative="greater")
    lo, hi, boot_p = paired_bootstrap_ci(delta)
    part1_results.append(dict(signal="POOLED", n=len(all_ralc), mean_ralc=all_ralc.mean(),
                               mean_beta=all_beta.mean(), mean_delta=delta.mean(),
                               pct_examples_ralc_higher=float(np.mean(delta > 0)) * 100,
                               paired_t_stat=t_stat, paired_t_p_onesided=t_p_one,
                               wilcoxon_stat=w_stat, wilcoxon_p_onesided=w_p_one,
                               boot_ci_low=lo, boot_ci_high=hi, boot_p_onesided=boot_p))
    print(f"\nPOOLED: RALC={all_ralc.mean():.4f} Beta-Guided={all_beta.mean():.4f} "
          f"delta={delta.mean():+.4f} paired-t p={t_p_one:.2e} wilcoxon p={w_p_one:.2e}")

    pd.concat(per_example_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "entailment_ralc_vs_beta_guided_per_example.csv"), index=False)
    pd.DataFrame(part1_results).to_csv(
        os.path.join(OUT_DIR, "entailment_ralc_vs_beta_guided_significance.csv"), index=False)

    print("\n=== Part 2: RALC vs Hedged QA (unpaired, Mann-Whitney U) ===")
    direct_resp, hedged_resp = load_direct_hedged_responses()
    n = min(len(direct_resp), len(hedged_resp))
    print(f"direct_qa responses: {len(direct_resp)}, hedged_qa responses: {len(hedged_resp)} "
          f"(pairing first {n} in file order, matching source notebook's zip behavior)")
    hedged_scores = entailment_scores(nli_model, hedged_resp.tolist()[:n], direct_resp.tolist()[:n])

    ralc_pooled = np.concatenate([r["entail_ralc"].to_numpy() for r in per_example_rows])
    u_stat, u_p = stats.mannwhitneyu(ralc_pooled, hedged_scores, alternative="greater")
    part2_row = dict(n_ralc=len(ralc_pooled), n_hedged=len(hedged_scores),
                      mean_ralc=ralc_pooled.mean(), mean_hedged=hedged_scores.mean(),
                      mean_delta=ralc_pooled.mean() - hedged_scores.mean(),
                      mannwhitney_u=u_stat, mannwhitney_p_onesided=u_p)
    print(f"  mean entail: RALC(pooled)={part2_row['mean_ralc']:.4f} Hedged QA={part2_row['mean_hedged']:.4f} "
          f"delta={part2_row['mean_delta']:+.4f} Mann-Whitney U p={u_p:.2e}")

    pd.Series(hedged_scores).to_csv(os.path.join(OUT_DIR, "entailment_hedged_qa_scores.csv"), index=False,
                                     header=["entail_hedged"])
    pd.DataFrame([part2_row]).to_csv(os.path.join(OUT_DIR, "entailment_ralc_vs_hedged_significance.csv"), index=False)

    print("\nSaved: entailment_ralc_vs_beta_guided_per_example.csv, "
          "entailment_ralc_vs_beta_guided_significance.csv, entailment_hedged_qa_scores.csv, "
          "entailment_ralc_vs_hedged_significance.csv")


if __name__ == "__main__":
    main()
