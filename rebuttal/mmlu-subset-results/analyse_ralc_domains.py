"""
Analyse RALC (post-calibration) performance across MMLU domains.
For each signal × subset: compare pre vs post ECE and FD.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = "/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/rebuttal/mmlu-subset-results"

pivot = pd.read_csv(f"{OUT_DIR}/subset_metrics_pivot.csv")

# ── MMLU coarse domain groups ────────────────────────────────────────────────
DOMAIN_MAP = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "world_religions",
        "high_school_government_and_politics",
    ],
    "Social Sciences": [
        "econometrics", "global_facts", "high_school_geography",
        "high_school_macroeconomics", "high_school_microeconomics",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Professional / Medical": [
        "anatomy", "clinical_knowledge", "college_medicine", "human_aging",
        "medical_genetics", "nutrition", "professional_medicine",
        "professional_accounting", "professional_law", "virology",
        "business_ethics", "management", "marketing", "miscellaneous",
    ],
}

# Build reverse mapping
subj_to_domain = {}
for domain, subjects in DOMAIN_MAP.items():
    for s in subjects:
        subj_to_domain[s] = domain


def safe_mean(s):
    vals = s.dropna()
    return vals.mean() if len(vals) > 0 else float("nan")


def main():
    print("=" * 70)
    print("RALC PERFORMANCE ACROSS MMLU DOMAINS")
    print("(averaged across 4 models: Llama-3.1-8B, Mistral-7B, gpt-oss-20b, Qwen3-8B)")
    print("=" * 70)

    pivot["domain"] = pivot["subject"].map(subj_to_domain).fillna("Other")

    results = []
    for signal in ["lc", "tp", "su"]:
        df_sig = pivot[pivot["signal"] == signal].copy()
        print(f"\n{'─'*60}")
        print(f"Signal: {signal.upper()}")
        print(f"{'─'*60}")
        print(f"{'Subject':<45} {'ECE pre':>8} {'ECE post':>9} {'ΔECE':>7} {'FD pre':>8} {'FD post':>8} {'ΔFD':>7}")
        print("-" * 100)

        for _, row in df_sig.sort_values("ece_mean_pre").iterrows():
            subj = row["subject"]
            ece_pre  = row.get("ece_mean_pre", float("nan"))
            ece_post = row.get("ece_mean_post", float("nan"))
            fd_pre   = row.get("fd_mean_pre",  float("nan"))
            fd_post  = row.get("fd_mean_post", float("nan"))
            d_ece = row.get("ece_delta", float("nan"))
            d_fd  = row.get("fd_delta",  float("nan"))
            domain = subj_to_domain.get(subj, "Other")
            print(
                f"  {subj:<43} {ece_pre:>8.4f} {ece_post:>9.4f} {d_ece:>+7.4f}"
                f"  {fd_pre:>8.4f} {fd_post:>8.4f} {d_fd:>+7.4f}  [{domain}]"
            )
            results.append({
                "signal": signal, "subject": subj, "domain": domain,
                "ece_pre": ece_pre, "ece_post": ece_post, "ece_delta": d_ece,
                "fd_pre": fd_pre, "fd_post": fd_post, "fd_delta": d_fd,
            })

    results_df = pd.DataFrame(results)

    # ── Domain-level summary ──────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("DOMAIN-LEVEL SUMMARY (mean across subjects in domain × signal)")
    print("=" * 70)
    print(f"\n{'Signal':<6} {'Domain':<25} {'ECE pre':>8} {'ECE post':>9} {'ΔECE':>8} {'FD pre':>8} {'FD post':>8} {'ΔFD':>8}")
    print("-" * 90)

    domain_summary = []
    for signal in ["lc", "tp", "su"]:
        for domain in sorted(results_df["domain"].unique()):
            sub = results_df[(results_df["signal"] == signal) & (results_df["domain"] == domain)]
            row = {
                "signal": signal,
                "domain": domain,
                "ece_pre":  safe_mean(sub["ece_pre"]),
                "ece_post": safe_mean(sub["ece_post"]),
                "ece_delta": safe_mean(sub["ece_delta"]),
                "fd_pre":   safe_mean(sub["fd_pre"]),
                "fd_post":  safe_mean(sub["fd_post"]),
                "fd_delta":  safe_mean(sub["fd_delta"]),
                "n_subjects": len(sub),
            }
            domain_summary.append(row)
            print(
                f"{signal.upper():<6} {domain:<25} "
                f"{row['ece_pre']:>8.4f} {row['ece_post']:>9.4f} {row['ece_delta']:>+8.4f} "
                f"{row['fd_pre']:>8.4f} {row['fd_post']:>8.4f} {row['fd_delta']:>+8.4f}"
            )

    domain_df = pd.DataFrame(domain_summary)
    domain_df.to_csv(f"{OUT_DIR}/domain_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/domain_summary.csv")

    # ── Key findings ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for signal in ["lc", "tp", "su"]:
        sub = results_df[results_df["signal"] == signal]
        print(f"\nSignal: {signal.upper()}")

        # ECE improvement
        best_ece = sub.nsmallest(3, "ece_delta")[["subject", "domain", "ece_delta"]]
        worst_ece = sub.nlargest(3, "ece_delta")[["subject", "domain", "ece_delta"]]
        print(f"  Best ECE improvement (most negative ΔECE):")
        for _, r in best_ece.iterrows():
            print(f"    {r['subject']:<40} ΔECE={r['ece_delta']:+.4f}  [{r['domain']}]")
        print(f"  Worst ECE (most positive ΔECE = calibration hurt):")
        for _, r in worst_ece.iterrows():
            print(f"    {r['subject']:<40} ΔECE={r['ece_delta']:+.4f}  [{r['domain']}]")

        # FD improvement
        best_fd = sub.nsmallest(3, "fd_delta")[["subject", "domain", "fd_delta"]]
        worst_fd = sub.nlargest(3, "fd_delta")[["subject", "domain", "fd_delta"]]
        print(f"  Best FD improvement:")
        for _, r in best_fd.iterrows():
            print(f"    {r['subject']:<40} ΔFD={r['fd_delta']:+.4f}  [{r['domain']}]")
        print(f"  Worst FD:")
        for _, r in worst_fd.iterrows():
            print(f"    {r['subject']:<40} ΔFD={r['fd_delta']:+.4f}  [{r['domain']}]")

        # % subsets improved
        pct_ece = (sub["ece_delta"] < 0).mean() * 100
        pct_fd  = (sub["fd_delta"]  < 0).mean() * 100
        print(f"  Subsets where calibration improved ECE: {pct_ece:.0f}%")
        print(f"  Subsets where calibration improved FD:  {pct_fd:.0f}%")

    # ── Heatmap: Δ ECE by domain × signal ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, label in [(axes[0], "ece_delta", "ΔECE (post−pre)"),
                               (axes[1], "fd_delta",  "ΔFD (post−pre)")]:
        heat = domain_df.pivot(index="domain", columns="signal", values=metric)
        im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=-0.05, vmax=0.05)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels([c.upper() for c in heat.columns], fontsize=11)
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index, fontsize=10)
        ax.set_title(label, fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                val = heat.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                            fontsize=9, color="black")

    plt.suptitle("RALC calibration Δ by domain × signal\n(green = improvement, red = degradation)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/domain_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved heatmap: {OUT_DIR}/domain_heatmap.png")

    # ── Scatter: pre vs post ECE per subset ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    domain_colors = {"STEM": "#1f77b4", "Humanities": "#ff7f0e",
                     "Social Sciences": "#2ca02c", "Professional / Medical": "#d62728",
                     "Other": "#9467bd"}
    for ax, signal in zip(axes, ["lc", "tp", "su"]):
        sub = results_df[results_df["signal"] == signal]
        for domain, grp in sub.groupby("domain"):
            ax.scatter(grp["ece_pre"], grp["ece_post"], label=domain,
                       color=domain_colors.get(domain, "gray"), alpha=0.7, s=40)
        lim = max(sub["ece_pre"].max(), sub["ece_post"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="no change")
        ax.set_xlabel("ECE pre-calibration", fontsize=10)
        ax.set_ylabel("ECE post-calibration", fontsize=10)
        ax.set_title(f"Signal: {signal.upper()}", fontsize=11)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("ECE pre vs post calibration per MMLU subset\n(points below diagonal = improved)", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f"{OUT_DIR}/ece_scatter.png", dpi=150, bbox_inches="tight")
    print(f"Saved scatter: {OUT_DIR}/ece_scatter.png")

    return results_df, domain_df


if __name__ == "__main__":
    results_df, domain_df = main()
