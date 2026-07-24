#!/usr/bin/env python
"""
Full comparison table: RALC (LC / SU / TP signals) vs Wang et al. (DOT).
Reads Wang results from wang_vs_ralc_results.csv and RALC from
calibration_performance.csv for each model.

Columns: gECE | ΔgECE | %ΔgECE | FD | ΔFD | %ΔFD | dAUROC
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_ROOT = "/hdd/ivny/direct_qa_in_domain_calibration/truthful_qa"
WANG_CSV  = "/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/comparison/wang_vs_ralc_results.csv"

MODELS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral-7B":   "mistralai/Mistral-7B-Instruct-v0.3",
    "GPT-OSS-20B":  "openai/gpt-oss-20b",
    "Qwen3-8B":     "qwen/Qwen3-8B",
}

SIGNALS = ["lc", "su", "tp"]
SIGNAL_LABEL = {"lc": "LC", "su": "SU", "tp": "TP"}
ESTIMATOR_TO_SIG = {"Linguistic Confidence": "lc", "Semantic Uncertainty": "su", "Token Probability": "tp"}

# ── RALC TruthfulQA values (from paper results dict) ──────────────────────────
# Keyed by (model_label, signal). dAUROC still read from CSV (not in this dict).
_MODEL_LABEL_MAP = {
    "Llama-3.1-8B": "Llama-3.1-8B-Inst.",
    "Mistral-7B":   "Mistral-7B-Inst.",
    "GPT-OSS-20B":  "GPT-OSS-20B",
    "Qwen3-8B":     "Qwen3-8B-Inst.",
}

_RALC_TRUTHFULQA = [
    {'Model': 'GPT-OSS-20B',       'Estimator': 'Linguistic Confidence',  'Original Faithfulness Divergence': 2.2651771441573527,  'Calibrated Faithfulness Divergence': 0.677997755319684,   'Original Generalised ECE': 0.3923309326778563,  'Calibrated Generalised ECE': 0.2298367006156856},
    {'Model': 'GPT-OSS-20B',       'Estimator': 'Semantic Uncertainty',   'Original Faithfulness Divergence': 2.2651771441573527,  'Calibrated Faithfulness Divergence': 0.4750813298421877,  'Original Generalised ECE': 0.3923309326778563,  'Calibrated Generalised ECE': 0.1786099145422159},
    {'Model': 'GPT-OSS-20B',       'Estimator': 'Token Probability',      'Original Faithfulness Divergence': 2.2651771441573527,  'Calibrated Faithfulness Divergence': 0.6453790604859229,  'Original Generalised ECE': 0.3923309326778563,  'Calibrated Generalised ECE': 0.2267115693249688},
    {'Model': 'Mistral-7B-Inst.',  'Estimator': 'Linguistic Confidence',  'Original Faithfulness Divergence': 1.781701991624782,   'Calibrated Faithfulness Divergence': 0.7101366963634791,  'Original Generalised ECE': 0.3740181445492032,  'Calibrated Generalised ECE': 0.2273091234891761},
    {'Model': 'Mistral-7B-Inst.',  'Estimator': 'Semantic Uncertainty',   'Original Faithfulness Divergence': 1.781701991624782,   'Calibrated Faithfulness Divergence': 0.4769694828200648,  'Original Generalised ECE': 0.3740181445492032,  'Calibrated Generalised ECE': 0.2376406809287461},
    {'Model': 'Mistral-7B-Inst.',  'Estimator': 'Token Probability',      'Original Faithfulness Divergence': 1.781701991624782,   'Calibrated Faithfulness Divergence': 0.7270842255792598,  'Original Generalised ECE': 0.3740181445492032,  'Calibrated Generalised ECE': 0.2478583195657958},
    {'Model': 'Llama-3.1-8B-Inst.','Estimator': 'Linguistic Confidence',  'Original Faithfulness Divergence': 1.8284753963139375,  'Calibrated Faithfulness Divergence': 0.7121433077421214,  'Original Generalised ECE': 0.4266761227177914,  'Calibrated Generalised ECE': 0.2686759653394905},
    {'Model': 'Llama-3.1-8B-Inst.','Estimator': 'Semantic Uncertainty',   'Original Faithfulness Divergence': 1.8284753963139375,  'Calibrated Faithfulness Divergence': 0.4668259789214957,  'Original Generalised ECE': 0.4266761227177914,  'Calibrated Generalised ECE': 0.2544532597415606},
    {'Model': 'Llama-3.1-8B-Inst.','Estimator': 'Token Probability',      'Original Faithfulness Divergence': 1.8284753963139375,  'Calibrated Faithfulness Divergence': 0.6363884084448393,  'Original Generalised ECE': 0.4266761227177914,  'Calibrated Generalised ECE': 0.2471196361662431},
    {'Model': 'Qwen3-8B-Inst.',    'Estimator': 'Linguistic Confidence',  'Original Faithfulness Divergence': 2.0537941094044863,  'Calibrated Faithfulness Divergence': 0.7069332612876178,  'Original Generalised ECE': 0.4421168887709607,  'Calibrated Generalised ECE': 0.2701187271085545},
    {'Model': 'Qwen3-8B-Inst.',    'Estimator': 'Semantic Uncertainty',   'Original Faithfulness Divergence': 2.0537941094044863,  'Calibrated Faithfulness Divergence': 0.4635585425900997,  'Original Generalised ECE': 0.4421168887709607,  'Calibrated Generalised ECE': 0.2466558556571673},
    {'Model': 'Qwen3-8B-Inst.',    'Estimator': 'Token Probability',      'Original Faithfulness Divergence': 2.0537941094044863,  'Calibrated Faithfulness Divergence': 0.6230594745270341,  'Original Generalised ECE': 0.4421168887709607,  'Calibrated Generalised ECE': 0.2421380813583741},
]

def _get_ralc_row(model_name: str, sig: str) -> dict:
    label = _MODEL_LABEL_MAP.get(model_name, model_name)
    estimator = {v: k for k, v in ESTIMATOR_TO_SIG.items()}[sig]
    for r in _RALC_TRUTHFULQA:
        if r["Model"] == label and r["Estimator"] == estimator:
            return r
    return {}

# ── helpers ────────────────────────────────────────────────────────────────────

def pct_change(orig, cal):
    if orig == 0 or np.isnan(orig) or np.isnan(cal):
        return float("nan")
    return 100.0 * (orig - cal) / orig   # positive = improvement (reduction)


def fmt(v, decimals=4):
    if np.isnan(v):
        return "  —   "
    return f"{v:.{decimals}f}"


def fmt_delta(v, decimals=4):
    if np.isnan(v):
        return "  —   "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}"


def fmt_pct(v, decimals=1):
    if np.isnan(v):
        return "   —  "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


# ── build per-model rows ────────────────────────────────────────────────────────

def load_model(model_name: str, model_path: str, wang_df: pd.DataFrame):
    perf_path = os.path.join(DATA_ROOT, model_path, "calibration_performance.csv")
    perf = dict(
        zip(
            pd.read_csv(perf_path)["metric"],
            pd.read_csv(perf_path)["value"],
        )
    )

    wang_row = wang_df[wang_df["model"] == model_name]

    rows = []

    # ── RALC signals ──────────────────────────────────────────────────────────
    for sig in SIGNALS:
        rd = _get_ralc_row(model_name, sig)
        orig_gece  = rd.get("Original Generalised ECE",            float("nan"))
        cal_gece   = rd.get("Calibrated Generalised ECE",          float("nan"))
        orig_fd    = rd.get("Original Faithfulness Divergence",    float("nan"))
        cal_fd     = rd.get("Calibrated Faithfulness Divergence",  float("nan"))
        orig_auroc = perf.get(f"original_{sig}_dAUROC",            float("nan"))
        cal_auroc  = perf.get(f"calibrated_{sig}_dAUROC",          float("nan"))

        rows.append({
            "label":       f"RALC-{SIGNAL_LABEL[sig]}",
            "stage":       "Uncal",
            "gece":        orig_gece,
            "d_gece":      float("nan"),
            "pct_gece":    float("nan"),
            "fd":          orig_fd,
            "d_fd":        float("nan"),
            "pct_fd":      float("nan"),
            "auroc":       orig_auroc,
        })
        rows.append({
            "label":       f"RALC-{SIGNAL_LABEL[sig]}",
            "stage":       "Cal",
            "gece":        cal_gece,
            "d_gece":      orig_gece - cal_gece,
            "pct_gece":    pct_change(orig_gece, cal_gece),
            "fd":          cal_fd,
            "d_fd":        orig_fd - cal_fd,
            "pct_fd":      pct_change(orig_fd, cal_fd),
            "auroc":       cal_auroc,
        })

    # ── Wang ──────────────────────────────────────────────────────────────────
    # Also check calibration_details to get the true total for coverage reporting
    details_path = os.path.join(DATA_ROOT, model_path, "calibration_details.csv")
    n_total_actual = len(pd.read_csv(details_path)) if os.path.exists(details_path) else 0

    KNOWN_MATCHES = {"Qwen3-8B": 9}  # models skipped due to <10 matches — store actual count
    if wang_row.empty:
        wu_gece = wu_fd = wu_auroc = wc_gece = wc_fd = wc_auroc = float("nan")
        n_matched = KNOWN_MATCHES.get(model_name, 0)
        n_total = n_total_actual
        pct_miss = 100.0 * (n_total - n_matched) / n_total if n_total > 0 else float("nan")
    else:
        wu = wang_row[wang_row["method"] == "wang_uncal"].iloc[0]
        wc = wang_row[wang_row["method"] == "wang_cal"].iloc[0]
        wu_gece, wu_fd, wu_auroc = wu["gece"], wu["fd"], wu["dauroc"]
        wc_gece, wc_fd, wc_auroc = wc["gece"], wc["fd"], wc["dauroc"]
        n_matched = int(wu["n_matched"])
        n_total   = int(wu["n_total"])
        pct_miss  = float(wu["pct_missing"])

    rows.append({
        "label":       "Wang et al.",
        "stage":       "Uncal",
        "gece":        wu_gece,
        "d_gece":      float("nan"),
        "pct_gece":    float("nan"),
        "fd":          wu_fd,
        "d_fd":        float("nan"),
        "pct_fd":      float("nan"),
        "auroc":       wu_auroc,
    })
    rows.append({
        "label":       "Wang et al.",
        "stage":       "Cal",
        "gece":        wc_gece,
        "d_gece":      wu_gece - wc_gece,
        "pct_gece":    pct_change(wu_gece, wc_gece),
        "fd":          wc_fd,
        "d_fd":        wu_fd - wc_fd,
        "pct_fd":      pct_change(wu_fd, wc_fd),
        "auroc":       wc_auroc,
    })

    return rows, n_matched, n_total, pct_miss


# ── print table ────────────────────────────────────────────────────────────────

COL_W = 9

def print_table(model_name: str, rows, n_matched, n_total, pct_miss):
    SEP   = "─" * 106
    SEP2  = "═" * 106
    HSEP  = "┄" * 106

    # header
    print(f"\n{SEP2}")
    print(f"  {model_name}   (Wang coverage: {n_matched}/{n_total} — {pct_miss:.1f}% excluded)")
    print(SEP2)

    hdr = (
        f"  {'Method':<14} {'Stage':<7}"
        f"  {'gECE↓':>{COL_W}} {'ΔgECE':>{COL_W}} {'%ΔgECE':>{COL_W+1}}"
        f"    {'FD↓':>{COL_W}} {'ΔFD':>{COL_W}} {'%ΔFD':>{COL_W+1}}"
        f"    {'dAUROC↑':>{COL_W}}"
    )
    print(hdr)
    print(SEP)

    prev_label = None
    for r in rows:
        if prev_label and r["label"] != prev_label:
            print(HSEP)
        prev_label = r["label"]

        line = (
            f"  {r['label']:<14} {r['stage']:<7}"
            f"  {fmt(r['gece']):>{COL_W}} {fmt_delta(r['d_gece']):>{COL_W}} {fmt_pct(r['pct_gece']):>{COL_W+1}}"
            f"    {fmt(r['fd']):>{COL_W}} {fmt_delta(r['d_fd']):>{COL_W}} {fmt_pct(r['pct_fd']):>{COL_W+1}}"
            f"    {fmt(r['auroc']):>{COL_W}}"
        )
        print(line)

    print(SEP2)


# ── main ───────────────────────────────────────────────────────────────────────

OUT_TXT = "/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/comparison/comparison_table.txt"

def main():
    wang_df = pd.read_csv(WANG_CSV)

    import io
    buf = io.StringIO()

    for model_name, model_path in MODELS.items():
        rows, n_matched, n_total, pct_miss = load_model(model_name, model_path, wang_df)
        # capture output
        import contextlib
        with contextlib.redirect_stdout(buf):
            print_table(model_name, rows, n_matched, n_total, pct_miss)
        # also print to terminal
        print_table(model_name, rows, n_matched, n_total, pct_miss)

    print()
    with open(OUT_TXT, "w") as f:
        f.write(buf.getvalue())
    print(f"Table saved to {OUT_TXT}")


if __name__ == "__main__":
    main()
