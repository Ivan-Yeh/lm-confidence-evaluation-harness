"""
Per-subset (MMLU category) computation of generalised ECE and faithfulness divergence,
pre and post calibration, for each signal (lc, tp, su), averaged across 4 models.
"""

import sys, os, pickle, re, ast
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.special import betaln, psi
from datasets import load_dataset
from collections import defaultdict

sys.path.insert(0, "/mnt/ssd_1/ivn/lm-confidence-evaluation-harness")
from lm_conf.confidence_metrics.distributionals import BetaDistribution

# ── Constants ───────────────────────────────────────────────────────────────
RESULTS_DIR = "/hdd/ivny/results/mmlu"
CALIB_DIR   = "/hdd/ivny/direct_qa_in_domain_calibration/mmlu"
OUT_DIR     = "/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/rebuttal/mmlu-subset-results"

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "openai/gpt-oss-20b",
    "qwen/Qwen3-8B",
]
SIGNALS = ["lc", "tp", "su"]

# ── BetaDistribution parsing ─────────────────────────────────────────────────
_BETA_RE = re.compile(
    r"BetaDistribution\(alpha=([\d.eE+\-]+),\s*beta=([\d.eE+\-]+),\s*mu=([\d.eE+\-]+),\s*sigma=([\d.eE+\-]+)\)"
)

def parse_beta(s) -> BetaDistribution | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, BetaDistribution):
        return s
    m = _BETA_RE.match(str(s).strip())
    if not m:
        return None
    mu, sigma = float(m.group(3)), float(m.group(4))
    try:
        bd = BetaDistribution(mu, sigma)
        return bd if bd.is_valid() else None
    except Exception:
        return None


# ── Metric implementations ───────────────────────────────────────────────────
def compute_generalised_ece(confs: list, accs: list, num_bins=10, num_samples=500) -> float:
    pairs = [(parse_beta(c), a) for c, a in zip(confs, accs)]
    pairs = [(c, a) for c, a in pairs if c is not None and a is not None]
    if len(pairs) < 2:
        return float("nan")
    dists, y = zip(*pairs)
    y = np.array(y, dtype=float)
    N = len(y)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    all_samples = np.array([d.sample(size=num_samples) for d in dists])  # (N, S)
    bin_indices = np.clip(np.digitize(all_samples, bins) - 1, 0, num_bins - 1)

    ece = 0.0
    total_p = 0.0
    for m in range(num_bins):
        mask = (bin_indices == m)
        p_nm = np.mean(mask, axis=1)
        pm = np.sum(p_nm)
        if pm > 0:
            rm = np.sum(p_nm * y) / pm
            bin_samples = all_samples[mask]
            gm = np.mean(bin_samples) if bin_samples.size > 0 else 0.0
            ece += pm * abs(rm - gm)
            total_p += pm
    return float(ece / total_p) if total_p > 0 else float("nan")


def compute_faithfulness_divergence(confs: list, accs: list) -> float:
    def kl_beta(a_post, b_post, a_prior, b_prior):
        return (
            betaln(a_prior, b_prior) - betaln(a_post, b_post)
            + (a_post - a_prior) * psi(a_post)
            + (b_post - b_prior) * psi(b_post)
            - (a_post + b_post - a_prior - b_prior) * psi(a_post + b_post)
        )

    divs = []
    for c, a in zip(confs, accs):
        bd = parse_beta(c)
        if bd is None or a is None:
            continue
        alpha, bparam = bd.alpha_param, bd.beta_param
        a_post = alpha + float(a)
        b_post = bparam + (1 - float(a))
        d = max(0.0, (alpha + bparam + 1e-8) * kl_beta(a_post, b_post, alpha, bparam))
        if not np.isnan(d):
            divs.append(d)
    return float(np.mean(divs)) if divs else float("nan")


# ── MMLU dataset: question text → subject ────────────────────────────────────
def load_mmlu_subjects() -> list[str]:
    """Returns list of subjects in MMLU test split order (14042 entries)."""
    print("Loading MMLU dataset ...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    subjects = [row["subject"] for row in ds]
    print(f"  Loaded {len(subjects)} questions, {len(set(subjects))} subjects")
    return subjects


# ── Data loading helpers ──────────────────────────────────────────────────────
def latest_dir(path: str) -> str:
    entries = sorted(
        [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]
    )
    return os.path.join(path, entries[-1])


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_graded_outputs(model: str, signal: str):
    task = f"direct_qa_unified_{signal}"
    run_dir = latest_dir(f"{RESULTS_DIR}/{task}/{model}")
    return load_pkl(f"{run_dir}/graded_outputs_0.pkl")


def get_calib_indices(model: str) -> np.ndarray:
    """
    Returns the MMLU dataset indices (into the 14042-length array) that are
    present in calibration_details.csv.

    Strategy: sequentially match calibration_details['original_response'] against
    graded_outputs extracted_answers in MMLU order, consuming each match once.
    This handles short responses ('D', 'B', etc.) that appear many times by
    always picking the next unused occurrence.
    """
    lc = load_graded_outputs(model, "lc")
    all_responses = lc.extracted_answers[0]  # 14042 in MMLU order

    calib_csv = f"{CALIB_DIR}/{model}/calibration_details.csv"
    df_calib = pd.read_csv(calib_csv)
    calib_responses = df_calib["original_response"].tolist()

    # Build index: response_text -> sorted list of MMLU positions
    from collections import defaultdict as _dd
    resp_to_positions: dict[str, list[int]] = _dd(list)
    for i, r in enumerate(all_responses):
        resp_to_positions[r].append(i)
    # Sort each list and keep a pointer
    resp_pointers: dict[str, int] = {k: 0 for k in resp_to_positions}

    calib_indices = []
    unmatched = 0
    for cr in calib_responses:
        positions = resp_to_positions.get(cr, [])
        ptr = resp_pointers.get(cr, 0)
        if ptr < len(positions):
            calib_indices.append(positions[ptr])
            resp_pointers[cr] = ptr + 1
        else:
            unmatched += 1
            calib_indices.append(-1)  # sentinel for unmatched

    calib_indices = np.array(calib_indices)
    print(
        f"  Calib indices for {model.split('/')[-1]}: "
        f"matched={np.sum(calib_indices >= 0)}/{len(calib_indices)}, "
        f"unmatched={unmatched}"
    )
    return calib_indices


# ── Per-model, per-signal metric computation ──────────────────────────────────
def compute_pre_calib_metrics(model: str, signal: str, subjects: list[str]):
    """
    Returns DataFrame with columns [subject, ece, fd] using pre-calibration signal.
    Uses graded_outputs pkl (same ordering as MMLU dataset).
    """
    graded = load_graded_outputs(model, signal)
    confs = graded.extracted_confidences[0]
    accs  = graded.accuracy_scores[0]

    assert len(confs) == len(subjects), f"Length mismatch: {len(confs)} vs {len(subjects)}"

    rows = defaultdict(lambda: {"confs": [], "accs": []})
    for subj, c, a in zip(subjects, confs, accs):
        if c is not None and a is not None and a != "":
            try:
                rows[subj]["confs"].append(c)
                rows[subj]["accs"].append(float(a))
            except (ValueError, TypeError):
                continue

    records = []
    for subj, data in rows.items():
        ece = compute_generalised_ece(data["confs"], data["accs"])
        fd  = compute_faithfulness_divergence(data["confs"], data["accs"])
        records.append({"subject": subj, "ece": ece, "fd": fd})
    return pd.DataFrame(records)


def compute_post_calib_metrics(model: str, signal: str, subjects: list[str], calib_indices: np.ndarray):
    """
    Returns DataFrame with columns [subject, ece, fd] using calibrated signal.
    calib_indices are MMLU dataset positions corresponding to calibration_details rows.
    """
    calib_csv = f"{CALIB_DIR}/{model}/calibration_details.csv"
    df = pd.read_csv(calib_csv)

    col = f"calibrated_{signal}"
    assert col in df.columns, f"Missing column {col} in {calib_csv}"

    subjects_arr = np.array(subjects + ["__unknown__"])  # -1 maps to __unknown__
    calib_subjects = subjects_arr[calib_indices]

    rows = defaultdict(lambda: {"confs": [], "accs": []})
    for subj, c, a in zip(calib_subjects, df[col], df["accuracy"]):
        if subj == "__unknown__":
            continue
        if pd.notna(c) and pd.notna(a):
            rows[subj]["confs"].append(c)
            rows[subj]["accs"].append(float(a))

    records = []
    for subj, data in rows.items():
        ece = compute_generalised_ece(data["confs"], data["accs"])
        fd  = compute_faithfulness_divergence(data["confs"], data["accs"])
        records.append({"subject": subj, "ece": ece, "fd": fd})
    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subjects = load_mmlu_subjects()

    all_records = []

    for model in MODELS:
        model_short = model.split("/")[-1]
        print(f"\n=== Model: {model_short} ===")

        # Build calibration indices once per model
        calib_indices = get_calib_indices(model)

        for signal in SIGNALS:
            print(f"  Signal: {signal}")

            # Pre-calibration
            print(f"    Computing pre-calib metrics ...")
            pre_df = compute_pre_calib_metrics(model, signal, subjects)
            for _, row in pre_df.iterrows():
                all_records.append({
                    "model": model_short,
                    "signal": signal,
                    "stage": "pre",
                    "subject": row["subject"],
                    "generalised_ece": row["ece"],
                    "faithfulness_divergence": row["fd"],
                })

            # Post-calibration
            print(f"    Computing post-calib metrics ...")
            post_df = compute_post_calib_metrics(model, signal, subjects, calib_indices)
            for _, row in post_df.iterrows():
                all_records.append({
                    "model": model_short,
                    "signal": signal,
                    "stage": "post",
                    "subject": row["subject"],
                    "generalised_ece": row["ece"],
                    "faithfulness_divergence": row["fd"],
                })

    # Save all per-model-signal-stage-subject records
    full_df = pd.DataFrame(all_records)
    full_df.to_csv(f"{OUT_DIR}/per_model_subset_metrics.csv", index=False)
    print(f"\nSaved per-model metrics: {OUT_DIR}/per_model_subset_metrics.csv")

    # Average across models → (subject, signal, stage) summary
    agg = (
        full_df
        .groupby(["subject", "signal", "stage"])[["generalised_ece", "faithfulness_divergence"]]
        .agg(["mean", "std"])
    )
    agg.columns = ["ece_mean", "ece_std", "fd_mean", "fd_std"]
    agg = agg.reset_index()
    agg.to_csv(f"{OUT_DIR}/subset_metrics_averaged.csv", index=False)
    print(f"Saved averaged metrics: {OUT_DIR}/subset_metrics_averaged.csv")

    # Pivot: pre vs post side by side for easy comparison
    pivot = agg.pivot(index=["subject", "signal"], columns="stage",
                      values=["ece_mean", "ece_std", "fd_mean", "fd_std"])
    pivot.columns = [f"{v}_{s}" for v, s in pivot.columns]
    pivot = pivot.reset_index()
    pivot["ece_delta"] = pivot["ece_mean_post"] - pivot["ece_mean_pre"]
    pivot["fd_delta"]  = pivot["fd_mean_post"]  - pivot["fd_mean_pre"]
    pivot.to_csv(f"{OUT_DIR}/subset_metrics_pivot.csv", index=False)
    print(f"Saved pivot table: {OUT_DIR}/subset_metrics_pivot.csv")

    return full_df, agg, pivot


if __name__ == "__main__":
    full_df, agg, pivot = main()
    print("\nDone.")
