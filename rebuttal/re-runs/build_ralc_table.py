"""
Reshape truthful_qa_ece_fd_summary.csv into a paper-style table: one row per
(model, signal), with Pre-RALC / Post-RALC generalised ECE and faithfulness
divergence as "mean +/- std" strings.

"Pre-RALC" = original (uncalibrated) signal; "Post-RALC" = in-domain
calibrated signal (stage="post_calibrated" -- the rewritten-text stage,
"post_rewritten", is excluded here since it isn't part of this comparison).

Reads rebuttal/re-runs/truthful_qa_ece_fd_summary.csv (written by
aggregate_results.py) and writes rebuttal/re-runs/truthful_qa_ralc_table.csv
next to it.
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "truthful_qa_ece_fd_summary.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "truthful_qa_ralc_table.csv")

MODEL_ORDER = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen/Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
SIGNAL_ORDER = ["lc", "tp", "su"]


def fmt(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def main():
    df = pd.read_csv(SUMMARY_PATH)

    rows = []
    for model in MODEL_ORDER:
        for signal in SIGNAL_ORDER:
            pre = df[(df["model"] == model) & (df["signal"] == signal) & (df["stage"] == "pre")]
            post = df[(df["model"] == model) & (df["signal"] == signal) & (df["stage"] == "post_calibrated")]
            if pre.empty or post.empty:
                print(f"WARNING: missing pre/post_calibrated rows for model={model} signal={signal}; skipping")
                continue
            pre = pre.iloc[0]
            post = post.iloc[0]
            rows.append({
                "model": model,
                "signal": signal,
                "Pre-RALC ECE": fmt(pre["generalised_ECE_mean"], pre["generalised_ECE_std"]),
                "Post-RALC ECE": fmt(post["generalised_ECE_mean"], post["generalised_ECE_std"]),
                "Pre-RALC FD": fmt(pre["faithfulness_divergence_mean"], pre["faithfulness_divergence_std"]),
                "Post-RALC FD": fmt(post["faithfulness_divergence_mean"], post["faithfulness_divergence_std"]),
            })

    table_df = pd.DataFrame(rows, columns=["model", "signal", "Pre-RALC ECE", "Post-RALC ECE", "Pre-RALC FD", "Post-RALC FD"])
    table_df.to_csv(OUT_PATH, index=False)
    print(f"Saved RALC table ({len(table_df)} rows) -> {OUT_PATH}")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
