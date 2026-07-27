"""
Reshape eli5_ece_fd_by_run.csv into a paper-style table: one row per
(model, signal), with pre- and post- in-domain-calibration generalised ECE
and faithfulness divergence side by side.

"pre" = original (uncalibrated) signal; "post" = in-domain calibrated signal
(stage="post_calibrated" -- the rewritten-text stage, "post_rewritten", is
excluded here since it isn't part of this pre/post comparison).

Reads rebuttal/re-runs-long/eli5_ece_fd_by_run.csv (written by
aggregate_results.py) and writes rebuttal/re-runs-long/eli5_pre_post_table.csv
next to it.
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BY_RUN_PATH = os.path.join(SCRIPT_DIR, "eli5_ece_fd_by_run.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "eli5_pre_post_table.csv")

MODEL_ORDER = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen/Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
SIGNAL_ORDER = ["lc", "tp", "su"]


def main():
    df = pd.read_csv(BY_RUN_PATH)

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
                "Pre-calibration generalised ECE": pre["generalised_ECE"],
                "Post-calibration generalised ECE": post["generalised_ECE"],
                "Pre-calibration faithfulness divergence": pre["faithfulness_divergence"],
                "Post-calibration faithfulness divergence": post["faithfulness_divergence"],
            })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(OUT_PATH, index=False)
    print(f"Saved pre/post table ({len(table_df)} rows) -> {OUT_PATH}")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
