"""
Tabulate generalised ECE / faithfulness divergence, pre- and post- in-domain
calibration, across the 5-seed x 4-model x 3-signal TruthfulQA rebuttal rerun.

Reads calibration_performance.csv from each
  {common_dir}/rebuttal_reruns_in_domain_calibration/truthful_qa/{model}/seed_{seed}/
(written by calibration/in_domain_calibration.py) and writes two CSVs next to
this script:
  truthful_qa_ece_fd_by_run.csv   - long format, one row per (model, seed, signal, stage)
  truthful_qa_ece_fd_summary.csv  - mean/std across the 5 seeds per (model, signal, stage)
"""
import os
import pandas as pd

MODELS = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "qwen/Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
SEEDS = [1, 2, 3, 4, 5]
SIGNALS = ["lc", "tp", "su"]

# (row-label-in-calibration_performance.csv, stage name for our tabulation)
STAGE_ROW_TEMPLATES = {
    "pre": "original_{signal}",
    "post_calibrated": "calibrated_{signal}",
    "post_rewritten": "calibrated_{signal}_rewritten_lc",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mirrors the common_dir logic in calibration/in_domain_calibration.py and
# run_seed_chain.sh: a dedicated re-runs tree, separate from production
# /hdd/ivny/results/.
if os.path.isdir("/hdd/ivny/re-runs/results"):
    common_dir = "/hdd/ivny/re-runs/results"
else:
    common_dir = os.path.join(SCRIPT_DIR, "results")


def calib_csv_path(model: str, seed: int) -> str:
    return os.path.join(
        common_dir, "in_domain_calibration", "truthful_qa",
        model, f"seed_{seed}", "calibration_performance.csv",
    )


def main():
    rows = []
    missing = []

    for model in MODELS:
        for seed in SEEDS:
            csv_path = calib_csv_path(model, seed)
            if not os.path.exists(csv_path):
                missing.append(csv_path)
                continue
            metrics_df = pd.read_csv(csv_path)
            metrics_lookup = dict(zip(metrics_df["metric"], metrics_df["value"]))

            for signal in SIGNALS:
                for stage, row_template in STAGE_ROW_TEMPLATES.items():
                    label = row_template.format(signal=signal)
                    ece_key = f"{label}_generalised_ECE"
                    fd_key = f"{label}_faithfulness_divergence"
                    if ece_key not in metrics_lookup or fd_key not in metrics_lookup:
                        continue
                    rows.append({
                        "model": model,
                        "seed": seed,
                        "signal": signal,
                        "stage": stage,
                        "generalised_ECE": metrics_lookup[ece_key],
                        "faithfulness_divergence": metrics_lookup[fd_key],
                    })

    if missing:
        print(f"WARNING: {len(missing)} expected calibration_performance.csv files not found:")
        for path in missing:
            print(f"  - {path}")

    if not rows:
        print("No results found; nothing to tabulate.")
        return

    by_run_df = pd.DataFrame(rows)
    by_run_path = os.path.join(SCRIPT_DIR, "truthful_qa_ece_fd_by_run.csv")
    by_run_df.to_csv(by_run_path, index=False)
    print(f"Saved long-format results ({len(by_run_df)} rows) -> {by_run_path}")

    summary_df = (
        by_run_df
        .groupby(["model", "signal", "stage"], sort=False)[["generalised_ECE", "faithfulness_divergence"]]
        .agg(["mean", "std", "count"])
    )
    summary_df.columns = ["_".join(col) for col in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_path = os.path.join(SCRIPT_DIR, "truthful_qa_ece_fd_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary ({len(summary_df)} rows) -> {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
