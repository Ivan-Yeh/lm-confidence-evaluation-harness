from lm_conf.post_processing.metrics import (
    accuracy_scalar_with_abstention,
    faithfulness_divergence,
    generalised_ece,
    dECE_point_mass,
    dAUROC,
    AUROC_point_mass
)
from lm_conf.default_utils.custom_types import OrganisedOutputs
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd

# find all leaf nodes in 
results_path = "/hdd/ivny/results"

# find all leaf nodes in results_path
def find_leaf_nodes(path):
    leaf_nodes = []
    for root, dirs, files in os.walk(path):
        if not dirs and files:
            leaf_nodes.append(root)
    return leaf_nodes

def recompute_leaf_metrics(leaf: str) -> str:
    if not os.path.exists(os.path.join(leaf, "graded_outputs_0.pkl")) or not os.path.exists(os.path.join(leaf, "eval_metrics.csv")):
        return f"Required files not found in {leaf}"


    metrics_df = pd.read_csv(os.path.join(leaf, "eval_metrics.csv"))
    if metrics_df.columns.tolist() == ["accuracy_scalar_with_abstention", "faithfulness_divergence", "generalised_ece", "dECE_point_mass", "dAUROC", "AUROC_point_mass"]:
        return f"Metrics already computed for {leaf}, skipping."


    organised_output_path = os.path.join(leaf, "graded_outputs_0.pkl")

    with open(organised_output_path, "rb") as f:
        organised_output: OrganisedOutputs = pd.read_pickle(f)

    for filename in ["dECE_reliability_diagram_0.pdf", "eval_metrics_summary.csv", "eval_metrics.csv", "gECE_reliability_diagram_0.pdf"]:
        file_path = os.path.join(leaf, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    cfg = {"results_path": leaf}

    pd.DataFrame({
        "accuracy_scalar_with_abstention": accuracy_scalar_with_abstention(cfg, organised_output),
        "faithfulness_divergence": faithfulness_divergence(cfg, organised_output),
        "generalised_ece": generalised_ece(cfg, organised_output),
        "dECE_point_mass": dECE_point_mass(cfg, organised_output),
        "dAUROC": dAUROC(cfg, organised_output),
        "AUROC_point_mass": AUROC_point_mass(cfg, organised_output),
    }).to_csv(os.path.join(leaf, "eval_metrics.csv"), index=False)

    return f"Recomputed metrics for {leaf}"


def main() -> None:
    leaves = find_leaf_nodes(results_path)
    max_workers = min(os.cpu_count() or 1, len(leaves))

    if max_workers == 0:
        print(f"No leaf nodes found under {results_path}")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(recompute_leaf_metrics, leaf): leaf for leaf in leaves}
        for future in as_completed(futures):
            leaf = futures[future]
            try:
                print(future.result())
            except Exception as exc:
                print(f"Failed to recompute metrics for {leaf}: {exc}")


if __name__ == "__main__":
    main()

    
