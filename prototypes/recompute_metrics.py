import pickle
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from lm_conf.post_processing.metrics import dECE_point_mass, AUROC_point_mass, dECE, dAUROC, accuracy_scalar_with_abstention, generalised_ece
from lm_conf.confidence_metrics.distributionals import BetaDistribution
from lm_conf.default_utils.custom_types import OrganisedOutputs


def compute_metrics(organised_output: OrganisedOutputs) -> dict:
    """Compute all metrics for a single OrganisedOutputs object."""
    cfg = {}
    
    # Compute each metric once and store in variables
    acc_result = accuracy_scalar_with_abstention(cfg, organised_output)
    accuracy = acc_result[0] if acc_result else None
    
    dece_result = dECE(cfg, organised_output)
    dece = dece_result[0] if dece_result else None
    
    dece_pm_result = dECE_point_mass(cfg, organised_output)
    dece_pm = dece_pm_result[0] if dece_pm_result else None

    generalised_ece_result = generalised_ece(cfg, organised_output)
    generalised_ece_val = generalised_ece_result[0] if generalised_ece_result else None
    
    dauroc_result = dAUROC(cfg, organised_output)
    dauroc = dauroc_result[0] if dauroc_result else None
    
    auroc_pm_result = AUROC_point_mass(cfg, organised_output)
    auroc_pm = auroc_pm_result[0] if auroc_pm_result else None
    
    results = {
        "accuracy": accuracy,
        "generalised_ece": generalised_ece_val,
        "dECE": dece,
        "dECE_point_mass": dece_pm,
        "dAUROC": dauroc,
        "AUROC_point_mass": auroc_pm,
    }
    return results


if __name__ == "__main__":
    base_path = "/hdd/ivny/results/"
    
    # Find all leaf nodes
    leaf_nodes = []
    
    for root, dirs, files in os.walk(base_path):
        # Check if this is a leaf node (no subdirectories)
        if not dirs and files:
            leaf_nodes.append(root)
    
    print(f"Found {len(leaf_nodes)} leaf nodes")
    
    # Process each leaf node
    for leaf_node in tqdm(leaf_nodes, desc="Processing leaf nodes"):

        print("Processing leaf node:", leaf_node)

        results_list = []
        
        # Check for graded_outputs_0.pkl and graded_outputs_1.pkl
        for pkl_idx in [0, 1]:
            pkl_path = os.path.join(leaf_node, f"graded_outputs_{pkl_idx}.pkl")
            
            if not os.path.exists(pkl_path):
                continue
            
            try:
                with open(pkl_path, "rb") as f:
                    organised_output: OrganisedOutputs = pickle.load(f)
                
                # Compute metrics
                metrics = compute_metrics(organised_output)
                # metrics["pkl_file"] = f"graded_outputs_{pkl_idx}.pkl"
                results_list.append(metrics)
            except Exception as e:
                print(f"Error processing {pkl_path}: {e}")
                continue
        
        # Save to CSV if we have results
        if results_list:
            df = pd.DataFrame(results_list)
            csv_path = os.path.join(leaf_node, "eval_metrics.csv")
            print(df)
            df.to_csv(csv_path, index=False)
            print(f"Saved metrics to {csv_path}")
            
            # Compute and save summary statistics
            summary_rows = []
            for col in df.columns:
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                summary_rows.append({
                    "metric": col,
                    "mean": numeric_vals.mean(),
                    "std": numeric_vals.std()
                })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_csv_path = os.path.join(leaf_node, "eval_metrics_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Saved summary metrics to {summary_csv_path}")

