import pandas as pd
import os
import pickle
from calibration.utils import estimate_linguistic_confidence

results_dir = "/hdd/ivny/results"

# list all leaf nodes in results_dir
leaf_dirs = []
for root, dirs, files in os.walk(results_dir):
    if not dirs:  # if there are no subdirectories, it's a leaf node
        leaf_dirs.append(root)


all_records = []
for leaf_dir in leaf_dirs:
    try:
        _, _, _, _, dataset_name, estimation_method, model_fam, model_name, _ = leaf_dir.split("/")
        save_path = f"/hdd/ivny/linguistic_confidence_prompt_test/{dataset_name}/{estimation_method}/{model_fam}/{model_name}"
        if estimation_method in ["dist_linguistic_confidence", "dist_lnll"]:
            os.makedirs(save_path, exist_ok=True)
            if os.path.exists(os.path.join(save_path, "linguistic_confidence.csv")):
                print(f"File already exists for {save_path}, skipping...")
                continue
            with open(os.path.join(leaf_dir, "graded_outputs_0.pkl"), "rb") as f:
                organised_outputs = pickle.load(f)
            responses = organised_outputs.extracted_answers[0]
            if len(responses[0].strip()) == 1:
                responses = ["Answer: " + r.strip() for r in responses]
            linguistic_confidence = estimate_linguistic_confidence(responses) # list of beta dists
            lc_means = [lc.mu for lc in linguistic_confidence] # list of means
            lc_stds = [lc.sigma for lc in linguistic_confidence] # list of means
            lc_alphas = [lc.alpha_param for lc in linguistic_confidence] # list of alphas
            lc_betas = [lc.beta_param for lc in linguistic_confidence] # list of betas

            df = pd.DataFrame({
                "response": responses,
                "linguistic_confidence_mean": lc_means,
                "linguistic_confidence_std": lc_stds,
                "linguistic_confidence_alpha": lc_alphas,
                "linguistic_confidence_beta": lc_betas,
            })
            df.to_csv(os.path.join(save_path, "linguistic_confidence.csv"), index=False)
            print(f"Saved linguistic confidence results to {save_path}")
    except Exception as e:
        print(f"Error processing {leaf_dir}: {e}")
        print("Continuing to next directory...")

