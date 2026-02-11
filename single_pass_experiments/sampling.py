import pickle
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy
from lm_conf.default_utils.custom_types import OrganisedOutputs, ModelOutputs
from lm_conf.confidence_metrics.distributionals import BetaDistribution
from lm_conf.post_processing.metrics import ece_scalar, auroc_scalar, dECE, dAUROC


parser = argparse.ArgumentParser(description="Evaluate linguistic calibration performance of a model")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'mmlu')")
parser.add_argument("--estimation_method", type=str, required=True, help="Estimation method to use")
parser.add_argument("--subset", type=int, required=True, help="Subset size to use for evaluation")

RESULTS_PATH = "/hdd/ivny/single_pass_results/{dataset}/{estimation_method}/"


def process_cache_path(cache_path: str, model_name: str, simulated_rounds: int) -> dict:
    lnll_cache_file_path = os.path.join(cache_path, "graded_outputs_0.pkl")

    with open(lnll_cache_file_path, "rb") as f:
        lnll_outputs: OrganisedOutputs = pickle.load(f)

    n_items = len(lnll_outputs.accuracy_scores[0])
    subset_size = 50
    subset_indices = np.random.choice(n_items, size=subset_size, replace=False)
    all_round_ece = []
    all_round_auroc = []
    all_round_dECE = []
    all_round_dAUROC = []

    for _ in range(simulated_rounds):
        conf_dists: list[BetaDistribution] = lnll_outputs.extracted_confidences[0]
        confidence_values = [float(dist.sample()[0]) for dist in conf_dists]
        round_organised_output: OrganisedOutputs = copy.deepcopy(lnll_outputs)
        round_organised_output.extracted_confidences = [confidence_values]

        subset_output: OrganisedOutputs = copy.deepcopy(round_organised_output)
        subset_output.accuracy_scores = [[round_organised_output.accuracy_scores[0][i] for i in subset_indices]]
        subset_output.extracted_confidences = [[round_organised_output.extracted_confidences[0][i] for i in subset_indices]]
        subset_output.extracted_answers = [[round_organised_output.extracted_answers[0][i] for i in subset_indices]]

        dist_subset_output: OrganisedOutputs = copy.deepcopy(lnll_outputs)
        dist_subset_output.accuracy_scores = [[lnll_outputs.accuracy_scores[0][i] for i in subset_indices]]
        dist_subset_output.extracted_confidences = [[lnll_outputs.extracted_confidences[0][i] for i in subset_indices]]
        dist_subset_output.extracted_answers = [[lnll_outputs.extracted_answers[0][i] for i in subset_indices]]

        print([x.sigma for x in conf_dists])

        ece = ece_scalar({}, subset_output)[0]
        auroc = auroc_scalar({}, subset_output)[0]
        dECE_value = dECE({}, dist_subset_output)[0]
        dAUROC_value = dAUROC({}, dist_subset_output)[0]

        all_round_ece.append(ece)
        all_round_auroc.append(auroc)

        all_round_dECE.append(dECE_value)
        all_round_dAUROC.append(dAUROC_value)

    return {
        "model_name": model_name,

        "ECE_mean": float(np.mean(all_round_ece)),
        "ECE_std": float(np.std(all_round_ece)),
        "dECE_mean": float(np.mean(all_round_dECE)),
        "dECE_std": float(np.std(all_round_dECE)),  
        
        "AUROC_mean": float(np.mean(all_round_auroc)),
        "AUROC_std": float(np.std(all_round_auroc)),
        "dAUROC_mean": float(np.mean(all_round_dAUROC)),
        "dAUROC_std": float(np.std(all_round_dAUROC)),
    }

if __name__ == "__main__":
    args = parser.parse_args()
    results_path = RESULTS_PATH.format(dataset=args.dataset, estimation_method=args.estimation_method)
    os.makedirs(results_path, exist_ok=True)

    # Find all leaf node cache paths and extract model names
    cache_base_path = f"/hdd/ivny/results/{args.dataset}/{args.estimation_method}/"
    
    cache_paths_and_models = []
    
    if os.path.exists(cache_base_path):
        for root, dirs, files in os.walk(cache_base_path):
            # Check if this is a leaf node (no subdirectories)
            if not dirs and files:
                # Extract model name from path: /base/provider/model_name/timestamp/
                # root will be: /hdd/ivny/results/{dataset}/dist_lnll/provider/model_name/timestamp/
                parts = root.replace(cache_base_path, "").split(os.sep)
                parts = [p for p in parts if p]  # Remove empty strings
                
                if len(parts) >= 2:
                    provider = parts[0]  # e.g., "openai"
                    model_name_part = parts[1]  # e.g., "gpt-oss-20b"
                    model_name = f"{provider}/{model_name_part}"  # e.g., "openai/gpt-oss-20b"
                    cache_paths_and_models.append((root, model_name))
        
        if not cache_paths_and_models:
            raise RuntimeError(f"No leaf node directories found in {cache_base_path}")
    else:
        raise RuntimeError(f"Cache path does not exist: {cache_base_path}")
    
    all_results = []
    
    simulated_rounds = 10

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_cache_path, cache_path, model_name, simulated_rounds)
            for cache_path, model_name in cache_paths_and_models
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing cache paths",
        ):
            all_results.append(future.result())
    
    pd.DataFrame(all_results).to_csv(f"{results_path}/results.csv", index=False)