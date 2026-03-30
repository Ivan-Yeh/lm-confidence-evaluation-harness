import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from lm_conf.default_utils.custom_types import OrganisedOutputs, PromptCollection
from lm_conf.models.model_manager import ModelManager
from lm_conf.post_processing.metrics import dECE, dECE_point_mass, AUROC_point_mass, dAUROC, generalised_ece
from calibration.utils import *

argparser = argparse.ArgumentParser(description="Calibrate linguistic confidence lexicon using empirical data.")
argparser.add_argument("--dataset", type=str, required=True, help="Dataset name")
argparser.add_argument("--model", type=str, required=True, help="Path to token probability cache as underlying signal.")
argparser.add_argument("--breakpoint", type=str, choices=["hedge", "eval", "metrics"], help="Whether to set a breakpoint after loading results for debugging.")

def load_pickled_results(dir, filename):
    with open(os.path.join(dir, filename), "rb") as f:
        return pickle.load(f)
    
def get_latest_leaf_node(dir) -> str:
    """Get the latest leaf node directory in a given directory."""
    subdirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {dir}")
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

if __name__ == "__main__":
    args = argparser.parse_args()

    print("Breakpoint set to:", args.breakpoint)

    common_dir = "/hdd/ivny"

    ling_path = get_latest_leaf_node(f"{common_dir}/results/{args.dataset}/hedged_qa_unified_lc/{args.model}/")
    tp_path = get_latest_leaf_node(f"{common_dir}/results/{args.dataset}/hedged_qa_unified_tp/{args.model}/")
    su_path = get_latest_leaf_node(f"{common_dir}/results/{args.dataset}/hedged_qa_unified_su/{args.model}/")

    linguistic_confidence_cache: OrganisedOutputs = load_pickled_results(ling_path, "graded_outputs_0.pkl")
    tp_signal_cache: OrganisedOutputs = load_pickled_results(tp_path, "graded_outputs_0.pkl")
    su_signal_cache: OrganisedOutputs = load_pickled_results(su_path, "graded_outputs_0.pkl")

    save_dir = f"{common_dir}/in_domain_calibration/{args.dataset}/{args.model}/"

    df = pd.DataFrame({
        "accuracy": [0.0 if (a is None or (isinstance(a, float) and np.isnan(a))) else a for a in linguistic_confidence_cache.accuracy_scores[0]],
        "original_response": linguistic_confidence_cache.extracted_answers[0],
        "original_lc": linguistic_confidence_cache.extracted_confidences[0],
        "original_tp": tp_signal_cache.extracted_confidences[0],
        "original_su": su_signal_cache.extracted_confidences[0],
    })

    df.dropna(inplace=True)

    # df = df.iloc[:100]  # limit to 100 samples for faster debugging; remove or adjust as needed

    # apply numerical post hoc calibration
    calibrated_lc = in_domain_numerical_post_hoc_calibration(
        np.array(df["original_lc"]),
        np.array(df["accuracy"]),
        method="platt_uni"
    )

    calibrated_tp = in_domain_numerical_post_hoc_calibration(
        np.array(df["original_tp"]),
        np.array(df["accuracy"]),
        method="platt_uni"
    )

    calibrated_su = in_domain_numerical_post_hoc_calibration(
        np.array(df["original_su"]),
        np.array(df["accuracy"]),
        method="platt_uni"
    )

    df["calibrated_lc"] = calibrated_lc
    df["calibrated_tp"] = calibrated_tp
    df["calibrated_su"] = calibrated_su
    
    df.dropna(inplace=True)

    os.makedirs(save_dir, exist_ok=True)

    # check if hedging words have been cached
    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        cache_file = os.path.join(save_dir, f"{conf_method}_hedging_words.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                hedging_words = pickle.load(f)
            print(f"Loaded cached hedging words for {conf_method}: {hedging_words}")
        else:
            with Pool(cpu_count() - 1) as pool:
                hedging_words = list(
                    tqdm(
                        pool.imap(
                            obtain_hedging_words,
                            df[conf_method],
                            chunksize=16,
                        ),
                        total=len(df),
                        desc=f"Finding hedging words for {conf_method}",
                    )
                )
            with open(cache_file, "wb") as f:
                pickle.dump(hedging_words, f)
            print(f"Saved hedging words to {cache_file}")

        df[f"{conf_method}_hedging_words"] = hedging_words

    if args.breakpoint == "hedge":
        print("Hedging words obtained. Breaking here for debugging.")
        sys.exit()

    rewrite_cfg = {
        "rewrite_model": {
            "backend": "vllm",
            "name": "openai/gpt-oss-20b",
            "max_model_len": 2048,
            "temperature": 1.0,
            "max_tokens": 512,
            "reasoning_effort": "low",
        }
    }

    # rewrite outputs based on calibrated signals
    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        model_manager: ModelManager = ModelManager(master_cfg=rewrite_cfg, model_config_type="rewrite_model")
        cache_file = os.path.join(save_dir, f"{conf_method}_rewrites.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                rewritten_answers = pickle.load(f)
            print(f"Loaded cached rewrites for {conf_method}: {len(rewritten_answers)}")
        else:
            prompts = []
            for _, row in df.iterrows():
                hedges = row[f"{conf_method}_hedging_words"] or []
                prompts.append(
                    REWRITE_PROMPT.format(
                        response=row["original_response"],
                        hedges=", ".join(hedges),
                    )
                )

            prompt_collection = PromptCollection(
                system_prompt="You are a linguistic expert.",
                context_texts=prompts,
            )
            outputs = model_manager.run_generation(prompt_collection)
            output_texts = outputs[0].output_texts if outputs else []
            if not output_texts:
                output_texts = [""] * len(df)
            rewritten_answers = [
                text.strip().split("assistantfinal")[-1].strip()
                for text in output_texts
            ]
            with open(cache_file, "wb") as f:
                pickle.dump(rewritten_answers, f)
            print(f"Saved rewrites to {cache_file}")

        df[f"{conf_method}_rewritten_response"] = rewritten_answers

        del model_manager
    
    # evaluate rewritten outputs
    eval_cfg = {
        "lc_eval_0": {
            "backend": "vllm",
            "name": "openai/gpt-oss-20b",
            "max_model_len": 2048,
            "temperature": 1.0,
            "max_tokens": 256,
            "repeat": 3,
            "reasoning_effort": "low",
        },
        "lc_eval_1": {
            "backend": "vllm",
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "max_model_len": 2048,
            "temperature": 1.0,
            "max_tokens": 256,
            "repeat": 3,
            "reasoning_effort": "low",
        },
        "lc_eval_2": {
            "backend": "vllm",
            "name": "mistralai/Mistral-7B-Instruct-v0.3",
            "max_model_len": 2048,
            "temperature": 1.0,
            "max_tokens": 256,
            "repeat": 3,
            "reasoning_effort": "low",
        },
    }

    evaluator_keys = ["lc_eval_0", "lc_eval_1", "lc_eval_2"]

    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        cache_file = os.path.join(save_dir, f"{conf_method}_rewritten_confidence.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                confidences = pickle.load(f)
            print(f"Loaded cached confidences for {conf_method}: {len(confidences)}")
        else:
            responses = df[f"{conf_method}_rewritten_response"].tolist()
            confidences = estimate_linguistic_confidence(
                responses,
                evaluators_cfg=eval_cfg,
                evaluator_keys=evaluator_keys,
            )
            with open(cache_file, "wb") as f:
                pickle.dump(confidences, f)
            print(f"Saved confidences to {cache_file}")
        df[f"{conf_method}_rewritten_lc"] = confidences

    df.dropna(inplace=True)
    
    if args.breakpoint == "eval":
        print("Evaluation completed. Breaking here for debugging.")
        sys.exit()


    print("Computing metrics...")
    from multiprocessing.pool import ThreadPool

    accuracy_list = df["accuracy"].tolist()

    def _compute_metrics(args_tuple):
        label, conf_list, answer_list = args_tuple
        organised_output = OrganisedOutputs(
            accuracy_scores=[accuracy_list],
            extracted_confidences=[conf_list],
            extracted_answers=[answer_list],
        )
        return [
            {"metric": f"{label}_generalised_ECE", "value": generalised_ece({}, organised_output)[0]},
            {"metric": f"{label}_dECE", "value": dECE({}, organised_output)[0]},
            {"metric": f"{label}_dECE_pt", "value": dECE_point_mass({}, organised_output)[0]},
            {"metric": f"{label}_dAUROC", "value": dAUROC({}, organised_output)[0]},
            {"metric": f"{label}_auroc_pt", "value": AUROC_point_mass({}, organised_output)[0]},
        ]

    original_answers = df["original_response"].tolist()

    metric_jobs = [
        ("original_lc", df["original_lc"].tolist(), original_answers),
        ("original_tp", df["original_tp"].tolist(), original_answers),
        ("original_su", df["original_su"].tolist(), original_answers),
        ("calibrated_lc", df["calibrated_lc"].tolist(), original_answers),
        ("calibrated_tp", df["calibrated_tp"].tolist(), original_answers),
        ("calibrated_su", df["calibrated_su"].tolist(), original_answers),
    ]

    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        metric_jobs.append(
            (
                f"{conf_method}_rewritten_lc",
                df[f"{conf_method}_rewritten_lc"].tolist(),
                df[f"{conf_method}_rewritten_response"].tolist(),
            )
        )

    metrics_rows = []
    with ThreadPool(min(8, len(metric_jobs))) as pool:
        for rows in tqdm(
            pool.imap(_compute_metrics, metric_jobs),
            total=len(metric_jobs),
            desc="Computing metrics",
        ):
            metrics_rows.extend(rows)

    metrics_df = pd.DataFrame(metrics_rows)

    details_csv = os.path.join(save_dir, "calibration_details.csv")
    details_pkl = os.path.join(save_dir, "calibration_details.pkl")
    metrics_csv = os.path.join(save_dir, "calibration_performance.csv")

    df.to_csv(details_csv, index=False)
    df.to_pickle(details_pkl)
    metrics_df.to_csv(metrics_csv, index=False)
    print(metrics_df)
    
            