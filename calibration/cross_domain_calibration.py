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
from lm_conf.post_processing.metrics import dECE, dECE_point_mass, AUROC_point_mass, dAUROC, faithfulness_divergence, generalised_ece
from calibration.utils import *

argparser = argparse.ArgumentParser(description="Calibrate linguistic confidence lexicon using empirical data.")
argparser.add_argument("--train", type=str, required=True, help="Training dataset name")
argparser.add_argument("--test", type=str, required=True, help="Test dataset name")
argparser.add_argument("--model", type=str, required=True, help="Path to token probability cache as underlying signal.")
argparser.add_argument("--breakpoint", type=str, choices=["hedge", "eval", "metrics"], help="Whether to set a breakpoint after loading results for debugging.")
argparser.add_argument("--prompt_type", type=str, choices=["direct_qa", "hedged_qa"], default="direct_qa", help="Type of prompts to use.")
argparser.add_argument("--re_estimate_lc", type=bool, default=False, help="Whether to re-estimate the original linguistic confidence on the test set.")


eval_cfg = {
    "lc_eval_0": {
        "backend": "vllm",
        "name": "qwen/Qwen3-8B",
        "max_model_len": 2048,
        "temperature": 1.0,
        "max_tokens": 256,
        "repeat": 3,
        "reasoning_effort": None,
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

    prompt_type = args.prompt_type

    print("Breakpoint set to:", args.breakpoint)

    common_dir = "cross_domain_calibration_results"

    ling_train_path = get_latest_leaf_node(f"{common_dir}/results/{args.train}/{prompt_type}_unified_lc/{args.model}/")
    tp_train_path = get_latest_leaf_node(f"{common_dir}/results/{args.train}/{prompt_type}_unified_tp/{args.model}/")
    su_train_path = get_latest_leaf_node(f"{common_dir}/results/{args.train}/{prompt_type}_unified_su/{args.model}/")

    ling_test_path = get_latest_leaf_node(f"{common_dir}/results/{args.test}/{prompt_type}_unified_lc/{args.model}/")
    tp_test_path = get_latest_leaf_node(f"{common_dir}/results/{args.test}/{prompt_type}_unified_tp/{args.model}/")
    su_test_path = get_latest_leaf_node(f"{common_dir}/results/{args.test}/{prompt_type}_unified_su/{args.model}/")

    linguistic_confidence_cache: OrganisedOutputs = load_pickled_results(ling_train_path, "graded_outputs_0.pkl")
    tp_signal_cache: OrganisedOutputs = load_pickled_results(tp_train_path, "graded_outputs_0.pkl")
    su_signal_cache: OrganisedOutputs = load_pickled_results(su_train_path, "graded_outputs_0.pkl")

    save_dir = f"{common_dir}/{prompt_type}_cross_domain_calibration/{args.train}--{args.test}/{args.model}/"

    os.makedirs(save_dir, exist_ok=True)

    # --- Construct DataFrames for train and test sets from loaded OrganisedOutputs for all signals ---
    train_df = pd.DataFrame({
        "accuracy": [0.0 if (a is None or (isinstance(a, float) and np.isnan(a))) else a for a in linguistic_confidence_cache.accuracy_scores[0]],
        "original_response": linguistic_confidence_cache.extracted_answers[0],
        "original_lc": linguistic_confidence_cache.extracted_confidences[0],
        "original_tp": tp_signal_cache.extracted_confidences[0],
        "original_su": su_signal_cache.extracted_confidences[0],
    })
    train_df.dropna(inplace=True)

    # Load test set
    test_ling_cache: OrganisedOutputs = load_pickled_results(ling_test_path, "graded_outputs_0.pkl")
    test_tp_cache: OrganisedOutputs = load_pickled_results(tp_test_path, "graded_outputs_0.pkl")
    test_su_cache: OrganisedOutputs = load_pickled_results(su_test_path, "graded_outputs_0.pkl")

    test_df = pd.DataFrame({
        "accuracy": [0.0 if (a is None or (isinstance(a, float) and np.isnan(a))) else a for a in test_ling_cache.accuracy_scores[0]],
        "original_response": test_ling_cache.extracted_answers[0],
        "original_lc": test_ling_cache.extracted_confidences[0],
        "original_tp": test_tp_cache.extracted_confidences[0],
        "original_su": test_su_cache.extracted_confidences[0],
    })
    test_df.dropna(inplace=True)
    start_idx = int(len(test_df) * 0.3)
    test_df = test_df.iloc[start_idx:].reset_index(drop=True)
    print(f"Using last 70% of test_df: {len(test_df)} rows")

    if args.re_estimate_lc:
        reestimate_cache_file = os.path.join(save_dir, "reestimated_original_lc.pkl")
        estimated_lc = None

        if os.path.exists(reestimate_cache_file):
            with open(reestimate_cache_file, "rb") as f:
                cached_estimated_lc = pickle.load(f)
            if len(cached_estimated_lc) == len(test_df):
                estimated_lc = cached_estimated_lc
                print(f"Loaded cached re-estimated original_lc: {len(estimated_lc)}")
            else:
                print(
                    "Cache length mismatch for re-estimated original_lc "
                    f"({len(cached_estimated_lc)} != {len(test_df)}). Recomputing."
                )

        if estimated_lc is None:
            print("Re-estimating original linguistic confidence from test responses...")
            estimated_lc = estimate_linguistic_confidence(
                responses=test_df["original_response"].tolist(),
                target_means=[0.99] * len(test_df),
                evaluators_cfg=eval_cfg,
                evaluator_keys=evaluator_keys,
            )
            with open(reestimate_cache_file, "wb") as f:
                pickle.dump(estimated_lc, f)
            print(f"Saved re-estimated original_lc to {reestimate_cache_file}")

        test_df["original_lc"] = estimated_lc

    # === Cross-domain calibration for all signals (lc, tp, su) ===
    # 1. Train calibration map using cross_domain_numerical_post_hoc_calibration
    # 2. Obtain hedging words in parallel
    # 3. Optionally break after hedging
    # 4. Rewrite responses using hedging words
    # 5. Evaluate rewritten responses
    # 6. Optionally break after eval
    # 7. Compute metrics

    # 1. Cross-domain calibration for each signal (fit on train, apply to test)
    for signal, orig_col in zip(
        ["lc", "tp", "su"],
        ["original_lc", "original_tp", "original_su"]
    ):
        train_conf = np.array(train_df[orig_col])
        train_acc = np.array(train_df["accuracy"])
        test_conf = np.array(test_df[orig_col])
        cache_file = os.path.join(save_dir, f"calibrated_{signal}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                calibrated = pickle.load(f)
            if len(calibrated) != len(test_df):
                print(
                    f"Cached calibrated_{signal} length mismatch "
                    f"({len(calibrated)} != {len(test_df)}). Recomputing."
                )
                calibrated = cross_domain_numerical_post_hoc_calibration(
                    train_conf, train_acc, test_conf, method="platt_uni"
                )
                with open(cache_file, "wb") as f:
                    pickle.dump(calibrated, f)
            else:
                print(f"Loaded cached calibrated_{signal}: {len(calibrated)}")
        else:
            calibrated = cross_domain_numerical_post_hoc_calibration(
                train_conf, train_acc, test_conf, method="platt_uni"
            )
            with open(cache_file, "wb") as f:
                pickle.dump(calibrated, f)
            print(f"Saved calibrated_{signal} to {cache_file}")
        test_df[f"calibrated_{signal}"] = calibrated

    test_df.dropna(inplace=True)

    # 2. Obtain hedging words for each calibrated signal (on test set)
    for signal in ["lc", "tp", "su"]:
        conf_col = f"calibrated_{signal}"
        cache_file = os.path.join(save_dir, f"{conf_col}_hedging_words.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                hedging_words = pickle.load(f)
            print(f"Loaded cached hedging words for {conf_col}: {len(hedging_words)}")
        else:
            with Pool(cpu_count() - 1) as pool:
                hedging_words = list(
                    tqdm(
                        pool.imap(
                            obtain_hedging_words,
                            test_df[conf_col],
                            chunksize=16,
                        ),
                        total=len(test_df),
                        desc=f"Finding hedging words for {conf_col}",
                    )
                )
            with open(cache_file, "wb") as f:
                pickle.dump(hedging_words, f)
            print(f"Saved hedging words to {cache_file}")
        test_df[f"{conf_col}_hedging_words"] = hedging_words

    if args.breakpoint == "hedge":
        print("Hedging words obtained. Breaking here for debugging.")
        sys.exit()

    # 3. Rewrite responses using hedging words (on test set)
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
    rewrite_targets = []
    for signal in ["lc", "tp", "su"]:
        conf_col = f"calibrated_{signal}"
        rewrite_targets.append({
            "conf_col": conf_col,
            "hedges_col": f"{conf_col}_hedging_words",
            "rewrite_col": f"{conf_col}_rewritten_response",
            "cache_file": os.path.join(save_dir, f"{conf_col}_rewrites.pkl"),
            "label": conf_col,
        })

    cached_rewrites = {}
    pending_rewrite_jobs = []
    batched_rewrite_prompts = []

    for target in rewrite_targets:
        cache_file = target["cache_file"]
        rewrite_col = target["rewrite_col"]
        label = target["label"]

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                rewritten_answers = pickle.load(f)
            if len(rewritten_answers) == len(test_df):
                cached_rewrites[rewrite_col] = rewritten_answers
                print(f"Loaded cached rewrites for {label}: {len(rewritten_answers)}")
                continue
            print(
                f"Cache length mismatch for rewrites {label} "
                f"({len(rewritten_answers)} != {len(test_df)}). Recomputing."
            )

        prompts = [
            REWRITE_PROMPT.format(
                response=row["original_response"],
                hedges=format_hedges_for_prompt(row[target["hedges_col"]] or []),
            )
            for _, row in test_df.iterrows()
        ]

        start_idx = len(batched_rewrite_prompts)
        batched_rewrite_prompts.extend(prompts)
        end_idx = len(batched_rewrite_prompts)
        pending_rewrite_jobs.append({
            "rewrite_col": rewrite_col,
            "cache_file": cache_file,
            "label": label,
            "start": start_idx,
            "end": end_idx,
        })

    if pending_rewrite_jobs:
        print(
            f"Generating rewrites for {len(pending_rewrite_jobs)} uncached targets "
            f"in one batched call ({len(batched_rewrite_prompts)} prompts)."
        )
        model_manager = ModelManager(master_cfg=rewrite_cfg, model_config_type="rewrite_model")
        prompt_collection = PromptCollection(
            system_prompt="You are a linguistic expert.",
            context_texts=batched_rewrite_prompts,
        )
        outputs = model_manager.run_generation(prompt_collection)
        output_texts = outputs[0].output_texts if outputs else []
        if len(output_texts) < len(batched_rewrite_prompts):
            output_texts = output_texts + [""] * (len(batched_rewrite_prompts) - len(output_texts))
        else:
            output_texts = output_texts[:len(batched_rewrite_prompts)]

        cleaned_texts = [
            text.strip().split("assistantfinal")[-1].strip()
            for text in output_texts
        ]

        for job in pending_rewrite_jobs:
            rewritten_answers = cleaned_texts[job["start"]:job["end"]]
            cached_rewrites[job["rewrite_col"]] = rewritten_answers
            with open(job["cache_file"], "wb") as f:
                pickle.dump(rewritten_answers, f)
            print(f"Saved rewrites to {job['cache_file']}")

        del model_manager

    for target in rewrite_targets:
        test_df[target["rewrite_col"]] = cached_rewrites[target["rewrite_col"]]

    # 4. Evaluate rewritten responses using multiple evaluators (on test set)
    confidence_targets = []
    for signal in ["lc", "tp", "su"]:
        conf_col = f"calibrated_{signal}"
        confidence_targets.append({
            "conf_col": conf_col,
            "rewrite_col": f"{conf_col}_rewritten_response",
            "lc_col": f"{conf_col}_rewritten_lc",
            "cache_file": os.path.join(save_dir, f"{conf_col}_rewritten_confidence.pkl"),
            "label": conf_col,
            "target_mean_col": conf_col,
        })

    cached_confidences = {}
    pending_confidence_jobs = []
    batched_responses = []
    batched_target_means = []

    for target in confidence_targets:
        cache_file = target["cache_file"]
        lc_col = target["lc_col"]
        label = target["label"]

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                confidences = pickle.load(f)
            if len(confidences) == len(test_df):
                cached_confidences[lc_col] = confidences
                print(f"Loaded cached confidences for {label}: {len(confidences)}")
                continue
            print(
                f"Cache length mismatch for confidences {label} "
                f"({len(confidences)} != {len(test_df)}). Recomputing."
            )

        responses = test_df[target["rewrite_col"]].tolist()
        target_means = test_df[target["target_mean_col"]].tolist()
        start_idx = len(batched_responses)
        batched_responses.extend(responses)
        batched_target_means.extend(target_means)
        end_idx = len(batched_responses)
        pending_confidence_jobs.append({
            "lc_col": lc_col,
            "cache_file": cache_file,
            "label": label,
            "start": start_idx,
            "end": end_idx,
        })

    if pending_confidence_jobs:
        print(
            f"Estimating linguistic confidence for {len(pending_confidence_jobs)} uncached targets "
            f"in one batched call ({len(batched_responses)} prompts)."
        )
        batched_confidences = estimate_linguistic_confidence(
            batched_responses,
            batched_target_means,
            evaluators_cfg=eval_cfg,
            evaluator_keys=evaluator_keys,
        )

        for job in pending_confidence_jobs:
            confidences = batched_confidences[job["start"]:job["end"]]
            cached_confidences[job["lc_col"]] = confidences
            with open(job["cache_file"], "wb") as f:
                pickle.dump(confidences, f)
            print(f"Saved confidences to {job['cache_file']}")

    for target in confidence_targets:
        test_df[target["lc_col"]] = cached_confidences[target["lc_col"]]

    test_df.dropna(inplace=True)

    if args.breakpoint == "eval":
        print("Evaluation completed. Breaking here for debugging.")
        sys.exit()

    # 5. Compute metrics for all signals and rewritten outputs (on test set)
    print("Computing metrics...")
    from multiprocessing.pool import ThreadPool

    accuracy_list = test_df["accuracy"].tolist()

    def _compute_metrics(args_tuple):
        label, conf_list, answer_list = args_tuple
        organised_output = OrganisedOutputs(
            accuracy_scores=[accuracy_list],
            extracted_confidences=[conf_list],
            extracted_answers=[answer_list],
        )
        return [
            {"metric": f"{label}_generalised_ECE", "value": generalised_ece({}, organised_output)[0]},
            {"metric": f"{label}_faithfulness_divergence", "value": faithfulness_divergence({}, organised_output)[0]},
            {"metric": f"{label}_ece_mean", "value": dECE_point_mass({}, organised_output)[0]},
            {"metric": f"{label}_dAUROC", "value": dAUROC({}, organised_output)[0]},
            {"metric": f"{label}_auroc_mean", "value": AUROC_point_mass({}, organised_output)[0]},
        ]

    original_answers = test_df["original_response"].tolist()

    metric_jobs = [
        ("original_lc", test_df["original_lc"].tolist(), original_answers),
        ("original_tp", test_df["original_tp"].tolist(), original_answers),
        ("original_su", test_df["original_su"].tolist(), original_answers),
        ("calibrated_lc", test_df["calibrated_lc"].tolist(), original_answers),
        ("calibrated_tp", test_df["calibrated_tp"].tolist(), original_answers),
        ("calibrated_su", test_df["calibrated_su"].tolist(), original_answers),
    ]
    for signal in ["lc", "tp", "su"]:
        conf_col = f"calibrated_{signal}_rewritten_lc"
        rewrite_col = f"calibrated_{signal}_rewritten_response"
        metric_jobs.append(
            (
                conf_col,
                test_df[conf_col].tolist(),
                test_df[rewrite_col].tolist(),
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

    test_df.to_csv(details_csv, index=False)
    test_df.to_pickle(details_pkl)
    metrics_df.to_csv(metrics_csv, index=False)
    print(metrics_df)

