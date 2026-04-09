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

    common_dir = "/hdd/ivny"

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
    start_idx = int(len(test_df) * 0.2)
    test_df = test_df.iloc[start_idx:].reset_index(drop=True)
    print(f"Using last 80% of test_df: {len(test_df)} rows")

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
        calibrated = cross_domain_numerical_post_hoc_calibration(
            train_conf, train_acc, test_conf, method="platt_uni"
        )
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
    for signal in ["lc", "tp", "su"]:
        conf_col = f"calibrated_{signal}"
        hedges_col = f"{conf_col}_hedging_words"
        rewrite_col = f"{conf_col}_rewritten_response"
        cache_file = os.path.join(save_dir, f"{conf_col}_rewrites.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                rewritten_answers = pickle.load(f)
            print(f"Loaded cached rewrites for {conf_col}: {len(rewritten_answers)}")
        else:
            prompts = [
                REWRITE_PROMPT.format(
                    response=row["original_response"],
                    hedges=", ".join(row[hedges_col] or []),
                )
                for _, row in test_df.iterrows()
            ]
            prompt_collection = PromptCollection(
                system_prompt="You are a linguistic expert.",
                context_texts=prompts,
            )
            model_manager = ModelManager(master_cfg=rewrite_cfg, model_config_type="rewrite_model")
            outputs = model_manager.run_generation(prompt_collection)
            output_texts = outputs[0].output_texts if outputs else []
            if not output_texts:
                output_texts = [""] * len(test_df)
            rewritten_answers = [
                text.strip().split("assistantfinal")[-1].strip()
                for text in output_texts
            ]
            with open(cache_file, "wb") as f:
                pickle.dump(rewritten_answers, f)
            print(f"Saved rewrites to {cache_file}")
            del model_manager
        test_df[rewrite_col] = rewritten_answers

    # 4. Evaluate rewritten responses using multiple evaluators (on test set)
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
    for signal in ["lc", "tp", "su"]:
        conf_col = f"calibrated_{signal}"
        rewrite_col = f"{conf_col}_rewritten_response"
        lc_col = f"{conf_col}_rewritten_lc"
        cache_file = os.path.join(save_dir, f"{conf_col}_rewritten_confidence.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                confidences = pickle.load(f)
            print(f"Loaded cached confidences for {conf_col}: {len(confidences)}")
        else:
            responses = test_df[rewrite_col].tolist()
            confidences = estimate_linguistic_confidence(
                responses,
                evaluators_cfg=eval_cfg,
                evaluator_keys=evaluator_keys,
            )
            with open(cache_file, "wb") as f:
                pickle.dump(confidences, f)
            print(f"Saved confidences to {cache_file}")
        test_df[lc_col] = confidences

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


    # df = pd.DataFrame({
    #     "accuracy": [0.0 if (a is None or (isinstance(a, float) and np.isnan(a))) else a for a in linguistic_confidence_cache.accuracy_scores[0]],
    #     "original_response": linguistic_confidence_cache.extracted_answers[0],
    #     "original_lc": linguistic_confidence_cache.extracted_confidences[0],
    #     "original_tp": tp_signal_cache.extracted_confidences[0],
    #     "original_su": su_signal_cache.extracted_confidences[0],
    # })

    # df.dropna(inplace=True)

    # # df = df.iloc[:100]  # limit to 100 samples for faster debugging; remove or adjust as needed

    # # apply numerical post hoc calibration
    # calibrated_lc = in_domain_numerical_post_hoc_calibration(
    #     np.array(df["original_lc"]),
    #     np.array(df["accuracy"]),
    #     method="platt_uni"
    # )

    # calibrated_tp = in_domain_numerical_post_hoc_calibration(
    #     np.array(df["original_tp"]),
    #     np.array(df["accuracy"]),
    #     method="platt_uni"
    # )

    # calibrated_su = in_domain_numerical_post_hoc_calibration(
    #     np.array(df["original_su"]),
    #     np.array(df["accuracy"]),
    #     method="platt_uni"
    # )

    # df["calibrated_lc"] = calibrated_lc
    # df["calibrated_tp"] = calibrated_tp
    # df["calibrated_su"] = calibrated_su
    
    # df.dropna(inplace=True)

    # os.makedirs(save_dir, exist_ok=True)

    # # check if hedging words have been cached
    # for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
    #     cache_file = os.path.join(save_dir, f"{conf_method}_hedging_words.pkl")
    #     if os.path.exists(cache_file):
    #         with open(cache_file, "rb") as f:
    #             hedging_words = pickle.load(f)
    #         print(f"Loaded cached hedging words for {conf_method}: {hedging_words}")
    #     else:
    #         with Pool(cpu_count() - 1) as pool:
    #             hedging_words = list(
    #                 tqdm(
    #                     pool.imap(
    #                         obtain_hedging_words,
    #                         df[conf_method],
    #                         chunksize=16,
    #                     ),
    #                     total=len(df),
    #                     desc=f"Finding hedging words for {conf_method}",
    #                 )
    #             )
    #         with open(cache_file, "wb") as f:
    #             pickle.dump(hedging_words, f)
    #         print(f"Saved hedging words to {cache_file}")

    #     df[f"{conf_method}_hedging_words"] = hedging_words

    # if args.breakpoint == "hedge":
    #     print("Hedging words obtained. Breaking here for debugging.")
    #     sys.exit()

    # rewrite_cfg = {
    #     "rewrite_model": {
    #         "backend": "vllm",
    #         "name": "openai/gpt-oss-20b",
    #         "max_model_len": 2048,
    #         "temperature": 1.0,
    #         "max_tokens": 512,
    #         "reasoning_effort": "low",
    #     }
    # }

    # # rewrite outputs based on calibrated signals
    # for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
    #     model_manager: ModelManager = ModelManager(master_cfg=rewrite_cfg, model_config_type="rewrite_model")
    #     cache_file = os.path.join(save_dir, f"{conf_method}_rewrites.pkl")
    #     if os.path.exists(cache_file):
    #         with open(cache_file, "rb") as f:
    #             rewritten_answers = pickle.load(f)
    #         print(f"Loaded cached rewrites for {conf_method}: {len(rewritten_answers)}")
    #     else:
    #         prompts = []
    #         for _, row in df.iterrows():
    #             hedges = row[f"{conf_method}_hedging_words"] or []
    #             prompts.append(
    #                 REWRITE_PROMPT.format(
    #                     response=row["original_response"],
    #                     hedges=", ".join(hedges),
    #                 )
    #             )

    #         prompt_collection = PromptCollection(
    #             system_prompt="You are a linguistic expert.",
    #             context_texts=prompts,
    #         )
    #         outputs = model_manager.run_generation(prompt_collection)
    #         output_texts = outputs[0].output_texts if outputs else []
    #         if not output_texts:
    #             output_texts = [""] * len(df)
    #         rewritten_answers = [
    #             text.strip().split("assistantfinal")[-1].strip()
    #             for text in output_texts
    #         ]
    #         with open(cache_file, "wb") as f:
    #             pickle.dump(rewritten_answers, f)
    #         print(f"Saved rewrites to {cache_file}")

    #     df[f"{conf_method}_rewritten_response"] = rewritten_answers

    #     del model_manager
    
    # # evaluate rewritten outputs
    # eval_cfg = {
    #     "lc_eval_0": {
    #         "backend": "vllm",
    #         "name": "openai/gpt-oss-20b",
    #         "max_model_len": 2048,
    #         "temperature": 1.0,
    #         "max_tokens": 256,
    #         "repeat": 3,
    #         "reasoning_effort": "low",
    #     },
    #     "lc_eval_1": {
    #         "backend": "vllm",
    #         "name": "meta-llama/Llama-3.1-8B-Instruct",
    #         "max_model_len": 2048,
    #         "temperature": 1.0,
    #         "max_tokens": 256,
    #         "repeat": 3,
    #         "reasoning_effort": "low",
    #     },
    #     "lc_eval_2": {
    #         "backend": "vllm",
    #         "name": "mistralai/Mistral-7B-Instruct-v0.3",
    #         "max_model_len": 2048,
    #         "temperature": 1.0,
    #         "max_tokens": 256,
    #         "repeat": 3,
    #         "reasoning_effort": "low",
    #     },
    # }

    # evaluator_keys = ["lc_eval_0", "lc_eval_1", "lc_eval_2"]

    # for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
    #     cache_file = os.path.join(save_dir, f"{conf_method}_rewritten_confidence.pkl")
    #     if os.path.exists(cache_file):
    #         with open(cache_file, "rb") as f:
    #             confidences = pickle.load(f)
    #         print(f"Loaded cached confidences for {conf_method}: {len(confidences)}")
    #     else:
    #         responses = df[f"{conf_method}_rewritten_response"].tolist()
    #         confidences = estimate_linguistic_confidence(
    #             responses,
    #             evaluators_cfg=eval_cfg,
    #             evaluator_keys=evaluator_keys,
    #         )
    #         with open(cache_file, "wb") as f:
    #             pickle.dump(confidences, f)
    #         print(f"Saved confidences to {cache_file}")
    #     df[f"{conf_method}_rewritten_lc"] = confidences

    # df.dropna(inplace=True)
    
    # if args.breakpoint == "eval":
    #     print("Evaluation completed. Breaking here for debugging.")
    #     sys.exit()


    # print("Computing metrics...")
    # from multiprocessing.pool import ThreadPool

    # accuracy_list = df["accuracy"].tolist()

    # def _compute_metrics(args_tuple):
    #     label, conf_list, answer_list = args_tuple
    #     organised_output = OrganisedOutputs(
    #         accuracy_scores=[accuracy_list],
    #         extracted_confidences=[conf_list],
    #         extracted_answers=[answer_list],
    #     )
    #     return [
    #         {"metric": f"{label}_generalised_ECE", "value": generalised_ece({}, organised_output)[0]},
    #         {"metric": f"{label}_faithfulness_divergence", "value": faithfulness_divergence({}, organised_output)[0]},
    #         {"metric": f"{label}_ece_mean", "value": dECE_point_mass({}, organised_output)[0]},
    #         {"metric": f"{label}_dAUROC", "value": dAUROC({}, organised_output)[0]},
    #         {"metric": f"{label}_auroc_mean", "value": AUROC_point_mass({}, organised_output)[0]},
    #     ]

    # original_answers = df["original_response"].tolist()

    # metric_jobs = [
    #     ("original_lc", df["original_lc"].tolist(), original_answers),
    #     ("original_tp", df["original_tp"].tolist(), original_answers),
    #     ("original_su", df["original_su"].tolist(), original_answers),
    #     ("calibrated_lc", df["calibrated_lc"].tolist(), original_answers),
    #     ("calibrated_tp", df["calibrated_tp"].tolist(), original_answers),
    #     ("calibrated_su", df["calibrated_su"].tolist(), original_answers),
    # ]

    # for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
    #     metric_jobs.append(
    #         (
    #             f"{conf_method}_rewritten_lc",
    #             df[f"{conf_method}_rewritten_lc"].tolist(),
    #             df[f"{conf_method}_rewritten_response"].tolist(),
    #         )
    #     )

    # metrics_rows = []
    # with ThreadPool(min(8, len(metric_jobs))) as pool:
    #     for rows in tqdm(
    #         pool.imap(_compute_metrics, metric_jobs),
    #         total=len(metric_jobs),
    #         desc="Computing metrics",
    #     ):
    #         metrics_rows.extend(rows)

    # metrics_df = pd.DataFrame(metrics_rows)

    # details_csv = os.path.join(save_dir, "calibration_details.csv")
    # details_pkl = os.path.join(save_dir, "calibration_details.pkl")
    # metrics_csv = os.path.join(save_dir, "calibration_performance.csv")

    # df.to_csv(details_csv, index=False)
    # df.to_pickle(details_pkl)
    # metrics_df.to_csv(metrics_csv, index=False)
    # print(metrics_df)
    
            