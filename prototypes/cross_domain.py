import numpy as np
import pandas as pd
import pickle
import argparse
import os
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams
from multiprocessing import Pool, cpu_count
from lm_conf.default_utils.custom_types import OrganisedOutputs
from lm_conf.post_processing.metrics import dECE, dECE_point_mass, AUROC_point_mass, dAUROC, generalised_ece
from calibration.utils import *


# Cross-domain numerical post-hoc calibration using isotonic regression or Platt scaling
# This answers the question of whether confidence signals and calibration maps are transferable across domains

argparser = argparse.ArgumentParser(description="Calibrate linguistic confidence lexicon using empirical data.")
argparser.add_argument("--cache_parent_path", type=str, required=True) # e.g. /hdd/ivny/results/
argparser.add_argument("--train_dataset", type=str, required=True) # mmlu, trivia_qa, squadv2
argparser.add_argument("--test_dataset", type=str, required=True) # mmlu, trivia_qa, squadv2
argparser.add_argument("--model", type=str, required=True, help="Target Model") # meta-llama/Llama-3.1-8B-Instruct
argparser.add_argument("--estimation_method", type=str, required=True, help="Estimation Method") # dist_lnll, dist_linguistic_confidence, dist_semantic_uncertainty
argparser.add_argument("--modifier", type=str, required=True, help="LLM calibrator (for rewriting raw responses).")
argparser.add_argument("--post-hoc-method", type=str, choices=["isotonic", "platt_uni", "platt_bi"], default="isotonic", help="Post-hoc calibration method.")
argparser.add_argument("--answer-prepend-test", type=str, required=False, help="Prefix to prepend to answers.", default="")
argparser.add_argument("--breakpoint", type=str, choices=["hedge", "eval", "metrics"])

if __name__ == "__main__":
    
    args = argparser.parse_args()
    print("Breakpoint set to:", args.breakpoint)

    # Construct cache paths from cache_parent_path
    # cache_parent_path is like /hdd/ivny/results/
    train_cache_path = os.path.join(args.cache_parent_path, args.train_dataset, args.estimation_method, args.model)
    test_cache_path = os.path.join(args.cache_parent_path, args.test_dataset, args.estimation_method, args.model)
    
    # Find the actual train and test paths (most recent directories)
    def get_latest_result_path(cache_path):
        if os.path.exists(cache_path):
            subdirs = [
                os.path.join(cache_path, d)
                for d in os.listdir(cache_path)
                if os.path.isdir(os.path.join(cache_path, d))
            ]
            if subdirs:
                return max(subdirs, key=os.path.getctime)
        return cache_path
    
    args.train_path = get_latest_result_path(train_cache_path)
    args.test_path = get_latest_result_path(test_cache_path)
    
    # Construct results path: /hdd/ivny/cross_domain_results/{model}/{train_dataset}_{test_dataset}/
    results_path = os.path.join(
        "/hdd/ivny/cross_domain_results/",
        args.estimation_method,
        args.model,
        f"{args.train_dataset}_{args.test_dataset}"
    )
    os.makedirs(results_path, exist_ok=True)

    # check if train path exists
    if not os.path.exists(args.train_path):
        parent_dir = os.path.dirname(args.train_path)
        subdirs = [
            os.path.join(parent_dir, d)
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]
        if not subdirs:
            raise RuntimeError(f"No directories found in {parent_dir}")
        newest_dir = max(subdirs, key=os.path.getctime)
        print(f"Train path {args.train_path} does not exist. Using newest directory: {newest_dir}")
        args.train_path = newest_dir

    # check if results path exists
    if not os.path.exists(args.test_path):
        # move to the parent directory and cd into the last dir in the path
        parent_dir = os.path.dirname(args.test_path)

        subdirs = [
            os.path.join(parent_dir, d)
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]

        if not subdirs:
            raise RuntimeError(f"No directories found in {parent_dir}")

        newest_dir = max(subdirs, key=os.path.getctime)

        print(f"Results path {args.test_path} does not exist. Using newest directory: {newest_dir}")
        args.test_path = newest_dir

    # Load training data
    with open(os.path.join(args.train_path, "graded_outputs_0.pkl"), "rb") as f:
        train_outputs: OrganisedOutputs = pickle.load(f)

    train_df = pd.DataFrame({
        "accuracy": train_outputs.accuracy_scores[0],
        "original_numerical_confidence": train_outputs.extracted_confidences[0],
    })
    train_df.dropna(inplace=True)

    # Load empirical results for calibration
    with open(os.path.join(args.test_path, "graded_outputs_0.pkl"), "rb") as f:
        graded_outputs: OrganisedOutputs = pickle.load(f)

    # truncate for debugging
    # limit = 500
    # graded_outputs.accuracy_scores[0] = graded_outputs.accuracy_scores[0][:limit]
    # graded_outputs.extracted_confidences[0] = graded_outputs.extracted_confidences[0][:limit]
    # graded_outputs.extracted_answers[0] = graded_outputs.extracted_answers[0][:limit]

    # use pd df to organise outputs
    test_df = pd.DataFrame({
        "accuracy": graded_outputs.accuracy_scores[0],
        "original_numerical_confidence": graded_outputs.extracted_confidences[0],
        "original_response": graded_outputs.extracted_answers[0]
    })

    test_df.dropna(inplace=True)

    # apply cross-domain numerical post hoc calibration
    calibrated_confidences = cross_domain_numerical_post_hoc_calibration(
        np.array(train_df["original_numerical_confidence"]),
        np.array(train_df["accuracy"]),
        np.array(test_df["original_numerical_confidence"]),
        method=args.post_hoc_method
    )

    test_df["calibrated_numerical_confidence"] = calibrated_confidences

    # find closest hedging words for calibrated confidences using multiprocessing
    hedging_words_cache = os.path.join(results_path, "hedging_words_cache.pkl")
    if os.path.exists(hedging_words_cache):
        print(f"Loading cached hedging words from {hedging_words_cache}")
        with open(hedging_words_cache, "rb") as f:
            hedging_words = pickle.load(f)
    else:
        print("Processing", results_path)
        with Pool(cpu_count() - 1) as pool:
            hedging_words = list(
                tqdm(
                    pool.imap(
                        obtain_hedging_words,
                        test_df["calibrated_numerical_confidence"],
                        chunksize=16
                    ),
                    total=len(test_df),
                    desc="Finding hedging words"
                )
            )
        with open(hedging_words_cache, "wb") as f:
            pickle.dump(hedging_words, f)
        print(f"Saved hedging words to {hedging_words_cache}")
    
    test_df["target_hedging_words"] = hedging_words

    if args.breakpoint == "hedge":
        print("Hedging words obtained. Breaking here for debugging.")
        sys.exit()

    # rewrite outputs with target hedging words
    rewrites_pkl_path = os.path.join(results_path, "linguistic_calibration_outputs_rewrites.pkl")
    if os.path.exists(rewrites_pkl_path):
        test_df = pd.read_pickle(rewrites_pkl_path)
        print("Rewritten outputs already exist, skipping rewriting step.")
    else:
        print("Rewriting outputs with target hedging words...")
        # rewrite original responses with target hedging words
        rewrite_prompts = [
            REWRITE_PROMPT.format(
                response=args.answer_prepend_test + row["original_response"],
                hedges=", ".join(row["target_hedging_words"])
            )
            for _, row in test_df.iterrows()
        ]

        llm = LLM(
            model=args.modifier,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=5096,
        )

        sampling_params = SamplingParams(temperature=1, max_tokens=1024)

        rewrite_outputs = llm.chat(
            messages=[
                [
                    {"role": "system", "content": "You are a linguistic expert."},
                    {"role": "user", "content": p},
                ]
                for p in rewrite_prompts
            ],
            sampling_params=sampling_params,
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        rewritten_answers = [
            out.outputs[0].text.strip().split("assistantfinal")[-1].strip()
            for out in rewrite_outputs
        ]

        test_df["calibrated_response"] = rewritten_answers

        test_df.dropna(inplace=True)

        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()

        test_df.to_pickle(rewrites_pkl_path)

    # evaluate linguistic confidence on original and calibrated responses
    outputs_csv_path = os.path.join(results_path, "linguistic_calibration_outputs.csv")
    outputs_pkl_path = os.path.join(results_path, "linguistic_calibration_outputs.pkl")
    if os.path.exists(outputs_pkl_path):
        print(f"Loading linguistic confidence outputs from {outputs_pkl_path}")
        test_df = pd.read_pickle(outputs_pkl_path)
    else:
        original_linguistic_confidences = estimate_linguistic_confidence(
            test_df["original_response"].tolist(),
            test_df["original_numerical_confidence"].tolist(),
        )

        calibrated_linguistic_confidences = estimate_linguistic_confidence(
            test_df["calibrated_response"].tolist(),
            test_df["calibrated_numerical_confidence"].tolist(),
        )

        test_df["original_linguistic_confidence"] = original_linguistic_confidences
        test_df["calibrated_linguistic_confidence"] = calibrated_linguistic_confidences

        test_df.dropna(inplace=True)

        test_df.to_csv(outputs_csv_path, index=False)
        test_df.to_pickle(outputs_pkl_path)

    if args.breakpoint == "eval":
        print("Eval complete. Breaking here for debugging.")
        sys.exit()

    # compute and save calibration metrics
    print("Computing calibration metrics...")
    original_organised_output = OrganisedOutputs(
        accuracy_scores=[test_df["accuracy"].tolist()],
        extracted_confidences=[test_df["original_numerical_confidence"].tolist()],
        extracted_answers=[test_df["original_response"].tolist()]
    )

    calibrated_organised_output = OrganisedOutputs(
        accuracy_scores=[test_df["accuracy"].tolist()],
        extracted_confidences=[test_df["calibrated_numerical_confidence"].tolist()],
        extracted_answers=[test_df["original_response"].tolist()]
    )

    original_linguistic_output = OrganisedOutputs(
        accuracy_scores=[test_df["accuracy"].tolist()],
        extracted_confidences=[test_df["original_linguistic_confidence"].tolist()],
        extracted_answers=[test_df["original_response"].tolist()]
    )

    calibrated_linguistic_output = OrganisedOutputs(
        accuracy_scores=[test_df["accuracy"].tolist()],
        extracted_confidences=[test_df["calibrated_linguistic_confidence"].tolist()],
        extracted_answers=[test_df["calibrated_response"].tolist()]
    )

    cfg = {"results_path": results_path}

    metrics_df = pd.DataFrame({
        "metric": [
            "original_signal_generalised_ECE",
            "original_signal_dECE",
            "original_signal_dECE_pt",
            "original_signal_dAUROC",
            "original_signal_auroc_pt",
            
            "calibrated_signal_generalised_ECE",
            "calibrated_signal_dECE",
            "calibrated_signal_dECE_pt",
            "calibrated_signal_dAUROC",
            "calibrated_signal_auroc_pt",
            
            "original_linguistic_generalised_ECE",
            "original_linguistic_dECE",
            "original_linguistic_dECE_pt",
            "original_linguistic_dAUROC",
            "original_linguistic_auroc_pt",
    
            "calibrated_linguistic_generalised_ECE",
            "calibrated_linguistic_dECE",
            "calibrated_linguistic_dECE_pt",
            "calibrated_linguistic_dAUROC",
            "calibrated_linguistic_auroc_pt",
        ],

        "value": [
            # original signal space metrics
            generalised_ece(cfg, original_organised_output)[0],
            dECE(cfg, original_organised_output)[0],
            dECE_point_mass(cfg, original_organised_output)[0],
            dAUROC(cfg, original_organised_output)[0],
            AUROC_point_mass(cfg, original_organised_output)[0],

            # calibrated signal space metrics 
            generalised_ece(cfg, calibrated_organised_output)[0],
            dECE(cfg, calibrated_organised_output)[0],
            dECE_point_mass(cfg, calibrated_organised_output)[0],
            dAUROC(cfg, calibrated_organised_output)[0],
            AUROC_point_mass(cfg, calibrated_organised_output)[0],

            # original linguistic space metrics
            generalised_ece(cfg, original_linguistic_output)[0],
            dECE(cfg, original_linguistic_output)[0],
            dECE_point_mass(cfg, original_linguistic_output)[0],
            dAUROC(cfg, original_linguistic_output)[0],
            AUROC_point_mass(cfg, original_linguistic_output)[0],
            
            # calibrated linguistic space metrics
            generalised_ece(cfg, calibrated_linguistic_output)[0],
            dECE(cfg, calibrated_linguistic_output)[0],
            dECE_point_mass(cfg, calibrated_linguistic_output)[0],
            dAUROC(cfg, calibrated_linguistic_output)[0],
            AUROC_point_mass(cfg, calibrated_linguistic_output)[0],
        ],
    })
    metrics_path = os.path.join(results_path, "linguistic_calibration_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(metrics_df)
    print(f"\nResults saved to {results_path}")