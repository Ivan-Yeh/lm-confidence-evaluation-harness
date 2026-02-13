import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from multiprocessing import Pool, cpu_count
from lm_conf.default_utils.custom_types import OrganisedOutputs
from lm_conf.post_processing.metrics import dECE, dECE_point_mass, AUROC_point_mass, dAUROC
from calibration.utils import *

argparser = argparse.ArgumentParser(description="Calibrate linguistic confidence lexicon using empirical data.")
argparser.add_argument("--results_path", type=str, required=True, help="Path to empirical results pickle file.")
argparser.add_argument("--save_path", type=str, required=True, help="Path to save calibrated results pickle file.")
argparser.add_argument("--model", type=str, required=True, help="LLM evaluator/calibrator name.")
argparser.add_argument("--post-hoc-method", type=str, choices=["isotonic", "platt_bi", "platt_uni"], default="platt_bi", help="Post-hoc calibration method.")
argparser.add_argument("--answer-prepend", type=str, required=False, help="Prefix to prepend to answers.", default="")


if __name__ == "__main__":
    args = argparser.parse_args()

    # check if results path exists
    if not os.path.exists(args.results_path):
        # move to the parent directory and cd into the last dir in the path
        parent_dir = os.path.dirname(args.results_path)

        subdirs = [
            os.path.join(parent_dir, d)
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]

        if not subdirs:
            raise RuntimeError(f"No directories found in {parent_dir}")

        newest_dir = max(subdirs, key=os.path.getctime)

        print(f"Results path {args.results_path} does not exist. Using newest directory: {newest_dir}")
        args.results_path = newest_dir

    save_dir = os.path.dirname(args.results_path.replace("/hdd/ivny/results", args.save_path))

    os.makedirs(save_dir, exist_ok=True)

    # Load empirical results
    with open(os.path.join(args.results_path, "graded_outputs_0.pkl"), "rb") as f:
        print(f"Loading graded_outputs_0.pkl from {args.results_path}")
        graded_outputs: OrganisedOutputs = pickle.load(f)

    # truncate for debugging
    # limit = 1000
    # graded_outputs.accuracy_scores[0] = graded_outputs.accuracy_scores[0][:limit]
    # graded_outputs.extracted_confidences[0] = graded_outputs.extracted_confidences[0][:limit]
    # graded_outputs.extracted_answers[0] = graded_outputs.extracted_answers[0][:limit]

    # use pd df to organise outputs
    output_df = pd.DataFrame({
        "accuracy": graded_outputs.accuracy_scores[0],
        "original_numerical_confidence": graded_outputs.extracted_confidences[0],
        "original_response": graded_outputs.extracted_answers[0]
    })

    output_df.dropna(inplace=True)

    # apply numerical post hoc calibration
    calibrated_confidences = in_domain_numerical_post_hoc_calibration(
        np.array(output_df["original_numerical_confidence"]),
        np.array(output_df["accuracy"]),
        method=args.post_hoc_method
    )

    output_df["calibrated_numerical_confidence"] = calibrated_confidences
    output_df.dropna(inplace=True)  # Remove training set rows with None

    # find closest hedging words for calibrated confidences using multiprocessing
    hedging_words_cache = os.path.join(save_dir, "hedging_words_cache.pkl")


    if os.path.exists(hedging_words_cache):
        print(f"Loading cached hedging words from {hedging_words_cache}")
        with open(hedging_words_cache, "rb") as f:
            hedging_words = pickle.load(f)
    else:
        with Pool(cpu_count() - 1) as pool:
            hedging_words = list(
                tqdm(
                    pool.imap(
                        obtain_hedging_words,
                        output_df["calibrated_numerical_confidence"],
                        chunksize=16
                    ),
                    total=len(output_df),
                    desc="Finding hedging words"
                )
            )
        with open(hedging_words_cache, "wb") as f:
            pickle.dump(hedging_words, f)
        print(f"Saved hedging words to {hedging_words_cache}")
    
    output_df["target_hedging_words"] = hedging_words

    # rewrite outputs with target hedging words
    if os.path.exists(os.path.join(save_dir, "linguistic_calibration_outputs_rewrites.pkl")):
        output_df = pd.read_pickle(os.path.join(save_dir, "linguistic_calibration_outputs_rewrites.pkl"))
        print("Rewritten outputs already exist, skipping rewriting step.")
    else:
        print("Rewriting outputs with target hedging words...")
        # rewrite original responses with target hedging words
        rewrite_prompts = [
            REWRITE_PROMPT.format(
                response=args.answer_prepend + row["original_response"],
                hedges=", ".join(row["target_hedging_words"])
            )
            for _, row in output_df.iterrows()
        ]

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=5096,
        )

        sampling_params = SamplingParams(temperature=1, max_tokens=512)

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

        output_df["calibrated_response"] = rewritten_answers

        output_df.dropna(inplace=True)

        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()

        output_df.to_pickle(os.path.join(save_dir, "linguistic_calibration_outputs_rewrites.pkl"))

    # evaluate linguistic confidence on original and calibrated responses
    if os.path.exists(os.path.join(save_dir, "linguistic_calibration_outputs.pkl")):
        output_df = pd.read_pickle(os.path.join(save_dir, "linguistic_calibration_outputs.pkl"))
    else:
        original_linguistic_confidences = estimate_linguistic_confidence(
            output_df["original_response"].tolist()
        )

        calibrated_linguistic_confidences = estimate_linguistic_confidence(
            output_df["calibrated_response"].tolist()
        )

        output_df["original_linguistic_confidence"] = original_linguistic_confidences
        output_df["calibrated_linguistic_confidence"] = calibrated_linguistic_confidences

        output_df.dropna(inplace=True)

        output_df.to_csv(os.path.join(save_dir, "linguistic_calibration_outputs.csv"), index=False)
        output_df.to_pickle(os.path.join(save_dir, "linguistic_calibration_outputs.pkl"))

    # print("LLM inference cache saved.")
    # sys.exit()
    # try:   
    #     llm = LLM(max_model_len=4000, model=args.model, trust_remote_code=True)
    #     sampling_params = SamplingParams(temperature=1, max_tokens=256)
    #     llm.chat([[{"role": "user", "content": "place holder"}]], sampling_params=sampling_params)
    #     print("Prepare to compute metrics...")
    # except:
    #     pass

    # print("Confidence cached.")
    # sys.exit()

    # compute and save calibration metrics
    print("Computing calibration metrics...")
    original_organised_output = OrganisedOutputs(
        accuracy_scores=[output_df["accuracy"].tolist()],
        extracted_confidences=[output_df["original_numerical_confidence"].tolist()],
        extracted_answers=[output_df["original_response"].tolist()]
    )

    calibrated_organised_output = OrganisedOutputs(
        accuracy_scores=[output_df["accuracy"].tolist()],
        extracted_confidences=[output_df["calibrated_numerical_confidence"].tolist()],
        extracted_answers=[output_df["original_response"].tolist()]
    )

    original_linguistic_output = OrganisedOutputs(
        accuracy_scores=[output_df["accuracy"].tolist()],
        extracted_confidences=[output_df["original_linguistic_confidence"].tolist()],
        extracted_answers=[output_df["original_response"].tolist()]
    )

    calibrated_linguistic_output = OrganisedOutputs(
        accuracy_scores=[output_df["accuracy"].tolist()],
        extracted_confidences=[output_df["calibrated_linguistic_confidence"].tolist()],
        extracted_answers=[output_df["calibrated_response"].tolist()]
    )

    metrics_df = pd.DataFrame({
        "metric": [
            "original_signal_dECE",
            "original_signal_dECE_pt",
            "original_signal_dAUROC",
            "original_signal_auroc_pt",
            
            "calibrated_signal_dECE",
            "calibrated_signal_dECE_pt",
            "calibrated_signal_dAUROC",
            "calibrated_signal_auroc_pt",
            
            "original_linguistic_dECE",
            "original_linguistic_dECE_pt",
            "original_linguistic_dAUROC",
            "original_linguistic_auroc_pt",

            "calibrated_linguistic_dECE",
            "calibrated_linguistic_dECE_pt",
            "calibrated_linguistic_dAUROC",
            "calibrated_linguistic_auroc_pt",
        ],

        "value": [
            # original signal space metrics
            dECE({"results_path": save_dir}, original_organised_output)[0],
            dECE_point_mass({"results_path": save_dir}, original_organised_output)[0],
            dAUROC({"results_path": save_dir}, original_organised_output)[0],
            AUROC_point_mass({"results_path": save_dir}, original_organised_output)[0],

            # calibrated signal space metrics 
            dECE({"results_path": save_dir}, calibrated_organised_output)[0],
            dECE_point_mass({"results_path": save_dir}, calibrated_organised_output)[0],
            dAUROC({"results_path": save_dir}, calibrated_organised_output)[0],
            AUROC_point_mass({"results_path": save_dir}, calibrated_organised_output)[0],

            # original linguistic space metrics
            dECE({"results_path": save_dir}, original_linguistic_output)[0],
            dECE_point_mass({"results_path": save_dir}, original_linguistic_output)[0],
            dAUROC({"results_path": save_dir}, original_linguistic_output)[0],
            AUROC_point_mass({"results_path": save_dir}, original_linguistic_output)[0],
            
            # calibrated linguistic space metrics
            dECE({"results_path": save_dir}, calibrated_linguistic_output)[0],
            dECE_point_mass({"results_path": save_dir}, calibrated_linguistic_output)[0],
            dAUROC({"results_path": save_dir}, calibrated_linguistic_output)[0],
            AUROC_point_mass({"results_path": save_dir}, calibrated_linguistic_output)[0],
        ],
    })
    metrics_df.to_csv(os.path.join(save_dir, "linguistic_calibration_metrics.csv"), index=False)
    print(metrics_df)