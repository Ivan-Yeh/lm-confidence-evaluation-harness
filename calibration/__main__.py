from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from multiprocessing import Pool, cpu_count
from linguistic_confidence_lexicon.linguistic_calibrator import find_closest_hedging_words
from lm_conf.confidence_metrics.distributionals import BetaDistribution
from lm_conf.default_utils.custom_types import OrganisedOutputs
from lm_conf.post_processing.metrics import dECE, dECE_point_mass, AUROC_point_mass


LINGUISTIC_LEXICON_PATH = "linguistic_confidence_lexicon/hedging_word_scores.pkl"
LEXICON = pd.read_pickle(LINGUISTIC_LEXICON_PATH)

argparser = argparse.ArgumentParser(description="Calibrate linguistic confidence lexicon using empirical data.")
argparser.add_argument("--results_path", type=str, required=True, help="Path to empirical results pickle file.")
argparser.add_argument("--model", type=str, required=True, help="LLM evaluator/calibrator name.")
argparser.add_argument("--post-hoc-method", type=str, choices=["isotonic", "platt"], default="isotonic", help="Post-hoc calibration method.")
argparser.add_argument("--answer-prepend", type=str, required=False, help="Prefix to prepend to answers.", default="")


rewrite_template = """
Given the following sentence and a list of target hedging words, rewrite the response to appropriately reflect the confidence level indicated by the set of target hedging words. Ensure the new sentence sounds natural and fluent. 
You may use words or expressions other than the provided hedging words, but the overall level of confidence expressed in the sentence must align with the target hedging words. 
If the original sentence is empty, return "No answer provided". 
If the original sentence suggests random guesses, abstention, or inability to answer already, return "I am not sure". 
If the hedging words list suggest high confidence, you may state the answer directly without hedging. 
You are encouraged to use first-person phrasing where appropriate. 

Original sentence: {response}
Target hedging words: {hedges}

Please return only the rewritten sentence without any explanation.
New sentence: 
""".strip()

DIRECT_PROMPT = """
Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge). 
Please pay attention to the hedging language used and the overall assertiveness of the statement. Do not consider the factual accuracy of the content or context outside the sentence itself.
If the sentence contains random guesses or abstention, the score should be towards 0. 
If the sentence does not contain any hedging expressions the score should be towards 100. 

Here is the sentence: {sentence} 

Confidence Score: [Return only a number between 0 and 100]
""".strip()


def process_conf(conf: BetaDistribution):
    if conf is None or not conf.is_valid():
        return None
    return find_closest_hedging_words(
        conf.alpha_param,
        conf.beta_param,
        LEXICON,
    )["hedging_word"].tolist()

def numerical_post_post_hoc_calibration(confidences, accuracies, method="isotonic"):
    """
    Calibrate beta distributions using first 10% for training, last 90% for prediction.
    
    Args:
        confidences: list of BetaDistribution objects
        accuracies: list of accuracy scores (0 or 1)
        method: "isotonic" or "platt"
    
    Returns:
        list of calibrated BetaDistribution objects (90% of input)
    """
    # Extract mu values from BetaDistributions
    mu_values = np.array([c.mu for c in confidences], dtype=float)
    acc_values = np.array([acc if acc != "" else 0 for acc in accuracies], dtype=float)
    
    n = len(mu_values)
    train_size = max(1, int(np.ceil(0.10 * n)))
    
    # Split: first 10% for training
    X_train = mu_values[:train_size]
    y_train = acc_values[:train_size]
    
    # Last 90% for prediction
    X_test = mu_values[train_size:]
    test_confidences = confidences[train_size:]
    
    # Fit calibration model
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X_train, y_train)
        calibrated_means = calibrator.predict(X_test)
    elif method == "platt":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(X_train.reshape(-1, 1), y_train)
        calibrated_means = calibrator.predict_proba(X_test.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown post-hoc calibration method: {method}")
    
    # Rebuild beta distributions with calibrated means, preserving original sigma
    calibrated_betas = []
    
    # Fill training set (first 10%) with None
    calibrated_betas.extend([None] * train_size)
    
    # Add calibrated betas for test set (last 90%)
    for calibrated_mu, original_beta in zip(calibrated_means, test_confidences):
        calibrated_betas.append(
            BetaDistribution(mu=float(calibrated_mu), sigma=original_beta.sigma)
        )
    
    return calibrated_betas


if __name__ == "__main__":
    args = argparser.parse_args()

    # Load empirical results
    with open(os.path.join(args.results_path, "graded_outputs_0.pkl"), "rb") as f:
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

    # apply numerical post hoc calibration
    calibrated_confidences = numerical_post_post_hoc_calibration(
        np.array(output_df["original_numerical_confidence"]),
        np.array(output_df["accuracy"]),
        method=args.post_hoc_method
    )

    output_df["calibrated_numerical_confidence"] = calibrated_confidences
    output_df.dropna(inplace=True)  # Remove training set rows with None

    # find closest hedging words for calibrated confidences using multiprocessing
    hedging_words_cache = os.path.join(args.results_path, "hedging_words_cache.pkl")
    
    if os.path.exists(hedging_words_cache):
        print(f"Loading cached hedging words from {hedging_words_cache}")
        with open(hedging_words_cache, "rb") as f:
            hedging_words = pickle.load(f)
    else:
        with Pool(cpu_count() - 1) as pool:
            hedging_words = list(
                tqdm(
                    pool.imap(
                        process_conf,
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


    # rewrite original responses with target hedging words
    rewrite_prompts = [
        rewrite_template.format(
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

    # evaluate linguistic confidence on original and calibrated responses
    direct_prompts_original = [
        DIRECT_PROMPT.format(sentence=row["original_response"])
        for _, row in output_df.iterrows()
    ]

    direct_prompts_calibrated = [
        DIRECT_PROMPT.format(sentence=row["calibrated_response"])
        for _, row in output_df.iterrows()
    ]

    # Evaluate 10 times for each prompt to fit beta distributions
    scores_original = [[] for _ in range(len(output_df))]
    scores_calibrated = [[] for _ in range(len(output_df))]

    sampling_params = SamplingParams(temperature=1.2, max_tokens=512)

    for _ in tqdm(range(10), desc="Evaluating linguistic confidence (10 repeats)"):
        # Original responses
        outputs_orig = llm.chat(
            messages=[
                [
                    {"role": "system", "content": "You are a linguistic confidence evaluator."},
                    {"role": "user", "content": p},
                ]
                for p in direct_prompts_original
            ],
            sampling_params=sampling_params,
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        for i, out in enumerate(outputs_orig):
            text = out.outputs[0].text.strip()
            match = re.search(r"\b\d{1,3}(?:\.\d+)?\b", text)
            if match:
                v = float(match.group(0))
                if 0 <= v <= 100:
                    scores_original[i].append(v / 100.0)
                else:
                    scores_original[i].append(np.nan)
            else:
                scores_original[i].append(np.nan)

        # Calibrated responses
        outputs_cal = llm.chat(
            messages=[
                [
                    {"role": "system", "content": "You are a linguistic confidence evaluator."},
                    {"role": "user", "content": p},
                ]
                for p in direct_prompts_calibrated
            ],
            sampling_params=sampling_params,
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        for i, out in enumerate(outputs_cal):
            text = out.outputs[0].text.strip()
            match = re.search(r"\b\d{1,3}(?:\.\d+)?\b", text)
            if match:
                v = float(match.group(0))
                if 0 <= v <= 100:
                    scores_calibrated[i].append(v / 100.0)
                else:
                    scores_calibrated[i].append(np.nan)
            else:
                scores_calibrated[i].append(np.nan)

    # Fit beta distributions from the 10 samples
    linguistic_confidences_original = []
    for scores in scores_original:
        if len(scores) == 0 or np.all(np.isnan(scores)):
            linguistic_confidences_original.append(None)
        else:
            mu = float(np.nanmean(scores))
            sigma = float(np.nanstd(scores)) if np.sum(~np.isnan(scores)) > 1 else 0.01
            beta_dist = BetaDistribution(mu=mu, sigma=sigma)
            linguistic_confidences_original.append(beta_dist if beta_dist.is_valid() else None)

    linguistic_confidences_calibrated = []
    for scores in scores_calibrated:
        if len(scores) == 0 or np.all(np.isnan(scores)):
            linguistic_confidences_calibrated.append(None)
        else:
            mu = float(np.nanmean(scores))
            sigma = float(np.nanstd(scores)) if np.sum(~np.isnan(scores)) > 1 else 0.01
            beta_dist = BetaDistribution(mu=mu, sigma=sigma)
            linguistic_confidences_calibrated.append(beta_dist if beta_dist.is_valid() else None)


    output_df["linguistic_confidence_original"] = linguistic_confidences_original
    output_df["linguistic_confidence_calibrated"] = linguistic_confidences_calibrated

    output_df.dropna(inplace=True)
    output_df.to_csv(os.path.join(args.results_path, "linguistic_calibration_outputs.csv"), index=False)
    

    original_organised_output = OrganisedOutputs(
        accuracy_scores=[output_df["accuracy"].tolist()],
        extracted_confidences=[output_df["original_numerical_confidence"].tolist()],
        extracted_answers=[output_df["original_response"].tolist()]
    )

    original_linguistic_output = OrganisedOutputs(
        accuracy_scores=[output_df["accuracy"].tolist()],
        extracted_confidences=[output_df["linguistic_confidence_original"].tolist()],
        extracted_answers=[output_df["original_response"].tolist()]
    )

    calibrated_linguistic_output = OrganisedOutputs(
        accuracy_scores=[output_df["accuracy"].tolist()],
        extracted_confidences=[output_df["linguistic_confidence_calibrated"].tolist()],
        extracted_answers=[output_df["calibrated_response"].tolist()]
    )

    metrics_df = pd.DataFrame({
        "metric": [
            "original_dECE",
            "original_dECE_pt",
            "original_auroc_pt",
            "original_linguistic_dECE",
            "original_linguistic_dECE_pt",
            "original_linguistic_auroc_pt",
            "calibrated_linguistic_dECE",
            "calibrated_linguistic_dECE_pt",
            "calibrated_linguistic_auroc_pt",
        ],
        "value": [
            dECE({}, original_organised_output)[0],
            dECE_point_mass({}, original_organised_output)[0],
            AUROC_point_mass({}, original_organised_output)[0],
            dECE({}, original_linguistic_output)[0],
            dECE_point_mass({}, original_linguistic_output)[0],
            AUROC_point_mass({}, original_linguistic_output)[0],
            dECE({}, calibrated_linguistic_output)[0],
            dECE_point_mass({}, calibrated_linguistic_output)[0],
            AUROC_point_mass({}, calibrated_linguistic_output)[0],
        ],
    })
    metrics_df.to_csv(os.path.join(args.results_path, "linguistic_calibration_metrics.csv"), index=False)

    
