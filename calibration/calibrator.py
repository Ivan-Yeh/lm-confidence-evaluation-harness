from scipy.stats import wasserstein_distance
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import re
from vllm import LLM, SamplingParams
from linguistic_confidence_lexicon.linguistic_calibrator import find_closest_hedging_words
from lm_conf.confidence_metrics.distributionals import BetaDistribution
from lm_conf.default_utils.custom_types import OrganisedOutputs
from lm_conf.post_processing.metrics import dECE, dECE_point_mass, AUROC_point_mass


LINGUISTIC_LEXICON_PATH = "linguistic_confidence_lexicon/linguistic_confidence_lexicon.pkl"


argparser = argparse.ArgumentParser(description="Calibrate linguistic confidence lexicon using empirical data.")
argparser.add_argument("--results_path", type=str, required=True, help="Path to empirical results pickle file.")
argparser.add_argument("--model", type=str, required=True, help="LLM evaluator/calibrator name.")
argparser.add_argument("--post-hoc-method", type=str, choices=["isotonic", "platt"], default="isotonic", help="Post-hoc calibration method.")


def post_hoc_calibrate(outputs: OrganisedOutputs, method="isotonic"):
    extracted_answers: list[str] = outputs.extracted_answers[0]  # assuming single round
    extracted_confidences: list[BetaDistribution] = outputs.extracted_confidences[0]  # assuming single round
    accuracy_scores: list[float] = outputs.accuracy_scores[0]  # assuming single round

    cleaned_answers: list[str] = []
    cleaned_confidences: list[BetaDistribution] = []
    cleaned_accuracies: list[float] = []

    # sanity check
    for resp, conf, acc in zip(extracted_answers, extracted_confidences, accuracy_scores):
        try:
            acc_float = float(acc)
            if conf.is_valid():
                cleaned_answers.append(resp)
                cleaned_confidences.append(conf)
                cleaned_accuracies.append(acc_float)
        except Exception:
            continue

    # Convert BetaDistribution to mean values
    numeric_confidences = []
    for conf in cleaned_confidences:
        numeric_confidences.append(conf.mu)
    
    numeric_confidences = np.array(numeric_confidences, dtype=float)
    numeric_accuracies = np.array(cleaned_accuracies, dtype=float)

    n = len(numeric_confidences)
    if n == 0:
        return [], [], [], [], []

    # Split: first 10% for training, last 90% for prediction
    train_size = max(1, int(np.ceil(0.10 * n)))
    
    X_train = numeric_confidences[:train_size]
    y_train = numeric_accuracies[:train_size]
    X_test = numeric_confidences[train_size:]
    
    # Split other arrays to match test set
    original_answers = cleaned_answers
    original_confidences = cleaned_confidences
    original_accuracies = cleaned_accuracies

    # Train calibration model
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X_train, y_train)
        calibrated_means = calibrator.transform(X_test)
    elif method == "platt":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(X_train.reshape(-1, 1), y_train)
        calibrated_means = calibrator.predict_proba(X_test.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    
    # Convert calibrated means back to BetaDistribution objects
    # Preserve the original sigma from the uncalibrated distributions
    calibrated_confidences = []
    for calibrated_mean, original_conf in zip(calibrated_means, original_confidences):
        calibrated_confidences.append(BetaDistribution(mu=calibrated_mean, sigma=original_conf.sigma))
    
    # Return: responses, original confidences, accuracies, calibrated confidences
    return original_answers, original_confidences, original_accuracies, calibrated_confidences


if __name__ == "__main__":
    # obtain raw confidence scores from cached results from results_path
    args = argparser.parse_args()
    original_results_path = os.path.join(args.results_path, "graded_outputs_0.pkl")
    with open(original_results_path, "rb") as f:
        graded_outputs: OrganisedOutputs = pickle.load(f)


    # post-hoc calibration
    post_hoc_calibrated_confidences: list[BetaDistribution]
    original_answers, original_confidences, original_accuracies, post_hoc_calibrated_confidences = post_hoc_calibrate(graded_outputs, args.post_hoc_method)


    # load linguistic confidence lexicon
    linguistic_lexicon: pd.DataFrame = pd.read_pickle(LINGUISTIC_LEXICON_PATH)


    # search for closest hedging words
    plausible_hedges = [
        find_closest_hedging_words(conf.alpha_param, conf.beta_param, linguistic_lexicon)["hedging_word"].tolist() 
        for conf in post_hoc_calibrated_confidences 
        if conf.is_valid()
    ]


    # prepare prompts for linguistic calibrator rewrites
    rewrite_template = """
    Given the following sentence and a list of plausible hedging words, rewrite the response to appropriately reflect the confidence level indicated by the set of hedging words.
    Please ensure that the rewritten sentence sounds natural and maintains the original meaning.
    Sentence: "{response}"
    Hedging words: {hedges}

    Please return only the rewritten sentence, with no additional explanation.
    """.strip()
    rewrite_prompts = []
    for response, hedges in zip(original_answers, plausible_hedges):
        prompt = rewrite_template.format(response=response, hedges=", ".join(hedges))
        rewrite_prompts.append(prompt)

    
    # linguistic calibrator rewrites with vllm
    calibrated_answers = []
    calibrated_confidences = []
    calibrated_accuracies = []

    llm = LLM(model=args.model,
              dtype="bfloat16",
              trust_remote_code=True,
              max_model_length=5096)
    sampling_params = SamplingParams(temperature=1, max_tokens=512)
    rewrite_outputs = llm.chat(
        requests=[{"role": "system", "content": "You are a linguistic expert."}, {"role": "user", "content": prompt} for prompt in rewrite_prompts],
        sampling_params=sampling_params,
        chat_template_kwargs={"reasoning_effort": "low"},
    )

    for i, output in enumerate(rewrite_outputs):
        rewrite = output.choices[0].message["content"].strip()
        if "assistantfinal" in rewrite.lower():
            calibrated_answers.append(rewrite)
            calibrated_confidences.append(post_hoc_calibrated_confidences[i])
            calibrated_accuracies.append(original_accuracies[i])

        


    # Estimate linguistic confidence score with linguistic evaluator
    DIRECT_PROMPT = """
    Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge). \
        Please pay attention to the hedging language used and the overall assertiveness of the statement. \
            If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. \
                If the sentence does not contain any linguistic cues or is a succinct, decisive short answer, the score should be towards 100, too.

    Here is the sentence:
    {sentence}

    Confidence Score: [Return only a number between 0 and 100]
    """.strip()
    evaluator_prompts = [DIRECT_PROMPT.format(sentence=response) for response in calibrated_answers]
    

    # Repeat 10 times for each prompt and extract numbers
    scores_per_prompt = [[] for _ in range(len(evaluator_prompts))]
    
    for _ in range(10):
        evaluator_outputs = llm.chat(
            messages=[[{"role": "system", "content": "You are a linguistic confidence evaluator."}, {"role": "user", "content": prompt}] for prompt in evaluator_prompts],
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        # Extract numbers from outputs
        for i, output in enumerate(evaluator_outputs):
            response_text = output.outputs[0].text.strip()
            # Extract number using regex
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                score = float(numbers[0]) / 100  # Normalize to [0, 1]
                scores_per_prompt[i].append(score)
            else:
                scores_per_prompt[i].append(np.nan)
    

    # Calculate mean and std for each prompt to construct beta distributions
    # Filter to only include prompts with valid scores
    linguistic_means = []
    linguistic_stds = []
    
    for i, scores in enumerate(scores_per_prompt):
        if len(scores) > 0:
            linguistic_means.append(np.nanmean(scores))
            linguistic_stds.append(np.nanstd(scores) if len(scores) > 1 else 0.01)  # Small default std if only 1 sample
    

    # Construct beta distributions from linguistic evaluation
    re_estimated_linguistic_beta_dists = [BetaDistribution(mu=mu, sigma=sigma) for mu, sigma in zip(linguistic_means, linguistic_stds)]

    organised_output = OrganisedOutputs(
        extracted_answers=[],
        extracted_confidences=[],
        accuracy_scores=[]
    )

    for i in range(len(re_estimated_linguistic_beta_dists)):
        if re_estimated_linguistic_beta_dists[i].is_valid():
            organised_output.extracted_answers[0].append(calibrated_answers[i])
            organised_output.extracted_confidences[0].append(re_estimated_linguistic_beta_dists[i])
            organised_output.accuracy_scores[0].append(calibrated_accuracies[i])
    

    # evaluate linguistic calibrated responses with dECE, auroc_point_mass (calculate with uncalibrated and calibrated confidences)
    original_dece = dECE({}, graded_outputs)
    original_dece_pt = dECE_point_mass({}, graded_outputs)
    original_auroc_pt = AUROC_point_mass({}, graded_outputs)
    
    cali_dECE = dECE({}, organised_output)
    cali_dECE_pt = dECE_point_mass({}, organised_output)
    cali_auroc_pt = AUROC_point_mass({}, organised_output)

    # Store results
    calibration_details_df = pd.DataFrame({
        "accuracy": graded_outputs.accuracy_scores[0],
        "original_response": graded_outputs.extracted_answers[0],
        "original_confidence": graded_outputs.extracted_confidences[0],
        "calibrated_response": organised_output.extracted_answers[0],
        "re_estimated_confidence": organised_output.extracted_confidences[0],
    })

    calibration_metrics_df = pd.DataFrame({
        "metric": ["original_dECE", "original_dECE_pt", "original_AUROC_pt", "calibrated_dECE", "calibrated_dECE_pt", "calibrated_AUROC_pt"],
        "value": [original_dece, original_dece_pt, original_auroc_pt, cali_dECE, cali_dECE_pt, cali_auroc_pt]
    })

    calibration_details_df.to_csv(os.path.join(args.results_path, "linguistic_calibration_details.csv"), index=False)
    calibration_metrics_df.to_csv(os.path.join(args.results_path, "linguistic_calibration_metrics.csv"), index=False)