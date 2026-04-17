from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import re
import gc
from multiprocessing import Pool, cpu_count
from linguistic_confidence_lexicon.linguistic_calibrator import find_closest_hedging_words
from lm_conf.confidence_metrics.distributionals import BetaDistribution
from lm_conf.default_utils.custom_types import PromptCollection
from lm_conf.models.model_manager import ModelManager

LINGUISTIC_LEXICON_PATH = "linguistic_confidence_lexicon/hedging_word_scores_reasonable.csv"
LEXICON = pd.read_csv(LINGUISTIC_LEXICON_PATH)


REWRITE_PROMPT = """
Given the an oriingal response and a list of target hedging words, rewrite the response to appropriately reflect the confidence level indicated by the set of target hedging words. 
You must preserve the original meaning of the response, as we are only adjusting the tone to match the confidence level suggested by the hedging words. Ensure the new response sounds natural and fluent. 
You do not need to explicitly include the hedging words in the rewritten response, but the tone and linguistic cues should reflect the confidence level indicated by those words.

You are encouraged to use words or expressions other than the provided hedging words, but the overall level of confidence expressed in the response must align with the target hedging words. 
If the original response is empty, return "No answer provided". 
If the original response suggests random guesses, abstention, or inability to answer already, you should return a response in the line of of "I'm not sure, but I guess...[original guess]". 
If the target hedging words suggest high confidence, you may state the answer directly without hedging to convey decisiveness.
When hedging is needed, it is better to use one or at most two representative hedging expression rather than incorporating many hedging words, to ensure the response remains natural and fluent.
You are encouraged to use first-person phrasing where appropriate.

Original response: ```My answer to the question is: "{response}"```
Target hedging words: {hedges}

Please return only the rewritten sentence without any explanation.
New response: 
""".strip()


LINGUISTIC_EVALUATOR_PROMPT = """
Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge). 
Please pay attention to the hedging language used and the overall assertiveness of the statement. 
If the sentence abstains from answering by pointing out the insufficiency of information with a firm tone, the score should be towards 100. 
If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. 
If the sentence does not contain any hedging language or is a succinct, decisive short answer, the score should be towards 100, too.

Here is the sentence:
{sentence}

Confidence Score: [Return only a number between 0 and 100]
""".strip()


def obtain_hedging_words(conf: BetaDistribution, top_k = 5) -> list[str] | None:
    if conf is None or not conf.is_valid():
        return None
    return find_closest_hedging_words(
        conf.alpha_param,
        conf.beta_param,
        LEXICON,
        top_k=top_k,
    )["hedging_word"].tolist()


evaluator_list = [
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def in_domain_numerical_post_hoc_calibration(confidences, accuracies, method="isotonic"):
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
    sigma_values = np.array([c.sigma for c in confidences], dtype=float)
    acc_values = np.array([acc if acc != "" else 0 for acc in accuracies], dtype=float)
    
    n = len(mu_values)
    train_size = max(1, int(np.ceil(0.2 * n)))
    
    # Split: first 20% for training
    X_train = mu_values[:train_size]
    y_train = acc_values[:train_size]
    
    # Last 80% for prediction
    X_test = mu_values[train_size:]
    test_confidences = confidences[train_size:]
    
    # Fit calibration model
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X_train, y_train)
        calibrated_means = calibrator.predict(X_test)
    elif method == "platt_uni":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(X_train.reshape(-1, 1), y_train)
        calibrated_means = calibrator.predict_proba(X_test.reshape(-1, 1))[:, 1]
    elif method == "platt_bi":
        # Use both mu and sigma as features
        X_train_features = np.column_stack([X_train, sigma_values[:train_size]])
        X_test_features = np.column_stack([X_test, sigma_values[train_size:]])
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(X_train_features, y_train)
        calibrated_means = calibrator.predict_proba(X_test_features)[:, 1]
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

def cross_domain_numerical_post_hoc_calibration(confidences_train, accuracies_train, raw_confidences, method="isotonic"):
    """
    Calibrate beta distributions using data from one domain for training, apply to another domain.
    
    Args:
        confidences_train: list of BetaDistribution objects for training
        accuracies_train: list of accuracy scores (0 or 1) for training
        raw_confidences: list of BetaDistribution objects to be calibrated
        method: "isotonic" or "platt"
    
    Returns:
        list of calibrated BetaDistribution objects (same length as raw_confidences)
    """
    # Filter to keep only valid pairs where both confidence and accuracy are valid
    valid_pairs = []
    for conf, acc in zip(confidences_train, accuracies_train):
        # Check if confidence is valid (not None and has valid mu)
        if conf is not None and hasattr(conf, 'mu') and not np.isnan(conf.mu):
            # Check if accuracy is valid (not empty string and not NaN)
            acc_value = acc if acc != "" else np.nan
            if not np.isnan(float(acc_value) if acc_value != "" else np.nan):
                valid_pairs.append((conf, float(acc_value)))
    
    if not valid_pairs:
        raise ValueError("No valid confidence-accuracy pairs found in training data")
    
    # Extract mu and sigma values and accuracies from valid pairs
    mu_train = np.array([c.mu for c, _ in valid_pairs], dtype=float)
    sigma_train = np.array([c.sigma for c, _ in valid_pairs], dtype=float)
    acc_train = np.array([acc for _, acc in valid_pairs], dtype=float)
    
    # Extract mu and sigma values from raw confidences to be calibrated
    mu_raw = np.array([c.mu for c in raw_confidences], dtype=float)
    sigma_raw = np.array([c.sigma for c in raw_confidences], dtype=float)
    
    # Fit calibration model on training data
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(mu_train, acc_train)
        calibrated_means = calibrator.predict(mu_raw)
    elif method == "platt_uni":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(mu_train.reshape(-1, 1), acc_train)
        calibrated_means = calibrator.predict_proba(mu_raw.reshape(-1, 1))[:, 1]
    elif method == "platt_bi":
        # Use both mu and sigma as features
        X_train_features = np.column_stack([mu_train, sigma_train])
        X_test_features = np.column_stack([mu_raw, sigma_raw])
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(X_train_features, acc_train)
        calibrated_means = calibrator.predict_proba(X_test_features)[:, 1]
    else:
        raise ValueError(f"Unknown post-hoc calibration method: {method}")
    
    # Rebuild beta distributions with calibrated means, preserving original sigma
    calibrated_betas = []
    for calibrated_mu, original_beta in zip(calibrated_means, raw_confidences):
        calibrated_betas.append(
            BetaDistribution(mu=float(calibrated_mu), sigma=original_beta.sigma)
        )
    
    return calibrated_betas

def estimate_linguistic_confidence(
    responses: list[str],
    evaluators_cfg: dict,
    evaluator_keys: list[str],
) -> list[BetaDistribution | None]:
    prompts = [LINGUISTIC_EVALUATOR_PROMPT.format(sentence=response) for response in responses]

    def extract_score(text: str) -> float:
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            score = float(match.group(1))
            score = min(max(score, 0.0), 100.0) / 100.0  # Normalize to [0, 1]
        else:
            score = np.nan
        return score

    scores_collection: list[list[float]] = [[] for _ in responses]

    for evaluator in evaluator_keys:
        model_manager: ModelManager = ModelManager(master_cfg=evaluators_cfg, model_config_type=evaluator)
        prompt_collection = PromptCollection(
            system_prompt="You are a careful assistant.",
            context_texts=prompts,
        )

        outputs = model_manager.run_generation(prompt_collection)
        for output in outputs:
            output_texts = output.output_texts
            for i, text in enumerate(output_texts):
                score = extract_score(text)
                if not np.isnan(score):
                    scores_collection[i].append(score)

        # del model_manager
        gc.collect()

    confidence_dists = []
    for text_idx, _ in enumerate(responses):
        try:
            # Get all scores from all evaluators for this specific text
            all_scores = scores_collection[text_idx]
            valid_scores = [s for s in all_scores if not np.isnan(s)]
            if valid_scores:
                mu = float(np.mean(valid_scores))
                sigma = float(np.std(valid_scores)) if len(valid_scores) > 1 else 1e-6
            else:
                mu = 0.5
                sigma = 1e-6
            
            confidence_dists.append(BetaDistribution(mu=mu, sigma=sigma))
        except Exception as e:
            print(f"Error processing text index {text_idx}: {e}")
            confidence_dists.append(None)
    
    return confidence_dists