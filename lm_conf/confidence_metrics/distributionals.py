import gc
import logging
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import beta
import re
from ..confidence_metrics.semantics import semantic_cluster_selection, EntailmentDeberta
from ..default_utils.registry import register_confidence
from ..default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
from ..models.model_manager import ModelManager

class BetaDistribution:
    def __init__(self, mu: float, sigma: float):
        self.mu = np.clip(mu, 1e-6, 1 - 1e-6)

        max_sigma2 = self.mu * (1 - self.mu)
        sigma2 = min(sigma**2, max_sigma2 * 0.999)

        self.sigma = np.sqrt(max(sigma2, 1e-8))
        self.alpha_param, self.beta_param = self._fit_parameters()

    def _fit_parameters(self):
        var = self.sigma ** 2
        common = self.mu * (1 - self.mu) / var - 1.0

        alpha = self.mu * common
        beta = (1 - self.mu) * common

        return max(alpha, 1e-6), max(beta, 1e-6)
    
    def sample(self, size: int = 1) -> np.ndarray:
        return beta.rvs(self.alpha_param, self.beta_param, size=size)
    
    def is_valid(self) -> bool:
        try:
            assert isinstance(float(self.mu), float)
            assert isinstance(float(self.sigma), float)
            assert isinstance(float(self.alpha_param), float)
            assert isinstance(float(self.beta_param), float)
            # Check that values are not NaN or inf
            assert not np.isnan(self.mu) and not np.isinf(self.mu)
            assert not np.isnan(self.sigma) and not np.isinf(self.sigma)
            assert not np.isnan(self.alpha_param) and not np.isinf(self.alpha_param)
            assert not np.isnan(self.beta_param) and not np.isinf(self.beta_param)
            # Check that parameters are positive
            assert self.alpha_param > 0 and self.beta_param > 0
            return True
        except:
            return False
    
    def __repr__(self):
        return f"BetaDistribution(alpha={self.alpha_param}, beta={self.beta_param}, mu={self.mu}, sigma={self.sigma})"


@register_confidence(name="distributional_length_normalised_log_likelihood")
def distributional_length_normalised_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    best_cluster_str, best_cluster_logprobs = semantic_cluster_selection(output_lst)
    selected_responses: list[str] = [cluster[0] for cluster in best_cluster_str]
    cluster_token_probs_dist = []
    for cluster_logprobs in best_cluster_logprobs:
        cluster_probs = [np.exp(np.mean(x)) for x in cluster_logprobs if x is not None and len(x) > 0]
        mu = np.mean(cluster_probs) if len(cluster_probs) > 0 else 0.0
        sigma = np.std(cluster_probs) if len(cluster_probs) > 1 else 0.0
        cluster_token_probs_dist.append(BetaDistribution(mu=mu, sigma=sigma))

    return OrganisedOutputs(
        extracted_answers=[selected_responses],
        extracted_confidences=[cluster_token_probs_dist]
    )


@register_confidence(name="distributional_semantic_uncertainty")
def distributional_semantic_uncertainty(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    response_lists: list[tuple[str]] = list(zip(*[outputs.output_texts for outputs in output_lst]))
    entailment_model = EntailmentDeberta()
    confidence_dists: list[BetaDistribution] = []
    selected_responses: list[str] = []

    for responses in tqdm(response_lists, desc="Processing Entailments (Entailment Probability)"):
        n = len(responses)
        premises: list[str] = []
        hypotheses: list[str] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    premises.append(responses[i])
                    hypotheses.append(responses[j])

        preds = entailment_model.entailment_probability_batch(premises, hypotheses) if premises else []
        preds = np.clip(np.array(preds), 1e-6, 1 - 1e-6)
        
        # Compute mean entailment probability for each response (as premise)
        mean_entailment_per_response = []
        for i in range(n):
            # Get entailment probs where response i is the premise
            idx_start = i * (n - 1)
            idx_end = idx_start + (n - 1)
            response_preds = preds[idx_start:idx_end] if preds else []
            mean_ent = np.mean(response_preds) if len(response_preds) > 0 else 0.0
            mean_entailment_per_response.append(mean_ent)
        
        # Select response with highest mean entailment probability
        best_idx = np.argmax(mean_entailment_per_response) if mean_entailment_per_response else 0
        selected_responses.append(responses[best_idx])
        
        confidence_dists.append(BetaDistribution(mu=np.mean(preds), sigma=np.std(preds, ddof=1)))

    entailment_model.model.to("cpu")
    del entailment_model.model
    del entailment_model.tokenizer
    del entailment_model
    gc.collect()
    torch.cuda.empty_cache()

    return OrganisedOutputs(
        extracted_answers=[selected_responses],
        extracted_confidences=[confidence_dists]
    )


@register_confidence(name="distributional_linguistic_confidence")
def distributional_linguistic_confidence(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # use llm judge to rate how decisive the model's answer is
    model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type="linguistic_confidence_judge_model")
    def per_round_estimator(output: ModelOutputs):
        def extract_score(text: str) -> float:
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                score = float(match.group(1))
                score = min(max(score, 0.0), 100.0) / 100.0  # Normalize to [0, 1]
            else:
                score = np.nan
            return score
        
        DIRECT_PROMPT = """
        Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge). \
            Please pay attention to the hedging language used and the overall assertiveness of the statement. \
                If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. \
                    If the sentence does not contain any linguistic cues or is a succinct, decisive short answer, the score should be towards 100, too.

        Here is the sentence:
        {sentence}

        Confidence Score: [Return only a number between 0 and 100]
        """.strip()
        prompt_collection = PromptCollection(context_texts=[DIRECT_PROMPT.format(sentence=response) for response in output.output_texts])
        judge_outputs: list[ModelOutputs] = model_manager.run_generation(prompt_collection)
        confidences: list[list[float]] = [[extract_score(text) for text in out.output_texts] for out in judge_outputs]
        # Transpose confidences: from [num_judge_outputs][num_texts] to [num_texts][num_judge_outputs]
        confidences = list(map(list, zip(*confidences)))
        # Average scores from different judge outputs
        confidence_dists = [BetaDistribution(mu=float(np.nanmean(scores)), sigma=float(np.nanstd(scores, ddof=1))) for scores in confidences]
        return confidence_dists
    
    all_confidences = []
    all_answers = []
    for output in output_lst:
        confidences = per_round_estimator(output)
        all_confidences.append(confidences)
        all_answers.append(output.output_texts)

    return OrganisedOutputs(
        extracted_answers=all_answers,
        extracted_confidences=all_confidences,
    )


@register_confidence(name="distributional_p_true_cont")
def distributional_p_true_mc(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    p_true_prompt_template = """
    Question: {question}
    Proposed Answer: {model_answer}
    Is the proposed answer:
    A) True
    B) False
    Return only A or B. The proposed answer is:
    """.strip()
    model = ModelManager(master_cfg=cfg, model_config_type="p_true_cont_model")
    # Get clusters per question (list[list[str]]) and ignore logprobs here
    clusters, _ = semantic_cluster_selection(output_lst)

    selected_answers: list[str] = []
    cluster_beta_dists: list[BetaDistribution | None] = [None for _ in clusters]

    # Build all prompts once, track which cluster each prompt belongs to
    all_prompts: list[str] = []
    prompt_cluster_indices: list[int] = []

    for q_idx, cluster in enumerate(clusters):
        if not cluster:
            selected_answers.append("")
            cluster_beta_dists[q_idx] = BetaDistribution(mu=0.5, sigma=0.0)
            continue

        selected_answers.append(cluster[0])

        question_text = prompts.context_texts[q_idx] if hasattr(prompts, "context_texts") else ""
        for resp in cluster:
            ctx = p_true_prompt_template.format(question=question_text, model_answer=resp)
            all_prompts.append(ctx)
            prompt_cluster_indices.append(q_idx)

    if not all_prompts:
        cluster_beta_dists = [beta_dist if beta_dist is not None else BetaDistribution(mu=0.5, sigma=0.0) for beta_dist in cluster_beta_dists]
        return OrganisedOutputs(
            extracted_answers=[selected_answers],
            extracted_confidences=[cluster_beta_dists],
        )

    cont_map = {ctx: [" A"] for ctx in all_prompts}
    p_true_prompt_collection = PromptCollection(context_texts=all_prompts, continuation_texts=cont_map)

    try:
        cont_results = model.run_continuation(p_true_prompt_collection)
        cont_out = cont_results[0]
    except Exception:
        cluster_beta_dists = [beta_dist if beta_dist is not None else BetaDistribution(mu=0.5, sigma=0.0) for beta_dist in cluster_beta_dists]
        return OrganisedOutputs(
            extracted_answers=[selected_answers],
            extracted_confidences=[cluster_beta_dists],
        )

    p_trues_by_cluster: list[list[float]] = [[] for _ in clusters]

    for logprob, cluster_idx in zip(cont_out.output_logprobs, prompt_cluster_indices):
        try:
            mean_a = np.mean(logprob)
            p_a = float(np.exp(mean_a))
        except Exception:
            p_a = 0.5
        p_a = float(np.clip(p_a, 1e-6, 1 - 1e-6))
        p_trues_by_cluster[cluster_idx].append(p_a)

    for q_idx, p_trues in enumerate(p_trues_by_cluster):
        if cluster_beta_dists[q_idx] is not None:
            continue
        if p_trues:
            mu = float(np.mean(p_trues))
            sigma = float(np.std(p_trues)) if len(p_trues) > 1 else 0.0
        else:
            mu = 0.5
            sigma = 0.0
        cluster_beta_dists[q_idx] = BetaDistribution(mu=mu, sigma=sigma)

    return OrganisedOutputs(
        extracted_answers=[selected_answers],
        extracted_confidences=[cluster_beta_dists],
    )

