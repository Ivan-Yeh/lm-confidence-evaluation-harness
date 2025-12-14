from ..default_utils.registry import register_confidence
from ..default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
import numpy as np

@register_confidence(name="length_normalised_log_likelihood")
def length_normalised_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    def per_round_estimator(outputs: ModelOutputs) -> list[float]: 
        length_normalised_probs: list[float] = []
        for logprobs in outputs.output_logprobs:
            try:
                length_normalised_prob = float(np.exp(np.mean(logprobs)))
            except:
                length_normalised_prob = None
            length_normalised_probs.append(length_normalised_prob)
        return length_normalised_probs
    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts for outputs in output_lst],
        extracted_confidences=[per_round_estimator(outputs) for outputs in output_lst]
    )


@register_confidence(name="continuation_candidate_normalised_log_likelihood")
def candidate_normalised_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    def per_round_estimator(outputs: ModelOutputs) -> list[float]: 
        normalised_probs: list[float] = []
        for candidates in outputs.continuation_candidates:
            candidate_logprobs = [np.exp(np.mean(candidate.get('logprobs', []))) for candidate in candidates]
            try:
                max_logprob = max(candidate_logprobs)
                normalised_prob = float(max_logprob / sum(candidate_logprobs))
            except:
                normalised_prob = None
            normalised_probs.append(normalised_prob)
        return normalised_probs
    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts for outputs in output_lst],
        extracted_confidences=[per_round_estimator(outputs) for outputs in output_lst]
    )



@register_confidence(name="volatility_adjusted_log_likelihood")
def volatility_adjusted_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    def confidence_with_sigma_amp(logprobs, k=1.5, alpha=0.1):
        logprobs = np.array(logprobs)
        n = len(logprobs)
        mu = logprobs.mean()
        sigma = logprobs.std()
        sigma_diff = np.std(np.diff(logprobs))
        
        # scale numerator by volatility amplifier
        adjusted = (mu) / (1 + k * sigma + sigma_diff)
        
        # apply length penalty
        adjusted /= n**alpha
        
        # sigmoid
        confidence = 1 / (1 + np.exp(-adjusted))
        return confidence

    
    def per_round_estimator(outputs: ModelOutputs) -> list[float]: 
        sharpe_sigmoid_probs: list[float] = []
        for logprobs in outputs.output_logprobs:
            try:
                confidence = confidence_with_sigma_amp(logprobs)
            except:
                confidence = None
            sharpe_sigmoid_probs.append(confidence)
        return sharpe_sigmoid_probs
    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts for outputs in output_lst],
        extracted_confidences=[per_round_estimator(outputs) for outputs in output_lst]
    )

