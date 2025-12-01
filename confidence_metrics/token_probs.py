from default_utils.registry import register_confidence
from default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
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