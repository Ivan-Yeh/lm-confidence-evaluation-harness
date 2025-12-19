from ..default_utils.registry import register_confidence
from ..default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
import numpy as np
from ..confidence_metrics.semantics import semantic_uncertainty_selection
from scipy.stats import beta

class BetaDistribution:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.alpha_param, self.beta_param = self._fit_parameters()

    def _fit_parameters(self) -> tuple[float, float]:
        # Using method of moments to estimate alpha and beta
        common_factor = (self.mu * (1 - self.mu) / (self.sigma ** 2)) - 1
        alpha_param = self.mu * common_factor
        beta_param = (1 - self.mu) * common_factor
        return alpha_param, beta_param
    
    def cdf(self, x: float) -> float:
        return beta.cdf(x, self.alpha_param, self.beta_param)
    
    def pdf(self, x: float) -> float:
        return beta.pdf(x, self.alpha_param, self.beta_param)
    
    def __repr__(self):
        return f"BetaDistribution(alpha={self.alpha_param}, beta={self.beta_param}, mu={self.mu}, sigma={self.sigma})"


@register_confidence(name="distributional_length_normalised_log_likelihood")
def distributional_length_normalised_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    selected_responses, _ = semantic_uncertainty_selection(output_lst)
    lnlls = [list(map(lambda x: float(np.exp(np.mean(x))), output.output_logprobs)) for output in output_lst]
    lnlls = list(zip(*lnlls))  # transpose to per-question

    all_confidence_dists = []
    for lnll in lnlls:
        # fit a distributional estimator
        mu = np.mean(lnll)
        sigma = np.std(lnll, ddof=1)
        try:
            estimator = BetaDistribution(mu, sigma)
        except:
            estimator = None
        all_confidence_dists.append(estimator)

    return OrganisedOutputs(
        extracted_answers=[selected_responses],
        extracted_confidences=[all_confidence_dists]
    )