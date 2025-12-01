from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
import numpy as np
from default_utils.registry import register_confidence


@register_confidence(name="verbalised_numerical_confidence")
def verbalised_numerical_confidence(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    pass


@register_confidence(name="discrete_verbalised_confidence_by_continuation")
def discrete_verbalised_confidence_by_continuation(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # for each vnc response, score the continuation of confidence values from 0 to 100 in steps of 5
    pass
