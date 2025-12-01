from default_utils.registry import register_confidence
from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
import numpy as np


@register_confidence(name="decisiveness_score")
def decisiveness_score(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # use llm judge to rate how decisive the model's answer is
    pass