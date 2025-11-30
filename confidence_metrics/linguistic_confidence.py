
from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
import numpy as np


def decisiveness_score(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # use llm judge to rate how decisive the model's answer is
    pass