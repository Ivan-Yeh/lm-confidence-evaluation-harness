from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
import numpy as np


def verbalised_numerical_confidence(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    outputs = output_lst[0]
    verbalised_confidences: list[float] = []

    return verbalised_confidences


def discrete_verbalised_confidence_by_continuation():
    # for each vnc response, score the continuation of confidence values from 0 to 100 in steps of 5
    pass