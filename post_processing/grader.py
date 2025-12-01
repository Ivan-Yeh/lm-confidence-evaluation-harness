from default_utils.custom_types import OrganisedOutputs, PromptCollection, ModelOutputs
from default_utils.registry import register_grader


@register_grader(name="exact_match")
def exact_match(cfg: dict, extracted_output: OrganisedOutputs, prompts: PromptCollection):
    exact_matches = []
    answer_keys = prompts.answer_keys
    for round_outputs in extracted_output.extracted_answers:
        round_matches = []
        for pred, ref in zip(round_outputs, answer_keys):
            round_matches.append(1 if pred.strip().lower() == ref.strip().lower() else 0)
        exact_matches.append(round_matches)
    return exact_matches


@register_grader(name="llm_grader")
def llm_grader(cfg: dict, extracted_output: OrganisedOutputs, prompts: PromptCollection):
    pass