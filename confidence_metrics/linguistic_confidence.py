from default_utils.registry import register_confidence
from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
from models.model_manager import ModelManager
import re


@register_confidence(name="scalar_linguistic_confidence")
def scalar_linguistic_confidence(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # use llm judge to rate how decisive the model's answer is
    model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type="linguistic_confidence_judge_model")
    def per_round_estimator(output: ModelOutputs):
        DIRECT_PROMPT = """
        Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge): 
        {sentence}

        Confidence Score: [Return only a number between 0 and 100]
        """.strip()
        prompt_collection = PromptCollection(context_texts=[DIRECT_PROMPT.format(sentence=response) for response in output.output_texts])
        judge_outputs: ModelOutputs = model_manager.run_generation(prompt_collection)[0]
        scores = []
        for text in judge_outputs.output_texts:
            # Extract confidence score using regex
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                score = float(match.group(1))
                score = min(max(score, 0.0), 100.0)
            else:
                score = 0.0
            scores.append(score / 100.0)  # normalise to [0, 1]
        return scores
    
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