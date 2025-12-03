import time
from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
from models.model_manager import ModelManager
from default_utils.registry import register_confidence
import numpy as np


@register_confidence(name="p_true_by_continuation")
def p_true_by_continuation(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    cfg.qa_model.temperature = 1.0 
    model = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    true_continuation: PromptCollection = PromptCollection(context_texts=[], continuation_texts={})
    false_continuation: PromptCollection = PromptCollection(context_texts=[], continuation_texts={})
    p_true_prompt_template = """
    Question: {question}
    Proposed Answer: {proposed_answer}
    Is the proposed answer:
    A) True
    B) False

    The proposed answer is:
    """.strip()
    def per_round_estimator(outputs: ModelOutputs) -> list[float]:
        for question, proposed_answer in zip(outputs.context_texts, outputs.output_texts):
            eval_prompt = p_true_prompt_template.format(question=question, proposed_answer=proposed_answer)
            true_continuation.context_texts.append(eval_prompt)
            true_continuation.continuation_texts[eval_prompt] = [" True"]
            false_continuation.context_texts.append(eval_prompt)
            false_continuation.continuation_texts[eval_prompt] = [" False"]
        print("Scoring P(True) 'True' continuation token")
        true_results: ModelOutputs = model.run_continuation(true_continuation)[0]
        print("Scoring P(True) 'False' continuation token")
        false_results: ModelOutputs = model.run_continuation(false_continuation)[0]
        extracted_true_probs: list[float] = [np.exp(np.mean(logprobs)) for logprobs in true_results.output_logprobs]
        extracted_false_probs: list[float] = [np.exp(np.mean(logprobs)) for logprobs in false_results.output_logprobs]
        p_trues: list[float] = []
        for p_true, p_false in zip(extracted_true_probs, extracted_false_probs):
            p_true_norm = p_true / (p_true + p_false)
            p_trues.append(p_true_norm)
        return p_trues
    
    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts for outputs in output_lst],
        extracted_confidences=[per_round_estimator(outputs) for outputs in output_lst]
    )