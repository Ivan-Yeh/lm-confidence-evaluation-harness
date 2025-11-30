from typer import prompt
from default_utils.custom_types import ModelOutputs, PromptCollection
from models.model_manager import ModelManager
import numpy as np


def p_true_by_continuation(cfg, outputs: ModelOutputs) -> list:

    model = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    true_continuation: PromptCollection = PromptCollection(context_texts=[], continuation_texts={})
    false_continuation: PromptCollection = PromptCollection(context_texts=[], continuation_texts={})

    for question, proposed_answer in zip(outputs.context_texts, outputs.output_texts):
        p_true_prompt_template = """
        Question: {question}
        Proposed Answer: {proposed_answer}
        Is the proposed answer:
         A) True
         B) False
        """.strip()
        eval_prompt = p_true_prompt_template.format(question=question, proposed_answer=proposed_answer)
        true_continuation.context_texts.append(eval_prompt)
        true_continuation.continuation_texts[eval_prompt] = ["A"]
        false_continuation.context_texts.append(eval_prompt)
        false_continuation.continuation_texts[eval_prompt] = ["B"]
    
    true_results: ModelOutputs = model.run_continuation(true_continuation)[0]
    false_results: ModelOutputs = model.run_continuation(false_continuation)[0]

    extracted_true_probs: list[float] = [np.exp(np.mean(logprobs)) for logprobs in true_results.output_logprobs]
    extracted_false_probs: list[float] = [np.exp(np.mean(logprobs)) for logprobs in false_results.output_logprobs]

    p_trues: list[float] = []
    for p_true, p_false in zip(extracted_true_probs, extracted_false_probs):
        p_true_norm = p_true / (p_true + p_false)
        p_trues.append(p_true_norm)

    return p_trues


