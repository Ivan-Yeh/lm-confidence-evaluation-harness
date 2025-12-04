import time
from default_utils.custom_types import ModelOutputs, PromptCollection, OrganisedOutputs
from models.model_manager import ModelManager
from default_utils.registry import register_confidence
import numpy as np


@register_confidence(name="p_true_by_continuation")
def p_true_by_continuation(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    model = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    true_continuation: PromptCollection = PromptCollection(context_texts=[], continuation_texts={})
    false_continuation: PromptCollection = PromptCollection(context_texts=[], continuation_texts={})
    p_true_prompt_template = """
    Question: Who was the third president of the United States?
    Here are some brainstormed ideas: James Monroe\n Thomas Jefferson\n Jefferson\nThomas Jefferson\n George Washington
    Possible Answer: James Monroe
    Is the possible answer:
    (A) True
    (B) False
    The possible answer is: (B)

    Question: Calculate 33 + 4
    Here are some brainstormed ideas: 37\n 37\n 40\n 36\n 37
    Possible Answer: 37
    Is the possible answer:
    (A) True
    (B) False
    The possible answer is: (A)

    Question: Fill in the blank in the sentence \"I went to the grocery and then to the pharmacy. I was disappointed that they didn't have any vegetarian sausage at the _____\."
    Here are some brainstormed ideas: grocery\n store\n grocery\n refrigerator\n grocery
    Possible Answer: grocery
    Is the possible answer:
    (A) True
    (B) False
    The possible answer is: (A)

    Question: Name a celebrated civil rights leader.
    Here are some brainstormed ideas: Martin Luther King\n Ghandhi\n Martin Luther
    King\n Barack Obama\n Martin Luther King
    Possible Answer: Martin Luther King
    25Is the possible answer:
    (A) True
    (B) False
    The possible answer is: (A)

    Question: Calculate 33 * 849
    Here are some brainstormed ideas: 28,347\n 1,490\n 27,488\n 3,409\n 34,561
    Possible Answer: 28347
    Is the possible answer:
    (A) True
    (B) False
    The possible answer is: (B)

    Question: Fill in the blank in the sentence \"I shot the _____ and it went swish. We walked away the winners of that battle!\"
    Here are some brainstormed ideas: gun\n bullet\n arrow\n basketball\n rifle
    Possible Answer: gun
    Is the possible answer:
    (A) True
    (B) False
    The possible answer is: (B)

    Question: {question}
    Possible Answer: {model_answer}
    Is the possible answer:
    (A) True
    (B) False
    The possible answer is:
    """.strip()
    def per_round_estimator(outputs: ModelOutputs) -> list[float]:
        for question, model_answer in zip(outputs.context_texts, outputs.output_texts):
            eval_prompt = p_true_prompt_template.format(question=question, model_answer=model_answer)
            true_continuation.context_texts.append(eval_prompt)
            true_continuation.continuation_texts[eval_prompt] = [" (A)", " True", "(A)"]
            false_continuation.context_texts.append(eval_prompt)
            false_continuation.continuation_texts[eval_prompt] = [" (B)", " False", "(B)"]
        print("Scoring P(True) 'True' continuation token")
        true_results: ModelOutputs = model.run_continuation(true_continuation)[0]
        print("Scoring P(True) 'False' continuation token")
        false_results: ModelOutputs = model.run_continuation(false_continuation)[0]
        extracted_true_probs: list[float] = [np.exp(logprobs[-1]) for logprobs in true_results.output_logprobs]
        extracted_false_probs: list[float] = [np.exp(logprobs[-1]) for logprobs in false_results.output_logprobs]
        p_trues: list[float] = []
        for p_true, p_false in zip(extracted_true_probs, extracted_false_probs):
            p_true_norm = p_true / (p_true + p_false + 1e-10)
            p_trues.append(float(p_true_norm))
        return p_trues
    
    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts for outputs in output_lst],
        extracted_confidences=[per_round_estimator(outputs) for outputs in output_lst]
    )