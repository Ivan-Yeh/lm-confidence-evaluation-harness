import pandas as pd
from jinja2 import Template
from .datasets_manager import DatasetsManager
from .custom_types import PromptCollection


def get_multiple_choice_prompts(cfg: dict, dataset_manager: DatasetsManager) -> PromptCollection:
    question_template: Template = Template(cfg.get("question_format", "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\n\n"))
    prompt_template: Template = Template(cfg.get("prompt_format", "{{few_shot_examples}}\n\n{{formatted_question}}\nAnswer:"))
    prompts = []
    few_shot = dataset_manager.few_shot_examples if dataset_manager.few_shot_examples else 0
    # Iterate through the dataset
    continuation_texts = dict()
    formatted_questions = []
    for _, row in dataset_manager.get_dataset().iterrows():
        # Build few-shot examples if requested
        few_shot_text = ""
        if few_shot > 0:
            few_shot_examples: pd.DataFrame = dataset_manager.get_few_shot_dataset().sample(n=few_shot, random_state=cfg.get("seed", 42))
            for _, example in few_shot_examples.iterrows():
                few_shot_text += question_template.render(**example.to_dict())
                few_shot_text += f"Answer: {chr(65 + example['answer_index'])}\n\n"
        formatted_question = question_template.render(**row.to_dict()).strip()
        formatted_questions.append(formatted_question)
        base_prompt = prompt_template.render(few_shot_examples=few_shot_text.strip(), formatted_question=formatted_question).strip()
        continuations = []
        for i in range(len(row['choices'])):
            continuations.append(chr(65 + i))  # Append choice letters A, B, C, ...
        prompts.append(base_prompt)
        continuation_texts[base_prompt] = continuations
    return PromptCollection(questions=formatted_questions, answer_keys=[chr(65 + idx) for idx in dataset_manager.get_dataset()['answer_index'].tolist()], system_prompt=cfg.get("system_prompt", ""), context_texts=prompts, continuation_texts=continuation_texts)