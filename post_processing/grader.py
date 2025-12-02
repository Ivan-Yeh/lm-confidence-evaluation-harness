from default_utils.custom_types import OrganisedOutputs, PromptCollection, ModelOutputs
from default_utils.registry import register_grader
from models.model_manager import ModelManager


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
    model = ModelManager(master_cfg=cfg, model_config_type="grader_model")
    correct_answers = prompts.answer_keys
    questions = prompts.context_texts
    all_scores = []
    for round_outputs in extracted_output.extracted_answers:
        grading_prompts = PromptCollection(context_texts=[])
        round_scores = []
        for question, predicted_answer, correct_answer in zip(questions, round_outputs, correct_answers):
            grading_prompt = f"""
            Your job is to look at a question with a correct answer and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
            If the predicted answer matches, implies or covers the correct answer, the grade is CORRECT.
            If the predicted answer does not match, imply or cover the correct answer, the grade is INCORRECT.
            If the predicted answer is empty or punts the question, the grade is NOT_ATTEMPTED.
            Ignore any explanation present in the predicted answer. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
            
            ```
            Question: {question}
            Correct answer: {correct_answer}
            Predicted answer: {predicted_answer}
            ```

            Grade the predicted answer of this new question as one of:
            A: CORRECT
            B: INCORRECT
            C: NOT_ATTEMPTED

            Just return the letters "A", "B", or "C", with no text around it.
            """.strip()
            grading_prompts.context_texts.append(grading_prompt)
        
        for output_text in model.run_generation(grading_prompts)[0].output_texts:
            if "A" in output_text or "CORRECT" in output_text.upper():
                round_scores.append(1)
            elif "B" in output_text or "INCORRECT" in output_text.upper():
                round_scores.append(0)
            else:
                round_scores.append(None)
        all_scores.append(round_scores)
    return all_scores