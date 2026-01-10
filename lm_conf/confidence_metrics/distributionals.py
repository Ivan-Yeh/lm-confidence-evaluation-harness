import gc
import logging
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import beta
import re
from ..confidence_metrics.semantics import semantic_uncertainty_selection, EntailmentDeberta
from ..default_utils.registry import register_confidence
from ..default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
from ..models.model_manager import ModelManager

class BetaDistribution:
    def __init__(self, mu: float, sigma: float):
        self.mu = np.clip(mu, 1e-6, 1 - 1e-6)

        max_sigma2 = self.mu * (1 - self.mu)
        sigma2 = min(sigma**2, max_sigma2 * 0.999)

        self.sigma = np.sqrt(max(sigma2, 1e-8))
        self.alpha_param, self.beta_param = self._fit_parameters()

    def _fit_parameters(self):
        var = self.sigma ** 2
        common = self.mu * (1 - self.mu) / var - 1.0

        alpha = self.mu * common
        beta = (1 - self.mu) * common

        return max(alpha, 1e-6), max(beta, 1e-6)
    
    def sample(self, size: int = 1) -> np.ndarray:
        return beta.rvs(self.alpha_param, self.beta_param, size=size)
    
    def is_valid(self) -> bool:
        try:
            assert isinstance(float(self.mu), float)
            assert isinstance(float(self.sigma), float)
            assert isinstance(float(self.alpha_param), float)
            assert isinstance(float(self.beta_param), float)
            # Check that values are not NaN or inf
            assert not np.isnan(self.mu) and not np.isinf(self.mu)
            assert not np.isnan(self.sigma) and not np.isinf(self.sigma)
            assert not np.isnan(self.alpha_param) and not np.isinf(self.alpha_param)
            assert not np.isnan(self.beta_param) and not np.isinf(self.beta_param)
            # Check that parameters are positive
            assert self.alpha_param > 0 and self.beta_param > 0
            return True
        except:
            return False
    
    def __repr__(self):
        return f"BetaDistribution(alpha={self.alpha_param}, beta={self.beta_param}, mu={self.mu}, sigma={self.sigma})"


@register_confidence(name="distributional_length_normalised_log_likelihood")
def distributional_length_normalised_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    outputs = output_lst[0]
    responses = outputs.output_texts
    # for each response, generate 10 semantically equivalent responses using an llm
    model = ModelManager(master_cfg=cfg, model_config_type="semantic_perturbation_model")
    perturbed_prompts = PromptCollection(context_texts=[
        f"""Generate 9 more semantically equivalent variations of the following response:
        Answer: {response}

        Do not change the meaning of the original response. Provide the variations as a list in the following format, including the provided answer itself. 
        Please only return the list of strings. Start with an opening bracket "[" and end with a closing bracket "]", 
        without any additional text. Place each item in a new line.
        [
            "variation 1",
            "variation 2",
            ...
            "variation 10"
        ]
        """
        for response in responses
    ])
    perturbed_outputs: ModelOutputs = model.run_generation(perturbed_prompts)[0]

    # extract the variations from the outputs using literal_eval
    from ast import literal_eval
    extracted_variations: list[list[str]] = []
    for out_text in perturbed_outputs.output_texts:
        try:
            parsed = literal_eval(out_text)
            if isinstance(parsed, list) and all(isinstance(v, str) for v in parsed):
                extracted_variations.append(parsed)
            else:
                extracted_variations.append(out_text.replace("[", "").replace("]", "").splitlines())
        except:
            extracted_variations.append(out_text.replace("[", "").replace("]", "").splitlines())

    continuations = {
        ctx: conts for ctx, conts in zip(outputs.context_texts, extracted_variations)
    }
    qa_model = ModelManager(master_cfg=cfg, model_config_type="qa_model")
    lnll_outputs = qa_model.run_continuation(PromptCollection(
        context_texts=outputs.context_texts,
        continuation_texts=continuations
    ))[0]

    confidence_dists: list[BetaDistribution] = []
    for response_conts in lnll_outputs.continuation_candidates:
        candidates_probs = [np.exp(d["mean"]) for d in response_conts]
        mu = float(np.mean(candidates_probs))
        sigma = float(np.std(candidates_probs))
        confidence_dists.append(BetaDistribution(mu=mu, sigma=sigma))

    return OrganisedOutputs(
        extracted_answers=[outputs.output_texts],
        extracted_confidences=[confidence_dists]
    )


@register_confidence(name="distributional_semantic_uncertainty")
def distributional_semantic_uncertainty(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    response_lists: list[tuple[str]] = list(zip(*[outputs.output_texts for outputs in output_lst]))
    entailment_model = EntailmentDeberta()
    confidence_dists: list[BetaDistribution] = []
    selected_responses: list[str] = []

    for responses in tqdm(response_lists, desc="Processing Entailments (Entailment Probability)"):
        n = len(responses)
        premises: list[str] = []
        hypotheses: list[str] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    premises.append(responses[i])
                    hypotheses.append(responses[j])

        preds = entailment_model.entailment_probability_batch(premises, hypotheses) if premises else []
        preds = np.clip(np.array(preds), 1e-6, 1 - 1e-6)
        
        # Compute mean entailment probability for each response (as premise)
        mean_entailment_per_response = []
        for i in range(n):
            # Get entailment probs where response i is the premise
            idx_start = i * (n - 1)
            idx_end = idx_start + (n - 1)
            response_preds = preds[idx_start:idx_end] if preds else []
            mean_ent = np.mean(response_preds) if len(response_preds) > 0 else 0.0
            mean_entailment_per_response.append(mean_ent)
        
        # Select response with highest mean entailment probability
        best_idx = np.argmax(mean_entailment_per_response) if mean_entailment_per_response else 0
        selected_responses.append(responses[best_idx])
        
        confidence_dists.append(BetaDistribution(mu=np.mean(preds), sigma=np.std(preds, ddof=1)))

    entailment_model.model.to("cpu")
    del entailment_model.model
    del entailment_model.tokenizer
    del entailment_model
    gc.collect()
    torch.cuda.empty_cache()

    return OrganisedOutputs(
        extracted_answers=[selected_responses],
        extracted_confidences=[confidence_dists]
    )


@register_confidence(name="distributional_linguistic_confidence")
def distributional_linguistic_confidence(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # use llm judge to rate how decisive the model's answer is
    model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type="linguistic_confidence_judge_model")
    def per_round_estimator(output: ModelOutputs):
        def extract_score(text: str) -> float:
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                score = float(match.group(1))
                score = min(max(score, 0.0), 100.0) / 100.0  # Normalize to [0, 1]
            else:
                score = np.nan
            return score
        
        DIRECT_PROMPT = """
        Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence (without using any external or prior knowledge). \
            Please pay attention to the hedging language used and the overall assertiveness of the statement. \
                If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. \
                    If the sentence does not contain any linguistic cues or is a succinct, decisive short answer, the score should be towards 100, too.

        Here is the sentence:
        {sentence}

        Confidence Score: [Return only a number between 0 and 100]
        """.strip()
        prompt_collection = PromptCollection(context_texts=[DIRECT_PROMPT.format(sentence=response) for response in output.output_texts])
        judge_outputs: list[ModelOutputs] = model_manager.run_generation(prompt_collection)
        confidences: list[list[float]] = [[extract_score(text) for text in out.output_texts] for out in judge_outputs]
        # Transpose confidences: from [num_judge_outputs][num_texts] to [num_texts][num_judge_outputs]
        confidences = list(map(list, zip(*confidences)))
        # Average scores from different judge outputs
        confidence_dists = [BetaDistribution(mu=float(np.nanmean(scores)), sigma=float(np.nanstd(scores, ddof=1))) for scores in confidences]
        return confidence_dists
    
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


# @register_confidence(name="distributional_p_true_mc")
# def distributional_p_true_mc(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
#     p_true_prompt_template = """
#     Question: {question}
#     Possible Answer: {model_answer}
#     Is the possible answer:
#     (A) True
#     (B) False
#     Return only (A) or (B). The possible answer is:
#     """.strip()
#     model = ModelManager(master_cfg=cfg, model_config_type="p_true_mc_model")

#     outputs = output_lst[0]
#     # Build prompts fresh per round to avoid leaking state across evaluations
#     p_true_prompt_collection: PromptCollection = PromptCollection(context_texts=[])
#     for question, model_answer in zip(outputs.context_texts, outputs.output_texts):
#         if model_answer:
#             eval_prompt = p_true_prompt_template.format(question=question, model_answer=model_answer)
#         else:
#             eval_prompt = p_true_prompt_template.format(question=question, model_answer="No answer provided.")
#         p_true_prompt_collection.context_texts.append(eval_prompt)

#     logging.info("Running P(True) by Monte Carlo generation") 
#     p_true_results: list[ModelOutputs] = model.run_generation(p_true_prompt_collection)

#     # Each ModelOutputs in p_true_results holds responses for the same set of
#     # questions; accumulate counts per question across all rounds.
#     num_questions = len(outputs.context_texts)
#     counts_a = [0] * num_questions
#     counts_b = [0] * num_questions

#     for result in p_true_results:
#         logging.debug(f"Monte Carlo generation result: {result}")
#         for idx, output in enumerate(result.output_texts):
#             # Extract last occurrence of ANSWER: (A)/(B) or ANSWER IS: (A)/(B)
#             match = None
#             for pattern in [r'ANSWER\s*IS\s*:*\s*\(([AB])\)', r'ANSWER\s*:*\s*\(([AB])\)', r'\s*\(([AB])\)']:
#                 matches = list(re.finditer(pattern, output.upper()))
#                 if matches:
#                     match = matches[-1]  # Get last occurrence
#                     break
            
#             if match:
#                 choice = match.group(1)
#                 if choice == 'A':
#                     counts_a[idx] += 1
#                 elif choice == 'B':
#                     counts_b[idx] += 1
#             else:
#                 # Fallback to simpler token matching
#                 upper = output.upper()
#                 if any(token in upper for token in ["(A)", "A", "TRUE"]):
#                     counts_a[idx] += 1
#                 elif any(token in upper for token in ["(B)", "B", "FALSE"]):
#                     counts_b[idx] += 1

#     extracted_true_probs_dists: list[BetaDistribution | None] = []
#     for a_count, b_count in zip(counts_a, counts_b):
#         try:
#             alpha_param = a_count + 1
#             beta_param = b_count + 1
#             mu = alpha_param / (alpha_param + beta_param)
#             sigma = np.sqrt((alpha_param * beta_param) / ((alpha_param + beta_param)**2 * (alpha_param + beta_param + 1)))
#             extracted_true_probs_dists.append(BetaDistribution(mu, sigma))
#         except:
#             extracted_true_probs_dists.append(None)
    
#     return OrganisedOutputs(
#         extracted_answers=[outputs.output_texts],
#         extracted_confidences=[extracted_true_probs_dists],
#     )

