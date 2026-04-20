import gc
import json
import csv
import logging
import os
import pickle
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import beta
import re
from ..confidence_metrics.semantics import semantic_cluster_selection, EntailmentDeberta
from ..default_utils.registry import register_confidence
from ..default_utils.custom_types import OrganisedOutputs, ModelOutputs, PromptCollection
from ..models.model_manager import ModelManager


SEMANTIC_CLUSTERING_PROMPT = """
You are a strict JSON generator. Group semantically equivalent candidate responses to the same question. Ignore any linguistic markers of uncertainty or hedging and focus solely on the core meaning of the responses. 

Return a JSON object with a single key `semantic_ids`, a list of integers aligned with the response order. Responses that are semantically equivalent (bidirectional entailment) must share the same integer id. Use 0-based ids. 
Semantic ids represent the semantic cluster assignment for each response. Return ONLY the JSON object, no extra text.

For instance, given the question and candidate responses:
Question: What is the capital of France?
Candidate responses:
    0: 'I guess Paris is the capital of France.'
    1: 'Paris is the capital city of France.'
    2: 'The capital of France is Berlin.' 

The correct JSON output would be:
{{"semantic_ids": [0, 0, 1]}}

Now, please group the following candidate responses to the given question and return the JSON object:
Question:
{question}

Candidate responses:
{responses}

{{"semantic_ids": [...]}}
"""

class BetaDistribution:
    def __init__(self, mu: float, sigma: float):

        __slots__ = ["mu", "sigma", "alpha_param", "beta_param"]

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


def _cache_file_candidates(cfg: dict, filename: str) -> list[str]:
    candidates: list[str] = []
    for base_path in [cfg.get("cache_path")]:
        if base_path:
            candidates.append(os.path.join(base_path, filename))
    return candidates


def _load_first_existing_pickle(paths: list[str]):
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def _save_pickle_to_results(cfg: dict, filename: str, payload) -> None:
    results_path = cfg.get("results_path")
    if not results_path:
        return
    os.makedirs(results_path, exist_ok=True)
    cache_file = os.path.join(results_path, filename)
    with open(cache_file, "wb") as f:
        pickle.dump(payload, f)


def _is_valid_majority_cluster_cache(cached_clusters, response_lists: list[tuple[str]]) -> bool:
    if not isinstance(cached_clusters, list) or len(cached_clusters) != len(response_lists):
        return False

    for cluster, responses in zip(cached_clusters, response_lists):
        if not isinstance(cluster, list):
            return False

        response_counts = Counter(list(responses))
        for text in cluster:
            if response_counts.get(text, 0) <= 0:
                return False
            response_counts[text] -= 1

    return True


def _load_or_build_majority_clusters(
    cfg: dict,
    response_lists: list[tuple[str]],
    question_texts: list[str] | None = None,
) -> tuple[list[list[str]], list[str]]:
    cache_path = cfg.get("cache_path")
    if cache_path:
        cache_file = os.path.join(cache_path, "majority_cluster_responses.pkl")
        if os.path.exists(cache_file):
            logging.info(f"Loading majority cluster responses from cache at {cache_file}")
            with open(cache_file, "rb") as f:
                cached_clusters = pickle.load(f)
            largest_clusters = cached_clusters
            selected_responses = [cluster[0] if cluster else "" for cluster in largest_clusters]
            _save_pickle_to_results(cfg, "majority_cluster_responses.pkl", largest_clusters)
            return largest_clusters, selected_responses

    cluster_cache_candidates = _cache_file_candidates(cfg, "majority_cluster_responses.pkl")
    cached_clusters = _load_first_existing_pickle(cluster_cache_candidates)

    if _is_valid_majority_cluster_cache(cached_clusters, response_lists):
        largest_clusters = cached_clusters
        selected_responses = [cluster[0] if cluster else "" for cluster in largest_clusters]
        _save_pickle_to_results(cfg, "majority_cluster_responses.pkl", largest_clusters)
        return largest_clusters, selected_responses

    if cached_clusters is not None:
        logging.warning("Ignoring incompatible majority_cluster_responses cache; recomputing semantic clusters.")

    _, largest_clusters, selected_responses = _build_semantic_majority_clusters(
        cfg,
        response_lists,
        question_texts=question_texts,
    )
    _save_pickle_to_results(cfg, "majority_cluster_responses.pkl", largest_clusters)
    return largest_clusters, selected_responses


def _build_semantic_majority_clusters(
    cfg: dict,
    response_lists: list[tuple[str]],
    question_texts: list[str] | None = None,
) -> tuple[list[list[int]], list[list[str]], list[str]]:
    semantic_ids_by_question: list[list[int]] = []
    largest_clusters: list[list[str]] = []
    selected_responses: list[str] = []

    def _parse_semantic_ids(text: str, expected_len: int) -> list[int] | None:
        if not text:
            return None

        stripped = text.strip()

        # Prefer a semantic_ids/cluster_ids list if present, otherwise fall back to the first list.
        key_match = re.search(
            r'"(?:semantic_ids|cluster_ids)"\s*:\s*\[([^\]]*)\]',
            stripped,
            flags=re.S,
        )
        list_match = key_match or re.search(r'\[([^\]]*)\]', stripped, flags=re.S)
        if not list_match:
            return None

        items = re.findall(r"-?\d+", list_match.group(1))
        if not items:
            return None

        ids = [int(x) for x in items]

        if len(ids) != expected_len or any(i < 0 for i in ids):
            return None

        return ids

    def _deberta_semantic_ids(entailment_model: EntailmentDeberta, responses: list[str]) -> list[int]:
        strict_entailment = False
        N = len(responses)
        entail_cache: dict[tuple[str, str], int] = {}

        def are_equivalent_from_scores(ij: int, ji: int) -> bool:
            if strict_entailment:
                return ij == 2 and ji == 2
            return (ij != 0) and (ji != 0) and not (ij == 1 and ji == 1)

        semantic_ids = [-1] * N
        cluster_representatives: list[int] = []

        for i in range(N):
            if not cluster_representatives:
                semantic_ids[i] = 0
                cluster_representatives.append(i)
                continue

            left, right, pair_keys = [], [], []
            for rep_idx in cluster_representatives:
                pairs = [
                    (responses[i], responses[rep_idx]),
                    (responses[rep_idx], responses[i]),
                ]
                for cache_key in pairs:
                    if cache_key not in entail_cache:
                        left.append(cache_key[0])
                        right.append(cache_key[1])
                        pair_keys.append(cache_key)

            if left:
                try:
                    results = entailment_model.check_implication_batch(
                        left, right, batch_size=256
                    )
                except Exception as e:
                    print(f"Error occurred: {e}. Falling back to smaller batch size.")
                    results = entailment_model.check_implication_batch(
                        left, right, batch_size=16
                    )

                for cache_key, res in zip(pair_keys, results):
                    entail_cache[cache_key] = res

            assigned_cluster = None
            for cluster_id, rep_idx in enumerate(cluster_representatives):
                ij = entail_cache[(responses[i], responses[rep_idx])]
                ji = entail_cache[(responses[rep_idx], responses[i])]
                if are_equivalent_from_scores(ij, ji):
                    assigned_cluster = cluster_id
                    break

            if assigned_cluster is None:
                assigned_cluster = len(cluster_representatives)
                cluster_representatives.append(i)

            semantic_ids[i] = assigned_cluster

        assert -1 not in semantic_ids
        return semantic_ids

    model_config_type = cfg.get("semantic_cluster_model_config", "clustering_model")
    model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type=model_config_type)

    prompt_texts: list[str] = []
    for idx, responses in enumerate(response_lists):
        question = ""
        if question_texts and idx < len(question_texts):
            question = question_texts[idx]
        response_lines = [f"    {idx}: '{text}'" for idx, text in enumerate(responses)]
        prompt_texts.append(
                SEMANTIC_CLUSTERING_PROMPT.strip().format(
                question=question,
                responses="\n".join(response_lines),
            )
        )

    prompt_collection = PromptCollection(
        system_prompt="You are a careful assistant.",
        context_texts=prompt_texts,
    )

    llm_outputs = model_manager.run_generation(prompt_collection)
    llm_texts = llm_outputs[0].output_texts if llm_outputs else []

    entailment_model = None

    for idx, responses in tqdm(
        list(enumerate(response_lists)),
        desc="Processing Semantic Clusters (LLM)",
    ):
        responses = list(responses)
        llm_text = llm_texts[idx] if idx < len(llm_texts) else ""
        parsed_ids = _parse_semantic_ids(llm_text, len(responses))

        if parsed_ids is None:
            if entailment_model is None:
                entailment_model = EntailmentDeberta()
            semantic_ids = _deberta_semantic_ids(entailment_model, responses)
        else:
            semantic_ids = parsed_ids

        counts = np.bincount(semantic_ids)
        majority_cluster = int(np.argmax(counts))
        majority_indices = np.where(np.array(semantic_ids) == majority_cluster)[0]
        majority_cluster_responses = [responses[idx] for idx in majority_indices]

        semantic_ids_by_question.append(semantic_ids)
        largest_clusters.append(majority_cluster_responses)
        selected_responses.append(majority_cluster_responses[0] if majority_cluster_responses else "")

    if entailment_model is not None:
        del entailment_model.model
        del entailment_model.tokenizer
        del entailment_model
        gc.collect()
        torch.cuda.empty_cache()

    del model_manager
    gc.collect()
    torch.cuda.empty_cache()

    return semantic_ids_by_question, largest_clusters, selected_responses


@register_confidence(name="distributional_length_normalised_log_likelihood")
def distributional_length_normalised_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    response_lists: list[tuple[str]] = list(zip(*[output.output_texts for output in output_lst]))
    logprob_lists: list[tuple] = list(zip(*[output.output_logprobs for output in output_lst]))
    largest_clusters, selected_responses = _load_or_build_majority_clusters(
        cfg,
        response_lists,
        question_texts=prompts.questions,
    )

    cluster_token_probs_dist = []
    for q_idx, (responses, response_logprobs) in enumerate(zip(response_lists, logprob_lists)):
        largest_cluster_responses = largest_clusters[q_idx] if q_idx < len(largest_clusters) else []

        if not largest_cluster_responses:
            cluster_token_probs_dist.append(BetaDistribution(mu=0.5, sigma=1e-6))
            continue

        # Match cached cluster texts to response/logprob pairs while preserving multiplicity for duplicates.
        remaining = {}
        for text in largest_cluster_responses:
            remaining[text] = remaining.get(text, 0) + 1

        largest_cluster_logprobs = []
        for text, logprob in zip(responses, response_logprobs):
            if remaining.get(text, 0) <= 0:
                continue
            remaining[text] -= 1

            if logprob is None:
                continue
            try:
                if len(logprob) == 0:
                    continue
            except TypeError:
                logprob = [logprob]
            largest_cluster_logprobs.append(logprob)

        cluster_probs = [np.exp(np.mean(x)) for x in largest_cluster_logprobs]
        mu = float(np.mean(cluster_probs)) if len(cluster_probs) > 0 else 0.5
        sigma = float(np.std(cluster_probs)) if len(cluster_probs) > 1 else 1e-6
        cluster_token_probs_dist.append(BetaDistribution(mu=mu, sigma=sigma))

    return OrganisedOutputs(
        extracted_answers=[selected_responses],
        extracted_confidences=[cluster_token_probs_dist]
    )


@register_confidence(name="distributional_semantic_uncertainty")
def distributional_semantic_uncertainty(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # Extract responses: response_lists[question_idx] = tuple of 20 responses (one per round)
    response_lists: list[tuple[str]] = list(zip(*[output.output_texts for output in output_lst]))
    selected_responses = []
    confidence_dists = []
    response_cache_candidates = _cache_file_candidates(cfg, "selected_responses.pkl")
    conf_cache_candidates = _cache_file_candidates(cfg, "confidence_cache.pkl")
    cached_selected = _load_first_existing_pickle(response_cache_candidates)
    cached_conf = _load_first_existing_pickle(conf_cache_candidates)
    largest_clusters, _ = _load_or_build_majority_clusters(
        cfg,
        response_lists,
        question_texts=prompts.questions,
    )
    if cached_selected is not None and cached_conf is not None:
        selected_responses = cached_selected
        confidence_dists = cached_conf
        logging.info("Loaded cached responses and confidences for semantic uncertainty")
    else:
        for responses, majority_cluster_responses in zip(response_lists, largest_clusters):
            if not responses or not majority_cluster_responses:
                confidence_dists.append(BetaDistribution(mu=0.5, sigma=1e-6))
                continue

            alpha = len(majority_cluster_responses)
            beta = len(responses) - alpha
            # Clamp to ensure valid Beta parameters
            alpha = max(alpha, 1)
            beta = max(beta, 1)
            mu = float(alpha) / (alpha + beta)
            sigma = np.sqrt((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)))
            confidence_dists.append(BetaDistribution(mu=mu, sigma=sigma))

        # Always select the first response from each largest cluster.
        selected_responses = [cluster[0] if cluster else "" for cluster in largest_clusters]

        _save_pickle_to_results(cfg, "selected_responses.pkl", selected_responses)
        _save_pickle_to_results(cfg, "confidence_cache.pkl", confidence_dists)
        logging.info(f"Cached responses and confidences saved to {cfg.get('results_path')}")

    return OrganisedOutputs(
        extracted_answers=[selected_responses],
        extracted_confidences=[confidence_dists]
    )


@register_confidence(name="distributional_linguistic_confidence")
def distributional_linguistic_confidence(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    # use llm judge to rate how decisive the model's answer is
    evaluator_lst = cfg.get("evaluator_list", ["linguistic_confidence_judge_model_0"])

    # Extract responses per question across rounds.
    response_lists: list[tuple[str]] = list(zip(*[output.output_texts for output in output_lst]))
    largest_clusters, _ = _load_or_build_majority_clusters(
        cfg,
        response_lists,
        question_texts=prompts.questions,
    )
    
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
                If the sentence abstains from answering by pointing out the insufficiency of information with a firm tone, the score should be towards 100. \
                    If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. \
                        If the sentence does not contain any hedging language or is a succinct, decisive short answer, the score should be towards 100, too.

    Here is the sentence:
    Answer: {sentence}

    Confidence Score: [Return only a number between 0 and 100]
    """.strip()
    
    # Flatten prompts while keeping the owning question index.
    all_cluster_responses: list[str] = []
    response_question_indices: list[int] = []
    selected_answers: list[str] = []

    for q_idx, cluster in enumerate(largest_clusters):
        if not cluster:
            selected_answers.append("")
            continue
        selected_answers.append(cluster[0])
        all_cluster_responses.append(cluster[0])
        response_question_indices.append(q_idx)

    if not all_cluster_responses:
        default_conf = [BetaDistribution(mu=0.5, sigma=1e-6) for _ in selected_answers]
        return OrganisedOutputs(
            extracted_answers=[selected_answers],
            extracted_confidences=[default_conf],
        )

    # scores_collection[question_idx] = list of scores pooled across all cluster responses and evaluators
    scores_collection: list[list[float]] = [[] for _ in selected_answers]

    def _load_cached_scores(cache_path: str, expected_len: int) -> list[float] | None:
        try:
            scores: list[float | None] = [None] * expected_len
            with open(cache_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row["response_index"])
                    score = float(row["score"])
                    if 0 <= idx < expected_len:
                        scores[idx] = score
            if any(s is None for s in scores):
                return None
            return [float(s) for s in scores]
        except Exception:
            return None

    def _save_cached_scores(cache_path: str, judge_outputs: list[ModelOutputs]) -> None:
        with open(cache_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["response_index", "score"])
            writer.writeheader()
            response_index = 0
            for judge_out in judge_outputs:
                for judge_text in judge_out.output_texts:
                    score = extract_score(judge_text)
                    writer.writerow({"response_index": response_index, "score": score})
                    response_index += 1
        logging.info(f"Cached scores saved to {cache_path}")
    
    # Loop through each evaluator
    results_path = cfg.get("results_path")

    for evaluator in evaluator_lst:
        cache_path = None
        if results_path:
            os.makedirs(results_path, exist_ok=True)
            cache_path = os.path.join(results_path, f"linguistic_confidence_scores_{evaluator.split('/')[-1]}.csv")

        if cache_path and os.path.exists(cache_path):
            cached_scores = _load_cached_scores(cache_path, len(all_cluster_responses))
            if cached_scores is not None:
                continue
            logging.warning(
                "Cached scores found but invalid for evaluator %s; recomputing.",
                evaluator,
            )

        model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type=evaluator)
        
        # Generate prompts for every response in each question's largest semantic cluster.
        prompt_collection = PromptCollection(context_texts=[DIRECT_PROMPT.format(sentence=response) for response in all_cluster_responses])
        judge_outputs: list[ModelOutputs] = model_manager.run_generation(prompt_collection)

        if cache_path:
            _save_cached_scores(cache_path, judge_outputs)
        else:
            logging.warning(
                "No results_path configured; cannot cache scores for evaluator %s.",
                evaluator,
            )

        del model_manager
        gc.collect()
        torch.cuda.empty_cache()

    # Load cached scores after loop to avoid holding them in memory during evaluation.
    if results_path:
        for evaluator in evaluator_lst:
            cache_path = os.path.join(results_path, f"linguistic_confidence_scores_{evaluator.split('/')[-1]}.csv")
            cached_scores = _load_cached_scores(cache_path, len(all_cluster_responses))
            if cached_scores is None:
                logging.warning(
                    "Missing or invalid cached scores for evaluator %s; skipping.",
                    evaluator,
                )
                continue
            for q_idx, score in zip(response_question_indices, cached_scores):
                if np.isnan(score):
                    continue
                scores_collection[q_idx].append(score)

    # Fit beta distributions from all collected scores for each question's largest cluster.
    confidence_dists = []
    for q_idx, _ in enumerate(selected_answers):
        all_scores = scores_collection[q_idx]
        valid_scores = [s for s in all_scores if not np.isnan(s)]

        if valid_scores:
            mu = float(np.mean(valid_scores))
            sigma = float(np.std(valid_scores)) if len(valid_scores) > 1 else 1e-6
        else:
            mu = 0.5
            sigma = 1e-6
        
        confidence_dists.append(BetaDistribution(mu=mu, sigma=sigma))

    return OrganisedOutputs(
        extracted_answers=[selected_answers],
        extracted_confidences=[confidence_dists],
    )

