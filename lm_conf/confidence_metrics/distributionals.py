import gc
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
    for base_path in [cfg.get("filtered_output_path"), cfg.get("results_path")]:
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


def _load_or_build_majority_clusters(cfg: dict, response_lists: list[tuple[str]]) -> tuple[list[list[str]], list[str]]:
    cluster_cache_candidates = _cache_file_candidates(cfg, "majority_cluster_responses.pkl")
    cached_clusters = _load_first_existing_pickle(cluster_cache_candidates)

    if _is_valid_majority_cluster_cache(cached_clusters, response_lists):
        largest_clusters = cached_clusters
        selected_responses = [cluster[0] if cluster else "" for cluster in largest_clusters]
        _save_pickle_to_results(cfg, "majority_cluster_responses.pkl", largest_clusters)
        return largest_clusters, selected_responses

    if cached_clusters is not None:
        logging.warning("Ignoring incompatible majority_cluster_responses cache; recomputing semantic clusters.")

    _, largest_clusters, selected_responses = _build_semantic_majority_clusters(cfg, response_lists)
    _save_pickle_to_results(cfg, "majority_cluster_responses.pkl", largest_clusters)
    return largest_clusters, selected_responses


def _build_semantic_majority_clusters(cfg: dict, response_lists: list[tuple[str]]) -> tuple[list[list[int]], list[list[str]], list[str]]:
    semantic_ids_by_question: list[list[int]] = []
    largest_clusters: list[list[str]] = []
    selected_responses: list[str] = []

    entailment_model = EntailmentDeberta()
    strict_entailment = False

    for responses in tqdm(
        response_lists,
        desc="Processing Entailments (Semantic Clusters)"
    ):
        responses = list(responses)
        N = len(responses)

        entail_cache = {}

        def are_equivalent_from_scores(ij, ji):
            if strict_entailment:
                return ij == 2 and ji == 2
            return (ij != 0) and (ji != 0) and not (ij == 1 and ji == 1)

        # Assign each response by checking only one representative per existing cluster.
        # Under transitivity assumption, this avoids building a full O(N^2) entailment graph.
        semantic_ids = [-1] * N
        cluster_representatives: list[int] = []

        for i in range(N):
            if not cluster_representatives:
                semantic_ids[i] = 0
                cluster_representatives.append(i)
                continue

            left, right = [], []
            pair_keys = []
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
                    batch_size = 256
                    results = entailment_model.check_implication_batch(
                        left, right, batch_size=batch_size
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

        counts = np.bincount(semantic_ids)
        majority_cluster = int(np.argmax(counts))
        majority_indices = np.where(np.array(semantic_ids) == majority_cluster)[0]
        majority_cluster_responses = [responses[idx] for idx in majority_indices]

        semantic_ids_by_question.append(semantic_ids)
        largest_clusters.append(majority_cluster_responses)
        selected_responses.append(majority_cluster_responses[0] if majority_cluster_responses else "")

    del entailment_model.model
    del entailment_model.tokenizer
    del entailment_model
    gc.collect()
    torch.cuda.empty_cache()

    return semantic_ids_by_question, largest_clusters, selected_responses


@register_confidence(name="distributional_length_normalised_log_likelihood")
def distributional_length_normalised_log_likelihood(cfg: dict, output_lst: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> OrganisedOutputs:
    response_lists: list[tuple[str]] = list(zip(*[output.output_texts for output in output_lst]))
    logprob_lists: list[tuple] = list(zip(*[output.output_logprobs for output in output_lst]))
    largest_clusters, selected_responses = _load_or_build_majority_clusters(cfg, response_lists)

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
    """
    Process 30 rounds of responses per question.
    Fix the first response as final answer, then:
    - Repeat 10 times: subsample 9 random responses from remaining 29
    - Calculate semantic cluster proportion for fixed response
    - Average confidences across 10 subsamples
    """
    # Extract responses: response_lists[question_idx] = tuple of 30 responses (one per round)
    response_lists: list[tuple[str]] = list(zip(*[output.output_texts for output in output_lst]))
    
    selected_responses = []
    confidence_dists = []

    response_cache_candidates = _cache_file_candidates(cfg, "selected_responses.pkl")
    conf_cache_candidates = _cache_file_candidates(cfg, "confidence_cache.pkl")
    cached_selected = _load_first_existing_pickle(response_cache_candidates)
    cached_conf = _load_first_existing_pickle(conf_cache_candidates)

    largest_clusters, _ = _load_or_build_majority_clusters(cfg, response_lists)

    if cached_selected is not None and cached_conf is not None:
        selected_responses = cached_selected
        confidence_dists = cached_conf
        logging.info("Loaded cached responses and confidences for semantic uncertainty")
    else:
        semantic_ids_by_question, majority_clusters, selected_responses = _build_semantic_majority_clusters(cfg, response_lists)
        _save_pickle_to_results(cfg, "majority_cluster_responses.pkl", majority_clusters)

        for semantic_ids in semantic_ids_by_question:
            semantic_ids_arr = np.array(semantic_ids)
            counts = np.bincount(semantic_ids_arr)
            majority_cluster = int(np.argmax(counts))

            subsample_confidences = []
            indices = np.arange(len(semantic_ids_arr))

            for _ in range(50):
                sampled = np.random.choice(indices, size=10, replace=False)
                support = np.mean([semantic_ids_arr[i] == majority_cluster for i in sampled])
                subsample_confidences.append(support)

            avg_confidence = float(np.mean(subsample_confidences))
            std_confidence = float(np.std(subsample_confidences)) if len(subsample_confidences) > 1 else 1e-6
            confidence_dists.append(BetaDistribution(mu=avg_confidence, sigma=std_confidence))

    # Keep selected answers aligned with the validated majority-cluster cache.
    if len(selected_responses) != len(largest_clusters):
        selected_responses = [cluster[0] if cluster else "" for cluster in largest_clusters]

    # pickle selected responses and confidence dists for this question
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
    largest_clusters, _ = _load_or_build_majority_clusters(cfg, response_lists)
    
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
    
    # Flatten prompts while keeping the owning question index.
    all_cluster_responses: list[str] = []
    response_question_indices: list[int] = []
    selected_answers: list[str] = []

    for q_idx, cluster in enumerate(largest_clusters):
        if not cluster:
            selected_answers.append("")
            continue
        selected_answers.append(cluster[0])
        for response in cluster:
            all_cluster_responses.append(response)
            response_question_indices.append(q_idx)

    if not all_cluster_responses:
        default_conf = [BetaDistribution(mu=0.5, sigma=1e-6) for _ in selected_answers]
        return OrganisedOutputs(
            extracted_answers=[selected_answers],
            extracted_confidences=[default_conf],
        )

    # scores_collection[question_idx] = list of scores pooled across all cluster responses and evaluators
    scores_collection: list[list[float]] = [[] for _ in selected_answers]
    
    # Loop through each evaluator
    for evaluator in evaluator_lst:
        model_manager: ModelManager = ModelManager(master_cfg=cfg, model_config_type=evaluator)
        
        # Generate prompts for every response in each question's largest semantic cluster.
        prompt_collection = PromptCollection(context_texts=[DIRECT_PROMPT.format(sentence=response) for response in all_cluster_responses])
        judge_outputs: list[ModelOutputs] = model_manager.run_generation(prompt_collection)

        generated_scores_texts: list[str] = []
        for judge_out in judge_outputs:
            generated_scores_texts.extend(judge_out.output_texts)

        # Assign judge outputs back to their question index via prompt order.
        for q_idx, judge_text in zip(response_question_indices, generated_scores_texts):
            score = extract_score(judge_text)
            if np.isnan(score):
                continue
            scores_collection[q_idx].append(score)
        del model_manager
        gc.collect()
        torch.cuda.empty_cache()

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

