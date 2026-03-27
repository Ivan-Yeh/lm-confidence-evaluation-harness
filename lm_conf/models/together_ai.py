from tqdm import tqdm
from ..default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection
from together import Together
from transformers import AutoTokenizer
import concurrent.futures
import os
import time
import logging


class TogetherAIModel(AbstractModel):

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        self.tokenizer = None
        if self.model_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True)
            except Exception as exc:
                logging.warning(
                    "Failed to load tokenizer for %s: %s",
                    self.model_name,
                    exc,
                )
        
        
    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        stop_seq = self.cfg.get("stop_sequences", [])
        temperature = self.cfg.get("temperature", 1.0)
        max_tokens = self.cfg.get("max_tokens", 256)
        top_k = self.cfg.get("logprobs", 5)

        model_outputs_list = []

        for repeat_idx in range(self.repeat):
            logging.info(
                "TogetherAI [%s] Generation Round %d/%d",
                self.model_name,
                repeat_idx + 1,
                self.repeat,
            )

            output_texts = []
            output_tokens = []
            output_logprobs = []
            all_top_k_tokens = []

            max_workers = self.cfg.get("num_workers")
            if not max_workers:
                max_workers = min(8, max(1, len(prompt_collection.context_texts)))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, context_text in enumerate(prompt_collection.context_texts):
                    futures.append(
                        executor.submit(
                            self._request_choices,
                            client,
                            prompt_collection.system_prompt,
                            context_text,
                            temperature,
                            max_tokens,
                            top_k,
                            stop_seq,
                        )
                    )
                    if idx < len(prompt_collection.context_texts) - 1:
                        time.sleep(0.1)

                for future in tqdm(
                    futures,
                    desc="TogetherAI generation",
                    total=len(futures),
                ):
                    try:
                        choice_results = future.result()
                    except Exception as exc:
                        logging.error("TogetherAI request failed: %s", exc)
                        output_texts.append("")
                        output_tokens.append([])
                        output_logprobs.append([])
                        all_top_k_tokens.append([])
                        continue

                    for text, tokens, logprobs, top_k_tokens in choice_results:
                        output_texts.append(text)
                        output_tokens.append(tokens)
                        output_logprobs.append(logprobs)
                        all_top_k_tokens.append(top_k_tokens)

            model_outputs_list.append(
                ModelOutputs(
                    context_texts=prompt_collection.context_texts,
                    output_texts=output_texts,
                    output_tokens=output_tokens,
                    output_logprobs=output_logprobs,
                    top_k_tokens=all_top_k_tokens,
                )
            )

        return model_outputs_list

    def _request_choices(
        self,
        client,
        system_prompt: str,
        context_text: str,
        temperature: float,
        max_tokens: int,
        top_k: int,
        stop_seq: list
    ) -> list[tuple[str, list[str], list[float], list[list[tuple[str, float]]]]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_text},
        ]

        response = None
        non_rate_attempts = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=top_k,
                    stop=list(stop_seq),
                    reasoning={"enabled": False},
                    reasoning_effort="low",
                )
                break
            except Exception as exc:
                error_text = str(exc).lower()
                is_rate_limit = (
                    "rate limit" in error_text
                    or "rate_limit" in error_text
                    or "429" in error_text
                )
                if is_rate_limit:
                    logging.warning("TogetherAI rate limit, retrying in 1s: %s", exc)
                    time.sleep(1)
                    continue

                non_rate_attempts += 1
                if non_rate_attempts >= 3:
                    raise
                logging.warning("TogetherAI request failed, retrying in 1s: %s", exc)
                time.sleep(1)

        results = []
        for choice in response.choices:
            text = ""
            if choice.message and choice.message.content:
                text = choice.message.content.strip()
            tokens, logprobs, top_k_tokens = self._extract_logprobs(choice, text)
            results.append((text, tokens, logprobs, top_k_tokens))

        return results

    def _extract_logprobs(self, choice, generated_text: str) -> tuple[list[str], list[float], list[list[tuple[str, float]]]]:
        tokens = []
        logprobs = []
        top_k_tokens = []

        if not hasattr(choice, "logprobs") or not choice.logprobs:
            return tokens, logprobs, top_k_tokens

        logprobs_data = choice.logprobs
        if hasattr(logprobs_data, "tokens"):
            tokens = list(logprobs_data.tokens or [])
            logprobs = list(logprobs_data.token_logprobs or [])
            raw_top_logprobs = list(logprobs_data.top_logprobs or [])
        else:
            tokens = list(logprobs_data.get("tokens", []))
            logprobs = list(logprobs_data.get("token_logprobs", []))
            raw_top_logprobs = list(logprobs_data.get("top_logprobs", []))

        if raw_top_logprobs:
            for token_top_k in raw_top_logprobs:
                if isinstance(token_top_k, dict):
                    top_k_tokens.append(list(token_top_k.items()))
                elif isinstance(token_top_k, list):
                    top_k_tokens.append(
                        [(item["token"], item["logprob"]) for item in token_top_k]
                    )
                elif hasattr(token_top_k, "items"):
                    top_k_tokens.append(list(token_top_k.items()))
                else:
                    top_k_tokens.append([])

        min_len = min(len(tokens), len(logprobs))
        tokens = tokens[:min_len]
        logprobs = logprobs[:min_len]
        if top_k_tokens:
            top_k_tokens = top_k_tokens[:min_len]

        if self.tokenizer and generated_text:
            expected_length = len(
                self.tokenizer.encode(generated_text, add_special_tokens=False)
            )
            if expected_length > 0:
                tokens = tokens[-expected_length:]
                logprobs = logprobs[-expected_length:]
                if top_k_tokens:
                    top_k_tokens = top_k_tokens[-expected_length:]

            filtered_tokens = []
            filtered_logprobs = []
            filtered_top_k = []
            for idx, token in enumerate(tokens):
                if token in self.tokenizer.all_special_tokens:
                    continue
                if token.startswith("<|") and token.endswith("|>"):
                    continue
                filtered_tokens.append(token)
                filtered_logprobs.append(logprobs[idx])
                if top_k_tokens:
                    filtered_top_k.append(top_k_tokens[idx])
            tokens = filtered_tokens
            logprobs = filtered_logprobs
            top_k_tokens = filtered_top_k

        return tokens, logprobs, top_k_tokens
    

    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        raise NotImplementedError("TogetherAIModel does not support continuation mode yet")