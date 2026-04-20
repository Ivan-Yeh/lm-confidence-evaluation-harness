import json
import logging
import os
import time

from together import Together

from ..default_utils.custom_types import AbstractModel, ModelOutputs, PromptCollection

class TogetherAIBatch(AbstractModel):
    def __init__(self, master_cfg, cfg):
        self.master_cfg = master_cfg
        self.cfg = cfg
        self.model_name = cfg.get("name", None)
        self.repeat = cfg.get("repeat", 1)
        
    def run_generation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

        temperature = self.cfg.get("temperature", 1.0)
        max_tokens = self.cfg.get("max_tokens", 256)
        top_k = self.cfg.get("logprobs", 5)
        stop_sequences = self.cfg.get("stop_sequences", [])
        max_requests = self.cfg.get("max_batch_requests", 10000)
        output_dir = self.master_cfg.get("results_path", "results/")
        os.makedirs(output_dir, exist_ok=True)

        task_file_paths = self._create_batch_files(
            prompt_collection=prompt_collection,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            stop_sequences=stop_sequences,
            output_dir=output_dir,
            repeat=self.repeat,
            max_requests=max_requests,
        )

        merged_responses: dict[tuple[int, int], dict] = {}
        batch_ids: list[str] = []
        for task_file_path in task_file_paths:
            batch_id = self._submit_batch(client, task_file_path)
            logging.info("Submitted Together batch job %s", batch_id)
            batch_ids.append(batch_id)

        self._wait_for_batches(client, batch_ids)

        for batch_id in batch_ids:
            merged_responses.update(self._retrieve_batch_output(client, batch_id, output_dir))

        total_prompts = len(prompt_collection.context_texts)
        model_outputs_list: list[ModelOutputs] = []
        for repeat_idx in range(self.repeat):
            output_texts = []
            output_tokens = []
            output_logprobs = []
            all_top_k_tokens = []

            for prompt_idx in range(total_prompts):
                entry = merged_responses.get((repeat_idx, prompt_idx), {})
                output_texts.append(entry.get("text", ""))
                output_tokens.append(entry.get("tokens", []))
                output_logprobs.append(entry.get("logprobs", []))
                all_top_k_tokens.append(entry.get("top_k_tokens", []))

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

    def _create_batch_files(
        self,
        prompt_collection: PromptCollection,
        temperature: float,
        max_tokens: int,
        top_k: int,
        stop_sequences: list,
        output_dir: str,
        repeat: int,
        max_requests: int,
    ) -> list[str]:
        task_file_paths = []
        total_prompts = len(prompt_collection.context_texts)
        total_requests = total_prompts * repeat
        chunk_count = (total_requests + max_requests - 1) // max_requests

        for chunk_idx in range(chunk_count):
            start_req = chunk_idx * max_requests
            end_req = min(start_req + max_requests, total_requests)
            tasks = []

            for req_idx in range(start_req, end_req):
                repeat_idx = req_idx // total_prompts
                prompt_idx = req_idx % total_prompts
                context_text = prompt_collection.context_texts[prompt_idx]

                if self.model_name and "qwen" in self.model_name.lower():
                    user_content = "/no_think\n" + context_text
                else:
                    user_content = context_text

                tasks.append(
                    {
                        "custom_id": f"r{repeat_idx}_p{prompt_idx}",
                        "body": {
                            "model": self.model_name,
                            "messages": [
                                {"role": "system", "content": prompt_collection.system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            "reasoning_effort": "low",
                            "enable_thinking": False,
                            "reasoning": {"enabled": False},
                            "logprobs": top_k,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "stop": list(stop_sequences),
                        },
                    }
                )

            task_file_path = os.path.join(output_dir, f"{self.master_cfg.get('dataset_name')}_batch_part_{chunk_idx}.jsonl")
            with open(task_file_path, "w", encoding="utf-8") as file_handle:
                for task in tasks:
                    file_handle.write(json.dumps(task, ensure_ascii=False) + "\n")

            task_file_paths.append(task_file_path)

        return task_file_paths

    def _submit_batch(self, client: Together, task_file_path: str) -> str:
        file_resp = client.files.upload(
            file=task_file_path,
            purpose="batch-api",
            check=False,
        )

        # Keep compatibility with SDK variants.
        try:
            batch = client.batches.create(input_file_id=file_resp.id, endpoint="/v1/chat/completions")
            if hasattr(batch, "job") and hasattr(batch.job, "id"):
                return batch.job.id
            if hasattr(batch, "id"):
                return batch.id
        except Exception:
            batch = client.batches.create_batch(file_id=file_resp.id, endpoint="/v1/chat/completions")
            return batch.id

        raise RuntimeError("Unable to create Together batch job")

    def _wait_for_batch(self, client: Together, batch_id: str, poll_interval: int = 30) -> None:
        while True:
            try:
                batch = client.batches.retrieve(batch_id)
            except Exception:
                batch = client.batches.get_batch(batch_id)

            if batch.status == "COMPLETED":
                return
            if batch.status in {"FAILED", "CANCELLED"}:
                raise RuntimeError(f"Batch {batch_id} ended with status {batch.status}")
            logging.info(f"Batch {batch_id} progress: {batch.progress}")
            time.sleep(poll_interval)

    def _wait_for_batches(self, client: Together, batch_ids: list[str], poll_interval: int = 30) -> None:
        pending = set(batch_ids)

        while pending:
            completed_now = []
            for batch_id in list(pending):
                try:
                    batch = client.batches.retrieve(batch_id)
                except Exception:
                    batch = client.batches.get_batch(batch_id)

                status = batch.status
                if status == "COMPLETED":
                    completed_now.append(batch_id)
                    continue

                if status in {"FAILED", "CANCELLED"}:
                    raise RuntimeError(f"Batch {batch_id} ended with status {status}")

                logging.info("Batch %s progress: %s", batch_id, getattr(batch, "progress", "unknown"))

            for batch_id in completed_now:
                pending.remove(batch_id)

            if pending:
                logging.info("Waiting on %d Together batch job(s)", len(pending))
                time.sleep(poll_interval)

    def _parse_logprobs(self, logprobs_data: dict) -> tuple[list[str], list[float], list[list[tuple[str, float]]]]:
        # Newer Together responses may return per-token entries under logprobs["content"].
        content_items = logprobs_data.get("content")
        if isinstance(content_items, list):
            tokens: list[str] = []
            token_logprobs: list[float] = []
            top_k_tokens: list[list[tuple[str, float]]] = []

            for item in content_items:
                if not isinstance(item, dict):
                    continue

                token = item.get("token")
                logprob = item.get("logprob")
                if token is None or logprob is None:
                    continue

                try:
                    token_str = str(token)
                    logprob_val = float(logprob)
                except (TypeError, ValueError):
                    continue

                tokens.append(token_str)
                token_logprobs.append(logprob_val)

                token_top_k: list[tuple[str, float]] = []
                raw_top = item.get("top_logprobs")
                if isinstance(raw_top, list):
                    for candidate in raw_top:
                        if not isinstance(candidate, dict):
                            continue
                        cand_token = candidate.get("token")
                        cand_logprob = candidate.get("logprob")
                        if cand_token is None or cand_logprob is None:
                            continue
                        try:
                            token_top_k.append((str(cand_token), float(cand_logprob)))
                        except (TypeError, ValueError):
                            continue
                elif isinstance(raw_top, dict):
                    for cand_token, cand_logprob in raw_top.items():
                        try:
                            token_top_k.append((str(cand_token), float(cand_logprob)))
                        except (TypeError, ValueError):
                            continue

                top_k_tokens.append(token_top_k)

            return tokens, token_logprobs, top_k_tokens

        tokens = list(logprobs_data.get("tokens", []) or [])
        token_logprobs = list(logprobs_data.get("token_logprobs", []) or [])
        raw_top_logprobs = list(logprobs_data.get("top_logprobs", []) or [])

        top_k_tokens = []
        for token_top_k in raw_top_logprobs:
            if isinstance(token_top_k, dict):
                top_k_tokens.append(list(token_top_k.items()))
            else:
                top_k_tokens.append([])

        min_len = min(len(tokens), len(token_logprobs))
        tokens = tokens[:min_len]
        token_logprobs = token_logprobs[:min_len]
        if top_k_tokens:
            top_k_tokens = top_k_tokens[:min_len]

        return tokens, token_logprobs, top_k_tokens

    def _retrieve_batch_output(self, client: Together, batch_id: str, output_dir: str) -> dict[tuple[int, int], dict]:
        try:
            batch = client.batches.retrieve(batch_id)
        except Exception:
            batch = client.batches.get_batch(batch_id)

        output_path = os.path.join(output_dir, f"batch_output_{batch_id}.jsonl")

        if batch.status != "COMPLETED":
            raise RuntimeError(f"Batch {batch_id} is not completed: {batch.status}")

        with client.files.with_streaming_response.content(id=batch.output_file_id) as response:
            with open(output_path, "wb") as file_handle:
                for chunk in response.iter_bytes():
                    file_handle.write(chunk)

        merged = {}
        with open(output_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                if not line.strip():
                    continue

                entry = json.loads(line)
                custom_id = entry.get("custom_id", "")

                try:
                    choice = entry["response"]["body"]["choices"][0]
                except Exception:
                    continue

                message = choice.get("message", {}) if isinstance(choice, dict) else {}
                text = (message.get("content") or "").strip() if isinstance(message, dict) else ""

                logprobs_data = choice.get("logprobs") or {}
                tokens, token_logprobs, top_k_tokens = self._parse_logprobs(logprobs_data)

                try:
                    repeat_part, prompt_part = custom_id.split("_", 1)
                    repeat_idx = int(repeat_part.lstrip("r"))
                    prompt_idx = int(prompt_part.lstrip("p"))
                except ValueError:
                    continue

                merged[(repeat_idx, prompt_idx)] = {
                    "text": text,
                    "tokens": tokens,
                    "logprobs": token_logprobs,
                    "top_k_tokens": top_k_tokens,
                }

        return merged
        
    def run_continuation(self, prompt_collection: PromptCollection) -> list[ModelOutputs]:
        raise NotImplementedError("TogetherAIBatch does not support continuation mode yet")