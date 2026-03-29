import argparse
import importlib
import json
import logging
import os
import pkgutil
import sys
import time
from datetime import datetime

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from together import Together

from lm_conf.default_utils.custom_types import ModelOutputs, PromptCollection
from lm_conf.default_utils.datasets_manager import DatasetsManager
from lm_conf.default_utils.logger import get_logger
from lm_conf.default_utils.registry import PROMPT_FORMATTER
from lm_conf.default_utils.utils import import_yaml_lib


def _auto_import_modules() -> None:
    from lm_conf import default_utils

    for package_path in default_utils.__path__:
        for module_info in pkgutil.iter_modules([package_path]):
            module_name = module_info.name
            if module_name.startswith("_"):
                continue
            importlib.import_module(
                f"{default_utils.__name__}.{module_name}"
            )


OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mmlu_pro batch inference via Together API")
    parser.add_argument("--task", required=True, help="Task config name under lm_conf/tasks/mmlu_pro")
    parser.add_argument("--model", required=True, help="Together model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset limit")
    parser.add_argument("--overrides", nargs="*", default=[], help="Hydra overrides in key=value format")
    parser.add_argument("--repeat", type=int, default=20, help="Number of batch repeats")
    return parser.parse_args()


def load_task_config(task_name: str, overrides: list[str]) -> tuple[str, str, dict]:
    dataset_name = "mmlu_pro"
    config_dir = os.path.abspath("./lm_conf/tasks/mmlu_pro")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name=task_name,
            overrides=overrides,
        )

    return dataset_name, task_name, cfg


def build_prompts(cfg: dict) -> PromptCollection:
    dataset_manager = DatasetsManager(cfg)
    prompt_formatter = PROMPT_FORMATTER.get(cfg.get("prompt_formatter"))
    if prompt_formatter is None:
        prompt_formatter = import_yaml_lib(cfg, "prompt_formatter")
    return prompt_formatter(cfg, dataset_manager)


def create_batch_files(
    prompts: PromptCollection,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    stop_sequences: list,
    output_dir: str,
    repeat: int,
    max_requests: int = 50000,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    task_file_paths = []

    total_prompts = len(prompts.context_texts)
    total_requests = total_prompts * repeat
    chunk_count = (total_requests + max_requests - 1) // max_requests

    for chunk_idx in range(chunk_count):
        start_req = chunk_idx * max_requests
        end_req = min(start_req + max_requests, total_requests)
        tasks = []

        for req_idx in range(start_req, end_req):
            repeat_idx = req_idx // total_prompts
            prompt_idx = req_idx % total_prompts
            context_text = prompts.context_texts[prompt_idx]
            if "qwen" in model_name.lower():
                user_content = "/no_think\n" + context_text
            else:
                user_content = context_text

            task = {
                "custom_id": f"r{repeat_idx}_p{prompt_idx}",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "enable_thinking": False,
                    "reasoning_effort": "low",
                    "logprobs": top_k,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            }
            tasks.append(task)

        task_file_path = os.path.join(
            output_dir,
            f"mmlu_pro_batch_tasks_part_{chunk_idx}.jsonl",
        )
        with open(task_file_path, "w", encoding="utf-8") as file_handle:
            for task in tasks:
                file_handle.write(json.dumps(task, ensure_ascii=False) + "\n")

        task_file_paths.append(task_file_path)

    return task_file_paths


def submit_batch(client: Together, task_file_path: str) -> str:
    file_resp = client.files.upload(
        file=task_file_path,
        purpose="batch-api",
        check=False
    )
    batch = client.batches.create(input_file_id=file_resp.id, endpoint="/v1/chat/completions")
    return batch.job.id


def wait_for_batch(client: Together, batch_id: str, poll_interval: int = 30) -> None:
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "COMPLETED":
            return
        if batch.status in {"FAILED", "CANCELLED"}:
            raise RuntimeError(f"Batch {batch_id} ended with status {batch.status}")
        time.sleep(poll_interval)


def _parse_logprobs(logprobs_data: dict) -> tuple[list[str], list[float], list[list[tuple[str, float]]]]:
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


def retrieve_batch_output(client: Together, batch_id: str, output_dir: str) -> dict[tuple[int, int], dict]:
    batch = client.batches.retrieve(batch_id)
    output_path = os.path.join(output_dir, f"batch_output_{batch_id}.jsonl")
    if batch.status == "COMPLETED":
        with client.files.with_streaming_response.content(
            id=batch.output_file_id
        ) as response:
            with open(output_path, "wb") as file_handle:
                for chunk in response.iter_bytes():
                    file_handle.write(chunk)
    else:
        raise RuntimeError(f"Batch {batch_id} is not completed: {batch.status}")

    merged = {}
    with open(output_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry["custom_id"]
            choice = entry["response"]["body"]["choices"][0]

            text = ""
            message = choice.get("message", {})
            if isinstance(message, dict):
                text = (message.get("content") or "").strip()

            logprobs_data = choice.get("logprobs") or {}
            tokens, token_logprobs, top_k_tokens = _parse_logprobs(logprobs_data)

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


def main() -> None:
    args = parse_args()
    dataset_name, task_name, cfg = load_task_config(args.task, args.overrides)

    if args.model:
        cfg.qa_model.name = args.model
    if args.limit is not None:
        cfg.limit = args.limit
    cfg.qa_model.repeat = args.repeat

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"/hdd/ivny/results/{dataset_name}/{task_name}/{cfg.qa_model.name}/{timestamp}"

    os.makedirs(output_path, exist_ok=True)
    cfg["results_path"] = output_path

    logger = get_logger(__name__, log_file=f"{output_path}/task.log")
    _auto_import_modules()

    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    prompts = build_prompts(cfg)

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    total_prompts = len(prompts.context_texts)
    task_file_paths = create_batch_files(
        prompts=prompts,
        model_name=cfg.qa_model.name,
        temperature=cfg.qa_model.temperature,
        max_tokens=cfg.qa_model.max_tokens,
        top_k=cfg.qa_model.get("logprobs", 5),
        stop_sequences=cfg.qa_model.stop_sequences,
        output_dir=output_path,
        repeat=cfg.qa_model.repeat,
    )

    merged_responses: dict[tuple[int, int], dict] = {}
    for task_file_path in task_file_paths:
        batch_id = submit_batch(client, task_file_path)
        logger.info("Submitted batch job %s", batch_id)

        wait_for_batch(client, batch_id)

        merged_responses.update(retrieve_batch_output(client, batch_id, output_path))

    outputs = []
    for repeat_idx in range(cfg.qa_model.repeat):
        output_texts = []
        output_tokens = []
        output_logprobs = []
        top_k_tokens = []

        for prompt_idx in range(total_prompts):
            entry = merged_responses.get((repeat_idx, prompt_idx), {})
            output_texts.append(entry.get("text", ""))
            output_tokens.append(entry.get("tokens", []))
            output_logprobs.append(entry.get("logprobs", []))
            top_k_tokens.append(entry.get("top_k_tokens", []))

        outputs.append(
            ModelOutputs(
                context_texts=prompts.context_texts,
                output_texts=output_texts,
                output_tokens=output_tokens,
                output_logprobs=output_logprobs,
                top_k_tokens=top_k_tokens,
            )
        )

    filtered_outputs_path = os.path.join(output_path, "filtered_outputs_0.pkl")
    with open(filtered_outputs_path, "wb") as file_handle:
        import pickle

        pickle.dump(outputs, file_handle)

    logger.info("Saved outputs to %s", filtered_outputs_path)


if __name__ == "__main__":
    main()
