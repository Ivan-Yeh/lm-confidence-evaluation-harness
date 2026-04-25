#!/usr/bin/env python3
"""
Fetch Together AI batch outputs for target models and reconstruct filtered_outputs_0.pkl.

For each leaf dir under /hdd/ivny/results/{dataset}/{direct|hedged}_qa_unified_lc/{model_fam}/{model_name}/{timestamp}
matching the target model list, reads task.log for batch IDs, downloads outputs from Together API,
parses into model_outputs_list, and saves to a sibling directory with a fresh timestamp.

All matching dirs are processed unconditionally (re-fetches if already done).

If {dataset}_batch_part_*.jsonl files are absent from a source dir, the script falls back to
any other model's timestamp dir under the same {dataset}/{task_pattern} tree — all models share
identical prompts for the same dataset+task combination.

Usage:
    conda run -n llm python prototypes/fetch_together.py
"""

import json
import logging
import os
import pickle
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from together import Together

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm_conf.default_utils.custom_types import ModelOutputs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("/hdd/ivny/results")
TASK_PATTERNS = ["direct_qa_unified_lc", "hedged_qa_unified_lc"]
TARGET_MODELS = {
    "gemma-4-31B-it",
    # "Llama-3.3-70B-Instruct-Turbo",
    "gpt-oss-120b",
    "Qwen3-235B-A22B-Instruct-2507-tput",
}
# Models that embed reasoning tokens in their logprob sequence (gpt-oss family)
REASONING_MODELS = {"gpt-oss-120b", "gpt-oss-20b"}


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def find_leaf_dirs() -> list[Path]:
    dirs = []
    for dataset_dir in RESULTS_ROOT.iterdir():
        if not dataset_dir.is_dir():
            continue
        for task_pattern in TASK_PATTERNS:
            task_dir = dataset_dir / task_pattern
            if not task_dir.exists():
                continue
            for model_fam_dir in task_dir.iterdir():
                if not model_fam_dir.is_dir():
                    continue
                for model_name_dir in model_fam_dir.iterdir():
                    if not model_name_dir.is_dir():
                        continue
                    if model_name_dir.name not in TARGET_MODELS:
                        continue
                    candidates = sorted(
                        d for d in model_name_dir.iterdir()
                        if d.is_dir() and d.name != "retrieved"
                    )
                    if candidates:
                        dirs.append(candidates[-1])  # latest timestamp
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Batch-ID extraction from task.log
# ---------------------------------------------------------------------------

def extract_batch_ids(log_path: Path) -> list[str]:
    batch_ids: list[str] = []
    seen: set[str] = set()
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Batch ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}) progress:", line)
            if m:
                bid = m.group(1)
                if bid not in seen:
                    seen.add(bid)
                    batch_ids.append(bid)
    return batch_ids


# ---------------------------------------------------------------------------
# Token / logprob parsing  (mirrors TogetherAIBatch._parse_logprobs)
# ---------------------------------------------------------------------------

def _parse_logprobs(
    logprobs_data: dict,
) -> tuple[list[str], list[float], list[list[tuple[str, float]]]]:
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
                    ct = candidate.get("token")
                    cl = candidate.get("logprob")
                    if ct is None or cl is None:
                        continue
                    try:
                        token_top_k.append((str(ct), float(cl)))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(raw_top, dict):
                for ct, cl in raw_top.items():
                    try:
                        token_top_k.append((str(ct), float(cl)))
                    except (TypeError, ValueError):
                        continue
            top_k_tokens.append(token_top_k)

        return tokens, token_logprobs, top_k_tokens

    # Legacy flat format
    tokens = list(logprobs_data.get("tokens", []) or [])
    token_logprobs = list(logprobs_data.get("token_logprobs", []) or [])
    raw_top = list(logprobs_data.get("top_logprobs", []) or [])
    top_k_tokens = [list(t.items()) if isinstance(t, dict) else [] for t in raw_top]
    min_len = min(len(tokens), len(token_logprobs))
    return tokens[:min_len], token_logprobs[:min_len], top_k_tokens[:min_len]


def _is_special_token(token: str) -> bool:
    return token.startswith("<|") and token.endswith("|>")


def _filter_reasoning_tokens(
    tokens: list[str],
    token_logprobs: list[float],
    top_k_tokens: list[list[tuple[str, float]]],
) -> tuple[list[str], list[float], list[list[tuple[str, float]]]]:
    """
    For gpt-oss: skip all tokens up to and including the one containing 'final'
    (mirrors vllm_model.py gpt-oss case), then drop remaining special tokens.
    """
    found_final = False
    out_tokens: list[str] = []
    out_lp: list[float] = []
    out_topk: list[list[tuple[str, float]]] = []

    for i, tok in enumerate(tokens):
        if not found_final:
            if "final" in tok.lower():
                found_final = True
            continue  # skip everything up to and including the 'final' token
        if _is_special_token(tok):
            continue
        out_tokens.append(tok)
        out_lp.append(token_logprobs[i] if i < len(token_logprobs) else 0.0)
        out_topk.append(top_k_tokens[i] if i < len(top_k_tokens) else [])

    return out_tokens, out_lp, out_topk



# ---------------------------------------------------------------------------
# Batch retrieval and parsing
# ---------------------------------------------------------------------------

def _validate_jsonl(path: str) -> bool:
    """Return True if every non-empty line in the file is valid JSON."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    json.loads(line)
        return True
    except Exception:
        return False


def retrieve_batch(client: Together, batch_id: str, output_dir: str,
                   max_retries: int = 5, backoff: float = 5.0) -> str:
    """Download batch output JSONL if not already cached (or if cached copy is corrupt)."""
    output_path = os.path.join(output_dir, f"batch_output_{batch_id}.jsonl")
    if os.path.exists(output_path):
        if _validate_jsonl(output_path):
            logging.info("Using cached batch output: %s", output_path)
            return output_path
        logging.warning("Cached file is corrupt, re-downloading: %s", output_path)
        os.remove(output_path)

    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Retrieving batch %s (attempt %d) ...", batch_id, attempt)
            batch = client.batches.retrieve(batch_id)
            if batch.status != "COMPLETED":
                raise RuntimeError(f"Batch {batch_id} status is '{batch.status}', expected COMPLETED")
            tmp_path = output_path + ".tmp"
            with client.files.with_streaming_response.content(id=batch.output_file_id) as resp:
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
            if not _validate_jsonl(tmp_path):
                raise ValueError(f"Downloaded file failed JSONL validation: {tmp_path}")
            os.replace(tmp_path, output_path)
            logging.info("Saved -> %s", output_path)
            return output_path
        except RuntimeError:
            raise  # non-retryable: wrong batch status
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = backoff * attempt
            logging.warning("Batch %s attempt %d failed (%s); retrying in %.0fs ...",
                            batch_id, attempt, exc, wait)
            time.sleep(wait)


def parse_batch_file(
    jsonl_path: str,
    model_name: str,
) -> dict[tuple[int, int], dict]:
    """Parse a batch output JSONL into {(repeat_idx, prompt_idx): entry}."""
    is_reasoning = model_name in REASONING_MODELS
    merged: dict[tuple[int, int], dict] = {}

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed JSON line in %s: %s", jsonl_path, exc)
                continue
            custom_id = entry.get("custom_id", "")

            try:
                choice = entry["response"]["body"]["choices"][0]
            except Exception:
                logging.warning("Skipping malformed entry: %s", custom_id)
                continue

            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            text = (message.get("content") or "").strip() if isinstance(message, dict) else ""

            logprobs_data = choice.get("logprobs") or {}
            tokens, token_logprobs, top_k_tokens = _parse_logprobs(logprobs_data)

            # gpt-oss embeds reasoning tokens in the logprob stream before the
            # "final" marker; strip those (and trailing special tokens).
            # All other models: return raw tokens as-is (matches original
            # together_ai_batch._retrieve_batch_output behaviour which does no
            # special-token filtering).
            if is_reasoning:
                tokens, token_logprobs, top_k_tokens = _filter_reasoning_tokens(
                    tokens, token_logprobs, top_k_tokens
                )

            try:
                repeat_part, prompt_part = custom_id.split("_", 1)
                repeat_idx = int(repeat_part.lstrip("r"))
                prompt_idx = int(prompt_part.lstrip("p"))
            except ValueError:
                logging.warning("Cannot parse custom_id '%s', skipping", custom_id)
                continue

            merged[(repeat_idx, prompt_idx)] = {
                "text": text,
                "tokens": tokens,
                "logprobs": token_logprobs,
                "top_k_tokens": top_k_tokens,
            }

    return merged


# ---------------------------------------------------------------------------
# Context-text reconstruction from batch-part prompt files
# ---------------------------------------------------------------------------

def _read_batch_parts(search_dir: Path, dataset_name: str) -> dict[int, str]:
    """Read all {dataset}_batch_part_*.jsonl in search_dir; return {prompt_idx: context_text}."""
    idx_to_text: dict[int, str] = {}
    part_idx = 0
    while True:
        part_path = search_dir / f"{dataset_name}_batch_part_{part_idx}.jsonl"
        if not part_path.exists():
            break
        with open(part_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                req = json.loads(line)
                cid = req.get("custom_id", "")
                try:
                    _, prompt_part = cid.split("_", 1)
                    prompt_idx = int(prompt_part.lstrip("p"))
                except ValueError:
                    continue
                if prompt_idx in idx_to_text:
                    continue  # same prompt appears once per repeat; keep first
                messages = req.get("body", {}).get("messages", [])
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if content.startswith("/no_think\n"):
                            content = content[len("/no_think\n"):]
                        idx_to_text[prompt_idx] = content
                        break
        part_idx += 1
    return idx_to_text


def load_context_texts(source_dir: Path, dataset_name: str) -> list[str]:
    """
    Reconstruct ordered context_texts from {dataset}_batch_part_{n}.jsonl files.

    If prompt files are absent from source_dir, falls back to any other model's
    timestamp dir under the same {dataset}/{task_pattern} tree — all models share
    identical prompts for the same dataset + task combination.
    """
    idx_to_text = _read_batch_parts(source_dir, dataset_name)

    if not idx_to_text:
        # Fallback: search sibling model dirs under the same dataset/task tree.
        # source_dir layout: results/{dataset}/{task}/{model_fam}/{model_name}/{timestamp}
        task_root = source_dir.parent.parent.parent  # results/{dataset}/{task}
        logging.warning(
            "No batch-part files in %s — searching sibling dirs under %s",
            source_dir, task_root,
        )
        for model_fam_dir in task_root.iterdir():
            if not model_fam_dir.is_dir():
                continue
            for model_name_dir in model_fam_dir.iterdir():
                if not model_name_dir.is_dir():
                    continue
                for ts_dir in model_name_dir.iterdir():
                    if not ts_dir.is_dir() or ts_dir == source_dir:
                        continue
                    idx_to_text = _read_batch_parts(ts_dir, dataset_name)
                    if idx_to_text:
                        logging.info("Using prompt files from %s", ts_dir)
                        break
                if idx_to_text:
                    break
            if idx_to_text:
                break

    if not idx_to_text:
        raise RuntimeError(
            f"Could not find batch-part prompt files for dataset '{dataset_name}' "
            f"in {source_dir} or any sibling model dir"
        )

    return [idx_to_text[i] for i in sorted(idx_to_text)]


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_leaf_dir(source_dir: Path, client: Together) -> None:
    log_path = source_dir / "task.log"
    if not log_path.exists():
        logging.warning("No task.log in %s — skipping", source_dir)
        return

    # parts: ['/', 'hdd', 'ivny', 'results', dataset, task, model_fam, model_name, timestamp]
    model_name = source_dir.parts[-2]   # e.g. gpt-oss-120b
    dataset_name = source_dir.parts[-5] # e.g. mmlu

    batch_ids = extract_batch_ids(log_path)
    if not batch_ids:
        logging.warning("No batch IDs found in %s — skipping", log_path)
        return

    # Create sibling output directory up front so batch JSONLs land there
    output_dir = source_dir.parent / "retrieved"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check whether all batch output files are already present and valid.
    cached_paths = {
        bid: output_dir / f"batch_output_{bid}.jsonl"
        for bid in batch_ids
    }
    all_cached = all(p.exists() and _validate_jsonl(str(p)) for p in cached_paths.values())

    # Load context_texts upfront so the final-batch thread can save immediately.
    context_texts = load_context_texts(source_dir, dataset_name)
    total_prompts = len(context_texts)

    merged_responses: dict[tuple[int, int], dict] = {}
    merge_lock = threading.Lock()
    completed_count = 0

    def _save_outputs() -> None:
        """Build model_outputs_list and write pkl + log; called by the last batch thread."""
        if not merged_responses:
            logging.warning("No parsed responses for %s — skipping save", source_dir)
            return
        max_repeat = max(k[0] for k in merged_responses) + 1
        logging.info("Saving: %d repeat(s) x %d prompts -> %s", max_repeat, total_prompts, output_dir)

        model_outputs_list: list[ModelOutputs] = []
        for repeat_idx in range(max_repeat):
            output_texts: list[str] = []
            output_tokens: list[list[str]] = []
            output_logprobs: list[list[float]] = []
            all_top_k_tokens: list[list[list[tuple[str, float]]]] = []
            for prompt_idx in range(total_prompts):
                entry = merged_responses.get((repeat_idx, prompt_idx), {})
                output_texts.append(entry.get("text", ""))
                output_tokens.append(entry.get("tokens", []))
                output_logprobs.append(entry.get("logprobs", []))
                all_top_k_tokens.append(entry.get("top_k_tokens", []))
            model_outputs_list.append(
                ModelOutputs(
                    context_texts=context_texts,
                    output_texts=output_texts,
                    output_tokens=output_tokens,
                    output_logprobs=output_logprobs,
                    top_k_tokens=all_top_k_tokens,
                )
            )

        pkl_path = output_dir / "filtered_outputs_0.pkl"
        pkl_path.unlink(missing_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(model_outputs_list, fh)
        logging.info("Saved %s", pkl_path)

        # Write provenance log immediately after pkl
        run_log_path = output_dir / "task.log"
        run_log_path.unlink(missing_ok=True)
        fetched_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(run_log_path, "w") as fh:
            fh.write(f"Source directory : {source_dir}\n")
            fh.write(f"Model            : {model_name}\n")
            fh.write(f"Dataset          : {dataset_name}\n")
            fh.write(f"Fetched at       : {fetched_at}\n")
            fh.write(f"Batch IDs ({len(batch_ids)}):\n")
            for bid in batch_ids:
                fh.write(f"  {bid}\n")
            fh.write(f"\nmodel_outputs_list: {max_repeat} round(s), {total_prompts} prompt(s) each\n")
            fh.write(f"\nTop 10 outputs (repeat=0):\n")
            for i in range(min(10, total_prompts)):
                fh.write(f"  [{i}] text    : {repr(model_outputs_list[0].output_texts[i][:200])}\n")
                fh.write(f"       logprobs: {model_outputs_list[0].output_logprobs[i][:8]}\n")
                fh.write(f"       tokens  : {model_outputs_list[0].output_tokens[i][:8]}\n")
        logging.info("Done: %s  ->  %s", source_dir.name, output_dir.name)

    def _fetch_parse_merge(batch_id: str) -> None:
        nonlocal completed_count
        cached_path = cached_paths[batch_id]
        if cached_path.exists() and _validate_jsonl(str(cached_path)):
            jsonl_path = str(cached_path)
            logging.info("Batch %s: using cached file", batch_id)
        else:
            jsonl_path = retrieve_batch(client, batch_id, str(output_dir))
        batch_data = parse_batch_file(jsonl_path, model_name)
        with merge_lock:
            merged_responses.update(batch_data)
            completed_count += 1
            done = completed_count == len(batch_ids)
        logging.info("Batch %s merged (%d/%d)", batch_id, completed_count, len(batch_ids))
        if done:
            _save_outputs()

    if all_cached:
        logging.info("--- %s | model=%s | dataset=%s | all %d batch file(s) cached — parsing only ---",
                     source_dir, model_name, dataset_name, len(batch_ids))
        for bid in batch_ids:
            batch_data = parse_batch_file(str(cached_paths[bid]), model_name)
            merged_responses.update(batch_data)
        _save_outputs()
        return

    logging.info("--- %s | model=%s | dataset=%s | %d batch(es) -> %s ---",
                 source_dir, model_name, dataset_name, len(batch_ids), output_dir.name)

    with ThreadPoolExecutor(max_workers=len(batch_ids)) as pool:
        futures = {pool.submit(_fetch_parse_merge, bid): bid for bid in batch_ids}
        for future in as_completed(futures):
            bid = futures[future]
            try:
                future.result()
            except Exception as exc:
                logging.error("Batch %s failed: %s", bid, exc)
                raise


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    )

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        logging.error("TOGETHER_API_KEY environment variable not set")
        sys.exit(1)

    client = Together(api_key=api_key)

    leaf_dirs = find_leaf_dirs()
    logging.info("Found %d leaf directories to process", len(leaf_dirs))

    processed = errors = 0
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(process_leaf_dir, leaf_dir, client): leaf_dir
                   for leaf_dir in leaf_dirs}
        for future in as_completed(futures):
            leaf_dir = futures[future]
            try:
                future.result()
                processed += 1
            except Exception as exc:
                logging.exception("Error processing %s: %s", leaf_dir, exc)
                errors += 1

    logging.info("Done. processed=%d  errors=%d", processed, errors)


if __name__ == "__main__":
    main()
