"""
Thin wrapper around the Together API for calling MiniMaxAI/MiniMax-M3 as the
disjoint linguistic-confidence judge.

The API key is read exclusively from the TOGETHER_API_KEY environment
variable at call time -- it is never written to disk or hard-coded here.
"""
import logging
import os
import random
import time

from together import Together
from together.error import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

MODEL_NAME = "MiniMaxAI/MiniMax-M3"
MAX_RETRIES = 8
BASE_BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 90.0

logger = logging.getLogger(__name__)

_client = None


def get_client() -> Together:
    global _client
    if _client is None:
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TOGETHER_API_KEY is not set in the environment. "
                "Export it in the same shell invocation that runs this script; "
                "never write it to a file."
            )
        _client = Together(api_key=api_key)
    return _client


def query_together(prompt: str, model: str = MODEL_NAME, max_tokens: int = 20, temperature: float = 1.0) -> str:
    """Single chat-completion call with retry/backoff on rate limits and transient errors.

    Reasoning is explicitly disabled: MiniMax-M3 is a reasoning model by default and will
    otherwise spend its token budget on a hidden chain-of-thought before the final number,
    which is unnecessary for a direct tone-reading judgment and inflates cost/latency.
    """
    client = get_client()
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning={"enabled": False},
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            wait = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
            wait *= 1.0 + random.random() * 0.25  # jitter
            logger.warning(
                "Together rate limit hit (attempt %d/%d): %s. Waiting %.1fs.",
                attempt, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError, APIError) as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            wait = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
            logger.warning(
                "Together transient error (attempt %d/%d): %s. Waiting %.1fs.",
                attempt, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)
