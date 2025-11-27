
from dataclasses import dataclass

@dataclass
class PromptCollection:
    system_prompt: str = ""  # optional system prompt
    context_texts: list[str] = None # a list of strings
    continuation_texts : dict[str, list[str]] = None # context (str) and continuations (list of strings) mapping


@dataclass
class ModelOutputs:
    output_texts: list[str] = None # a list of strings
    output_tokens: list[str] = None # a list of strings
    output_logprobs: list[float] = None # a list of floats
    token_logprobs: list[dict[str, float]] = None # a list token-logprob mappings