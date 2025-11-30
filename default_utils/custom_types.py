from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class PromptCollection:
    system_prompt: str = ""  # optional system prompt
    context_texts: list[str] = None # a list of strings
    continuation_texts : dict[str, list[str]] = None # context (str) and continuations (list of strings) mapping


@dataclass
class ModelOutputs:
    context_texts: list[str] = None # a list of strings
    output_texts: list[str] = None # a list of strings
    output_tokens: list[list[str]] = None # a list of strings
    output_logprobs: list[list[float]] = None # a list of floats
    token_logprobs: list[dict[str, float]] = None # a list token-logprob mappings


class AbstractModel(ABC):
    @abstractmethod
    def run_generation(self, prompts: PromptCollection) -> "ModelOutputs":
        """Generate outputs given prompts (e.g., free-form generation)."""
        raise NotImplementedError

    @abstractmethod
    def run_continuation(self, prompts: PromptCollection) -> "ModelOutputs":
        """Score or generate continuations conditioned on contexts."""
        raise NotImplementedError


