from default_utils.custom_types import ModelOutputs, PromptCollection
from default_utils.registry import register_filter
import re


def output_substring_extractor(regexes: list[str], model_outputs: list[ModelOutputs]) -> list[ModelOutputs]:
    potential_answer_regex = [re.compile(rf"{rgx}") for rgx in regexes]

    def locate_substring_tokens(decoded_tokens, token_logprobs, substring):
        """Return token-level slice for the substring in decoded tokens."""
        full_text = "".join(decoded_tokens)
        start_char = full_text.find(substring)
        if start_char == -1:
            return None, None

        end_char = start_char + len(substring)
        running = 0
        token_start = None
        token_end = None

        for i, tok in enumerate(decoded_tokens):
            tok_len = len(tok)
            if token_start is None and running + tok_len > start_char:
                token_start = i
            if token_end is None and running + tok_len >= end_char:
                token_end = i + 1
                break
            running += tok_len

        if token_start is None or token_end is None:
            return None, None

        return decoded_tokens[token_start:token_end], token_logprobs[token_start:token_end]

    filtered_outputs = []

    for output in model_outputs:
        new_texts = []
        new_tokens = []
        new_logprobs = []

        for text, tokens, logprobs in zip(output.output_texts, output.output_tokens, output.output_logprobs):
            captured_text = None

            for rgx in potential_answer_regex:
                m = rgx.search(text)
                if m:
                    captured_text = m.group(1)  # capture group (answer letter)
                    break

            if captured_text is None:
                new_texts.append(None)
                new_tokens.append(None)
                new_logprobs.append(None)
                continue

            matched_tokens, matched_lp = locate_substring_tokens(tokens, logprobs, captured_text)

            new_texts.append(captured_text)
            new_tokens.append(matched_tokens)
            new_logprobs.append(matched_lp)

        filtered_outputs.append(
            ModelOutputs(
                context_texts=output.context_texts,
                output_texts=new_texts,
                output_tokens=new_tokens,
                output_logprobs=new_logprobs,
            )
        )
    return filtered_outputs


@register_filter(name="multiple_choice_regex_extractor")
def multiple_choice_regex_extractor(cfg: dict, model_outputs: list[ModelOutputs], prompts: PromptCollection, **kwargs) -> list[ModelOutputs]:
    potential_answer_regex = kwargs.get("regexes", [])
    print(potential_answer_regex)
    return output_substring_extractor(potential_answer_regex, model_outputs)