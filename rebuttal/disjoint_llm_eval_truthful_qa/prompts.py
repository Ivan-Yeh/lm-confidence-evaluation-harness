from human_cues import format_cues_for_prompt, sample_human_cues

LINGUISTIC_EVALUATOR_PROMPT = """
Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence, without using any external or prior knowledge to assess the knowledge conveyed by the sentence.
Please pay attention to the hedging language used and the overall assertiveness of the statement.
If the sentence abstains from answering by pointing out the insufficiency of information with a firm tone, the score should be towards 100.
If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100.
If the sentence does not contain any hedging language or is a succinct, decisive short answer, the score should be towards 100, too.

To align with human perception, here are some human-annotated linguistic cues with perceived confidence profiles (mean and standard deviation) for your reference:
{human_annotated_cues}

For additional reference, this exact statement's linguistic confidence was previously estimated by another automatic method as a distribution with mean={orig_conf_mean:.2f} and standard deviation={orig_conf_std:.2f} (on a 0-100 scale). Your score MUST fall within the range [{range_low:.2f}, {range_high:.2f}] (mean +/- 3 standard deviations of that reference estimate) -- use your own independent reading of the tone to pick where within that range the score falls, rather than defaulting to the mean.

Here is the sentence:
{sentence}

Confidence Score: [Return only a number between {range_low:.2f} and {range_high:.2f} without any additional text or explanation]
""".strip()


def hint_range(orig_conf_mean: float, orig_conf_std: float, n_std: float = 3.0) -> tuple[float, float]:
    low = max(0.0, orig_conf_mean - n_std * orig_conf_std)
    high = min(100.0, orig_conf_mean + n_std * orig_conf_std)
    if high - low < 1.0:  # guard against a degenerate near-zero-width range
        low = max(0.0, low - 0.5)
        high = min(100.0, high + 0.5)
    return low, high


def build_evaluator_prompt(
    sentence: str,
    orig_conf_mean: float,
    orig_conf_std: float,
    cues_text: str | None = None,
) -> str:
    if cues_text is None:
        cues_text = format_cues_for_prompt(sample_human_cues())
    range_low, range_high = hint_range(orig_conf_mean, orig_conf_std)
    return LINGUISTIC_EVALUATOR_PROMPT.format(
        human_annotated_cues=cues_text,
        sentence=sentence,
        orig_conf_mean=orig_conf_mean,
        orig_conf_std=orig_conf_std,
        range_low=range_low,
        range_high=range_high,
    )


if __name__ == "__main__":
    print(build_evaluator_prompt("Paris is the capital of France.", orig_conf_mean=91.7, orig_conf_std=4.7))
