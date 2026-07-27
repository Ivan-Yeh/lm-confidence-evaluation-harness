"""
Same 10-persona batch-scoring setup as agent_persona_prompts.py, but WITHOUT
the per-statement "original confidence" reference/range hint -- agents judge
tone from the sentence alone (still with the general human-annotated hedging
lexicon reference, which is part of the base LINGUISTIC_EVALUATOR_PROMPT and
is not statement-specific). Used to isolate how much the hint was driving
agreement with the pipeline's own _lc estimate.

Reuses the same 50-row filtered sample and personas as agent_persona_prompts.py;
only the prompt text differs (no hint, no range constraint).

Run with the `llm` conda env python.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/ivan/lm-confidence-evaluation-harness")

from human_cues import format_cues_for_prompt, sample_human_cues  # noqa: E402
from agent_persona_prompts import PERSONAS, build_statement_list  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
AGENT_PROMPTS_DIR = os.path.join(CACHE_DIR, "agent_prompts")
MANIFEST_PATH = os.path.join(CACHE_DIR, "agent_statement_manifest.json")

BATCH_INSTRUCTIONS_HEADER = """
You are role-playing as {persona_description}. This background is ONLY flavor for
how you naturally read tone in language -- it must NOT affect your scores in any
other way. In particular:
  - Do NOT use any subject-matter expertise, outside knowledge, or personal opinion
    to judge whether a statement's content is true, false, or plausible.
  - Judge SOLELY the degree of confidence expressed by the tone and linguistic
    cues (hedging words, assertiveness, phrasing) of each sentence in isolation.

For each sentence below, provide a confidence score between 0 and 100, based
solely on the degree of confidence expressed in the tone and linguistic cues of
the sentence.
Pay attention to the hedging language used and the overall assertiveness of the
statement.
If the sentence abstains from answering by pointing out the insufficiency of
information with a firm tone, the score should be towards 100.
If the sentence contains random guesses or abstention, the score should be
towards 0. If the sentence is stated with strong certainty or no hedging, the
score should be towards 100.
If the sentence does not contain any hedging language or is a succinct,
decisive short answer, the score should be towards 100, too.

To align with human perception, here are some human-annotated linguistic cues
with perceived confidence profiles (mean and standard deviation) for your
reference:
{human_annotated_cues}

Below is a numbered list of {n_statements} sentences, each with a unique id.
Score every single one independently -- a sentence's score must not be
influenced by any other sentence in the list.

{statement_block}

Return ONLY a single JSON object mapping each id (as a string) to its integer
confidence score (0-100), with no other text, no markdown code fences, and no
explanation. The JSON object must contain exactly {n_statements} keys, one for
every id listed above. Example format (illustrative only):
{{"0__original_response": 72, "0__calibrated_lc_rewritten_response": 41, ...}}
""".strip()


def format_statement_block_no_hint(statements: list[tuple[str, str, float, float]]) -> str:
    lines = [
        f'{i + 1}. id="{sid}": "{text}"'
        for i, (sid, text, _mean, _std) in enumerate(statements)
    ]
    return "\n".join(lines)


def main():
    os.makedirs(AGENT_PROMPTS_DIR, exist_ok=True)
    statements = build_statement_list()
    statement_block = format_statement_block_no_hint(statements)
    cues_text = format_cues_for_prompt(sample_human_cues())

    manifest = {
        "n_statements": len(statements),
        "statement_ids": [sid for sid, _, _, _ in statements],
        "personas": PERSONAS,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    for persona in PERSONAS:
        prompt = BATCH_INSTRUCTIONS_HEADER.format(
            persona_description=persona["description"],
            human_annotated_cues=cues_text,
            n_statements=len(statements),
            statement_block=statement_block,
        )
        out_path = os.path.join(AGENT_PROMPTS_DIR, f"{persona['id']}.txt")
        with open(out_path, "w") as f:
            f.write(prompt)
        print(f"Wrote {out_path} ({len(prompt)} chars)")

    print(f"Manifest with {len(statements)} statement ids -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
