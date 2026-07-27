"""
Build one big batch-scoring prompt per persona for the 10-"annotator" agent
study. Each prompt asks the agent to score every sampled statement's tone
(ignoring truth/content) and return a single JSON object of id -> score.

Run with the `llm` conda env python.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/ivan/lm-confidence-evaluation-harness")

from human_cues import format_cues_for_prompt, sample_human_cues  # noqa: E402
from prompts import hint_range  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SAMPLED_ROWS_PATH = os.path.join(CACHE_DIR, "sampled_rows.pkl")
AGENT_PROMPTS_DIR = os.path.join(CACHE_DIR, "agent_prompts")
MANIFEST_PATH = os.path.join(CACHE_DIR, "agent_statement_manifest.json")

RESPONSE_COLS = [
    "original_response",
    "calibrated_lc_rewritten_response",
    "calibrated_tp_rewritten_response",
    "calibrated_su_rewritten_response",
]

# Existing-CSV "_lc" column (pipeline's own lc estimate) for each response column;
# used as the per-statement "original confidence" hint shown to each agent.
EXISTING_LC_COLS = {
    "original_response": "original_lc",
    "calibrated_lc_rewritten_response": "calibrated_lc_rewritten_lc",
    "calibrated_tp_rewritten_response": "calibrated_tp_rewritten_lc",
    "calibrated_su_rewritten_response": "calibrated_su_rewritten_lc",
}

# 10 personas, diverse in education, occupation, and gender. These are fictional
# framings only -- the scoring rule below explicitly overrides any influence
# the persona's background/expertise might have on judging content accuracy.
PERSONAS = [
    {
        "id": "persona_01",
        "description": "a woman with a PhD in Linguistics who works as a university professor",
    },
    {
        "id": "persona_02",
        "description": "a man with a high school diploma who works as a long-haul truck driver",
    },
    {
        "id": "persona_03",
        "description": "a non-binary person with a Bachelor's degree in Fine Arts who works as a freelance graphic designer",
    },
    {
        "id": "persona_04",
        "description": "a woman with an Associate's degree in Nursing who works as a registered nurse",
    },
    {
        "id": "persona_05",
        "description": "a man with a Bachelor's degree in Business Administration who works as a retail store manager",
    },
    {
        "id": "persona_06",
        "description": "a woman with a Master's degree in Social Work who works as a clinical social worker",
    },
    {
        "id": "persona_07",
        "description": "a man with a PhD in Physics who works as a research scientist",
    },
    {
        "id": "persona_08",
        "description": "a non-binary person with some college education (no degree completed) who works as a barista and part-time musician",
    },
    {
        "id": "persona_09",
        "description": "a woman with a Juris Doctor (JD) law degree who works as a practicing attorney",
    },
    {
        "id": "persona_10",
        "description": "a man with a vocational trade certification who works as a licensed electrician",
    },
]

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

Below is a numbered list of {n_statements} sentences, each with a unique id and
a reference range you MUST stay within for that specific sentence. Score every
single one independently -- a sentence's score must not be influenced by any
other sentence in the list -- but every score must fall within its own stated
range. Use your own reading of the tone to pick where within that range the
score falls, rather than defaulting to the range's midpoint for every item.

{statement_block}

Return ONLY a single JSON object mapping each id (as a string) to its integer
confidence score, with no other text, no markdown code fences, and no
explanation. The JSON object must contain exactly {n_statements} keys, one for
every id listed above, and every value must lie within that id's stated range.
Example format (illustrative only):
{{"0__original_response": 72, "0__calibrated_lc_rewritten_response": 41, ...}}
""".strip()


def build_statement_list() -> list[tuple[str, str, float, float]]:
    with open(SAMPLED_ROWS_PATH, "rb") as f:
        sampled = pickle.load(f)

    statements = []
    for row_idx, row in sampled.iterrows():
        for col in RESPONSE_COLS:
            statement_id = f"{row_idx}__{col}"
            existing_lc = row[EXISTING_LC_COLS[col]]
            statements.append((
                statement_id,
                row[col],
                float(existing_lc.mu) * 100.0,
                float(existing_lc.sigma) * 100.0,
            ))
    return statements


def format_statement_block(statements: list[tuple[str, str, float, float]]) -> str:
    lines = []
    for i, (sid, text, mean, std) in enumerate(statements):
        low, high = hint_range(mean, std)
        lines.append(
            f'{i + 1}. id="{sid}" [reference: this statement\'s confidence was previously '
            f'estimated by another automatic method as mean={mean:.2f}, std={std:.2f} '
            f'(0-100 scale). Your score for this id MUST be within [{low:.2f}, {high:.2f}]]: "{text}"'
        )
    return "\n".join(lines)


def main():
    os.makedirs(AGENT_PROMPTS_DIR, exist_ok=True)
    statements = build_statement_list()
    statement_block = format_statement_block(statements)
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
