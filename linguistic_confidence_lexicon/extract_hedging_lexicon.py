"""
Extract dominant hedging cues from uncertainty_expression and build:
  distilled_hedging_lexicon.csv  - one row per benchmark entry with list of annotator scores
  hedging_word_aggregated.csv    - pooled stats per hedging word
"""

import re
import csv
import math
import ast
from pathlib import Path
from collections import defaultdict

SRC = Path("/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/llm_evaluator_test/linguistic_confidence_benchmark.csv")
LEX_DIR = Path("/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/linguistic_confidence_lexicon")
SCORES_CSV = LEX_DIR / "hedging_word_scores.csv"
OUT_LEXICON = LEX_DIR / "distilled_hedging_lexicon.csv"
OUT_AGG = LEX_DIR / "hedging_word_aggregated.csv"

ANNOTATOR_COLS = [
    "annotater_confidence_score1",
    "annotater_confidence_score2",
    "annotater_confidence_score3",
    "annotater_confidence_score4",
    "annotater_confidence_score5",
]


def normalize_quotes(text: str) -> str:
    """Replace curly/smart apostrophes and quotes with ASCII equivalents."""
    return (
        text
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2032", "'")
    )


def load_known_vocab() -> list[str]:
    """Load hedging words from hedging_word_scores.csv, sorted longest-first."""
    words = []
    with open(SCORES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            words.append(row["hedging_word"])
    # Longest phrases first so specific matches beat short ones
    words.sort(key=lambda w: -len(w))
    return words


def build_matchers(vocab: list[str]) -> list[tuple[re.Pattern, str]]:
    """Compile a regex for each hedging word. Multi-word → substring, single-word → word boundary."""
    compiled = []
    for word in vocab:
        norm = normalize_quotes(word)
        # Escape for regex
        escaped = re.escape(norm)
        if " " in norm:
            pat = re.compile(escaped, re.IGNORECASE)
        else:
            pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
        compiled.append((pat, word))
    return compiled


# Fallback patterns for expressions not covered by the known vocab,
# ordered from most to least specific.
FALLBACK_PATTERNS = [
    (r"i(?:'m| am) not (?:at all )?(?:sure|certain)", "I am not sure"),
    (r"i(?:'m| am) (?:very |extremely )?doubtful", "I am doubtful"),
    (r"i(?:'m| am) doubting myself", "I am doubtful"),
    (r"i(?:'m| am) (?:a bit |somewhat )?uncertain", "I am uncertain"),
    (r"i(?:'m| am) (?:a bit |somewhat )?unsure", "I am unsure"),
    (r"i(?:'m| am) (?:almost |nearly )?(?:positive|certain)", "I am almost sure"),
    (r"i(?:'m| am) (?:pretty|quite|very|extremely|so) (?:sure|certain)", "I am pretty sure"),
    (r"i(?:'m| am) reasonably (?:sure|certain)", "I am reasonably confident that"),
    (r"i(?:'m| am) fairly (?:sure|certain)", "I am fairly sure"),
    (r"i(?:'m| am) fairly confident", "I am fairly confident that"),
    (r"i(?:'m| am) reasonably confident", "I am reasonably confident that"),
    (r"i(?:'m| am) (?:pretty|quite|very) confident", "I am confident that"),
    (r"i(?:'m| am) not (?:entirely|fully|completely|100%) (?:sure|certain|confident)", "I am not sure"),
    (r"i(?:'m| am) not confident", "I am not sure"),
    (r"don'?t quote me on this", "correct me if I am wrong"),
    (r"i have a faint idea", "I have a feeling"),
    (r"if i (?:had to )?guess", "I would guess"),
    (r"i(?:'d| would) (?:venture|wager|say with)", "I would venture to say"),
    (r"i(?:'d| would) guess", "I would guess"),
    (r"i(?:'d| would) say", "I'd say"),
    (r"i(?:'d| would) estimate", "I would estimate"),
    (r"my tentative answer", "I would guess"),
    (r"my (?:best )?guess", "I guess"),
    (r"my understanding is", "my understanding is"),
    (r"my impression is", "my sense is"),
    (r"i(?:'m| am) inclined to (?:say|think|believe)", "I am inclined to think"),
    (r"i(?:'m| am) guessing", "I guess"),
    (r"according to (?:most|many|some|all) (?:sources|accounts|records|reports)", "according to some scholars"),
    (r"if my information is correct", "if I am not mistaken"),
    (r"it should be", "it is certain that"),
    (r"it(?:'s| is) (?:clear|undeniable|unquestionable|obvious) (?:that|and)", "it is clear that"),
    (r"\bundoubtedly\b", "undoubtedly"),
    (r"\bdefinitely\b", "definitely"),
    (r"\bprecisely\b", "certainly"),
    (r"\bconclusively\b", "certainly"),
    (r"\bunequivocally\b", "univocally"),
    (r"\bcategorically\b", "categorically"),
    (r"from what i (?:can )?gather", "from what I gather"),
    (r"from what i (?:can )?understand", "from what I understand"),
    (r"i (?:understand|gather) that", "from what I understand"),
    (r"i (?:have no hesitation|hesitate not)", "I am certain"),
    (r"i(?:'m| am) leaning (?:towards?|to)", "I am inclined to think"),
    (r"there(?:'s| is) a chance", "there is a chance that"),
    (r"it(?:'s| is) possible that", "it is possible that"),
    (r"it(?:'s| is) plausible", "it is plausible that"),
    (r"george|albert|the (?:correct|right|true) answer is", "it is certain that"),
    (r"she might have", "might be"),
    (r"may well have", "may be"),
    (r"i(?:'m| am) (?:almost |nearly )?certain", "I am certain"),
    (r"i (?:would |'d )?say with (?:some |high )?(?:confidence|certainty)", "I am confident that"),
    (r"i(?:'m| am) (?:fairly |pretty |reasonably |quite )?(?:sure|confident)", "I am fairly sure"),
    (r"\bapparently\b", "apparently"),
    (r"\bostensibly\b", "ostensibly"),
    (r"\bseemingly\b", "seemingly"),
    (r"\ballegedly\b", "allegedly"),
    (r"\bpurportedly\b", "purportedly"),
    (r"\breportedly\b|is reported to be", "reportedly"),
    # recall / memory variants not in vocab
    (r"(?:as |from what |) ?i recall(?: that| correctly)?", "if I recall correctly"),
    (r"if i (?:remember|recollect)(?: correctly)?", "if I recall correctly"),
    (r"(?:as |from what |)i (?:remember|recollect)(?: correctly)?", "if I recall correctly"),
    (r"from what i (?:recall|remember|recollect)", "if I recall correctly"),
    (r"(?:as |) ?i recall\b", "if I recall correctly"),
    # knowledge/understanding variants
    (r"from (?:my |what i can |)understanding", "my understanding is"),
    (r"based on (?:the |all |my |)(?:available |current |)?(?:information|sources?|evidence|data|records?)", "based on current evidence"),
    (r"according to (?:most|many|some|all|available|historical)? ?(?:sources?|accounts?|records?|reports?|evidence)", "according to some scholars"),
    # certainty / fact variants
    (r"it(?:'s| is) (?:a )?(?:known )?fact that", "it goes without saying that"),
    (r"(?:the )?(?:historical|available) record(?:s)? (?:is|are) clear", "it is clear that"),
    (r"(?:the )?most widely accepted (?:answer|view|interpretation) is", "it is clear that"),
    (r"\bconclusively\b|\bprecisely\b|\bunequivocally\b|\bunambiguously\b", "univocally"),
    (r"\bundoubtedly\b|\bundeniably\b|\bindisputably\b|\bincontrovertibly\b", "undoubtedly"),
    (r"\bdefinitely\b|\babsolutely\b(?! no)", "definitely"),
    (r"\bcategorically\b", "categorically"),
    # appearance/suggestion variants
    (r"(?:the )?(?:name|answer|solution) (?:that )?comes? to mind", "I would guess"),
    (r"(?:the )?(?:evidence|data|record|history) (?:is |are )?clear", "it is clear that"),
    (r"there is no (?:doubt|question) that", "without doubt"),
    # "from what I know/'ve gathered" variants
    (r"from what i(?:'ve| have) (?:gathered|seen|read|heard)", "from what I gather"),
    (r"from what i know\b", "as far as I know"),
    # "it's my understanding / as per my information"
    (r"it(?:'s| is) my understanding", "my understanding is"),
    (r"as per my (?:information|knowledge|records?|understanding)", "to my knowledge"),
    # recollection/reason to think
    (r"i have (?:some |a )?recollection(?: that)?", "if I recall correctly"),
    (r"i have (?:some |a )?reason to (?:think|believe|suppose)", "I think"),
    # "comes to mind" in any position
    (r"comes? to mind", "I would guess"),
    # passive/impersonal evidentials
    (r"\bis believed\b|\bis thought\b|\bis said\b|\bis reported\b", "is believed to"),
    (r"according to (?:available|the available|current|my|our) (?:data|information|evidence|records?)", "based on current evidence"),
    # "the records I have / based on what I have"
    (r"(?:the )?records? i have|based on what i have", "based on current evidence"),
    # "really not sure"
    (r"(?:really |very )?not (?:sure|certain)(?: at all)?", "I am not sure"),
    # evidence / consensus pointing
    (r"(?:the )?(?:evidence|data|findings?|results?) points? to", "the evidence suggests"),
    (r"(?:the )?(?:general |broad |clear )?consensus (?:is|points? to|suggests?)", "there is consensus that"),
    (r"(?:the )?indication is(?: that)?", "indicates"),
    # learned / memory
    (r"from what i(?:'ve| have) learned", "from what I gather"),
    (r"if my memory serves(?: me)?(?: right| correctly| well)?", "if memory serves"),
    # high certainty factual statements
    (r"(?:the )?definitive (?:answer|result|figure|date|value) is", "it is certain that"),
    (r"\bdefinitively\b", "definitely"),
    (r"widely reported", "reportedly"),
    (r"i have the impression(?: that)?", "my sense is"),
    # abstention / unable
    (r"(?:sorry,? )?i(?:'m| am) unable to (?:determine|say|answer|confirm|identify)", "I am unable to answer"),
    # bare factual statements (no hedge word — treat as high confidence)
    (r"^(?:it was|they were|he was|she was|the (?:year|date|answer|figure|value|result|founding)(?: was| is))", "it is certain that"),
]

FALLBACK_COMPILED = [
    (re.compile(p, re.IGNORECASE), label) for p, label in FALLBACK_PATTERNS
]


def extract_hedge(text: str, matchers: list[tuple[re.Pattern, str]]) -> str:
    norm = normalize_quotes(text)
    for pat, label in matchers:
        if pat.search(norm):
            return label
    # Fallback patterns
    for pat, label in FALLBACK_COMPILED:
        if pat.search(norm):
            return label
    return "other"


def fit_beta(scores: list[float]) -> tuple[float, float]:
    """Method of moments Beta fit."""
    if len(scores) < 2:
        return float("nan"), float("nan")
    n = len(scores)
    mu = sum(scores) / n
    var = sum((x - mu) ** 2 for x in scores) / (n - 1)
    mu = max(1e-6, min(1 - 1e-6, mu))
    if var <= 0 or var >= mu * (1 - mu):
        common = max(1.0, 10 * mu * (1 - mu))
        return round(mu * common, 4), round((1 - mu) * common, 4)
    factor = mu * (1 - mu) / var - 1
    return round(mu * factor, 4), round((1 - mu) * factor, 4)


def main():
    LEX_DIR.mkdir(exist_ok=True)

    vocab = load_known_vocab()
    matchers = build_matchers(vocab)
    print(f"Loaded {len(vocab)} known hedging words")

    # --- Pass 1: build distilled lexicon ----------------------------------
    lexicon_rows = []
    # bucket maps hedging_word -> flat list of all individual annotator scores
    bucket: dict[str, list[float]] = defaultdict(list)

    with open(SRC, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expr = row["uncertainty_expression"].strip()

            # Collect individual annotator scores (skip missing)
            raw_scores = []
            for col in ANNOTATOR_COLS:
                val = row.get(col, "").strip()
                if val:
                    try:
                        raw_scores.append(float(val))
                    except ValueError:
                        pass

            if not raw_scores:
                continue

            norm_scores = [round(s / 100.0, 6) for s in raw_scores]
            hedge = extract_hedge(expr, matchers)

            lexicon_rows.append({
                "uncertainty_expression": expr,
                "extracted_hedging_expression": hedge,
                "normalized_confidence_scores": norm_scores,
            })
            bucket[hedge].extend(norm_scores)

    with open(OUT_LEXICON, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uncertainty_expression", "extracted_hedging_expression", "normalized_confidence_scores"],
        )
        writer.writeheader()
        writer.writerows(lexicon_rows)

    print(f"Wrote {len(lexicon_rows)} rows -> {OUT_LEXICON}")

    # --- Pass 2: aggregate ------------------------------------------------
    agg_rows = []
    for hedge in sorted(bucket.keys()):
        scores = bucket[hedge]
        n = len(scores)
        mu = sum(scores) / n
        std = math.sqrt(sum((x - mu) ** 2 for x in scores) / (n - 1)) if n > 1 else 0.0
        alpha, beta_p = fit_beta(scores)
        agg_rows.append({
            "hedging_word": hedge,
            "all_scores": scores,
            "count": n,
            "mean": round(mu, 6),
            "std": round(std, 6),
            "alpha_param": alpha,
            "beta_param": beta_p,
        })

    agg_rows.sort(key=lambda r: -r["count"])

    with open(OUT_AGG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["hedging_word", "all_scores", "count", "mean", "std", "alpha_param", "beta_param"],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Wrote {len(agg_rows)} hedge types -> {OUT_AGG}")
    print("\nTop hedge types by count:")
    for r in agg_rows[:25]:
        print(
            f"  {r['count']:5d}  mean={r['mean']:.3f}  std={r['std']:.3f}  "
            f"α={r['alpha_param']}  β={r['beta_param']}  '{r['hedging_word']}'"
        )

    unmatched = bucket.get("other", [])
    if unmatched:
        expr_other = [
            row["uncertainty_expression"]
            for row in lexicon_rows
            if row["extracted_hedging_expression"] == "other"
        ]
        pct = len(expr_other) / len(lexicon_rows) * 100
        print(f"\n  'other' (unmatched): {len(expr_other)} rows ({pct:.1f}%) — sample:")
        for e in expr_other[:15]:
            print(f"    {repr(e[:100])}")


if __name__ == "__main__":
    main()
