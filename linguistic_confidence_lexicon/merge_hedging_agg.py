"""
Merge synonym/inflection groups in hedging_word_aggregated.csv,
recompute all statistics, and overwrite the file.
"""

import csv
import ast
import math
from pathlib import Path
from collections import defaultdict

AGG_PATH = Path("/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/linguistic_confidence_lexicon/hedging_word_aggregated.csv")

# ---------------------------------------------------------------------------
# Merge map: canonical -> [aliases to absorb].
# Aliases are removed as standalone rows; their scores are pooled into canonical.
# ---------------------------------------------------------------------------
MERGE_MAP: dict[str, list[str]] = {
    # Same verb, different inflection / optional subject or complement
    "suggests":                 ["suggest", "suggests that"],
    "indicates":                ["indicate", "indicates that"],
    "appears to":               ["appear to", "it appears", "it appears that"],
    "it seems":                 ["seem to", "seems to", "it seems that", "it seems to me"],
    "could be":                 ["could", "it could be", "it could be that"],
    "might be":                 ["might", "it might be", "it might be that"],
    # Exact contraction of the same phrase
    "there is a chance that":   ["there's a chance that"],
    "if I'm not mistaken":      ["if I am not mistaken"],
    "I'd say":                  ["I would say"],
    "I guess":                  ["I would guess"],
    # "not X" vs "un-X" — morphological negation of same root
    "I am not sure":            ["I am unsure"],
    # Adjective vs adverb form of same phrase
    "almost certain":           ["almost certainly"],
    "vague":                    ["vaguely"],
    # Adverb vs prepositional phrase of same root word
    "with certainty":           ["certainly"],
    # Clausal vs adverbial form of same root word
    "conceivably":              ["it's conceivable that"],
}

# Build reverse index: alias -> canonical
ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in MERGE_MAP.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias] = canonical


def fit_beta(scores: list[float]) -> tuple[float, float]:
    n = len(scores)
    if n < 2:
        return float("nan"), float("nan")
    mu = sum(scores) / n
    var = sum((x - mu) ** 2 for x in scores) / (n - 1)
    mu = max(1e-6, min(1 - 1e-6, mu))
    if var <= 0 or var >= mu * (1 - mu):
        common = max(1.0, 10 * mu * (1 - mu))
        return round(mu * common, 4), round((1 - mu) * common, 4)
    factor = mu * (1 - mu) / var - 1
    return round(mu * factor, 4), round((1 - mu) * factor, 4)


def recompute(scores: list[float]) -> dict:
    n = len(scores)
    mu = sum(scores) / n
    std = math.sqrt(sum((x - mu) ** 2 for x in scores) / (n - 1)) if n > 1 else 0.0
    alpha, beta_p = fit_beta(scores)
    return {
        "count": n,
        "mean": round(mu, 6),
        "std": round(std, 6),
        "alpha_param": alpha,
        "beta_param": beta_p,
    }


def main():
    # Load existing aggregated rows
    rows: dict[str, list[float]] = {}
    with open(AGG_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores = ast.literal_eval(row["all_scores"])
            rows[row["hedging_word"]] = scores

    print(f"Loaded {len(rows)} hedge types")

    # Pool aliases into their canonical buckets
    merged_scores: dict[str, list[float]] = defaultdict(list)
    absorbed: dict[str, str] = {}  # alias -> canonical (for reporting)
    skipped_aliases: list[str] = []

    for word, scores in rows.items():
        canonical = ALIAS_TO_CANONICAL.get(word)
        if canonical is not None:
            if canonical not in rows:
                print(f"  WARNING: canonical '{canonical}' not in data — skipping merge of '{word}'")
                merged_scores[word].extend(scores)  # keep as-is
            else:
                merged_scores[canonical].extend(scores)
                absorbed[word] = canonical
        else:
            merged_scores[word].extend(scores)

    # Report merges
    print("\nMerges applied:")
    for canonical, aliases in MERGE_MAP.items():
        actually_absorbed = [a for a in aliases if a in absorbed]
        if actually_absorbed:
            counts = {a: len(rows[a]) for a in actually_absorbed if a in rows}
            total_added = sum(counts.values())
            print(f"  '{canonical}' (+{total_added} scores) ← {actually_absorbed}")

    # Build output rows
    output_rows = []
    for word, scores in merged_scores.items():
        stats = recompute(scores)
        output_rows.append({
            "hedging_word": word,
            "all_scores": scores,
            **stats,
        })

    output_rows.sort(key=lambda r: -r["count"])

    with open(AGG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["hedging_word", "all_scores", "count", "mean", "std", "alpha_param", "beta_param"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    n_before = len(rows)
    n_after = len(output_rows)
    print(f"\nRows: {n_before} → {n_after} (collapsed {n_before - n_after} aliases)")
    print("\nTop 25 by count:")
    for r in output_rows[:25]:
        print(
            f"  {r['count']:5d}  mean={r['mean']:.3f}  std={r['std']:.3f}  "
            f"α={r['alpha_param']}  β={r['beta_param']}  '{r['hedging_word']}'"
        )


if __name__ == "__main__":
    main()
