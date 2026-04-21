import ast
import math
import numpy as np
import pandas as pd


def _coerce_scores(value):
	"""Return a list of numeric scores from stored value."""
	if isinstance(value, list):
		return [float(v) for v in value if isinstance(v, (int, float)) and not math.isnan(v)]
	if isinstance(value, str):
		try:
			parsed = ast.literal_eval(value)
			if isinstance(parsed, list):
				return [float(v) for v in parsed if isinstance(v, (int, float)) and not math.isnan(v)]
		except Exception:
			return []
	return []


def aggregate_scores(df):
	df = df.copy()
	df["scores"] = df["scores"].apply(_coerce_scores)

	aggregated = (
		df.groupby("hedging_word")["scores"]
		.apply(lambda lists: [score for sub in lists for score in sub])
		.reset_index(name="all_scores")
	)

	aggregated["count"] = aggregated["all_scores"].apply(len)
	aggregated["mean"] = aggregated["all_scores"].apply(lambda xs: sum(xs) / len(xs) if xs else None)
	aggregated["std"] = aggregated["all_scores"].apply(lambda xs: float(np.std(xs)) if xs else None)

	return aggregated

from scipy.stats import beta
import numpy as np

def fit_beta_to_scores(scores_list):
	"""Fit beta distribution to confidence scores (0-100 scale)."""
	if not scores_list or len(scores_list) < 2:
		return None, None

	# Normalize scores to [0, 1] for beta distribution
	normalized = np.array(scores_list)

	# Clip to avoid boundary issues
	normalized = np.clip(normalized, 1e-6, 1 - 1e-6)

	normalized = normalized[~np.isnan(normalized)]

	# filter out any values outside the 0-1 range and exactly 0
	normalized = normalized[(normalized > 0) & (normalized <= 1)]

	# Fit beta distribution
	try:
		alpha, beta_param, loc, scale = beta.fit(normalized, floc=0, fscale=1)
		return alpha, beta_param
	except Exception:
		return None, None



def main():
	confidence_scores = pd.read_pickle("linguistic_confidence_lexicon/confidence_scores.pkl")
	# confidence_scores_1 = pd.read_pickle("linguistic_confidence_lexicon/combined_batch_results.pkl")
	# confidence_scores = pd.concat([confidence_scores_0, confidence_scores_1], ignore_index=True)
	print(f"Loaded {len(confidence_scores)} rows of confidence scores")
	aggregated = aggregate_scores(confidence_scores)
	aggregated[['alpha_param', 'beta_param']] =	aggregated['all_scores'].apply(lambda x: pd.Series(fit_beta_to_scores(x))).clip(lower=1e-4)
	aggregated["alpha_param"] = aggregated["alpha_param"].clip(lower=1e-4)
	aggregated["beta_param"] = aggregated["beta_param"].clip(lower=1e-4)
	aggregated.dropna(inplace=True)
	output_base = "linguistic_confidence_lexicon/hedging_word_scores"
	aggregated.to_pickle(f"{output_base}.pkl")
	aggregated.to_csv(f"{output_base}.csv", index=False)

	print(f"Aggregated {len(aggregated)} hedging words")
	print(aggregated.head())


if __name__ == "__main__":
	main()
