import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.stats import beta

lexicon_df = pd.read_pickle("hedging_word_scores.pkl")

def find_closest_hedging_words(target_alpha, target_beta, lexicon_df, top_k=10) -> pd.DataFrame:
    """
    Find the top K closest hedging words to a target beta distribution.
    
    Args:
        target_alpha: alpha parameter of target beta distribution
        target_beta: beta parameter of target beta distribution
        lexicon_df: DataFrame with hedging words and their beta parameters
        top_k: number of closest matches to return
    
    Returns:
        DataFrame with top K closest hedging words and their distances
    """
    # Generate samples from target beta distribution
    sample_size = 3000
    target_samples = beta.rvs(target_alpha, target_beta, size=sample_size)
    
    distances = []
    
    for idx, row in lexicon_df.iterrows():
        if pd.isna(row['alpha_param']) or pd.isna(row['beta_param']):
            continue
        
        # Generate samples from hedging word's beta distribution
        word_samples = beta.rvs(row['alpha_param'], row['beta_param'], size=sample_size)
        
        # Compute Wasserstein distance
        w_dist = wasserstein_distance(target_samples, word_samples)
        
        distances.append({
            'hedging_word': row['hedging_word'],
            'alpha': row['alpha_param'],
            'beta': row['beta_param'],
            'mean': row['mean'],
            'wasserstein_distance': w_dist
        })
    
    # Sort by distance and return top K
    result_df = pd.DataFrame(distances).sort_values('wasserstein_distance').head(top_k)
    return result_df.reset_index(drop=True)