import os

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import re
import gc
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, digamma, betainc, expit
from linguistic_confidence_lexicon.linguistic_calibrator import find_closest_hedging_words
from lm_conf.confidence_metrics.distributionals import BetaDistribution
from lm_conf.default_utils.custom_types import OrganisedOutputs, PromptCollection
from lm_conf.models.model_manager import ModelManager
from lm_conf.post_processing.metrics import faithfulness_divergence, generalised_ece


MIN_ALPHA_BETA = 1e-4    # floor for alpha/beta; keeps Beta pdf well-defined
MAX_KAPPA      = 1_000.0 # cap on kappa=alpha+beta; above this the Beta pdf is
                         # a near-delta spike that Q=100 quadrature cannot resolve
MU_EPS         = 1e-6
MIN_KAPPA      = 2.0 * MIN_ALPHA_BETA

LINGUISTIC_LEXICON_PATH = "/home/ivan/lm-confidence-evaluation-harness/linguistic_confidence_lexicon/hedging_word_scores.csv"
LEXICON = pd.read_csv(LINGUISTIC_LEXICON_PATH)
LEXICON["alpha_param"] = LEXICON["alpha_param"].astype(float).clip(MIN_ALPHA_BETA)
LEXICON["beta_param"] = LEXICON["beta_param"].astype(float).clip(MIN_ALPHA_BETA)
LEXICON["mean"] = LEXICON["mean"].astype(float)
LEXICON["std"] = LEXICON["std"].astype(float)
LEXICON.dropna(inplace=True)


REWRITE_PROMPT = """
Given an original response and a list of target hedging words with their confidence profiles (Beta Distributions), rewrite the response to appropriately reflect the confidence level indicated by the set of target hedging words. 
You must preserve the original meaning of the response, as we are only adjusting the tone to match the confidence level suggested by the hedging words. Ensure the new response sounds natural and fluent. 

Original response: ```My answer to the question is: "{response}"```
Target hedging words with confidence profiles: {hedges}

Please return only the rewritten sentence without any explanation.
New response: 
""".strip()


human_annotated_cues = pd.read_csv(os.path.join("/home/ivan/lm-confidence-evaluation-harness/linguistic_confidence_lexicon", "hedging_word_aggregated.csv"))[["hedging_word", "mean", "std"]]
human_annotated_cues["mean"] *= 100.0
human_annotated_cues["std"] *= 100.0
human_annotated_cues = human_annotated_cues.sort_values("mean").round(2).to_dict(orient="records")
_HUMAN_CUE_MEANS = np.array([row["mean"] for row in human_annotated_cues], dtype=float)

LINGUISTIC_EVALUATOR_PROMPT = """
Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence, without using any external or prior knowledge to assess the knowledge conveyed by the sentence. 
Please pay attention to the hedging language used and the overall assertiveness of the statement. 
If the sentence abstains from answering by pointing out the insufficiency of information with a firm tone, the score should be towards 100. 
If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. 
If the sentence does not contain any hedging language or is a succinct, decisive short answer, the score should be towards 100, too.

To align with human perception, here are some human-annotated linguistic cues with perceived confidence profiles (mean and standard deviation) for your reference: 
{human_annotated_cues}

Here is the sentence:
{sentence}

Confidence Score: [Return only a number between 0 and 100 without any additional text or explanation]
""".strip()


def obtain_hedging_words(conf: BetaDistribution, top_k=5) -> list[tuple[str, BetaDistribution]] | None:
    """Return top-k hedging words paired with their lexicon BetaDistribution profiles."""
    if conf is None or not conf.is_valid():
        return None
    df = find_closest_hedging_words(
        conf.alpha_param,
        conf.beta_param,
        LEXICON,
        top_k=top_k,
    )
    result = []
    for _, row in df.iterrows():
        mu, sigma = _to_mu_sigma(float(row["alpha"]), float(row["beta"]))
        result.append((row["hedging_word"], BetaDistribution(mu=mu, sigma=sigma)))
    return result


def format_hedges_for_prompt(hedges: list[tuple[str, BetaDistribution]] | None) -> str:
    """Format hedging word + confidence profile pairs for the rewrite prompt."""
    if not hedges:
        return "none"
    parts = [
        f'"{word}": Beta(alpha={dist.alpha_param:.2f}, beta={dist.beta_param:.2f}) with mean={dist.mu*100:.2f}% and std={dist.sigma*100:.2f}%'
        for word, dist in hedges
    ]
    return "; ".join(parts)


evaluator_list = [
    "qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


# =============================================================================
# PARAMETER CONVERSION UTILITIES
# =============================================================================

def _to_alpha_beta(mu: float, sigma: float) -> tuple[float, float]:
    """
    Convert (mu, sigma) to (alpha, beta).

    Uses the method-of-moments identities:
        kappa = mu * (1 - mu) / sigma^2 - 1
        alpha = mu * kappa
        beta  = (1 - mu) * kappa

    Raises ValueError if sigma is too large for the given mu (i.e. if
    sigma^2 >= mu * (1 - mu), which would give kappa <= 0).
    """
    var = sigma ** 2
    max_var = mu * (1.0 - mu)
    if var >= max_var:
        raise ValueError(
            f"sigma={sigma:.4f} too large for mu={mu:.4f}; "
            f"requires sigma < {max_var**0.5:.4f}"
        )
    kappa = max_var / var - 1.0
    return mu * kappa, (1.0 - mu) * kappa


def _to_mu_sigma(alpha: float, beta: float) -> tuple[float, float]:
    """
    Convert (alpha, beta) to (mu, sigma).

    Always produces a consistent (mu, sigma) pair derived from the same
    Beta distribution. Never call BetaDistribution(mu=mu_new, sigma=old_sigma)
    after shifting mu -- always go through this function instead.
    """
    mu    = alpha / (alpha + beta)
    kappa = alpha + beta
    sigma = np.sqrt(mu * (1.0 - mu) / (kappa + 1.0))
    return mu, sigma


# =============================================================================
# KERNEL-SMOOTHED ECE  (Wang 2025, continuous limit)
#
# Each sample's Beta(alpha_i, beta_i) acts as its own input-dependent kernel
# via the Nadaraya-Watson estimator:
#
#     r_hat(s) = sum_i f(s | alpha_i, beta_i) * y_i
#                -----------------------------------
#                sum_i f(s | alpha_i, beta_i)
#
#     f_hat(s) = (1/N) * sum_i f(s | alpha_i, beta_i)
#
#     KS-ECE   = integral |r_hat(s) - s| * f_hat(s) ds
#
# Approximated by Q-point uniform quadrature over (0, 1).
# =============================================================================

def _beta_pdf(s, a, b):
    """Beta pdf. s:(Q,), a,b:(N,) -> (N,Q)"""
    log_K = (
        (a[:, None] - 1) * np.log(s[None, :].clip(1e-9))
        + (b[:, None] - 1) * np.log((1 - s[None, :]).clip(1e-9))
        - betaln(a[:, None], b[:, None])
    )
    return np.exp(log_K)


def _beta_kl(a_p, b_p, a_q, b_q):
    """KL( Beta(a_p,b_p) || Beta(a_q,b_q) ) >= 0."""
    return (
          betaln(a_q, b_q) - betaln(a_p, b_p)
        + (a_p - a_q) * digamma(a_p)
        + (b_p - b_q) * digamma(b_p)
        + (a_q - a_p + b_q - b_p) * digamma(a_p + b_p)
    )


def _mean_fd(a, b, y):
    """Mean Faithfulness Divergence. FD_i = (a+b)*KL(post||prior). Always >= 0."""
    kl = _beta_kl(a + y, b + (1 - y), a, b)
    return ((a + b) * kl).mean()


def _ks_ece_sq(a, b, y, Q):
    """Kernel-smoothed squared ECE."""
    s     = np.linspace(1 / (2 * Q), 1 - 1 / (2 * Q), Q)
    K     = _beta_pdf(s, a, b)
    r_hat = (K * y[:, None]).sum(0) / K.sum(0).clip(1e-8)
    f_hat = K.mean(0)
    return (f_hat * (r_hat - s) ** 2).sum() / Q


def _ks_ece_abs(a, b, y, Q):
    """Kernel-smoothed absolute ECE (L1). Reporting metric."""
    s     = np.linspace(1 / (2 * Q), 1 - 1 / (2 * Q), Q)
    K     = _beta_pdf(s, a, b)
    r_hat = (K * y[:, None]).sum(0) / K.sum(0).clip(1e-8)
    f_hat = K.mean(0)
    return (f_hat * np.abs(r_hat - s)).sum() / Q


# =============================================================================
# SOFT-BINNED HISTOGRAM CALIBRATION MAP
#
# Training:  divide [0,1] into B equal-width bins.  For bin b = [lo, hi],
#   compute soft membership weights via the Beta CDF:
#     w_{n,b} = I_{hi}(α_n, β_n) - I_{lo}(α_n, β_n)
#   then set the bin's calibrated accuracy to:
#     μ'_b = Σ_n w_{n,b} · y_n  /  Σ_n w_{n,b}
#
# Inference: apply the same Beta CDF weighting to a new distribution:
#     μ' = Σ_b w_b · μ'_b,   w_b = I_{hi}(α, β) - I_{lo}(α, β)
#   then reconstruct α' = μ'·κ, β' = (1-μ')·κ with κ unchanged.
# =============================================================================

def _fit_soft_hist_bins(
    mu: np.ndarray,
    kappa: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Fit soft-binned histogram calibration on training data.

    Returns bin_accuracies of shape (n_bins,), one calibrated accuracy per bin.
    Bins with negligible total weight fall back to their bin midpoint.
    """
    a = (mu * kappa).clip(MIN_ALPHA_BETA)
    b = ((1.0 - mu) * kappa).clip(MIN_ALPHA_BETA)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.empty(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        w = betainc(a, b, hi) - betainc(a, b, lo)   # (N,)
        w_sum = w.sum()
        bin_accs[i] = (w @ y) / w_sum if w_sum > 1e-8 else (lo + hi) / 2.0
    # Enforce monotonicity: calibrated accuracy should be non-decreasing with bin index
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    bin_accs = iso.fit_transform(midpoints, bin_accs)
    return bin_accs


def _apply_soft_hist_bin(
    mu: float,
    kappa: float,
    bin_accs: np.ndarray,
) -> float:
    """
    Apply fitted soft-binned histogram map to a single (mu, kappa).

    Returns calibrated μ' = Σ_b w_b · μ'_b, normalised by Σ_b w_b ≈ 1.
    """
    n_bins = len(bin_accs)
    a = float(np.clip(mu * kappa, MIN_ALPHA_BETA, None))
    b = float(np.clip((1.0 - mu) * kappa, MIN_ALPHA_BETA, None))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    w = betainc(a, b, edges[1:]) - betainc(a, b, edges[:-1])   # (n_bins,)
    w_sum = w.sum()
    return float((w @ bin_accs) / w_sum) if w_sum > 1e-8 else mu


# =============================================================================
# MIN-ECE CALIBRATION MAP
#   kernel-smoothed reliability curve  →  Beta CDF fit  →  distributional map
#
# Training (on {(μ_i, y_i)}):
#   1. Select bandwidth h* via LOO-CV: h* = argmin_h Σ_i (y_i - ȳ_{-i}(μ_i))²
#   2. Evaluate smoothed reliability curve on M-point grid:
#        ȳ(c) = Σ_i K_h(c - μ_i) y_i  /  Σ_i K_h(c - μ_i),  K_h = Gaussian
#   3. Fit (α*, β*) by least squares:
#        (α*, β*) = argmin_{α,β} Σ_m ( I_{c_m}(α, β) - ȳ(c_m) )²
#
# Test-time (per instance Beta(α_i, β_i)):
#   4. Sample T points {x_t} ~ Beta(α_i, β_i)
#   5. Push through the Beta CDF map: x'_t = I_{x_t}(α*, β*)
#   6. Refit Beta(α'_i, β'_i) to {x'_t} via method of moments
#
# Hyperparameters: none manual — h* is fully data-driven.
# =============================================================================

def _nw_smooth(c: np.ndarray, mu: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """Nadaraya-Watson with Gaussian kernel. c:(M,), mu:(N,), y:(N,) -> (M,)."""
    diff = (c[:, None] - mu[None, :]) / h   # (M, N)
    K    = np.exp(-0.5 * diff ** 2)
    return (K @ y) / K.sum(axis=1).clip(1e-8)


def _loo_mse(h: float, mu: np.ndarray, y: np.ndarray) -> float:
    """LOO-CV MSE for NW bandwidth h. Off-diagonal kernel, zero diagonal."""
    diff = (mu[:, None] - mu[None, :]) / h  # (N, N)
    K    = np.exp(-0.5 * diff ** 2)
    np.fill_diagonal(K, 0.0)
    y_hat = (K @ y) / K.sum(axis=1).clip(1e-8)
    return float(np.mean((y - y_hat) ** 2))


def _select_bandwidth(mu: np.ndarray, y: np.ndarray) -> float:
    """Grid search over 40 log-spaced bandwidths in [0.01, 0.5]."""
    h_grid = np.logspace(-2, np.log10(0.5), 40)
    mse    = [_loo_mse(h, mu, y) for h in h_grid]
    return float(h_grid[int(np.argmin(mse))])


def _fit_beta_cdf_map(
    mu_train: np.ndarray, y_train: np.ndarray, h_star: float, M: int = 100,
) -> tuple[float, float]:
    """
    Fit (α*, β*) so that I_c(α*, β*) ≈ ȳ(c) on an M-point grid.
    Optimises log(α), log(β) to enforce strict positivity.
    """
    c_grid   = np.linspace(0.01, 0.99, M)
    y_smooth = _nw_smooth(c_grid, mu_train, y_train, h_star)

    def loss(log_params: np.ndarray) -> float:
        a, b = np.exp(log_params)
        return float(np.sum((betainc(a, b, c_grid) - y_smooth) ** 2))

    # bounds keep α*, β* ∈ [exp(-2), exp(6)] ≈ [0.14, 403] — prevents collapse to 0
    result = minimize(
        loss, x0=np.zeros(2),
        method="L-BFGS-B",
        bounds=[(-2.0, 6.0), (-2.0, 6.0)],
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
    )
    a_star, b_star = np.exp(result.x)
    return float(a_star), float(b_star)


def fit_calibration_map(
    mu_train: np.ndarray,
    sigma_train: np.ndarray,
    y_train: np.ndarray,
    M: int = 300,
) -> tuple[float, float, float]:
    """
    Fit kernel-smoothed Beta CDF calibration map.

    Args:
        mu_train:    predicted confidence means, shape (N,)
        sigma_train: predicted confidence stds, shape (N,)  [unused in fit; kept for API compat]
        y_train:     binary correctness labels, shape (N,)
        M:           grid resolution for CDF fitting (default 100)

    Returns:
        (alpha_star, beta_star, h_star)
    """
    mu_tr = np.asarray(mu_train, dtype=float).clip(MU_EPS, 1.0 - MU_EPS)
    y_tr  = np.asarray(y_train,  dtype=float)
    h_star = _select_bandwidth(mu_tr, y_tr)
    alpha_star, beta_star = _fit_beta_cdf_map(mu_tr, y_tr, h_star, M)
    return alpha_star, beta_star, h_star


def _mom_fit(samples: np.ndarray) -> tuple[float, float] | None:
    """
    Method-of-moments Beta fit to samples. Returns (alpha, beta) or None on failure.
    """
    mu_s  = float(samples.mean())
    var_s = float(samples.var())
    if var_s <= 0.0 or mu_s <= MU_EPS or mu_s >= 1.0 - MU_EPS:
        return None
    kappa = mu_s * (1.0 - mu_s) / var_s - 1.0
    if kappa <= 0.0:
        return None
    return max(mu_s * kappa, MIN_ALPHA_BETA), max((1.0 - mu_s) * kappa, MIN_ALPHA_BETA)


def apply_calibration_map(
    alpha_star: float,
    beta_star: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    T: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply fitted Beta CDF map to arrays of (alpha, beta).

    Per instance: sample T points from Beta(α_i, β_i), push through
    I_x(α*, β*), refit via MoM. Falls back to input params on failure.
    """
    alpha_out = alpha.copy().astype(float)
    beta_out  = beta.copy().astype(float)

    for i, (a_i, b_i) in enumerate(zip(alpha, beta)):
        a_i = max(float(a_i), MIN_ALPHA_BETA)
        b_i = max(float(b_i), MIN_ALPHA_BETA)
        samples = np.random.beta(a_i, b_i, size=T).clip(1e-9, 1.0 - 1e-9)
        mapped  = betainc(alpha_star, beta_star, samples)
        fit     = _mom_fit(mapped)
        if fit is not None:
            alpha_out[i], beta_out[i] = fit

    return alpha_out, beta_out


def evaluate(
    alpha_star: float,
    beta_star: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    y: np.ndarray,
    Q: int = 200,
    T: int = 500,
) -> dict:
    """Evaluate calibration quality. Returns ece, fd_mean, fd_median, alpha_star, beta_star."""
    a2, b2 = apply_calibration_map(alpha_star, beta_star, alpha, beta, T)
    kl     = _beta_kl(a2 + y, b2 + (1 - y), a2, b2)
    fd     = (a2 + b2) * kl
    return {
        "ece":        float(_ks_ece_abs(a2, b2, y, Q)),
        "fd_mean":    float(fd.mean()),
        "fd_median":  float(np.median(fd)),
        "alpha_star": alpha_star,
        "beta_star":  beta_star,
    }


def _apply_min_ece_calibration(
    confidences: list[BetaDistribution | None],
    alpha_star: float,
    beta_star: float,
    T: int = 1000,
) -> list[BetaDistribution | None]:
    """
    Apply fitted Beta CDF map to a list of BetaDistributions.

    Per instance: sample T points from the input Beta, push each through
    I_x(α*, β*), refit via MoM. Falls back to original on failure.
    """
    calibrated = []
    for conf in confidences:
        if conf is None:
            calibrated.append(None)
            continue
        try:
            alpha_in, beta_in = _to_alpha_beta(conf.mu, conf.sigma)
        except Exception:
            calibrated.append(conf)
            continue

        a_in    = max(alpha_in, MIN_ALPHA_BETA)
        b_in    = max(beta_in,  MIN_ALPHA_BETA)
        samples = np.random.beta(a_in, b_in, size=T).clip(1e-9, 1.0 - 1e-9)
        mapped  = betainc(alpha_star, beta_star, samples)
        fit     = _mom_fit(mapped)

        if fit is None:
            calibrated.append(conf)
            continue

        mu_out, sigma_out = _to_mu_sigma(fit[0], fit[1])
        calibrated.append(BetaDistribution(mu=mu_out, sigma=sigma_out))

    return calibrated


def _rebuild_with_consistent_sigma(
    calibrated_mu: float,
    original_beta: BetaDistribution | None,
) -> BetaDistribution:
    """
    Rebuild a BetaDistribution with a new mu but the same concentration kappa
    as the original. Sigma is always recomputed from (mu', kappa) so the output
    is guaranteed to satisfy σ'² < μ'(1-μ'), keeping alpha' and beta' strictly
    positive. Falls back to kappa=2 (a near-uniform Beta) when the original
    cannot be parsed.
    """
    mu_c = float(np.clip(calibrated_mu, MU_EPS, 1.0 - MU_EPS))
    kappa = 2.0  # safe fallback: kappa=2 gives a gently informative Beta
    if original_beta is not None:
        try:
            orig_a, orig_b = _to_alpha_beta(original_beta.mu, original_beta.sigma)
            kappa = max(float(orig_a + orig_b), MIN_KAPPA)
        except Exception:
            pass
    # sigma² = mu'(1-mu')/(kappa+1) < mu'(1-mu') always → alpha', beta' > 0
    sigma_c = float(np.sqrt(mu_c * (1.0 - mu_c) / (kappa + 1.0)))
    return BetaDistribution(mu=mu_c, sigma=sigma_c)


# =============================================================================
# FAITHFULNESS DIVERGENCE  (evaluation only, never used in training loss)
#
#   FD_i = (alpha_i + beta_i) * KL( Beta_post || Beta_prior )
#
#   prior = Beta(alpha_i, beta_i)
#   post  = Beta(alpha_i + y_i, beta_i + 1 - y_i)   [single Bayesian update]
#
#   KL( Beta(a_P, b_P) || Beta(a_Q, b_Q) ):
#       = log B(a_Q, b_Q) - log B(a_P, b_P)
#         + (a_P - a_Q) * psi(a_P)
#         + (b_P - b_Q) * psi(b_P)
#         + (a_Q - a_P + b_Q - b_P) * psi(a_P + b_P)
#
#   P's parameters go into the digammas.
#   Q's parameters appear only in log B(Q).
#   Always >= 0 by non-negativity of KL divergence.
#
#   Interpretation:
#     FD is large when the model is sharp (high kappa) and wrong
#     (post differs greatly from prior). It captures the severity of
#     overconfident errors. Low FD means the distribution anticipated its
#     label, i.e. it was appropriately hedged.
# =============================================================================

def _safe_kappa(conf: "BetaDistribution") -> float | None:
    """Extract kappa = alpha + beta from a BetaDistribution; returns None on failure."""
    if conf is None:
        return None
    try:
        a, b = _to_alpha_beta(conf.mu, conf.sigma)
        return float(a + b)
    except Exception:
        return None


def compute_faithfulness_divergence(
    confidences: list[BetaDistribution | None],
    accuracies: list[float | int],
) -> dict:
    """
    Compute Faithfulness Divergence for a list of BetaDistributions.

    FD_i = (alpha_i + beta_i) * KL( Beta_post || Beta_prior )

    where:
        prior = Beta(alpha_i, beta_i)
        post  = Beta(alpha_i + y_i, beta_i + 1 - y_i)

    Args:
        confidences: list of BetaDistribution objects (None entries skipped)
        accuracies:  list of binary correctness labels in {0, 1}

    Returns:
        dict with keys:
            "fd_per_instance": np.ndarray of FD_i for each valid instance
            "fd_mean":         float, mean FD across valid instances
            "fd_median":       float, median FD
            "valid_indices":   list[int], positions of non-None valid entries
    """
    alphas, betas, ys, valid_idx = [], [], [], []

    for i, (conf, acc) in enumerate(zip(confidences, accuracies)):
        if conf is None:
            continue
        try:
            alpha_i, beta_i = _to_alpha_beta(conf.mu, conf.sigma)
            acc_f = float(acc) if acc != "" else np.nan
            if np.isnan(acc_f):
                continue
            alphas.append(alpha_i)
            betas.append(beta_i)
            ys.append(acc_f)
            valid_idx.append(i)
        except Exception:
            continue

    if not alphas:
        return {
            "fd_per_instance": np.array([]),
            "fd_mean":         float("nan"),
            "fd_median":       float("nan"),
            "valid_indices":   [],
        }

    a = np.array(alphas, dtype=np.float64)
    b = np.array(betas,  dtype=np.float64)
    y = np.array(ys,     dtype=np.float64)

    a_post = a + y
    b_post = b + (1.0 - y)
    kl = _beta_kl(a_post, b_post, a, b)
    fd_np = (a + b) * kl
    return {
        "fd_per_instance": fd_np,
        "fd_mean":         float(fd_np.mean()),
        "fd_median":       float(np.median(fd_np)),
        "valid_indices":   valid_idx,
    }


def _print_signal_metrics_from_lists(
    label: str,
    accuracies: list[float | int] | np.ndarray,
    confidences: list[BetaDistribution | None],
) -> None:
    valid_acc: list[float] = []
    valid_conf: list[BetaDistribution] = []

    for acc, conf in zip(accuracies, confidences):
        if conf is None:
            continue
        try:
            acc_f = float(acc) if acc != "" else np.nan
        except (TypeError, ValueError):
            continue
        if not np.isfinite(acc_f):
            continue
        valid_acc.append(acc_f)
        valid_conf.append(conf)

    if not valid_acc:
        print(f"[{label}] no valid rows for metric computation")
        return

    organised_output = OrganisedOutputs(
        accuracy_scores=[valid_acc],
        extracted_confidences=[valid_conf],
        extracted_answers=[[""] * len(valid_acc)],
    )
    ece_val = generalised_ece({}, organised_output)[0]
    fd_val = faithfulness_divergence({}, organised_output)[0]
    print(f"[{label}] n={len(valid_acc)} | ECE={ece_val:.6f} | FD={fd_val:.6f}")


# =============================================================================
# IN-DOMAIN CALIBRATION
# =============================================================================

def in_domain_numerical_post_hoc_calibration(
    confidences: list[BetaDistribution],
    accuracies: list[float | int],
    method: str = "min_ece",
) -> list[BetaDistribution | None]:
    """
    Calibrate Beta distributions using first 25% for training, last 75%
    for prediction.

    Args:
        confidences: list of BetaDistribution objects
        accuracies:  list of accuracy scores (0 or 1)
        method:      "min_ece" (recommended), "isotonic", "platt_uni", "platt_bi"

    Returns:
        list of calibrated BetaDistribution objects.
        First 20% are None (training set, not calibrated).

    Method notes:
        "min_ece":    Two-stage: scale mean first via Platt(mu), then
                      scale concentration via w_kap to minimise KS-ECE.
                      Uses clipping/fallbacks to avoid degenerate parameters.
                      Always produces a valid, strictly-positive Beta. Recommended.
        "isotonic":   Isotonic regression on mu only. Sigma recomputed from
                      original kappa for consistency.
        "platt_uni":  Logistic regression on mu only. Sigma recomputed.
        "platt_bi":   Logistic regression on (mu, sigma). Sigma recomputed.
    """
    mu_values    = np.array([c.mu    for c in confidences], dtype=float)
    sigma_values = np.array([c.sigma for c in confidences], dtype=float)
    acc_values   = np.array(
        [acc if acc != "" else 0 for acc in accuracies], dtype=float
    )

    n          = len(mu_values)
    train_size = max(1, int(np.ceil(0.3 * n)))

    y_train          = acc_values[:train_size]
    test_confidences = confidences[train_size:]
    X_test_mu        = mu_values[train_size:]
    X_test_sigma     = sigma_values[train_size:]

    calibrated_betas: list[BetaDistribution | None] = [None] * train_size

    if method == "min_ece":
        alpha_star, beta_star, h_star = fit_calibration_map(
            mu_values[:train_size],
            sigma_values[:train_size],
            y_train,
        )
        print(f"[min_ece] h*={h_star:.4f}  α*={alpha_star:.4f}  β*={beta_star:.4f}")
        train_calibrated = _apply_min_ece_calibration(
            list(confidences[:train_size]), alpha_star, beta_star
        )
        _print_signal_metrics_from_lists("min_ece_train_calibrated", y_train, train_calibrated)
        calibrated_betas.extend(
            _apply_min_ece_calibration(test_confidences, alpha_star, beta_star)
        )
        return calibrated_betas

    elif method == "hist_bin":
        train_confs = list(confidences[:train_size])
        train_kappa = np.array(
            [_safe_kappa(c) if c is not None else np.nan for c in train_confs],
            dtype=float,
        )
        valid_mask  = np.isfinite(train_kappa) & np.isfinite(mu_values[:train_size])
        mu_tr       = mu_values[:train_size][valid_mask].clip(MU_EPS, 1.0 - MU_EPS)
        kappa_tr    = train_kappa[valid_mask].clip(MIN_KAPPA, MAX_KAPPA)
        bin_accs    = _fit_soft_hist_bins(mu_tr, kappa_tr, y_train[valid_mask], 20)
        print(f"[hist_bin] bin_accuracies={np.round(bin_accs, 3).tolist()}")

        train_calibrated: list[BetaDistribution | None] = []
        for conf in train_confs:
            if conf is None:
                train_calibrated.append(None)
                continue
            kappa = _safe_kappa(conf)
            if kappa is None:
                train_calibrated.append(conf)
                continue
            mu_i = float(np.clip(conf.mu, MU_EPS, 1.0 - MU_EPS))
            kap_i = float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))
            mu_p = float(np.clip(_apply_soft_hist_bin(mu_i, kap_i, bin_accs), MU_EPS, 1.0 - MU_EPS))
            train_calibrated.append(_rebuild_with_consistent_sigma(mu_p, conf))
        _print_signal_metrics_from_lists("hist_bin_train_calibrated", y_train, train_calibrated)

        for conf in test_confidences:
            if conf is None:
                calibrated_betas.append(None)
                continue
            kappa = _safe_kappa(conf)
            if kappa is None:
                calibrated_betas.append(conf)
                continue
            mu_i   = float(np.clip(conf.mu, MU_EPS, 1.0 - MU_EPS))
            kap_i  = float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))
            mu_p   = float(np.clip(_apply_soft_hist_bin(mu_i, kap_i, bin_accs), MU_EPS, 1.0 - MU_EPS))
            calibrated_betas.append(_rebuild_with_consistent_sigma(mu_p, conf))

        return calibrated_betas

    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(mu_values[:train_size], y_train)
        calibrated_means_train = calibrator.predict(mu_values[:train_size])
        calibrated_means = calibrator.predict(X_test_mu)

    elif method == "platt_uni":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(mu_values[:train_size].reshape(-1, 1), y_train)
        calibrated_means_train = calibrator.predict_proba(
            mu_values[:train_size].reshape(-1, 1)
        )[:, 1]
        calibrated_means = calibrator.predict_proba(
            X_test_mu.reshape(-1, 1)
        )[:, 1]

    elif method == "platt_bi":
        X_tr = np.column_stack([mu_values[:train_size], sigma_values[:train_size]])
        X_te = np.column_stack([X_test_mu, X_test_sigma])
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(X_tr, y_train)
        calibrated_means_train = calibrator.predict_proba(X_tr)[:, 1]
        calibrated_means = calibrator.predict_proba(X_te)[:, 1]

    elif method == "temperature_scaling":
        mu_tr_raw = mu_values[:train_size]
        valid_tr = np.isfinite(mu_tr_raw) & np.isfinite(acc_values[:train_size])
        mu_tr = mu_tr_raw[valid_tr].clip(MU_EPS, 1.0 - MU_EPS)
        y_tr_ts = acc_values[:train_size][valid_tr]
        logit_tr = np.log(mu_tr / (1.0 - mu_tr))

        def _ts_nll(log_T: float) -> float:
            p = expit(logit_tr / np.exp(log_T)).clip(1e-9, 1.0 - 1e-9)
            return float(-np.mean(y_tr_ts * np.log(p) + (1.0 - y_tr_ts) * np.log(1.0 - p)))

        if len(mu_tr) >= 2:
            res = minimize_scalar(_ts_nll, bounds=(-4.0, 4.0), method="bounded")
            T_star = float(np.exp(res.x)) if np.isfinite(res.x) else 1.0
        else:
            T_star = 1.0
        print(f"[temperature_scaling] T*={T_star:.4f}")

        mu_tr_full = mu_tr_raw.clip(MU_EPS, 1.0 - MU_EPS)
        np.nan_to_num(mu_tr_full, nan=0.5, copy=False)
        calibrated_means_train = expit(np.log(mu_tr_full / (1.0 - mu_tr_full)) / T_star)
        X_te_clipped = np.where(np.isfinite(X_test_mu), X_test_mu, 0.5).clip(MU_EPS, 1.0 - MU_EPS)
        logit_te = np.log(X_te_clipped / (1.0 - X_te_clipped))
        calibrated_means = expit(logit_te / T_star)

    elif method == "two_stage_isotonic":
        B = 3
        train_confs = list(confidences[:train_size])
        train_mu_raw = mu_values[:train_size]
        train_kappa = np.array(
            [_safe_kappa(c) if c is not None else np.nan for c in train_confs],
            dtype=float,
        )
        valid_mask = np.isfinite(train_kappa)
        kappa_qs = np.quantile(
            train_kappa[valid_mask], np.linspace(0, 1, B + 1)[1:-1]
        )

        def _assign_bin(kappa_val: float) -> int:
            if not np.isfinite(kappa_val):
                return 1
            for b_idx, q in enumerate(kappa_qs):
                if kappa_val <= q:
                    return b_idx
            return B - 1

        bin_regressors: list = []
        for b_idx in range(B):
            mask = np.array(
                [_assign_bin(k) == b_idx for k in train_kappa], dtype=bool
            )
            mask &= valid_mask
            if mask.sum() >= 2:
                reg = IsotonicRegression(out_of_bounds="clip")
                reg.fit(train_mu_raw[mask], y_train[mask])
                bin_regressors.append(reg)
            else:
                bin_regressors.append(None)
            print(
                f"[two_stage_isotonic] bin {b_idx}: "
                f"κ ≤ {kappa_qs[b_idx - 1] if b_idx > 0 else '-∞'} ... "
                f"κ ≤ {kappa_qs[b_idx] if b_idx < B - 1 else '∞'}, "
                f"n={int(mask.sum())}"
            )

        train_calibrated: list[BetaDistribution | None] = []
        for conf in train_confs:
            if conf is None:
                train_calibrated.append(None)
                continue
            kappa = _safe_kappa(conf)
            if kappa is None:
                train_calibrated.append(conf)
                continue
            b_idx = _assign_bin(kappa)
            reg = bin_regressors[b_idx]
            mu_prime = float(reg.predict(np.array([conf.mu]))[0]) if reg is not None else conf.mu
            train_calibrated.append(_rebuild_with_consistent_sigma(mu_prime, conf))
        # _print_signal_metrics_from_lists("two_stage_isotonic_train_calibrated", y_train, train_calibrated)

        for conf in test_confidences:
            if conf is None:
                calibrated_betas.append(None)
                continue
            kappa = _safe_kappa(conf)
            if kappa is None:
                calibrated_betas.append(conf)
                continue
            b_idx = _assign_bin(kappa)
            reg = bin_regressors[b_idx]
            mu_prime = float(reg.predict(np.array([conf.mu]))[0]) if reg is not None else conf.mu
            calibrated_betas.append(_rebuild_with_consistent_sigma(mu_prime, conf))

        return calibrated_betas

    elif method == "ece_fd_opt":
        train_confs  = list(confidences[:train_size])
        train_kappa  = np.array(
            [_safe_kappa(c) if c is not None else np.nan for c in train_confs],
            dtype=float,
        )
        valid_mask = np.isfinite(train_kappa) & np.isfinite(mu_values[:train_size])
        mu_tr      = mu_values[:train_size][valid_mask].clip(MU_EPS, 1.0 - MU_EPS)
        kappa_tr   = train_kappa[valid_mask].clip(MIN_KAPPA, MAX_KAPPA)
        y_tr       = y_train[valid_mask]
        logit_mu_tr = np.log(mu_tr / (1.0 - mu_tr))
        log_kap_tr  = np.log(kappa_tr)

        # Normalise by identity-params baseline so ECE and FD start at 1.0 each
        a0    = (mu_tr * kappa_tr).clip(MIN_ALPHA_BETA)
        b0    = ((1.0 - mu_tr) * kappa_tr).clip(MIN_ALPHA_BETA)
        ece0  = max(_ks_ece_sq(a0, b0, y_tr, Q=100), 1e-10)
        fd0   = max(_mean_fd(a0, b0, y_tr), 1e-10)
        print(f"[ece_fd_opt] baseline ECE={ece0:.6f}  FD={fd0:.6f}")

        # Joint optimisation: normalised ECE + normalised FD over all 4 params
        def _joint_loss(params: np.ndarray) -> float:
            w_k, b_k, w, b = params
            kap_p = np.exp(np.clip(w_k * log_kap_tr + b_k, -20.0, 20.0)).clip(MIN_KAPPA, MAX_KAPPA)
            mu_p  = expit(w * logit_mu_tr + b).clip(MU_EPS, 1.0 - MU_EPS)
            a_p   = (mu_p * kap_p).clip(MIN_ALPHA_BETA)
            b_p   = ((1.0 - mu_p) * kap_p).clip(MIN_ALPHA_BETA)
            return _ks_ece_sq(a_p, b_p, y_tr, Q=100) / ece0 + _mean_fd(a_p, b_p, y_tr) / fd0

        res = minimize(
            _joint_loss, x0=np.array([1.0, 0.0, 1.0, 0.0]), method="L-BFGS-B",
            bounds=[(-10.0, 10.0)] * 4,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        w_kap, b_kap, w_mu, b_mu = (float(x) for x in res.x)
        print(f"[ece_fd_opt] w_kap={w_kap:.4f}  b_kap={b_kap:.4f}  "
              f"w_mu={w_mu:.4f}  b_mu={b_mu:.4f}  loss={res.fun:.6f}")

        train_calibrated: list[BetaDistribution | None] = []
        for conf in train_confs:
            if conf is None:
                train_calibrated.append(None)
                continue
            kappa = _safe_kappa(conf)
            if kappa is None:
                train_calibrated.append(conf)
                continue
            mu_i = float(np.clip(conf.mu, MU_EPS, 1.0 - MU_EPS))
            kap_i = float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))
            kap_p = float(
                np.clip(
                    np.exp(np.clip(w_kap * np.log(kap_i) + b_kap, -20.0, 20.0)),
                    MIN_KAPPA,
                    MAX_KAPPA,
                )
            )
            mu_p = float(expit(w_mu * np.log(mu_i / (1.0 - mu_i)) + b_mu).clip(MU_EPS, 1.0 - MU_EPS))
            a_p = max(mu_p * kap_p, MIN_ALPHA_BETA)
            b_p = max((1.0 - mu_p) * kap_p, MIN_ALPHA_BETA)
            mu_out, sigma_out = _to_mu_sigma(a_p, b_p)
            train_calibrated.append(BetaDistribution(mu=mu_out, sigma=sigma_out))
        # _print_signal_metrics_from_lists("ece_fd_opt_train_calibrated", y_train, train_calibrated)

        for conf in test_confidences:
            if conf is None:
                calibrated_betas.append(None)
                continue
            kappa = _safe_kappa(conf)
            if kappa is None:
                calibrated_betas.append(conf)
                continue
            mu_i  = float(np.clip(conf.mu, MU_EPS, 1.0 - MU_EPS))
            kap_i = float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))
            kap_p = float(np.clip(np.exp(np.clip(w_kap * np.log(kap_i) + b_kap, -20.0, 20.0)),
                                   MIN_KAPPA, MAX_KAPPA))
            mu_p  = float(expit(w_mu * np.log(mu_i / (1.0 - mu_i)) + b_mu).clip(MU_EPS, 1.0 - MU_EPS))
            a_p   = max(mu_p * kap_p, MIN_ALPHA_BETA)
            b_p   = max((1.0 - mu_p) * kap_p, MIN_ALPHA_BETA)
            mu_out, sigma_out = _to_mu_sigma(a_p, b_p)
            calibrated_betas.append(BetaDistribution(mu=mu_out, sigma=sigma_out))

        return calibrated_betas

    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    train_calibrated: list[BetaDistribution | None] = []
    for cal_mu, orig_beta in zip(calibrated_means_train, confidences[:train_size]):
        if orig_beta is None:
            train_calibrated.append(None)
        else:
            train_calibrated.append(_rebuild_with_consistent_sigma(cal_mu, orig_beta))
    # _print_signal_metrics_from_lists(f"{method}_train_calibrated", y_train, train_calibrated)

    for cal_mu, orig_beta in zip(calibrated_means, test_confidences):
        if orig_beta is None:
            calibrated_betas.append(None)
        else:
            calibrated_betas.append(_rebuild_with_consistent_sigma(cal_mu, orig_beta))

    return calibrated_betas


# =============================================================================
# CROSS-DOMAIN CALIBRATION
# =============================================================================

def cross_domain_numerical_post_hoc_calibration(
    confidences_train: list[BetaDistribution | None],
    accuracies_train: list[float | int],
    raw_confidences: list[BetaDistribution],
    method: str = "affine",
) -> list[BetaDistribution | None]:
    """
    Calibrate Beta distributions using one domain for training, apply to
    another.

    Args:
        confidences_train: BetaDistribution objects for training
        accuracies_train:  binary correctness labels for training
        raw_confidences:   BetaDistribution objects to calibrate
        method:            "min_ece" (recommended), "isotonic", "platt_uni",
                           "platt_bi"

    Returns:
        list of calibrated BetaDistribution objects, same length as
        raw_confidences. None entries in raw_confidences are passed through.

    Method notes: see in_domain_numerical_post_hoc_calibration.
    None entries and NaN accuracies in the training set are filtered out.
    """
    # Filter valid training pairs
    valid_pairs: list[tuple[BetaDistribution, float]] = []
    for conf, acc in zip(confidences_train, accuracies_train):
        if conf is None or not hasattr(conf, "mu") or np.isnan(conf.mu):
            continue
        try:
            acc_f = float(acc) if acc != "" else np.nan
        except (TypeError, ValueError):
            continue
        if not np.isnan(acc_f):
            valid_pairs.append((conf, acc_f))

    if not valid_pairs:
        raise ValueError(
            "No valid confidence-accuracy pairs found in training data."
        )

    train_confs = [c for c, _ in valid_pairs]
    acc_train   = np.array([a for _, a in valid_pairs], dtype=float)
    mu_train    = np.array([c.mu    for c in train_confs], dtype=float)
    sigma_train = np.array([c.sigma for c in train_confs], dtype=float)

    # Separate valid raw entries; Nones are passed through unchanged.
    valid_raw_idx   = [i for i, c in enumerate(raw_confidences) if c is not None]
    valid_raw_confs = [raw_confidences[i] for i in valid_raw_idx]
    mu_raw    = np.array([c.mu    for c in valid_raw_confs], dtype=float)
    sigma_raw = np.array([c.sigma for c in valid_raw_confs], dtype=float)

    if method == "min_ece":
        alpha_star, beta_star, h_star = fit_calibration_map(mu_train, sigma_train, acc_train)
        print(f"[min_ece] h*={h_star:.4f}  α*={alpha_star:.4f}  β*={beta_star:.4f}")
        return _apply_min_ece_calibration(raw_confidences, alpha_star, beta_star)

    elif method == "hist_bin":
        train_kappa = np.array(
            [_safe_kappa(c) if c is not None else np.nan for c in train_confs],
            dtype=float,
        )
        valid_k_mask = np.isfinite(train_kappa) & np.isfinite(mu_train)
        mu_tr        = mu_train[valid_k_mask].clip(MU_EPS, 1.0 - MU_EPS)
        kappa_tr     = train_kappa[valid_k_mask].clip(MIN_KAPPA, MAX_KAPPA)
        bin_accs     = _fit_soft_hist_bins(mu_tr, kappa_tr, acc_train[valid_k_mask])
        print(f"[hist_bin] bin_accuracies={np.round(bin_accs, 3).tolist()}")

        result: list[BetaDistribution | None] = [None] * len(raw_confidences)
        for i, orig_beta in zip(valid_raw_idx, valid_raw_confs):
            kappa = _safe_kappa(orig_beta)
            if kappa is None:
                result[i] = orig_beta
                continue
            mu_i  = float(np.clip(orig_beta.mu, MU_EPS, 1.0 - MU_EPS))
            kap_i = float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))
            mu_p  = float(np.clip(_apply_soft_hist_bin(mu_i, kap_i, bin_accs), MU_EPS, 1.0 - MU_EPS))
            result[i] = _rebuild_with_consistent_sigma(mu_p, orig_beta)
        return result

    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(mu_train, acc_train)
        calibrated_means = calibrator.predict(mu_raw)

    elif method == "platt_uni":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(mu_train.reshape(-1, 1), acc_train)
        calibrated_means = calibrator.predict_proba(mu_raw.reshape(-1, 1))[:, 1]

    elif method == "platt_bi":
        X_tr = np.column_stack([mu_train, sigma_train])
        X_te = np.column_stack([mu_raw,   sigma_raw])
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(X_tr, acc_train)
        calibrated_means = calibrator.predict_proba(X_te)[:, 1]

    elif method == "temperature_scaling":
        valid_cd = np.isfinite(mu_train) & np.isfinite(acc_train)
        mu_tr_cd = mu_train[valid_cd].clip(MU_EPS, 1.0 - MU_EPS)
        y_tr_cd = acc_train[valid_cd]
        logit_tr_cd = np.log(mu_tr_cd / (1.0 - mu_tr_cd))

        def _ts_nll_cd(log_T: float) -> float:
            p = expit(logit_tr_cd / np.exp(log_T)).clip(1e-9, 1.0 - 1e-9)
            return float(-np.mean(y_tr_cd * np.log(p) + (1.0 - y_tr_cd) * np.log(1.0 - p)))

        if len(mu_tr_cd) >= 2:
            res = minimize_scalar(_ts_nll_cd, bounds=(-4.0, 4.0), method="bounded")
            T_star = float(np.exp(res.x)) if np.isfinite(res.x) else 1.0
        else:
            T_star = 1.0
        print(f"[temperature_scaling] T*={T_star:.4f}")
        mu_raw_safe = np.where(np.isfinite(mu_raw), mu_raw, 0.5).clip(MU_EPS, 1.0 - MU_EPS)
        calibrated_means = expit(np.log(mu_raw_safe / (1.0 - mu_raw_safe)) / T_star)

    elif method == "two_stage_isotonic":
        B = 3
        train_kappa = np.array(
            [_safe_kappa(c) if c is not None else np.nan for c in train_confs],
            dtype=float,
        )
        valid_k_mask = np.isfinite(train_kappa)
        kappa_qs = np.quantile(
            train_kappa[valid_k_mask], np.linspace(0, 1, B + 1)[1:-1]
        )

        def _assign_bin(kappa_val: float) -> int:
            if not np.isfinite(kappa_val):
                return 1
            for b_idx, q in enumerate(kappa_qs):
                if kappa_val <= q:
                    return b_idx
            return B - 1

        bin_regressors: list = []
        for b_idx in range(B):
            mask = np.array(
                [_assign_bin(k) == b_idx for k in train_kappa], dtype=bool
            )
            mask &= valid_k_mask
            if mask.sum() >= 2:
                reg = IsotonicRegression(out_of_bounds="clip")
                reg.fit(mu_train[mask], acc_train[mask])
                bin_regressors.append(reg)
            else:
                bin_regressors.append(None)
            print(
                f"[two_stage_isotonic] bin {b_idx}: "
                f"κ ≤ {kappa_qs[b_idx - 1] if b_idx > 0 else '-∞'} ... "
                f"κ ≤ {kappa_qs[b_idx] if b_idx < B - 1 else '∞'}, "
                f"n={int(mask.sum())}"
            )

        result: list[BetaDistribution | None] = [None] * len(raw_confidences)
        for i, orig_beta in zip(valid_raw_idx, valid_raw_confs):
            kappa = _safe_kappa(orig_beta)
            if kappa is None:
                result[i] = orig_beta
                continue
            b_idx = _assign_bin(kappa)
            reg = bin_regressors[b_idx]
            mu_prime = float(reg.predict(np.array([orig_beta.mu]))[0]) if reg is not None else orig_beta.mu
            result[i] = _rebuild_with_consistent_sigma(mu_prime, orig_beta)
        return result

    elif method == "ece_fd_opt":
        train_kappa = np.array(
            [_safe_kappa(c) if c is not None else np.nan for c in train_confs],
            dtype=float,
        )
        valid_k_mask = np.isfinite(train_kappa) & np.isfinite(mu_train)
        mu_tr        = mu_train[valid_k_mask].clip(MU_EPS, 1.0 - MU_EPS)
        kappa_tr     = train_kappa[valid_k_mask].clip(MIN_KAPPA, MAX_KAPPA)
        y_tr         = acc_train[valid_k_mask]
        logit_mu_tr  = np.log(mu_tr / (1.0 - mu_tr))
        log_kap_tr   = np.log(kappa_tr)

        # Normalise by identity-params baseline so ECE and FD start at 1.0 each
        a0   = (mu_tr * kappa_tr).clip(MIN_ALPHA_BETA)
        b0   = ((1.0 - mu_tr) * kappa_tr).clip(MIN_ALPHA_BETA)
        ece0 = max(_ks_ece_sq(a0, b0, y_tr, Q=100), 1e-10)
        fd0  = max(_mean_fd(a0, b0, y_tr), 1e-10)
        print(f"[ece_fd_opt] baseline ECE={ece0:.6f}  FD={fd0:.6f}")

        # Joint optimisation: normalised ECE + normalised FD over all 4 params
        def _joint_loss(params: np.ndarray) -> float:
            w_k, b_k, w, b = params
            kap_p = np.exp(np.clip(w_k * log_kap_tr + b_k, -20.0, 20.0)).clip(MIN_KAPPA, MAX_KAPPA)
            mu_p  = expit(w * logit_mu_tr + b).clip(MU_EPS, 1.0 - MU_EPS)
            a_p   = (mu_p * kap_p).clip(MIN_ALPHA_BETA)
            b_p   = ((1.0 - mu_p) * kap_p).clip(MIN_ALPHA_BETA)
            return _ks_ece_sq(a_p, b_p, y_tr, Q=100) / ece0 + _mean_fd(a_p, b_p, y_tr) / fd0

        res = minimize(
            _joint_loss, x0=np.array([1.0, 0.0, 1.0, 0.0]), method="L-BFGS-B",
            bounds=[(-10.0, 10.0)] * 4,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        w_kap, b_kap, w_mu, b_mu = (float(x) for x in res.x)
        print(f"[ece_fd_opt] w_kap={w_kap:.4f}  b_kap={b_kap:.4f}  "
              f"w_mu={w_mu:.4f}  b_mu={b_mu:.4f}  loss={res.fun:.6f}")

        result: list[BetaDistribution | None] = [None] * len(raw_confidences)
        for i, orig_beta in zip(valid_raw_idx, valid_raw_confs):
            kappa = _safe_kappa(orig_beta)
            if kappa is None:
                result[i] = orig_beta
                continue
            mu_i  = float(np.clip(orig_beta.mu, MU_EPS, 1.0 - MU_EPS))
            kap_i = float(np.clip(kappa, MIN_KAPPA, MAX_KAPPA))
            kap_p = float(np.clip(np.exp(np.clip(w_kap * np.log(kap_i) + b_kap, -20.0, 20.0)),
                                   MIN_KAPPA, MAX_KAPPA))
            mu_p  = float(expit(w_mu * np.log(mu_i / (1.0 - mu_i)) + b_mu).clip(MU_EPS, 1.0 - MU_EPS))
            a_p   = max(mu_p * kap_p, MIN_ALPHA_BETA)
            b_p   = max((1.0 - mu_p) * kap_p, MIN_ALPHA_BETA)
            mu_out, sigma_out = _to_mu_sigma(a_p, b_p)
            result[i] = BetaDistribution(mu=mu_out, sigma=sigma_out)
        return result

    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    result: list[BetaDistribution | None] = [None] * len(raw_confidences)
    for i, (cal_mu, orig_beta) in zip(valid_raw_idx, zip(calibrated_means, valid_raw_confs)):
        result[i] = _rebuild_with_consistent_sigma(cal_mu, orig_beta)
    return result


# =============================================================================
# LINGUISTIC CONFIDENCE ESTIMATION
# =============================================================================

def estimate_linguistic_confidence(
    responses: list[str],
    target_means: list[float | BetaDistribution | None],
    evaluators_cfg: dict,
    evaluator_keys: list[str],
) -> list[BetaDistribution | None]:
    prompts = build_linguistic_evaluator_prompts(responses, target_means)

    def extract_score(text: str) -> float:
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            score = float(match.group(1))
            score = min(max(score, 0.0), 100.0) / 100.0
        else:
            score = np.nan
        return score

    scores_collection: list[list[float]] = [[] for _ in responses]

    for evaluator in evaluator_keys:
        model_manager: ModelManager = ModelManager(
            master_cfg=evaluators_cfg, model_config_type=evaluator
        )
        prompt_collection = PromptCollection(
            system_prompt="You are a careful assistant.",
            context_texts=prompts,
        )

        outputs = model_manager.run_generation(prompt_collection)
        for output in outputs:
            for i, text in enumerate(output.output_texts):
                score = extract_score(text)
                if not np.isnan(score):
                    scores_collection[i].append(score)

        gc.collect()

    confidence_dists = []
    for text_idx, _ in enumerate(responses):
        try:
            all_scores   = scores_collection[text_idx]
            valid_scores = [s for s in all_scores if not np.isnan(s)]
            if valid_scores:
                mu    = float(np.mean(valid_scores))
                sigma = float(np.std(valid_scores)) if len(valid_scores) > 1 else 1e-6
            else:
                mu    = 0.5
                sigma = 1e-6
            confidence_dists.append(BetaDistribution(mu=mu, sigma=sigma))
        except Exception as e:
            print(f"Error processing text index {text_idx}: {e}")
            confidence_dists.append(None)

    return confidence_dists


def build_linguistic_evaluator_prompts(
    responses: list[str],
    target_means: list[float | BetaDistribution | None],
    top_k: int = 20,
) -> list[str]:
    if len(responses) != len(target_means):
        raise ValueError(
            f"responses and target_means length mismatch: {len(responses)} vs {len(target_means)}"
        )

    def _normalize_mean(value: float | BetaDistribution | None) -> float:
        if value is None:
            return 50.0
        if isinstance(value, BetaDistribution):
            mean_val = float(value.mu)
        else:
            try:
                mean_val = float(value)
            except (TypeError, ValueError):
                return 50.0
        if not np.isfinite(mean_val):
            return 50.0
        return mean_val * 100.0 if mean_val <= 1.0 else mean_val

    def _select_human_cues(target_mean: float) -> list[dict]:
        diffs = np.abs(_HUMAN_CUE_MEANS - target_mean)
        idx = np.argsort(diffs)[:top_k]
        return [human_annotated_cues[i] for i in idx]

    prompts = []
    for response, mean_val in zip(responses, target_means):
        normalized = _normalize_mean(mean_val)
        cues = _select_human_cues(normalized)
        prompts.append(
            LINGUISTIC_EVALUATOR_PROMPT.format(
                sentence=response,
                human_annotated_cues=cues,
            )
        )
    return prompts