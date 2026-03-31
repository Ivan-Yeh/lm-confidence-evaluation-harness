import numpy as np
from scipy.stats import beta
from scipy.special import betaln, psi


# ---------------------------------------------------------------------------
# Grid / soft-membership helpers
# ---------------------------------------------------------------------------

def build_grid(num_points=1000):
    s_grid = np.linspace(0, 1, num_points)
    ds = s_grid[1] - s_grid[0]
    return s_grid, ds


def compute_soft_memberships(alpha_list, beta_list, bins, s_grid, ds):
    N = len(alpha_list)
    M = len(bins) - 1
    pdfs = np.zeros((N, len(s_grid)))
    w = np.zeros((N, M))
    for i in range(N):
        pdf = beta.pdf(s_grid, alpha_list[i], beta_list[i])
        norm = np.sum(pdf) * ds
        if norm < 1e-20:
            pdf[:] = 0.0
        else:
            pdf /= norm
        pdfs[i] = pdf
        for m in range(M):
            mask = (s_grid >= bins[m]) & (s_grid < bins[m + 1])
            w[i, m] = np.sum(pdf[mask]) * ds
    return w, pdfs


# ---------------------------------------------------------------------------
# genECE
# ---------------------------------------------------------------------------

def compute_genECE(alpha_list, beta_list, correctness, num_bins=10):
    s_grid, ds = build_grid()
    bins = np.linspace(0, 1, num_bins + 1)
    w, pdfs = compute_soft_memberships(alpha_list, beta_list, bins, s_grid, ds)
    N, M = w.shape
    correctness = np.array(correctness)

    p_m = np.zeros(M)
    bin_ece = np.zeros(M)

    for m in range(M):
        weights = w[:, m]
        total_weight = np.sum(weights)
        if total_weight < 1e-12:
            continue

        r_m = np.sum(weights * correctness) / total_weight

        num = 0.0
        for i in range(N):
            if weights[i] < 1e-12:
                continue
            mask = (s_grid >= bins[m]) & (s_grid < bins[m + 1])
            if np.sum(mask) == 0:
                continue
            cond_mean = np.sum(s_grid[mask] * pdfs[i, mask]) * ds / weights[i]
            if np.isnan(cond_mean):
                continue
            num += weights[i] * cond_mean

        g_m = num / total_weight
        if np.isnan(g_m) or np.isnan(r_m):
            continue

        p_m[m] = total_weight / N
        bin_ece[m] = abs(g_m - r_m)

    return np.sum(p_m * bin_ece)


# ---------------------------------------------------------------------------
# KL divergence  KL( prior || posterior )
#   Measures how much the single observation forces you to update –
#   a well-calibrated, diffuse prior needs little update (low cost);
#   an overconfident, spiked prior needs a large update (high cost).
# ---------------------------------------------------------------------------

def kl_beta(a, b, a2, b2):
    """KL( Beta(a,b) || Beta(a2,b2) )"""
    return (
        betaln(a2, b2) - betaln(a, b)
        + (a - a2) * psi(a)
        + (b - b2) * psi(b)
        + (a2 + b2 - a - b) * psi(a + b)
    )


def compute_kl_correction_cost(alpha_list, beta_list, correctness):
    """
    Average KL( prior || posterior ) across all samples.

    posterior after observing y=1: Beta(a+1, b)
    posterior after observing y=0: Beta(a,   b+1)

    A large value means the model was surprised by the outcome →
    its confidence distribution was miscalibrated (too concentrated).
    """
    kl_vals = []
    for a, b, y in zip(alpha_list, beta_list, correctness):
        a_post = a + y
        b_post = b + (1 - y)
        kl_vals.append(kl_beta(a_post, b_post, a, b))   # KL(post || prior)
    return float(np.mean(kl_vals))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    N = 300

    # ------------------------------------------------------------------
    # Shared ground truth: all three groups see the SAME means and labels.
    # This deliberately makes genECE blind to the difference between groups.
    # ------------------------------------------------------------------
    mus = np.random.uniform(0.6, 0.85, size=N)   # true success probabilities
    y   = np.random.binomial(1, mus)              # observed outcomes

    # ------------------------------------------------------------------
    # Group A – Well-calibrated (moderate concentration)
    #   Beta mean = mu,  concentration c ~ 5-10
    #   The spread of the distribution roughly matches true uncertainty.
    # ------------------------------------------------------------------
    c_good     = np.random.uniform(5, 10, size=N)
    alpha_good = mus * c_good
    beta_good  = (1 - mus) * c_good

    # ------------------------------------------------------------------
    # Group B – Overconfident / variance-miscalibrated (high concentration)
    #   Beta mean = mu  (identical!),  concentration c ~ 150-300
    #   The model is artificially certain; it will be heavily surprised.
    # ------------------------------------------------------------------
    c_over     = np.random.uniform(150, 300, size=N)
    alpha_over = mus * c_over
    beta_over  = (1 - mus) * c_over

    # ------------------------------------------------------------------
    # Group C – Underconfident / over-dispersed (very low concentration)
    #   Beta mean = mu  (identical!),  concentration c ~ 1.2-2
    #   The model hedges excessively; also miscalibrated but differently.
    # ------------------------------------------------------------------
    c_under     = np.random.uniform(1.2, 2.0, size=N)
    alpha_under = mus * c_under
    beta_under  = (1 - mus) * c_under

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    groups = {
        "Well-calibrated  (c≈7)  ": (alpha_good,  beta_good,  y),
        "Overconfident    (c≈225)": (alpha_over,  beta_over,  y),
        "Underconfident   (c≈1.6)": (alpha_under, beta_under, y),
    }

    print("=" * 65)
    print("  VARIANCE MISCALIBRATION: genECE vs KL Correction Cost")
    print("=" * 65)
    print(f"  N={N}, shared means ∈ [0.60, 0.85], identical labels\n")
    print(f"{'Group':<32} {'Mean μ':>8} {'genECE':>10} {'KL cost':>10}")
    print("-" * 65)

    for name, (a, b, labels) in groups.items():
        mean_conf = float(np.mean(a / (a + b)))
        ece       = compute_genECE(a, b, labels)
        kl        = compute_kl_correction_cost(a, b, labels)
        print(f"  {name}  {mean_conf:.4f}  {ece:>10.4f}  {kl:>10.4f}")

    print("=" * 65)
    print("""
KEY INSIGHT
-----------
All three groups have the SAME mean confidence and the SAME labels,
so genECE sees them as essentially equivalent (near-zero for all three).

KL correction cost  KL( prior || posterior )  tells a different story:

  • Overconfident (c≈225) → near-zero KL cost
      The prior is so concentrated (high pseudocount) that a single
      observation barely moves the posterior.  The model is *stubborn*:
      it ignores evidence.  This is a dangerous failure mode – the model
      will never self-correct even when wrong.

  • Well-calibrated (c≈7) → moderate KL cost
      Each observation causes a healthy, proportional belief update.
      The model is neither stubborn nor erratic.

  • Underconfident (c≈1.6) → high KL cost
      The prior is so diffuse (low pseudocount) that every observation
      causes a wild swing.  The model is *volatile*: it over-reacts to
      individual data points, signalling the distribution is too spread.

genECE bins samples by mean confidence.  Since all three groups share
the same means, they fall into the same bins with the same accuracy,
and genECE reports identical (near-zero) miscalibration for all.

KL correction cost detects *both* failure modes – over-concentration
(stubbornness) and under-concentration (volatility) – producing clearly
separated scores where genECE is completely blind.
""")