import numpy as np
from scipy.stats import beta

# ---------- Grid ----------
def build_grid(num_points=2000, eps=1e-6):
    s = np.linspace(eps, 1 - eps, num_points)
    ds = s[1] - s[0]
    return s, ds


# ---------- Soft binning ----------
def compute_soft_memberships(alpha_list, beta_list, bins, s_grid, ds):
    N = len(alpha_list)
    M = len(bins) - 1

    pdfs = np.array([beta.pdf(s_grid, a, b) for a, b in zip(alpha_list, beta_list)])

    w = np.zeros((N, M))

    for m in range(M):
        mask = (s_grid >= bins[m]) & (s_grid < bins[m+1])
        w[:, m] = np.sum(pdfs[:, mask], axis=1) * ds

    return w, pdfs


# ---------- genECE ----------
def compute_genECE(alpha_list, beta_list, correctness, num_bins=10):
    s_grid, ds = build_grid()
    bins = np.linspace(0, 1, num_bins + 1)

    w, pdfs = compute_soft_memberships(alpha_list, beta_list, bins, s_grid, ds)
    N, M = w.shape
    correctness = np.array(correctness)

    p_m = np.zeros(M)
    bin_ece = np.zeros(M)
    bin_stats = []

    for m in range(M):
        weights = w[:, m]
        total_weight = np.sum(weights)
        if total_weight < 1e-12:
            bin_stats.append({
                "bin_index": m,
                "bin_start": bins[m],
                "bin_end": bins[m + 1],
                "total_weight": 0.0,
                "p_m": 0.0,
                "accuracy": np.nan,
                "mean_confidence": np.nan,
                "miscalibration": 0.0,
            })
            continue

        # accuracy
        r_m = np.sum(weights * correctness) / total_weight

        # mean confidence
        num = 0.0
        for i in range(N):
            if weights[i] < 1e-12:
                continue
            mask = (s_grid >= bins[m]) & (s_grid < bins[m+1])
            cond_mean = np.sum(s_grid[mask] * pdfs[i, mask]) * ds / weights[i]
            num += weights[i] * cond_mean

        g_m = num / total_weight
        p_m[m] = total_weight / N
        bin_ece[m] = abs(g_m - r_m)

        bin_stats.append({
            "bin_index": m,
            "bin_start": bins[m],
            "bin_end": bins[m + 1],
            "total_weight": total_weight,
            "p_m": p_m[m],
            "accuracy": r_m,
            "mean_confidence": g_m,
            "miscalibration": bin_ece[m],
        })
    total_gen_ece = np.sum(p_m * bin_ece)
    return bin_ece, total_gen_ece, bin_stats


def compute_dECE(alpha_list, beta_list, correctness, num_bins=10):
    import numpy as np

    s_grid, ds  = build_grid()
    bins        = np.linspace(0, 1, num_bins + 1)
    w, pdfs     = compute_soft_memberships(alpha_list, beta_list, bins, s_grid, ds)

    alpha_arr   = np.array(alpha_list)
    beta_arr    = np.array(beta_list)
    correctness = np.array(correctness)
    mu          = alpha_arr / (alpha_arr + beta_arr)

    N, M = w.shape

    bin_dECE  = np.zeros(M)
    p_m       = np.zeros(M)
    bin_stats = []

    for m in range(M):
        weights_m      = w[:, m]
        total_weight_m = np.sum(weights_m)

        if total_weight_m < 1e-12:
            bin_stats.append({
                "bin_index":        m,
                "bin_start":        bins[m],
                "bin_end":          bins[m + 1],
                "total_weight":     0.0,
                "p_m":              0.0,
                "accuracy":         np.nan,
                "conf":             np.nan,
                "raw_gap":          np.nan,
                "impurity_bonus":   np.nan,
                "miscalibration":   0.0,
                "native_purity":    np.nan,
            })
            continue

        # accuracy
        acc_m = np.sum(weights_m * correctness) / total_weight_m

        # conditional mean confidence
        conf_num = 0.0
        mask     = (s_grid >= bins[m]) & (s_grid < bins[m + 1])
        for i in range(N):
            if weights_m[i] < 1e-12:
                continue
            cond_mean  = np.sum(s_grid[mask] * pdfs[i, mask]) * ds / weights_m[i]
            conf_num  += weights_m[i] * cond_mean
        conf_m = conf_num / total_weight_m

        # native purity
        from scipy.stats import beta as beta_dist

        # compute probability mass of each distribution inside the bin
        p_bin_i = np.array([
            beta_dist.cdf(bins[m + 1], alpha_arr[i], beta_arr[i]) -
            beta_dist.cdf(bins[m],     alpha_arr[i], beta_arr[i])
            for i in range(N)
        ])

        # mass-weighted purity
        native_purity = np.sum(weights_m * p_bin_i) / total_weight_m
        impurity      = 1.0 - native_purity                         # in [0, 1]

        raw_gap = abs(conf_m - acc_m)                               # in [0, 1], = genECE bin term

        # impurity_bonus: extra penalty for wide distributions, in [0, 1]
        # = 0 for sharp distributions (recovers genECE exactly)
        # = raw_gap for maximally impure bin (doubles the penalty)
        # additive, same units as raw_gap, no free parameter
        impurity_bonus = raw_gap * impurity                         # in [0, 1]

        # dECE = genECE term + impurity bonus
        # range: [0, 1] since raw_gap ∈ [0,1], impurity ∈ [0,1], and raw_gap * (1 + impurity) ≤ 2*raw_gap ≤ 2
        # BUT: when impurity=0, dECE = raw_gap exactly — sharp recovery holds
        # normalise by 2 only the bonus term so total stays in [0, 1]:
        #   raw_gap + impurity_bonus/1 can hit 2, so normalise whole expression by (1 + impurity)
        # Cleanest: express as weighted average between raw_gap and its doubled value
        bin_dECE[m] = raw_gap + impurity_bonus * (1.0 - raw_gap)   # in [0, 1] always

        p_m[m] = total_weight_m / N

        bin_stats.append({
            "bin_index":      m,
            "bin_start":      bins[m],
            "bin_end":        bins[m + 1],
            "total_weight":   total_weight_m,
            "p_m":            p_m[m],
            "accuracy":       acc_m,
            "conf":           conf_m,
            "raw_gap":        raw_gap,
            "impurity_bonus": impurity_bonus,
            "miscalibration": bin_dECE[m],
            "native_purity":  native_purity,
        })

    dataset_dECE = np.sum(p_m * bin_dECE)
    return bin_dECE, dataset_dECE, bin_stats


def compute_ECE_mean(alpha_list, beta_list, correctness, num_bins=10):
    alpha_arr = np.array(alpha_list)
    beta_arr = np.array(beta_list)
    correctness = np.array(correctness)

    mean_conf = alpha_arr / (alpha_arr + beta_arr)
    bins = np.linspace(0, 1, num_bins + 1)

    ece = 0.0
    bin_stats = []
    N = len(mean_conf)

    for m in range(num_bins):
        in_bin = (mean_conf >= bins[m]) & (mean_conf < bins[m + 1])
        count = int(np.sum(in_bin))
        if count == 0:
            bin_stats.append({
                "bin_index": m,
                "bin_start": bins[m],
                "bin_end": bins[m + 1],
                "count": 0,
                "p_m": 0.0,
                "accuracy": np.nan,
                "mean_confidence": np.nan,
                "miscalibration": 0.0,
            })
            continue

        acc = float(np.mean(correctness[in_bin]))
        conf = float(np.mean(mean_conf[in_bin]))
        p_m = count / N
        gap = abs(conf - acc)
        ece += p_m * gap

        bin_stats.append({
            "bin_index": m,
            "bin_start": bins[m],
            "bin_end": bins[m + 1],
            "count": count,
            "p_m": p_m,
            "accuracy": acc,
            "mean_confidence": conf,
            "miscalibration": gap,
        })

    return ece, bin_stats


def print_bin_stats(name, stats, include_confidence=False):
    print(f"{name} per-bin stats:")
    if include_confidence:
        print("bin\trange\tweight\tp_m\tacc\tconf\t|gap|")
    else:
        print("bin\trange\tweight\tp_m\tacc\tdECE_m")

    for row in stats:
        acc = "nan" if np.isnan(row["accuracy"]) else f"{row['accuracy']:.4f}"
        base = (
            f"{row['bin_index']}\t"
            f"[{row['bin_start']:.2f}, {row['bin_end']:.2f})\t"
            f"{row['total_weight']:.4f}\t"
            f"{row['p_m']:.4f}\t"
            f"{acc}"
        )

        if include_confidence:
            conf = "nan" if np.isnan(row["mean_confidence"]) else f"{row['mean_confidence']:.4f}"
            print(f"{base}\t{conf}\t{row['miscalibration']:.4f}")
        else:
            print(f"{base}\t{row['miscalibration']:.4f}")


# ---------- 🔥 Experiment ----------
def generate_miscalibrated_pairs(N=50):
    wide_alpha, wide_beta = [], []
    narrow_alpha, narrow_beta = [], []
    correctness = []

    for _ in range(N):
        # fixed mean (high confidence region)
        mu = np.random.uniform(0.3, 0.6)

        y = np.random.choice([0, 1], p=[0.5, 0.5])

        # narrow: very confident
        k_narrow = 1000
        a_n = mu * k_narrow
        b_n = (1 - mu) * k_narrow

        # wide: uncertain
        k_wide = 10
        a_w = mu * k_wide
        b_w = (1 - mu) * k_wide

        narrow_alpha.append(a_n)
        narrow_beta.append(b_n)

        wide_alpha.append(a_w)
        wide_beta.append(b_w)

        correctness.append(y)

    return (wide_alpha, wide_beta), (narrow_alpha, narrow_beta), correctness


# ---------- Run ----------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N = 500
    num_bins = 15

    # Shared means, different variances via concentration k.
    means = rng.uniform(0.3, 0.6, size=N)
    correctness = rng.integers(0, 2, size=N)

    k_low_var = 1200.0
    k_high_var = 20.0

    low_alpha = means * k_low_var
    low_beta = (1 - means) * k_low_var

    high_alpha = means * k_high_var
    high_beta = (1 - means) * k_high_var

    ece_low, _ = compute_ECE_mean(low_alpha, low_beta, correctness, num_bins=num_bins)
    ece_high, _ = compute_ECE_mean(high_alpha, high_beta, correctness, num_bins=num_bins)

    _, gen_low, _ = compute_genECE(low_alpha, low_beta, correctness, num_bins=num_bins)
    _, gen_high, _ = compute_genECE(high_alpha, high_beta, correctness, num_bins=num_bins)

    _, d_low, _ = compute_dECE(low_alpha, low_beta, correctness, num_bins=num_bins)
    _, d_high, _ = compute_dECE(high_alpha, high_beta, correctness, num_bins=num_bins)

    print("=== Low Variance Subset ===")
    print("ECE_mean:", ece_low)
    print("genECE:", gen_low)
    print("dECE:", d_low)

    print("\n=== High Variance Subset ===")
    print("ECE_mean:", ece_high)
    print("genECE:", gen_high)
    print("dECE:", d_high)