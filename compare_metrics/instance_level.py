import numpy as np
from scipy.stats import beta
from scipy.special import gammaln

def beta_bernoulli_prob(a, b, y):
    assert y in (0, 1)
    log_p = gammaln(a + b) - gammaln(a + b + 1) + \
            (gammaln(a + y) + gammaln(b + 1 - y) - gammaln(a) - gammaln(b))
    return np.exp(log_p)

def compute_desired_behavior_score(a, b, y,
                                   lambda_Q=0.1, lambda_R=0.05,
                                   max_variance=0.1,
                                   sharpness_threshold=0.05):
    mu = a / (a + b)
    v_raw = (a * b) / ((a + b)**2 * (a + b + 1))
    v = min(v_raw, max_variance)
    rho = 1.0 / v

    # 1. NLL
    p_y = beta_bernoulli_prob(a, b, y)
    nll = -np.log(p_y)

    # 2. Wrong‑penalty term (high when low‑var + wrong)
    miscalib2 = (mu - y)**2
    P_wrong = rho * miscalib2

    # 3. Right‑reward term (only if variance < sharpness_threshold, and positive only as a negative reward)
    if v < sharpness_threshold:
        R = rho * miscalib2
    else:
        R = 0.0

    # 4. Composite Q_i
    Q_i = nll + lambda_Q * P_wrong - lambda_R * R

    # 5. Also report components for debugging
    return {
        'nll': nll,
        'P_wrong': P_wrong,
        'R': R,
        'Q_i': Q_i,
        'mu': mu,
        'v': v,
        'rho': rho
    }

# Example: Beta(2,8), Beta(20,80), y=0 and y=1
betas = [(2, 8), (20, 80)]
labels = [0, 1]

print("Case                                  NLL     P_wrong     R     Q_i")
print("-" * 75)

for a, b in betas:
    for y in labels:
        res = compute_desired_behavior_score(a, b, y,
                                             lambda_Q=0.1,
                                             lambda_R=0.1,
                                             max_variance=0.1,
                                             sharpness_threshold=0.05)

        label_str = "right" if y == 1 else "wrong"
        var_size = "low" if res['v'] < 0.05 else "high"
        case = f" Beta({a},{b}) y={y} ({label_str}, {var_size} var)"

        print(f"{case:<50} {res['nll']:7.3f}  {res['P_wrong']:8.4f}  {res['R']:8.4f}  {res['Q_i']:8.4f}")