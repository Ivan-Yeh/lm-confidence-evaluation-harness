import numpy as np
from scipy.special import gammaln

def beta_bernoulli_prob(a, b, y):
    assert y in (0, 1)
    log_p = (
        gammaln(a + b)
        - gammaln(a + b + 1)
        + (gammaln(a + y) + gammaln(b + 1 - y) - gammaln(a) - gammaln(b))
    )
    return np.exp(log_p)

def beta_variance(a, b):
    return (a * b) / ((a + b)**2 * (a + b + 1))

# def entropy(mu, eps=1e-12):
#     mu = np.clip(mu, eps, 1 - eps)
#     return -mu * np.log(mu) - (1 - mu) * np.log(1 - mu)

# def entropy_confidence(mu):
#     H = entropy(mu)
#     return 1 - H / np.log(2)  # normalize to [0,1]

def expected_brier_score(a, b, y):
    p = a / (a + b)
    return (p - y) ** 2 + beta_variance(a, b)

def hybrid_brier_nll(a, b, y):
    p = a / (a + b)
    nll = -np.log(beta_bernoulli_prob(a, b, y))
    brier = expected_brier_score(a, b, y)
    z_score = (p - y) / np.sqrt(beta_variance(a, b) + 1e-12)
    return  abs(p-y) * (1 + nll)



if __name__ == "__main__":

        test_cases = [
            (8, 2, 1),
            (8, 2, 0),
            (80, 20, 1),
            (80, 20, 0),
        ]

        for a, b, y in test_cases:
            mu = a / (a + b)
            var = beta_variance(a, b)
            brier = expected_brier_score(a, b, y)
            nll_val = -np.log(beta_bernoulli_prob(a, b, y))
            hybrid = hybrid_brier_nll(a, b, y)
            print(f"a={a}, b={b}, y={y} => mu={mu:.4f}, var={var:.6f}, expected_brier={brier:.6f}, nll={nll_val:.6f}, hybrid={hybrid:.6f}")
