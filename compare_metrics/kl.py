import numpy as np
from scipy.special import betaln, psi

# -----------------------------
# KL divergence: KL(post || prior)
# -----------------------------
def kl_beta(a_post, b_post, a_prior, b_prior):
    return (
        betaln(a_prior, b_prior) - betaln(a_post, b_post)
        + (a_post - a_prior) * psi(a_post)
        + (b_post - b_prior) * psi(b_post)
        + (a_prior + b_prior - a_post - b_post) * psi(a_post + b_post)
    )

# -----------------------------
# Compute KL correction cost
# -----------------------------
def kl_correction_cost(a, b, y):
    a_post = a + y
    b_post = b + (1 - y)
    return (a + b + 1e-8) * kl_beta(a_post, b_post, a, b)

# -----------------------------
# Experiment
# -----------------------------
priors = {
    "Beta(80,1)": (8, 1),
    "Beta(800,10)": (800, 10),
}

labels = [1, 0]

print("=" * 60)
print("KL Correction Cost Comparison")
print("=" * 60)

results = []

for name, (a, b) in priors.items():
    for y in labels:
        kl = kl_correction_cost(a, b, y)
        results.append((name, y, kl))

print(f"{'Prior':<15} {'Label':<6} {'KL(post || prior)':>20}")
print("-" * 60)

for name, y, kl in results:
    print(f"{name:<15} {y:<6} {kl:>20.6f}")

print("=" * 60)

# Optional: summary comparison
print("\nSummary (higher KL = larger update / more surprise):")
for name, y, kl in sorted(results, key=lambda x: x[2], reverse=True):
    print(f"{name}, y={y} -> KL={kl:.6f}")