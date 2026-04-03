import os
from typing import Literal
import logging
import random

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
import numpy as np
from scipy.stats import beta, wasserstein_distance
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.special import betaln, psi

from ..confidence_metrics.distributionals import BetaDistribution
from ..default_utils.custom_types import OrganisedOutputs
from ..default_utils.registry import register_metric



@register_metric(name="accuracy_scalar_with_abstention")
def accuracy_scalar_with_abstention(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    """
    Treat abstentions as incorrect and exclude "None" in accuracy scores.
    """
    all_results = []
    for accuracies in extracted_output.accuracy_scores:
        accuracies_filtered = [acc if acc is not None else None for acc in accuracies]
        accuracies_filtered = [0.0 if acc == "" else acc for acc in accuracies_filtered]
        accuracies_filtered = [acc for acc in accuracies_filtered if acc is not None]
        if len(accuracies_filtered) == 0:
            all_results.append(0.0)
        else:
            all_results.append(float(np.sum(accuracies_filtered) / len(accuracies_filtered)))
    return all_results


@register_metric(name="ece_scalar")
def ece_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    n_bins = cfg.get("ece_n_bins", 10)

    def to_number(val):
        if val is None or val == "" or isinstance(val, list):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def compute_ece(accuracies_raw, confidences_raw) -> float:
        accuracies = [to_number(acc) for acc in accuracies_raw]
        confidences = [to_number(conf) for conf in confidences_raw]

        valid_pairs = [
            (acc, conf)
            for acc, conf in zip(accuracies, confidences)
            if acc is not None and conf is not None
        ]

        if len(valid_pairs) == 0:
            return float("nan")

        accuracies_arr = np.array([pair[0] for pair in valid_pairs], dtype=float)
        confidences_arr = np.array([pair[1] for pair in valid_pairs], dtype=float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0

        for i in range(n_bins):
            lower, upper = bins[i], bins[i + 1]
            bin_mask = (confidences_arr > lower) & (confidences_arr <= upper)

            if np.any(bin_mask):
                acc_bin = np.mean(accuracies_arr[bin_mask])
                conf_bin = np.mean(confidences_arr[bin_mask])
                weight = np.sum(bin_mask) / len(confidences_arr)
                ece_val += weight * abs(acc_bin - conf_bin)

        return float(ece_val)

    eces = [
        compute_ece(accuracies_raw, confidences_raw)
        for accuracies_raw, confidences_raw in zip(
            extracted_output.accuracy_scores, extracted_output.extracted_confidences
        )
    ]

    finite_eces = [ece for ece in eces if not np.isnan(ece)]
    if len(finite_eces) == 0:
        return [float("nan")]

    return finite_eces


@register_metric(name="dECE_point_mass")
def dECE_point_mass(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    confidence_dists: list[BetaDistribution] = extracted_output.extracted_confidences[0]  # list[BetaDistribution]
    confs = [bd.mu for bd in confidence_dists]
    dECE_point_mass = ece_scalar(cfg, OrganisedOutputs(
        extracted_answers=extracted_output.extracted_answers,
        extracted_confidences=[confs],
        accuracy_scores=extracted_output.accuracy_scores
    ))
    return dECE_point_mass

def precompute_stats_worker(args):
    samples, bins = args
    B = len(bins) - 1

    bin_idx = np.digitize(samples, bins) - 1
    bin_probs = np.zeros(B)
    bin_means = np.zeros(B)
    bin_values = [None] * B

    for m in range(B):
        mask = bin_idx == m
        if np.any(mask):
            vals = samples[mask]
            bin_probs[m] = vals.size / samples.size
            bin_means[m] = vals.mean()
            bin_values[m] = vals
        else:
            bin_values[m] = np.empty(0)

    return bin_probs, bin_means, bin_values

@register_metric(name="dECE")
def dECE(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    logging.info("Computing dECE.")
    accuracies = []
    confidence_dists = []

    for acc, conf in zip(
        extracted_output.accuracy_scores[0],
        extracted_output.extracted_confidences[0],
    ):
        try:
            if acc is None or conf is None or not conf.is_valid():
                continue
            accuracies.append(float(acc))
            confidence_dists.append(conf)
        except Exception:
            continue

    if len(accuracies) == 0:
        return [float("nan")]

    accuracies = np.asarray(accuracies)
    alpha_list = [bd.alpha_param for bd in confidence_dists]
    beta_list = [bd.beta_param for bd in confidence_dists]

    num_bins = cfg.get("num_bins", 10)
    bins = np.linspace(0.0, 1.0, num_bins + 1)

    s_grid = np.linspace(1e-6, 1 - 1e-6, 2000)
    ds = s_grid[1] - s_grid[0]

    pdfs = np.array([beta.pdf(s_grid, a, b) for a, b in zip(alpha_list, beta_list)])
    N = len(alpha_list)
    M = len(bins) - 1

    w = np.zeros((N, M))
    for m in range(M):
        mask = (s_grid >= bins[m]) & (s_grid < bins[m + 1])
        w[:, m] = np.sum(pdfs[:, mask], axis=1) * ds

    bin_dECE = np.zeros(M)
    p_m = np.zeros(M)
    plot_bin_confidences = []
    plot_bin_accuracies = []
    plot_lower_bounds = []
    plot_upper_bounds = []
    plot_bin_purities = []
    plot_bin_left = []
    plot_bin_right = []

    for m in range(M):
        weights_m = w[:, m]
        total_weight_m = np.sum(weights_m)

        if total_weight_m < 1e-12:
            continue

        acc_m = np.sum(weights_m * accuracies) / total_weight_m

        conf_num = 0.0
        mask = (s_grid >= bins[m]) & (s_grid < bins[m + 1])
        for i in range(N):
            if weights_m[i] < 1e-12:
                continue
            cond_mean = np.sum(s_grid[mask] * pdfs[i, mask]) * ds / weights_m[i]
            conf_num += weights_m[i] * cond_mean
        conf_m = conf_num / total_weight_m

        p_bin_i = np.array([
            beta.cdf(bins[m + 1], alpha_list[i], beta_list[i]) -
            beta.cdf(bins[m], alpha_list[i], beta_list[i])
            for i in range(N)
        ])

        native_purity = np.sum(weights_m * p_bin_i) / total_weight_m
        impurity = 1.0 - native_purity

        raw_gap = abs(conf_m - acc_m)
        impurity_bonus = raw_gap * impurity
        bin_dECE[m] = raw_gap + impurity_bonus * (1.0 - raw_gap)

        p_m[m] = total_weight_m / N

        ws = weights_m / total_weight_m
        n_obs = len(accuracies)
        if n_obs > 0:
            n_bootstrap = cfg.get("n_bootstrap_samples", 2000)
            bootstrap_estimates = []
            for _ in range(n_bootstrap):
                idx = np.random.randint(0, n_obs, size=n_obs)
                ws_b = ws[idx]
                ys_b = accuracies[idx]
                ws_b = ws_b / ws_b.sum()
                bootstrap_estimates.append(np.sum(ws_b * ys_b))
            lower, upper = np.percentile(bootstrap_estimates, [5, 95])
        else:
            lower = acc_m
            upper = acc_m

        plot_bin_confidences.append(conf_m)
        plot_bin_accuracies.append(acc_m)
        plot_lower_bounds.append(lower)
        plot_upper_bounds.append(upper)
        plot_bin_purities.append(native_purity)
        plot_bin_left.append(bins[m])
        plot_bin_right.append(bins[m + 1])

    dataset_dECE = np.sum(p_m * bin_dECE)

    plot_path = cfg.get("results_path")
    if plot_path:
        os.makedirs(plot_path, exist_ok=True)

        plot_bin_confidences = np.array(plot_bin_confidences)
        plot_bin_accuracies = np.array(plot_bin_accuracies)
        plot_lower_bounds = np.array(plot_lower_bounds)
        plot_upper_bounds = np.array(plot_upper_bounds)
        plot_bin_purities = np.array(plot_bin_purities)

        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="Perfect Calibration")

        if plot_bin_confidences.size > 1:
            order = np.argsort(plot_bin_confidences)
            x_sorted = plot_bin_confidences[order]
            lower_sorted = plot_lower_bounds[order]
            upper_sorted = plot_upper_bounds[order]
            purity_sorted = plot_bin_purities[order]

            dense_x = np.linspace(0.0, 1.0, 200)
            dense_lower = np.interp(dense_x, x_sorted, lower_sorted, left=lower_sorted[0], right=lower_sorted[-1])
            dense_upper = np.interp(dense_x, x_sorted, upper_sorted, left=upper_sorted[0], right=upper_sorted[-1])
            dense_purity = np.interp(dense_x, x_sorted, purity_sorted, left=purity_sorted[0], right=purity_sorted[-1])

            verts = []
            colors = []
            for i in range(len(dense_x) - 1):
                alpha = 0.1 + 0.6 * float(
                    np.clip(0.5 * (dense_purity[i] + dense_purity[i + 1]), 0.0, 1.0)
                )
                verts.append([
                    (dense_x[i], dense_lower[i]),
                    (dense_x[i], dense_upper[i]),
                    (dense_x[i + 1], dense_upper[i + 1]),
                    (dense_x[i + 1], dense_lower[i + 1]),
                ])
                colors.append(to_rgba("#2ca02c", alpha=alpha))

            band = PolyCollection(verts, facecolors=colors, edgecolors="none")
            plt.gca().add_collection(band)

            plt.plot(
                plot_bin_confidences,
                plot_bin_accuracies,
                "o-",
                color="black",
                markersize=4,
                label="Bin Accuracy",
            )

        plt.xlabel("Mean Predicted Confidence")
        plt.ylabel("Accuracy")
        # plt.title("dECE Reliability Diagram")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()

        name = "dECE_reliability_diagram_0"
        if os.path.exists(plot_path + f"/{name}.pdf"):
            idx = 1
            while os.path.exists(plot_path + f"/dECE_reliability_diagram_{idx}.pdf"):
                idx += 1
            name = f"dECE_reliability_diagram_{idx}"
        plt.savefig(plot_path + f"/{name}.pdf", bbox_inches="tight", dpi=300)
        plt.close()

    return [float(dataset_dECE)]


@register_metric(name="auroc_scalar")
def auroc_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    """
    Computes AUROC (Area Under the Receiver Operating Characteristic curve)
    from the extracted accuracies and confidence scores.
    """
    # Ensure all values are numbers, exclude None and lists
    def to_number(val):
        if val is None or val == "" or isinstance(val, list):
            return None
        try:
            num = float(val)
            if np.isnan(num):
                return None
            return num
        except (ValueError, TypeError):
            return None

    def compute_auroc(accuracies_raw, confidences_raw) -> float:
        accuracies = [to_number(acc) for acc in accuracies_raw]
        confidences = [to_number(conf) for conf in confidences_raw]

        valid_pairs = [
            (acc, conf)
            for acc, conf in zip(accuracies, confidences)
            if acc is not None and conf is not None
        ]

        if len(valid_pairs) == 0:
            return float("nan")

        accuracies_arr = np.array([pair[0] for pair in valid_pairs], dtype=float)
        confidences_arr = np.array([pair[1] for pair in valid_pairs], dtype=float)

        if len(confidences_arr) == 0 or len(np.unique(accuracies_arr)) < 2:
            return float("nan")

        return float(roc_auc_score(accuracies_arr, confidences_arr))

    aurocs = [
        compute_auroc(accuracies_raw, confidences_raw)
        for accuracies_raw, confidences_raw in zip(
            extracted_output.accuracy_scores, extracted_output.extracted_confidences
        )
    ]

    finite_aurocs = [auc for auc in aurocs if not np.isnan(auc)]
    if len(finite_aurocs) == 0:
        return [float("nan")]

    return finite_aurocs


@register_metric(name="AUROC_point_mass")
def AUROC_point_mass(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    logging.info("Computing AUROC_point_mass")
    confidence_dists: list[BetaDistribution] = extracted_output.extracted_confidences[0]  # list[BetaDistribution]
    confs = [bd.mu for bd in confidence_dists]
    dauroc_point_mass = auroc_scalar(cfg, OrganisedOutputs(
        extracted_answers=extracted_output.extracted_answers,
        extracted_confidences=[confs],
        accuracy_scores=extracted_output.accuracy_scores
    ))[0]
    return [float(dauroc_point_mass)]


def compute_batch_wins(args):
    batch_idx, pos_dists_batch, neg_dists_batch, batch_size_batch = args
    batch_wins = 0
    pos_samples = np.array([
        pos_dists_batch[random.randrange(len(pos_dists_batch))].sample(1)[0]
        for _ in range(batch_size_batch)
    ])
    neg_samples = np.array([
        neg_dists_batch[random.randrange(len(neg_dists_batch))].sample(1)[0]
        for _ in range(batch_size_batch)
    ])
    batch_wins = np.sum(pos_samples > neg_samples)
    return batch_wins


def compute_wins(pos_vals_flat, neg_vals_flat):
    """
    Compute total wins for AUROC given flattened sampled values.
    Vectorized using broadcasting.
    """
    # Shape (n_pos, 1) - (1, n_neg) broadcasting
    diff = pos_vals_flat[:, None] - neg_vals_flat[None, :]
    wins = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return wins

def compute_wins_wrapper(args):
    pos_vals_flat, neg_vals_flat = args
    return compute_wins(pos_vals_flat, neg_vals_flat)

@register_metric(name="dAUROC")
def dAUROC(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    logging.info("Computing dAUROC")
    accuracies = []
    confidence_dists = []
    
    # Collect valid pairs
    for acc, conf in zip(extracted_output.accuracy_scores[0], extracted_output.extracted_confidences[0]):
        try:
            if acc is None or conf is None or conf.is_valid() is False:
                continue
            cleaned_acc = float(acc)
            accuracies.append(cleaned_acc)
            confidence_dists.append(conf)
        except:
            continue
    
    # Get indices of positive and negative examples
    pos_idxs = [i for i, y in enumerate(accuracies) if y == 1]
    neg_idxs = [i for i, y in enumerate(accuracies) if y == 0 or y == ""]
    
    if not pos_idxs or not neg_idxs:
        return [float("nan")]
    
    num_iterations = 10_000
    batch_size = 10_000
    total_wins = 0.0
    
    for _ in tqdm(range(num_iterations // batch_size), desc="Computing dAUROC using Monte Carlo"):
        # Sample INDICES of predictions (not distributions)
        p_idxs_batch = np.random.choice(pos_idxs, size=batch_size, replace=True)
        n_idxs_batch = np.random.choice(neg_idxs, size=batch_size, replace=True)
        
        # Draw one sample from each selected prediction's distribution
        s_p = np.array([confidence_dists[i].sample() for i in p_idxs_batch]).flatten()
        s_n = np.array([confidence_dists[i].sample() for i in n_idxs_batch]).flatten()
        
        # Vectorized comparison
        total_wins += np.sum(s_p > s_n) + 0.5 * np.sum(s_p == s_n)
    
    return [float(total_wins / num_iterations)]



# https://arxiv.org/pdf/2410.04315
@register_metric(name="generalised_ece")
def generalised_ece(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    logging.info("Computing generalised ECE")
    # 1. Data Cleaning
    accuracies = []
    confidence_dists = []
    
    # Collect valid pairs
    for acc, conf in zip(extracted_output.accuracy_scores[0], extracted_output.extracted_confidences[0]):
        try:
            if acc is None or conf is None or conf.is_valid() is False:
                continue
            if acc == "":
                acc = 0
            cleaned_acc = float(acc)
            accuracies.append(cleaned_acc)
            confidence_dists.append(conf)
        except:
            continue

    y = np.array(accuracies)
    N = len(y)
    if N == 0:
        return [float("nan")]
    
    num_bins = cfg.get("num_bins", 10)
    num_samples = cfg.get("num_samples", 1000) # Balanced for speed/accuracy
    bins = np.linspace(0.0, 1.0, num_bins + 1)

    # 2. Vectorized Sampling
    # Shape: (N, num_samples)
    all_samples = np.array([dist.sample(size=num_samples) for dist in confidence_dists])

    # 3. Compute Probabilities and Expectations (Vectorized across N and Samples)
    # We use broadcasting to find which bin every sample falls into
    # bin_indices shape: (N, num_samples) -> values from 0 to num_bins-1
    bin_indices = np.digitize(all_samples, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    rm = np.zeros(num_bins)
    pm = np.zeros(num_bins)
    gm = np.zeros(num_bins)
    plot_bin_confidences = []
    plot_bin_accuracies = []
    plot_lower_bounds = []
    plot_upper_bounds = []

    for m in range(num_bins):
        # Create a boolean mask for samples in bin m
        mask = (bin_indices == m)
        
        # P(S in Im | X = xn) for each n (Shape: N,)
        p_nm = np.mean(mask, axis=1)
        
        # Calculate pm: Sum over all n
        pm[m] = np.sum(p_nm)

        if pm[m] > 0:
            # Calculate rm: Weighted average of labels y
            rm[m] = np.sum(p_nm * y) / pm[m]
            
            # Calculate gm: Expected value of samples within the bin
            # We sum only the samples that fell in the bin and divide by total samples in bin
            bin_samples_mask = all_samples[mask]
            gm[m] = np.mean(bin_samples_mask) if bin_samples_mask.size > 0 else 0.0

            ws = p_nm / pm[m]
            n_obs = len(y)
            if n_obs > 0:
                n_bootstrap = cfg.get("n_bootstrap_samples", 2000)
                bootstrap_estimates = []
                for _ in range(n_bootstrap):
                    idx = np.random.randint(0, n_obs, size=n_obs)
                    ws_b = ws[idx]
                    ys_b = y[idx]
                    ws_sum = ws_b.sum()
                    if ws_sum > 0:
                        ws_b = ws_b / ws_sum
                        bootstrap_estimates.append(np.sum(ws_b * ys_b))
                if len(bootstrap_estimates) > 0:
                    lower, upper = np.percentile(bootstrap_estimates, [5, 95])
                else:
                    lower = rm[m]
                    upper = rm[m]
            else:
                lower = rm[m]
                upper = rm[m]

            plot_bin_confidences.append(gm[m])
            plot_bin_accuracies.append(rm[m])
            plot_lower_bounds.append(lower)
            plot_upper_bounds.append(upper)

    # 4. Final gECE
    # gECE = \sum (pm / N) * |rm - gm|
    gece_value = np.sum((pm / N) * np.abs(rm - gm))

    plot_path = cfg.get("results_path")
    if plot_path and len(plot_bin_confidences) > 0:
        os.makedirs(plot_path, exist_ok=True)

        plot_bin_confidences = np.array(plot_bin_confidences)
        plot_bin_accuracies = np.array(plot_bin_accuracies)
        plot_lower_bounds = np.array(plot_lower_bounds)
        plot_upper_bounds = np.array(plot_upper_bounds)

        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="Perfect Calibration")

        if plot_bin_confidences.size > 1:
            order = np.argsort(plot_bin_confidences)
            x_sorted = plot_bin_confidences[order]
            lower_sorted = plot_lower_bounds[order]
            upper_sorted = plot_upper_bounds[order]

            dense_x = np.linspace(0.0, 1.0, 200)
            dense_lower = np.interp(dense_x, x_sorted, lower_sorted, left=lower_sorted[0], right=lower_sorted[-1])
            dense_upper = np.interp(dense_x, x_sorted, upper_sorted, left=upper_sorted[0], right=upper_sorted[-1])

            plt.fill_between(
                dense_x,
                dense_lower,
                dense_upper,
                color="#2ca02c",
                alpha=0.25,
                linewidth=0,
                label="90% CI",
            )

            plt.plot(
                plot_bin_confidences,
                plot_bin_accuracies,
                "o-",
                color="black",
                markersize=4,
                label="Bin Accuracy",
            )

        plt.xlabel("Mean Predicted Confidence")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()

        name = "gECE_reliability_diagram_0"
        if os.path.exists(plot_path + f"/{name}.pdf"):
            idx = 1
            while os.path.exists(plot_path + f"/gECE_reliability_diagram_{idx}.pdf"):
                idx += 1
            name = f"gECE_reliability_diagram_{idx}"
        plt.savefig(plot_path + f"/{name}.pdf", bbox_inches="tight", dpi=300)
        plt.close()

    return [float(gece_value)]



@register_metric(name="faithfulness_divergence")
def faithfulness_divergence(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    logging.info("Computing faithfulness divergence")
    def kl_beta(a_post, b_post, a_prior, b_prior):
        return (
            betaln(a_prior, b_prior) - betaln(a_post, b_post)
            + (a_post - a_prior) * psi(a_post)
            + (b_post - b_prior) * psi(b_post)
            + (a_prior + b_prior - a_post - b_post) * psi(a_post + b_post)
        )
    def faithfulness_divergence_single(dist: BetaDistribution, y):
        # Placeholder: Implement the actual divergence calculation based on the paper
        a = dist.alpha_param
        b = dist.beta_param
        a_post = a + y
        b_post = b + (1 - y)
        return (a + b + 1e-8) * kl_beta(a_post, b_post, a, b)
    
    accuracies = []
    confidence_dists = []
    
    # Collect valid pairs
    for acc, conf in zip(extracted_output.accuracy_scores[0], extracted_output.extracted_confidences[0]):
        try:
            if acc is None or conf is None or conf.is_valid() is False:
                continue
            if acc == "":
                acc = 0
            cleaned_acc = float(acc)
            accuracies.append(cleaned_acc)
            confidence_dists.append(conf)
        except:
            continue

    divergences = []
    for dist, acc in zip(confidence_dists, accuracies):
        divergence = faithfulness_divergence_single(dist, acc)
        if not np.isnan(divergence):
            divergences.append(divergence)

    return [float(np.mean(divergences))]