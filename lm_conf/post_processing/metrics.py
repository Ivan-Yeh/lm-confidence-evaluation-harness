import os
from typing import Literal
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, wasserstein_distance
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
def dECE_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    confidence_dists: list[BetaDistribution] = extracted_output.extracted_confidences[0]  # list[BetaDistribution]
    confs = [bd.mu for bd in confidence_dists]
    dECE_point_mass = ece_scalar(cfg, OrganisedOutputs(
        extracted_answers=extracted_output.extracted_answers,
        extracted_confidences=[confs],
        accuracy_scores=extracted_output.accuracy_scores
    ))
    return dECE_point_mass


@register_metric(name="dECE")
def dECE(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    accuracies = []
    confidence_dists: list[BetaDistribution] = [] 

    # --- Existing sanity check and data cleaning ---
    for acc, conf in zip(extracted_output.accuracy_scores[0], extracted_output.extracted_confidences[0]):
        try:
            if acc is None or conf is None or conf.is_valid() is False:
                continue
            accuracies.append(float(acc))
            confidence_dists.append(conf)
        except:
            continue

    N = len(accuracies)
    num_bins = cfg.get("num_bins", 10)
    num_samples = cfg.get("num_wasserstein_samples", 100000)
    samples_list = [bd.sample(size=num_samples) for bd in confidence_dists]
    bins = np.linspace(0, 1, num_bins + 1)

    dECE_bins = []
    bin_total_weights = []
    bin_accuracies = []
    bin_confidences = []

    for m in range(len(bins) - 1):
        s_min, s_max = bins[m], bins[m + 1]
        weighted_W = []
        total_weight = 0.0
        weighted_correct_sum = 0.0
        weighted_conf_sum = 0.0

        for samples, y in zip(samples_list, accuracies):
            # Soft membership weight for this distribution in this bin
            w = np.mean((samples >= s_min) & (samples < s_max))
            if w == 0: continue
            
            weighted_correct_sum += w * y
            # Mean of samples that actually fell into this bin
            weighted_conf_sum += w * np.mean(samples[(samples >= s_min) & (samples < s_max)])
            total_weight += w

        # Step 2: Bin Metrics
        bin_acc = weighted_correct_sum / total_weight if total_weight > 0 else 0.0
        bin_conf = weighted_conf_sum / total_weight if total_weight > 0 else (s_min + s_max) / 2
        
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)

        # Step 3: Wasserstein calculation (from your original code)
        for samples in samples_list:
            mask = (samples >= s_min) & (samples < s_max)
            w = np.mean(mask)
            if w == 0: continue
            samples_bin = samples[mask]
            acc_samples = np.full(len(samples_bin), bin_acc)
            W = wasserstein_distance(samples_bin, acc_samples)
            weighted_W.append(w * W)

        dECE_bin = np.sum(weighted_W) / total_weight if total_weight > 0 else 0.0
        dECE_bins.append(dECE_bin)
        bin_total_weights.append(total_weight)

    # Calculate final dECE
    bin_total_weights = np.array(bin_total_weights)
    dataset_dECE = np.sum(np.array(dECE_bins) * bin_total_weights) / bin_total_weights.sum() if bin_total_weights.sum() > 0 else float("nan")

    # --- NEW: Plotting Logic with 95% Vertical Confidence Interval ---
    plot_path = cfg.get("results_path")
    if plot_path:
        plt.figure(figsize=(8, 8))
        
        # 1. Calculate the 95% CI (2.5th and 97.5th percentiles) for each bin
        lower_bounds = []
        upper_bounds = []
        
        for m in range(len(bins) - 1):
            s_min, s_max = bins[m], bins[m + 1]
            all_samples_in_bin = []
            
            for samples in samples_list:
                mask = (samples >= s_min) & (samples < s_max)
                if np.any(mask):
                    all_samples_in_bin.extend(samples[mask])
            
            if all_samples_in_bin:
                lower_bounds.append(np.percentile(all_samples_in_bin, 2.5))
                upper_bounds.append(np.percentile(all_samples_in_bin, 97.5))
            else:
                lower_bounds.append(bin_accuracies[m]) # Fallback
                upper_bounds.append(bin_accuracies[m])

        # Convert to relative errors for matplotlib: [lower_offset, upper_offset]
        y_err = [
            np.array(bin_accuracies) - np.array(lower_bounds),
            np.array(upper_bounds) - np.array(bin_accuracies)
        ]

        # 2. Perfect calibration line
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration", alpha=0.7)
        
        # 3. Plot the Bin Accuracy with the 95% Confidence Interval vertically
        plt.errorbar(
            bin_confidences, 
            bin_accuracies, 
            yerr=y_err, 
            fmt='o', 
            color='#1f77b4', 
            ecolor='#1f77b4', 
            elinewidth=2, 
            capsize=4, 
            alpha=0.8,
            label=f"Mean Accuracy (95% Confidence CI)\nTotal dECE: {dataset_dECE:.4f}"
        )

        # 4. Optional: Shaded band for visual continuity
        plt.fill_between(bin_confidences, lower_bounds, upper_bounds, color='#1f77b4', alpha=0.1)
        
        # 5. Aesthetics
        plt.xlabel("Mean Predicted Confidence")
        plt.ylabel("Accuracy")
        plt.title("dECE Reliability Diagram")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return [dataset_dECE]


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


@register_metric(name="dAUROC_point_mass")
def dAUROC_point_mass(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    logging.info("Computing dAUROC_point_mass")
    confidence_dists: list[BetaDistribution] = extracted_output.extracted_confidences[0]  # list[BetaDistribution]
    confs = [bd.mu for bd in confidence_dists]
    dauroc_point_mass = auroc_scalar(cfg, OrganisedOutputs(
        extracted_answers=extracted_output.extracted_answers,
        extracted_confidences=[confs],
        accuracy_scores=extracted_output.accuracy_scores
    ))[0]
    return [float(dauroc_point_mass)]


@register_metric(name="dAUROC")
def dAUROC(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    logging.info("Computing dAUROC")
    accuracies = []
    confidence_dists = []

    # sanity check
    for acc, conf in zip(extracted_output.accuracy_scores[0], extracted_output.extracted_confidences[0]):
        try:
            if acc is None or conf is None or conf.is_valid() is False:
                continue
            cleaned_acc = float(acc)
            accuracies.append(cleaned_acc)
            confidence_dists.append(conf)
        except:
            continue

    pos_dists = [bd for y, bd in zip(accuracies, confidence_dists) if y == 1]
    neg_dists = [bd for y, bd in zip(accuracies, confidence_dists) if y == 0]

    if not pos_dists or not neg_dists:
        return [float("nan")]

    num_dauroc_mc = cfg.get("num_dauroc_mc", 1000000)
    seed = cfg.get("seed", None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    wins = 0
    comparisons = 0
    batch_size = 100000

    for _ in tqdm(range(0, num_dauroc_mc, batch_size), desc="Computing dAUROC with global Monte Carlo sampling"):
        pos_samples = np.array([
            pos_dists[random.randrange(len(pos_dists))].sample(1)[0]
            for _ in range(batch_size)
        ])
        neg_samples = np.array([
            neg_dists[random.randrange(len(neg_dists))].sample(1)[0]
            for _ in range(batch_size)
        ])
        wins += np.sum(pos_samples > neg_samples)
        comparisons += batch_size

    return [float(wins / comparisons)]