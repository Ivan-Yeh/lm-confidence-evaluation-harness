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
def dECE_point_mass(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
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

    # --- NEW: Plotting Logic with Bootstrap Confidence Interval and Confidence Violins ---
    plot_path = cfg.get("results_path")
    if plot_path:
        plt.figure(figsize=(6, 6))
        
        # Use serif fonts for all text
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman']

        n_bootstrap = cfg.get("n_bootstrap_samples", 5000)
        max_violin_samples = cfg.get("max_violin_samples", 5000)

        plot_bin_confidences = []
        plot_bin_accuracies = []
        plot_lower_bounds = []
        plot_upper_bounds = []
        bin_conf_samples = []

        # Step 1: collect per-bin accuracy and confidence samples
        for m in range(len(bins) - 1):
            s_min, s_max = bins[m], bins[m + 1]

            ys = []
            ws = []
            conf_samples_bin = []

            for samples, y in zip(samples_list, accuracies):
                mask = (samples >= s_min) & (samples < s_max)
                w = np.mean(mask)
                if w > 0:
                    ys.append(y)
                    ws.append(w)
                    conf_samples_bin.append(samples[mask])

            # Combine confidence samples for horizontal violin
            if len(conf_samples_bin) > 0:
                conf_samples_bin = np.concatenate(conf_samples_bin)
                # Subsample for plotting efficiency
                if len(conf_samples_bin) > max_violin_samples:
                    conf_samples_bin = np.random.choice(conf_samples_bin, size=max_violin_samples, replace=False)
                bin_conf_samples.append(conf_samples_bin)
            else:
                bin_conf_samples.append(None)

            # Skip empty bins
            if len(ys) == 0:
                continue

            ys = np.asarray(ys)
            ws = np.asarray(ws)
            ws = ws / ws.sum()  # normalize soft weights

            # Soft-bin accuracy
            r_hat = bin_accuracies[m]

            # Bootstrap CI for accuracy
            bootstrap_estimates = []
            n_obs = len(ys)
            for _ in range(n_bootstrap):
                idx = np.random.randint(0, n_obs, size=n_obs)
                ws_b = ws[idx]
                ys_b = ys[idx]
                ws_b = ws_b / ws_b.sum()
                bootstrap_estimates.append(np.sum(ws_b * ys_b))

            lower, upper = np.percentile(bootstrap_estimates, [2.5, 97.5])

            plot_bin_confidences.append(bin_confidences[m])
            plot_bin_accuracies.append(r_hat)
            plot_lower_bounds.append(lower)
            plot_upper_bounds.append(upper)

        # Step 2: Convert to arrays for plotting
        plot_bin_confidences = np.array(plot_bin_confidences)
        plot_bin_accuracies = np.array(plot_bin_accuracies)
        plot_lower_bounds = np.array(plot_lower_bounds)
        plot_upper_bounds = np.array(plot_upper_bounds)
        y_err = np.vstack([
            np.maximum(0, plot_bin_accuracies - plot_lower_bounds),
            np.maximum(0, plot_upper_bounds - plot_bin_accuracies),
        ])

        # Step 3: Horizontal violins for confidence distributions
        violin_label_added = False
        for m, conf_samples in enumerate(bin_conf_samples):
            if conf_samples is None or m >= len(plot_bin_accuracies):
                continue

            y_center = plot_bin_accuracies[m]

            parts = plt.violinplot(
                conf_samples,
                positions=[y_center],
                vert=False,
                widths=0.03,          # controls vertical thickness of the violin
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for pc in parts["bodies"]:
                pc.set_facecolor("#10d1a1")
                pc.set_alpha(0.15)
                # Add legend label only once
                if not violin_label_added:
                    pc.set_label("Confidence Distribution")
                    violin_label_added = True

        # Step 4: Perfect calibration line
        plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="Perfect Calibration")

        # Step 5: Accuracy points with vertical CI
        plt.errorbar(
            plot_bin_confidences,
            plot_bin_accuracies,
            yerr=y_err,
            fmt="o",
            capsize=4,
            elinewidth=2,
            alpha=0.85,
            label=f"Bin Accuracy (95% CI)\nTotal dECE: {dataset_dECE:.4f}",
        )

        # Optional shaded vertical band for visual continuity
        plt.fill_between(
            plot_bin_confidences,
            plot_lower_bounds,
            plot_upper_bounds,
            alpha=0.15,
            color="#1f77b4"
        )

        # Step 6: Plot aesthetics
        plt.xlabel("Mean Predicted Confidence")
        plt.ylabel("Accuracy")
        plt.title("dECE Reliability Diagram with Confidence Uncertainty")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()

        # Step 7: Save figure
        name = "dECE_reliability_diagram_0"
        if os.path.exists(plot_path + f"/{name}.pdf"):
            idx = 1
            while os.path.exists(plot_path + f"/dECE_reliability_diagram_{idx}.pdf"):
                idx += 1
            name = f"dECE_reliability_diagram_{idx}"
        plt.savefig(plot_path + f"/{name}.pdf", bbox_inches="tight", dpi=300)
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