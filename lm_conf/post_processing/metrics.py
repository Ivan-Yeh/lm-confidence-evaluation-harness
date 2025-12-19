from lm_conf.confidence_metrics.distributionals import BetaDistribution
from ..default_utils.custom_types import OrganisedOutputs
from ..default_utils.registry import register_metric
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance, beta
import matplotlib.pyplot as plt
from typing import Literal


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
        return float("nan")

    return (finite_eces)


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
        return float("nan")

    return finite_aurocs



@register_metric(name="dECE_scalar")
def dECE_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    confidence_dists: list[BetaDistribution] = extracted_output.extracted_confidences[0]  # list[BetaDistribution]
    confs = [bd.mu for bd in confidence_dists]
    dece_scalar = ece_scalar(cfg, OrganisedOutputs(
        extracted_answers=extracted_output.extracted_answers,
        extracted_confidences=[confs],
        accuracy_scores=extracted_output.accuracy_scores
    ))[0]
    return [float(dece_scalar)]

def dECE(cfg: dict, extracted_output: OrganisedOutputs, binning_mode: Literal["equal_width", "equal_mass"] = "equal_width") -> list[float]:
    accuracies = extracted_output.accuracy_scores[0]            # list[int] in {0,1}
    confidence_dists: list[BetaDistribution] = extracted_output.extracted_confidences[0]  # list[BetaDistribution]

    N = len(accuracies)
    assert N == len(confidence_dists)

    num_bins = cfg.get("num_bins", 10)
    num_samples = cfg.get("num_wasserstein_samples", 1000)

    # bin by expected confidence (mu)
    mean_conf = np.array([bd.mu for bd in confidence_dists])
    
    if binning_mode == "equal_width":
        # equal-width bins on [0, 1]
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        # equal-mass (quantile) bins
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        bin_edges = np.quantile(mean_conf, quantiles)

    dECE_value = 0.0

    # 2. per-bin Wasserstein distance
    for m in range(num_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]

        # include right edge for last bin
        if m == num_bins - 1:
            idx = np.where((mean_conf >= lo) & (mean_conf <= hi))[0]
        else:
            idx = np.where((mean_conf >= lo) & (mean_conf < hi))[0]

        if len(idx) == 0:
            continue

        conf_samples = []
        for i in idx:
            bd: BetaDistribution = confidence_dists[i]
            conf_samples.append(
                beta.rvs(bd.alpha_param, bd.beta_param, size=num_samples)
            )
        conf_samples = np.concatenate(conf_samples)

        k = sum(accuracies[i] for i in idx)
        n = len(idx)
        alpha_y = k + 1
        beta_y = (n - k) + 1
        acc_samples = beta.rvs(alpha_y, beta_y, size=len(conf_samples))

        w1 = wasserstein_distance(conf_samples, acc_samples)

        dECE_value += (len(idx) / N) * w1

        # Collect data for optional plotting
        if 'dece_plot_bins' not in locals():
            dece_plot_bins = []
        dece_plot_bins.append({
            "bin": m,
            "lo": lo,
            "hi": hi,
            "n": n,
            "conf": conf_samples,
            "acc": acc_samples,
        })

    # Create a single figure with one subplot per non-empty bin
    if 'dece_plot_bins' in locals() and len(dece_plot_bins) > 0:
        num_plots = len(dece_plot_bins)
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        x_range = (0.0, 1.0)
        for ax, data in zip(axes, dece_plot_bins):
            ax.hist(data["conf"], bins=30, range=x_range, alpha=0.6, density=True, label="Confidence")
            ax.hist(data["acc"], bins=30, range=x_range, alpha=0.6, density=True, label="Accuracy")
            ax.set_title(f"Bin {data['bin']} [{data['lo']:.2f},{data['hi']:.2f}] n={data['n']}")
            ax.set_xlim(*x_range)
        # Hide any unused subplots
        for ax in axes[num_plots:]:
            ax.axis('off')
        axes[0].legend()
        fig.suptitle("dECE: Confidence vs Accuracy distributions per bin")
        fig.tight_layout()
        results_path = cfg.get("results_path") + f"/dECE_{binning_mode}_distributions.png"
        plt.savefig(results_path, dpi=150)
        plt.close(fig)
    return [float(dECE_value)]


@register_metric(name="dECE_equal_width")
def dECE_equal_width(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    return dECE(cfg, extracted_output, binning_mode="equal_width")


@register_metric(name="dECE_equal_mass")
def dECE_equal_mass(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    return dECE(cfg, extracted_output, binning_mode="equal_mass")


@register_metric(name="dAUROC_scalar")
def dAUROC_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    confidence_dists: list[BetaDistribution] = extracted_output.extracted_confidences[0]  # list[BetaDistribution]
    confs = [bd.mu for bd in confidence_dists]
    d_auroc_scalar = auroc_scalar(cfg, OrganisedOutputs(
        extracted_answers=extracted_output.extracted_answers,
        extracted_confidences=[confs],
        accuracy_scores=extracted_output.accuracy_scores
    ))[0]
    return [float(d_auroc_scalar)]


@register_metric(name="dAUROC")
def dAUROC(cfg: dict, extracted_output: OrganisedOutputs) -> list[float]:
    accuracies = extracted_output.accuracy_scores[0]            # list[int] in {0,1}
    confidence_dists = extracted_output.extracted_confidences[0]  # list[BetaDistribution]

    num_samples = cfg.get("num_dauroc_samples", 1000)

    # Split into correct / incorrect examples
    pos_dists = [
        bd for y, bd in zip(accuracies, confidence_dists) if y == 1
    ]
    neg_dists = [
        bd for y, bd in zip(accuracies, confidence_dists) if y == 0
    ]

    # Edge cases
    if len(pos_dists) == 0 or len(neg_dists) == 0:
        return [float("nan")]

    wins = 0
    total = 0

    for bd_pos in pos_dists:
        pos_samples = beta.rvs(
            bd_pos.alpha_param,
            bd_pos.beta_param,
            size=num_samples
        )

        for bd_neg in neg_dists:
            neg_samples = beta.rvs(
                bd_neg.alpha_param,
                bd_neg.beta_param,
                size=num_samples
            )

            # Probability C+ > C-
            wins += np.sum(pos_samples[:, None] > neg_samples[None, :])
            total += num_samples * num_samples

    d_auroc = wins / total
    return [float(d_auroc)]
