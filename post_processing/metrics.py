from default_utils.custom_types import OrganisedOutputs
from default_utils.registry import register_metric
import numpy as np
from sklearn.metrics import roc_auc_score


@register_metric(name="accuracy_scalar_with_na")
def accuracy_scalar_with_na(cfg: dict, extracted_output: OrganisedOutputs) -> float:
    accuracies = extracted_output.accuracy_scores[0]
    return float(np.nansum(accuracies) / len(accuracies))


@register_metric(name="accuracy_scalar_without_na")
def accuracy_scalar_without_na(cfg: dict, extracted_output: OrganisedOutputs) -> float:
    accuracies = extracted_output.accuracy_scores[0]
    return float(np.nanmean(accuracies))


@register_metric(name="ece_scalar")
def ece_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> float:
    n_bins = cfg.get("ece_n_bins", 10)
    accuracies = np.array(extracted_output.accuracy_scores[0], dtype=float)
    confidences = np.array(extracted_output.extracted_confidences[0], dtype=float)

    # --- Fix accuracy: NA → 0 ---
    accuracies = np.nan_to_num(accuracies, nan=0.0)

    # --- Skip NA confidence by masking them out ---
    valid_mask = ~np.isnan(confidences)
    accuracies = accuracies[valid_mask]
    confidences = confidences[valid_mask]

    # No valid confidence values → undefined ECE
    if len(confidences) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        bin_mask = (confidences > lower) & (confidences <= upper)

        if np.any(bin_mask):
            acc_bin = np.mean(accuracies[bin_mask])
            conf_bin = np.mean(confidences[bin_mask])
            weight = np.sum(bin_mask) / len(confidences)
            ece += weight * abs(acc_bin - conf_bin)

    return float(ece)


@register_metric(name="auroc_scalar")
def auroc_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> float:
    """
    Computes AUROC (Area Under the Receiver Operating Characteristic curve)
    from the extracted accuracies and confidence scores.
    """
    # Convert to numpy arrays
    accuracies = np.array(extracted_output.accuracy_scores[0], dtype=float)
    confidences = np.array(extracted_output.extracted_confidences[0], dtype=float)

    # Treat NA accuracies as 0 (missed/incorrect)
    accuracies = np.nan_to_num(accuracies, nan=0.0)

    # Skip NA confidence scores
    valid_mask = ~np.isnan(confidences)
    accuracies = accuracies[valid_mask]
    confidences = confidences[valid_mask]

    # If no valid data, return nan
    if len(confidences) == 0 or len(np.unique(accuracies)) < 2:
        return float("nan")
    # Compute AUROC
    return float(roc_auc_score(accuracies, confidences))