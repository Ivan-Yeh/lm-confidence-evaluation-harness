from ..default_utils.custom_types import OrganisedOutputs
from ..default_utils.registry import register_metric
import numpy as np
from sklearn.metrics import roc_auc_score


@register_metric(name="accuracy_scalar_with_abstention")
def accuracy_scalar_with_abstention(cfg: dict, extracted_output: OrganisedOutputs) -> float:
    """
    Treat abstentions as incorrect and exclude "None" in accuracy scores.
    """
    accuracies = extracted_output.accuracy_scores[0]
    accuracies_filtered = [acc if acc is not None else None for acc in accuracies]
    accuracies_filtered = [0.0 if acc == "" else acc for acc in accuracies_filtered]
    accuracies_filtered = [acc for acc in accuracies_filtered if acc is not None]
    if len(accuracies_filtered) == 0:
        return 0.0
    return float(np.sum(accuracies_filtered) / len(accuracies_filtered))


@register_metric(name="ece_scalar")
def ece_scalar(cfg: dict, extracted_output: OrganisedOutputs) -> float:
    n_bins = cfg.get("ece_n_bins", 10)
    accuracies_raw = extracted_output.accuracy_scores[0]
    confidences_raw = extracted_output.extracted_confidences[0]
    
    # Treat "" as 0.0, exclude None, and ensure numbers only
    def to_number(val):
        if val is None or val == "" or isinstance(val, list):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    
    accuracies = [to_number(acc) for acc in accuracies_raw]
    confidences = [to_number(conf) for conf in confidences_raw]
    
    # Filter out None values from both lists
    valid_pairs = [(acc, conf) for acc, conf in zip(accuracies, confidences) 
                   if acc is not None and conf is not None]
    
    if len(valid_pairs) == 0:
        return float("nan")
    
    accuracies = np.array([pair[0] for pair in valid_pairs], dtype=float)
    confidences = np.array([pair[1] for pair in valid_pairs], dtype=float)
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
    accuracies_raw = extracted_output.accuracy_scores[0]
    confidences_raw = extracted_output.extracted_confidences[0]
    
    # Ensure all values are numbers, exclude None and lists
    def to_number(val):
        if val is None or val == "" or isinstance(val, list):
            return None
        try:
            num = float(val)
            # Exclude NaN values
            if np.isnan(num):
                return None
            return num
        except (ValueError, TypeError):
            return None
    
    accuracies = [to_number(acc) for acc in accuracies_raw]
    confidences = [to_number(conf) for conf in confidences_raw]
    
    # Filter out None values from both lists
    valid_pairs = [(acc, conf) for acc, conf in zip(accuracies, confidences) 
                   if acc is not None and conf is not None]
    
    if len(valid_pairs) == 0:
        return float("nan")
    
    accuracies = np.array([pair[0] for pair in valid_pairs], dtype=float)
    confidences = np.array([pair[1] for pair in valid_pairs], dtype=float)
    
    # If no valid data or only one class, return nan
    if len(confidences) == 0 or len(np.unique(accuracies)) < 2:
        return float("nan")
    # Compute AUROC
    return float(roc_auc_score(accuracies, confidences))