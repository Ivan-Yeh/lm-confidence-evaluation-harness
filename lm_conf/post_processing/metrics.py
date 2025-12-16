from ..default_utils.custom_types import OrganisedOutputs
from ..default_utils.registry import register_metric
import numpy as np
from sklearn.metrics import roc_auc_score


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