import sys
import numpy as np
import pickle
from pathlib import Path
import os
import pandas as pd

# Add parent directory to path to import lm_conf
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.special import betaln, psi, expit, logit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from scipy.stats import beta
from typing import Literal

from lm_conf.default_utils.custom_types import OrganisedOutputs
from lm_conf.post_processing.metrics import BetaDistribution
from lm_conf.post_processing.metrics import dAUROC, dECE_equal_width, dECE_equal_mass, dAUROC_scalar, dECE_scalar

BIN = 10

path = """
/hdd/ivny/results/trivia_qa/dist_lnll/meta-llama/Llama-3.1-8B-Instruct/2025-12-24_10-25-36
""".strip()
with open(f"{path}/graded_outputs_0.pkl", "rb") as f:
    graded_outputs: OrganisedOutputs = pickle.load(f)

def build_outputs(conf_list, acc_list):
    return OrganisedOutputs(
        extracted_confidences=[conf_list],
        accuracy_scores=[acc_list],
    )

def construct_calibration_data(conf_lst: list[BetaDistribution], acc_lst: list[float]):
    # Step 1: Bin confidence scores into equal-width bins
    num_bins = BIN
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    features = []
    targets = []
    target_dists = []

    # Bin the training data
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        
        # Include right edge for last bin
        if i == num_bins - 1:
            mask = np.array([(c.mu >= lo and c.mu <= hi) for c in conf_lst])
        else:
            mask = np.array([(c.mu >= lo and c.mu < hi) for c in conf_lst])

        conf_in_bin: list[BetaDistribution] = [conf_lst[j] for j in range(len(conf_lst)) if mask[j]]
        acc_in_bin: list[float] = [acc_lst[j] for j in range(len(acc_lst)) if mask[j]]
        # bin accuracy distribution
        if len(acc_in_bin) > 0:
            alpha_param = np.sum(acc_in_bin) + 1
            beta_param = len(acc_in_bin) - np.sum(acc_in_bin) + 1
            mu = alpha_param / (alpha_param + beta_param)
            sigma = np.sqrt((alpha_param * beta_param) / ((alpha_param + beta_param)**2 * (alpha_param + beta_param + 1)))
            bin_acc_dist = BetaDistribution(mu=mu, sigma=sigma)
            # build training data for this bin
            for i, c in enumerate(conf_in_bin):
                features.append((c.mu, c.sigma, c.alpha_param, c.beta_param, c.alpha_param + c.beta_param))
                target_dists.append((bin_acc_dist.mu, bin_acc_dist.sigma, bin_acc_dist.alpha_param, bin_acc_dist.beta_param, bin_acc_dist.alpha_param + bin_acc_dist.beta_param)) 
                targets.append(acc_in_bin[i])
        else:
            continue
    return features, targets, target_dists



def parameter_scaling(
    features, targets, target_dist, X, eps=1e-6
):
    """
    Mean calibration via Platt Scaling (Logistic) and 
    Concentration calibration via Isotonic Regression.
    """

    def unpack(arr):
        mu    = np.array([x[0] for x in arr], dtype=float)
        sigma = np.array([x[1] for x in arr], dtype=float)
        conc  = np.array([x[4] for x in arr], dtype=float)
        return mu, sigma, conc

    # --- unpack ---
    mu_f, sigma_f, conc_f = unpack(features)
    mu_x, sigma_x, conc_x = unpack(X)
    _, _, conc_t = unpack(target_dist) # Ground truth concentrations

    # --- numerical safety ---
    mu_f = np.clip(mu_f, eps, 1 - eps)
    mu_x = np.clip(mu_x, eps, 1 - eps)
    sigma_f = np.clip(sigma_f, eps, None)
    sigma_x = np.clip(sigma_x, eps, None)

    # =====================
    # Mean calibration (Platt Scaling)
    # =====================
    logits_f = np.log(mu_f / (1 - mu_f))
    logits_x = np.log(mu_x / (1 - mu_x))
    
    Z_f = np.column_stack([logits_f, np.log(sigma_f)])
    Z_x = np.column_stack([logits_x, np.log(sigma_x)])

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(Z_f, targets)
    mu_cal = lr.predict_proba(Z_x)[:, 1]

    # =========================
    # Concentration calibration (Isotonic)
    # =========================
    # IsotonicRegression fits a non-decreasing step function.
    # 'out_of_bounds=clip' ensures that if X has higher concentration 
    # than anything in 'features', it gets the max seen value.
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(conc_f, conc_t)
    
    conc_cal = iso_reg.predict(conc_x)
    
    # Ensure conc_cal doesn't hit zero (Beta dist requirement)
    conc_cal = np.clip(conc_cal, eps, None)

    # --- reconstruct Beta ---
    alpha_cal = mu_cal * conc_cal
    beta_cal  = (1 - mu_cal) * conc_cal
    
    # sigma = sqrt((alpha * beta) / (nu^2 * (nu + 1)))
    sigma_cal = np.sqrt(
        (alpha_cal * beta_cal) / ((conc_cal ** 2) * (conc_cal + 1))
    )

    return list(zip(mu_cal, sigma_cal, alpha_cal, beta_cal, conc_cal))

def kl_beta(alpha_p, beta_p, alpha_q, beta_q):
    """Analytic KL Divergence: KL(Target || Scaled)"""
    term1 = betaln(alpha_q, beta_q) - betaln(alpha_p, beta_p)
    term2 = (alpha_p - alpha_q) * psi(alpha_p)
    term3 = (beta_p - beta_q) * psi(beta_p)
    term4 = (alpha_q + beta_q - (alpha_p + beta_p)) * psi(alpha_p + beta_p)
    return term1 + term2 + term3 + term4

def parameter_scaling_kl(
    features, targets, target_dist, X, eps=1e-6
):
    """
    PDSS with Bivariate Logistic Mean Scaling and KL Concentration Scaling.
    
    Stage 1: Location Scaling - Maps (mu, sigma) to targets via bivariate logistic.
    Stage 2: Dispersion Scaling - Optimizes concentration scaling via KL divergence.
    """
    def unpack(arr):
        arr = np.array(arr)
        # 0: mu, 1: sigma, 2: alpha, 3: beta, 4: conc
        return arr[:, 0], arr[:, 1], arr[:, 4]

    mu_f, sigma_f, conc_f = unpack(features)
    mu_x, sigma_x, conc_x = unpack(X)
    mu_t, _, conc_t = unpack(target_dist)
    y_f = np.array(targets)

    # ==========================================
    # STAGE 1: Location Scaling (Bivariate Platt)
    # ==========================================
    logits_f = np.log(np.clip(mu_f, eps, 1 - eps) / (1 - np.clip(mu_f, eps, 1 - eps)))
    logits_x = np.log(np.clip(mu_x, eps, 1 - eps) / (1 - np.clip(mu_x, eps, 1 - eps)))
    
    # Feature matrices: combining location (logit) and dispersion (log-sigma)
    Z_f = np.column_stack([logits_f, np.log(np.clip(sigma_f, eps, None))])
    Z_x = np.column_stack([logits_x, np.log(np.clip(sigma_x, eps, None))])
    
    def logistic_obj(params):
        # params: [w_logit, w_sigma, bias]
        w = params[:2]
        b = params[2]
        # dot product + bias
        z = np.dot(Z_f, w) + b
        p = expit(z)
        return -np.mean(y_f * np.log(p + eps) + (1 - y_f) * np.log(1 - p + eps))

    # Initialize with 1.0 for mu weight, 0.0 for sigma weight and bias
    res_mu = minimize(logistic_obj, x0=[1.0, 0.0, 0.0], method="L-BFGS-B")
    w_opt, b_opt = res_mu.x[:2], res_mu.x[2]
    
    mu_cal = expit(np.dot(Z_x, w_opt) + b_opt)

    # ==========================================
    # STAGE 2: Dispersion Scaling (KL Optimization)
    # ==========================================
    def concentration_obj(params):
        # params: [intercept, coefficient]
        a, b = params
        # Power-law relationship for concentration mapping
        conc_scaled = np.exp(a + b * np.log(np.clip(conc_f, eps, None)))
        
        alpha_s = mu_t * conc_scaled
        beta_s  = (1 - mu_t) * conc_scaled
        
        # Target distribution reconstructed from target_dist parameters
        alpha_t = mu_t * conc_t
        beta_t  = (1 - mu_t) * conc_t
        
        return np.mean(kl_beta(alpha_t, beta_t, alpha_s, beta_s))

    res_conc = minimize(concentration_obj, x0=[0.0, 1.0], method="L-BFGS-B")
    a_opt, b_opt = res_conc.x
    
    conc_cal = np.exp(a_opt + b_opt * np.log(np.clip(conc_x, eps, None)))

    # --- Reconstruction ---
    alpha_cal = mu_cal * conc_cal
    beta_cal  = (1 - mu_cal) * conc_cal
    sigma_cal = np.sqrt((alpha_cal * beta_cal) / ((conc_cal ** 2) * (conc_cal + 1)))

    return list(zip(mu_cal, sigma_cal, alpha_cal, beta_cal, conc_cal))

def dist_reliability_diagram(
    cfg: dict,
    uncalibrated_outputs: OrganisedOutputs,
    calibrated_outputs_reg: OrganisedOutputs,
    calibrated_outputs_kl: OrganisedOutputs,
    binning_mode: Literal["equal_width", "equal_mass"] = "equal_width",
    ci_level: float = 0.9,
):
    """
    Distributional reliability diagrams with confidence intervals.
    """

    outputs_dict = {
        "Uncalibrated": uncalibrated_outputs,
        "Platt Scaling + Isotonic Regression": calibrated_outputs_reg,
        "Platt Scaling + KL Optimisation": calibrated_outputs_kl,
    }

    num_bins = cfg.get("num_bins", 10)
    num_samples = cfg.get("num_wasserstein_samples", 2000)

    alpha_ci = (1.0 - ci_level) / 2.0
    lo_q, hi_q = alpha_ci, 1.0 - alpha_ci

    # ------------------------------------------------------------
    # 1. Clean UNCALIBRATED data (used for binning)
    # ------------------------------------------------------------
    acc_u = []
    conf_u = []

    for acc, conf in zip(
        uncalibrated_outputs.accuracy_scores[0],
        uncalibrated_outputs.extracted_confidences[0],
    ):
        try:
            if acc is None or conf is None or not conf.is_valid():
                continue
            acc_u.append(float(acc))
            conf_u.append(conf)
        except Exception:
            continue

    mean_conf = np.array([bd.mu for bd in conf_u])

    # ------------------------------------------------------------
    # 2. Define bins ONCE
    # ------------------------------------------------------------
    if binning_mode == "equal_width":
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        bin_edges = np.quantile(mean_conf, np.linspace(0.0, 1.0, num_bins + 1))

    # ------------------------------------------------------------
    # 3. Compute per-bin statistics
    # ------------------------------------------------------------
    bin_stats = []

    for m in range(num_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]

        if m == num_bins - 1:
            idx = np.where((mean_conf >= lo) & (mean_conf <= hi))[0]
        else:
            idx = np.where((mean_conf >= lo) & (mean_conf < hi))[0]

        if len(idx) == 0:
            continue

        bin_info = {
            "bin": m,
            "n": len(idx),
            "stats": {},
        }

        for name, outputs in outputs_dict.items():
            acc = outputs.accuracy_scores[0]
            conf = outputs.extracted_confidences[0]

            # --- confidence samples ---
            conf_samples = []
            acc_vals = []

            for i in idx:
                bd: BetaDistribution = conf[i]
                conf_samples.append(
                    beta.rvs(bd.alpha_param, bd.beta_param, size=num_samples)
                )
                acc_vals.append(acc[i])

            conf_samples = np.concatenate(conf_samples)

            conf_mean = np.mean(conf_samples)
            conf_lo, conf_hi = np.quantile(conf_samples, [lo_q, hi_q])

            # --- accuracy posterior ---
            k = int(np.sum(acc_vals))
            n = len(acc_vals)

            acc_mean = (k + 1) / (n + 2)
            acc_lo = beta.ppf(lo_q, k + 1, n - k + 1)
            acc_hi = beta.ppf(hi_q, k + 1, n - k + 1)

            bin_info["stats"][name] = {
                "conf_mean": conf_mean,
                "conf_err": (conf_mean - conf_lo, conf_hi - conf_mean),
                "acc_mean": acc_mean,
                "acc_err": (acc_mean - acc_lo, acc_hi - acc_mean),
            }

        bin_stats.append(bin_info)

    # ------------------------------------------------------------
    # 4. Plot reliability diagrams
    # ------------------------------------------------------------
    methods = list(outputs_dict.keys())

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
    })

    fig, axes = plt.subplots(
        1, len(methods),
        figsize=(4 * len(methods), 4),
        sharex=True,
        sharey=True,
    )

    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        xs, ys = [], []

        # Relative errors (for error bars)
        x_err = [[], []]
        y_err = [[], []]

        # Absolute CI endpoints (for envelopes)
        x_low, x_high = [], []
        y_low, y_high = [], []

        # -------------------------
        # Collect bin statistics
        # -------------------------
        for b in bin_stats:
            s = b["stats"][method]

            x = s["conf_mean"]
            y = s["acc_mean"]

            xs.append(x)
            ys.append(y)

            # Absolute CI endpoints
            y_low.append(y - s["acc_err"][0])
            y_high.append(y + s["acc_err"][1])
            x_low.append(x - s["conf_err"][0])
            x_high.append(x + s["conf_err"][1])

            # Relative offsets (for error bars)
            y_err[0].append(s["acc_err"][0])
            y_err[1].append(s["acc_err"][1])
            x_err[0].append(s["conf_err"][0])
            x_err[1].append(s["conf_err"][1])

        # -------------------------
        # Convert to arrays & sort
        # -------------------------
        xs = np.array(xs)
        ys = np.array(ys)

        y_low = np.array(y_low)
        y_high = np.array(y_high)
        x_low = np.array(x_low)
        x_high = np.array(x_high)

        x_err = np.array(x_err)
        y_err = np.array(y_err)

        order = np.argsort(xs)

        xs = xs[order]
        ys = ys[order]
        y_low = y_low[order]
        y_high = y_high[order]
        x_low = x_low[order]
        x_high = x_high[order]
        x_err = x_err[:, order]
        y_err = y_err[:, order]

        # -------------------------
        # Plotting layers
        # -------------------------

        # 1. Confidence envelopes (connect error bar ends)
        ax.fill_between(
            xs, y_low, y_high,
            color="teal",
            alpha=0.15,
            linewidth=0,
            label="Acc. CI Band"
        )

        ax.fill_betweenx(
            ys, x_low, x_high,
            color="orange",
            alpha=0.15,
            linewidth=0,
            label="Conf. CI Band"
        )

        # 2. Error bars (lighter, dashed)
        ax.errorbar(
            xs, ys,
            yerr=y_err,
            fmt="none",
            ecolor="teal",
            elinewidth=1,
            capsize=2,
            alpha=0.35,
            linestyle="--"
        )

        ax.errorbar(
            xs, ys,
            xerr=x_err,
            fmt="none",
            ecolor="orange",
            elinewidth=1,
            capsize=2,
            alpha=0.35,
            linestyle="--"
        )

        # 3. Mean calibration curve
        ax.plot(
            xs, ys,
            color="black",
            marker="o",
            markersize=4,
            linewidth=1,
            label="Mean Calibration"
        )

        # 4. Perfect calibration reference
        ax.plot(
            [0, 1], [0, 1],
            "--",
            color="gray",
            linewidth=1,
            label="Perfect Calibration"
        )

        # -------------------------
        # Formatting
        # -------------------------
        ax.set_title(method)
        ax.set_xlabel("Mean confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Empirical accuracy")

    # Single legend (avoid clutter)
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        fontsize="x-small",
        frameon=True
    )

    fig.tight_layout()


    out_path = f"{cfg.get('results_path', '.')}/dist_reliability_{binning_mode}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved distributional reliability diagram to {out_path}")


def dist_calibrator(graded_outputs: OrganisedOutputs, output_path: str = "."):

    os.makedirs(output_path, exist_ok=True)

    conf: list[BetaDistribution] = graded_outputs.extracted_confidences[0]
    acc: list = graded_outputs.accuracy_scores[0]

    def to_float_acc(x):
        try:
            if x is None or x == "" or isinstance(x, list):
                return None
            val = float(x)
            if np.isnan(val):
                return None
            return val
        except (ValueError, TypeError):
            return None

    clean_conf = []
    clean_acc = []
    for c, a in zip(conf, acc):
        fa = to_float_acc(a)
        if fa is None:
            continue
        if c is None:
            continue
        try:
            valid = c.is_valid()
        except Exception:
            valid = True
        if not valid:
            continue
        clean_conf.append(c)
        clean_acc.append(fa)

    # 1. Split the raw samples first
    conf_train, conf_test, acc_train, acc_test = train_test_split(
        clean_conf,
        clean_acc,
        test_size=0.9, 
        random_state=42,
    )

    # 2. Build calibration targets ONLY for the training set
    X_train, y_train, y_train_dists = construct_calibration_data(conf_train, acc_train)

    # 3. DO NOT bin the test set for features. 
    # Your calibrator needs to predict on individual samples (X_test_raw)
    X_test_raw = np.array([(c.mu, c.sigma, c.alpha_param, c.beta_param, c.alpha_param + c.beta_param) for c in conf_test])

    calibrated_X_reg = parameter_scaling(X_train, y_train, y_train_dists, X_test_raw)
    calibrated_X_kl = parameter_scaling_kl(X_train, y_train, y_train_dists, X_test_raw)
    calibrated_distributions_reg = [BetaDistribution(x[0], x[1]) for x in calibrated_X_reg]
    calibrated_distributions_kl = [BetaDistribution(x[0], x[1]) for x in calibrated_X_kl]

    uncalibrated_outputs: OrganisedOutputs = build_outputs(conf_test, acc_test)
    calibrated_outputs_reg: OrganisedOutputs = build_outputs(calibrated_distributions_reg, acc_test)
    calibrated_outputs_kl: OrganisedOutputs = build_outputs(calibrated_distributions_kl, acc_test)

    dist_reliability_diagram(
        {"results_path": output_path, "num_bins": BIN, "num_wasserstein_samples": 2000},
        uncalibrated_outputs,
        calibrated_outputs_reg,
        calibrated_outputs_kl,
        binning_mode="equal_width",
        ci_level=0.95,
    )

    return {
        "raw_dECE": dECE_equal_width({"results_path": None}, uncalibrated_outputs),
        "raw_dECE_pt": dECE_scalar({"results_path": None}, uncalibrated_outputs),
        "dECE_reg": dECE_equal_width({"results_path": None}, calibrated_outputs_reg),
        "dECE_pt_reg": dECE_scalar({"results_path": None}, calibrated_outputs_reg),
        "dECE_kl": dECE_equal_width({"results_path": None}, calibrated_outputs_kl),
        "dECE_pt_kl": dECE_scalar({"results_path": None}, calibrated_outputs_kl),

        "raw_dAUROC": dAUROC({"results_path": None}, uncalibrated_outputs),
        "raw_dAUROC_pt": dAUROC_scalar({"results_path": None}, uncalibrated_outputs),
        "dAUROC_reg": dAUROC({"results_path": None}, calibrated_outputs_reg),
        "dAUROC_pt_reg": dAUROC_scalar({"results_path": None}, calibrated_outputs_reg),
        "dAUROC_kl": dAUROC({"results_path": None}, calibrated_outputs_kl),
        "dAUROC_pt_kl": dAUROC_scalar({"results_path": None}, calibrated_outputs_kl),
    }


BASE_PATH = "/hdd/ivny/results/"
DATASETS = ["trivia_qa", "mmlu", "squadv2"]

# Map base method name → paper name
BASE_METHOD_MAP = {
    "lnll": "Token Probability",
    "lnll_gen": "Token Probability",
    "p_true_mc": "Self-Evaluation",
    "semantic_uncertainty": "Semantic Uncertainty",
    "linguistic_confidence": "Linguistic Confidence",
}

def run_dist_calibration(task):
    (
        model_name,
        estimation_method,
        graded_outputs_path,
        output_dir,
    ) = task

    # Load graded outputs
    with open(graded_outputs_path, "rb") as f:
        graded_outputs = pickle.load(f)

    # Run expensive calibration
    calibration_metrics = dist_calibrator(
        graded_outputs,
        output_path=output_dir,
    )

    # Initialise row
    row = {
        "Model": model_name,
        "Estimation Method": estimation_method,
        "raw_dECE": None,
        "raw_dECE_pt": None,
        "dECE_reg": None,
        "dECE_pt_reg": None,
        "dECE_kl": None,
        "dECE_pt_kl": None,
        "raw_dAUROC": None,
        "raw_dAUROC_pt": None,
        "dAUROC_reg": None,
        "dAUROC_pt_reg": None,
        "dAUROC_kl": None,
        "dAUROC_pt_kl": None,
    }

    # Fill metrics
    for key in calibration_metrics:
        row[key] = calibration_metrics[key][0]

    return row


# for DATASET in DATASETS:
#     rows = []
#     dataset_path = os.path.join(BASE_PATH, DATASET)

#     for conf_mode in os.listdir(dataset_path):
#         conf_path = os.path.join(dataset_path, conf_mode)
#         if not os.path.isdir(conf_path):
#             continue

#         # -----------------------------
#         # Detect scalar vs dist
#         # -----------------------------
#         is_dist = conf_mode.startswith("dist_")
#         base_mode = conf_mode.replace("dist_", "")

#         if not is_dist:
#             continue

#         if base_mode not in BASE_METHOD_MAP:
#             continue

#         estimation_method = BASE_METHOD_MAP[base_mode]

#         for model_family in os.listdir(conf_path):
#             family_path = os.path.join(conf_path, model_family)
#             if not os.path.isdir(family_path):
#                 continue

#             for model_name in os.listdir(family_path):
#                 model_path = os.path.join(family_path, model_name)
#                 if not os.path.isdir(model_path):
#                     continue

#                 timestamps = sorted(os.listdir(model_path))
#                 if not timestamps:
#                     continue

#                 timestamp = timestamps[-1]
#                 graded_outputs_path = os.path.join(
#                     model_path, timestamp, "graded_outputs_0.pkl"
#                 )

#                 if not os.path.exists(graded_outputs_path):
#                     continue
#                 with open(graded_outputs_path, "rb") as f:
#                     graded_outputs: OrganisedOutputs = pickle.load(f)

#                 calibration_metrics = dist_calibrator(graded_outputs, output_path=os.path.join(model_path, timestamp, "dist_calibrator_outputs"))

#                 row = {
#                     "Model": model_name,
#                     "Estimation Method": estimation_method,
#                     "raw_dECE": None,
#                     "raw_dECE_pt": None,
#                     "dECE_reg": None,
#                     "dECE_pt_reg": None,
#                     "dECE_kl": None,
#                     "dECE_pt_kl": None,
#                     "raw_dAUROC": None,
#                     "raw_dAUROC_pt": None,
#                     "dAUROC_reg": None,
#                     "dAUROC_pt_reg": None,
#                     "dAUROC_kl": None,
#                     "dAUROC_pt_kl": None,
#                 }

#                 # -----------------------------
#                 # Fill metrics
#                 # -----------------------------
#                 for key in calibration_metrics:
#                     row[key] = calibration_metrics[key][0]
#                 rows.append(row)

#     # =========================
#     # Merge scalar + dist rows
#     # =========================
#     results_df = (
#         pd.DataFrame(rows)
#         .groupby(["Model", "Estimation Method"], as_index=False)
#         .first()
#     )

#     results_df.sort_values(
#         by=["Model", "Estimation Method"],
#         inplace=True
#     )

#     results_df.reset_index(drop=True, inplace=True)

#     results_df["Model"] = results_df["Model"].str.upper()
#     results_df.to_csv(f"paper_materials/calibration_table_{DATASET}.csv", index=True)
#     results_df

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp

for DATASET in DATASETS:
    tasks = []

    dataset_path = os.path.join(BASE_PATH, DATASET)

    for conf_mode in os.listdir(dataset_path):
        conf_path = os.path.join(dataset_path, conf_mode)
        if not os.path.isdir(conf_path):
            continue

        # Only distributional modes
        if not conf_mode.startswith("dist_"):
            continue

        base_mode = conf_mode.replace("dist_", "")
        if base_mode not in BASE_METHOD_MAP:
            continue

        estimation_method = BASE_METHOD_MAP[base_mode]

        for model_family in os.listdir(conf_path):
            family_path = os.path.join(conf_path, model_family)
            if not os.path.isdir(family_path):
                continue

            for model_name in os.listdir(family_path):
                model_path = os.path.join(family_path, model_name)
                if not os.path.isdir(model_path):
                    continue

                timestamps = sorted(os.listdir(model_path))
                if not timestamps:
                    continue

                timestamp = timestamps[-1]
                graded_outputs_path = os.path.join(
                    model_path, timestamp, "graded_outputs_0.pkl"
                )

                if not os.path.exists(graded_outputs_path):
                    continue

                output_dir = os.path.join(
                    model_path, timestamp, "dist_calibrator_outputs"
                )

                tasks.append((
                    model_name,
                    estimation_method,
                    graded_outputs_path,
                    output_dir,
                ))
    
    rows = []
    num_workers = max(1, mp.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for row in tqdm(
            executor.map(run_dist_calibration, tasks),
            total=len(tasks),
            desc=f"Running dist calibration ({DATASET})",
        ):
            rows.append(row)
    
        results_df = (
        pd.DataFrame(rows)
        .groupby(["Model", "Estimation Method"], as_index=False)
        .first()
    )

    results_df.sort_values(
        by=["Model", "Estimation Method"],
        inplace=True
    )

    results_df.reset_index(drop=True, inplace=True)
    results_df["Model"] = results_df["Model"].str.upper()

    results_df.to_csv(
        f"paper_materials/calibration_table_{DATASET}.csv",
        index=False
    )

