import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import os
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lm_conf.default_utils.custom_types import OrganisedOutputs, PromptCollection
from lm_conf.models.model_manager import ModelManager
from lm_conf.post_processing.metrics import dECE_point_mass, AUROC_point_mass, dAUROC, faithfulness_divergence, generalised_ece
from calibration.utils import *
from calibration.utils import _select_bandwidth, _nw_smooth

argparser = argparse.ArgumentParser(description="Calibrate linguistic confidence lexicon using empirical data.")
argparser.add_argument("--dataset", type=str, required=True, help="Dataset name")
argparser.add_argument("--model", type=str, required=True, help="Path to token probability cache as underlying signal.")
argparser.add_argument("--breakpoint", type=str, choices=["hedge", "eval", "metrics"], help="Whether to set a breakpoint after loading results for debugging.")
argparser.add_argument("--prompt_type", type=str, choices=["direct_qa", "hedged_qa"], default="direct_qa", help="Type of prompts to use.")
argparser.add_argument("--top_k", type=int, default=5, help="Top-k hedging words to retrieve.")

BETA_GUIDED_PROMPT = """
Given an original response and a Beta distribtution, rewrite the response to appropriately reflect the confidence level indicated by the given Beta distribution by using hedging language. 
You must preserve the original meaning of the response, as we are only adjusting the tone to match the confidence level suggested by the hedging words. Ensure the new response sounds natural and fluent. 

Original response: ```My answer to the question is: "{response}"```
Target Beta distribution: ```Beta(alpha={alpha:.2f}, beta={beta:.2f})```

Please return only the rewritten sentence without any explanation.
New response: 
"""

def load_pickled_results(dir, filename):
    with open(os.path.join(dir, filename), "rb") as f:
        return pickle.load(f)
    
def get_latest_leaf_node(dir) -> str:
    """Get the latest leaf node directory in a given directory."""
    subdirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {dir}")
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir


_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = _TOKENIZER_CACHE.get(model_name)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def _compute_max_model_len(
    prompts: list[str],
    model_name: str,
    max_tokens: int,
    min_len: int = 1024,
    pad: int = 64,
) -> int:
    if not prompts:
        return max(min_len, max_tokens + pad)
    tokenizer = _get_tokenizer(model_name)
    try:
        enc = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)
        max_prompt_len = max(len(ids) for ids in enc["input_ids"]) if enc["input_ids"] else 0
    except Exception:
        max_prompt_len = max(
            len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts
        )
    target = max(max_prompt_len + max_tokens + pad + 256, min_len)
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and model_max > 0 and model_max < 1_000_000:
        target = min(target, model_max)
    return int(target)


def _update_max_model_len(cfg: dict, prompts: list[str], min_len: int = 1024, pad: int = 64) -> None:
    model_name = cfg.get("name")
    if not model_name:
        return
    max_tokens = int(cfg.get("max_tokens", 256))
    cfg["max_model_len"] = _compute_max_model_len(
        prompts, model_name, max_tokens, min_len=min_len, pad=pad
    )

if __name__ == "__main__":
    args = argparser.parse_args()

    print("Breakpoint set to:", args.breakpoint)

    if os.path.exists("/hdd"):
        print("Using /hdd/ivny as common directory for results.")
        common_dir = "/hdd/ivny"
    else:
        print("Using ivny as common directory for results.")
        common_dir = "ivny"

    prompt_type = args.prompt_type
    top_k = args.top_k

    signal_calibration_method = "platt_uni" 

    ling_path = get_latest_leaf_node(f"{common_dir}/results/{args.dataset}/{prompt_type}_unified_lc/{args.model}/")
    tp_path = get_latest_leaf_node(f"{common_dir}/results/{args.dataset}/{prompt_type}_unified_tp/{args.model}/")
    su_path = get_latest_leaf_node(f"{common_dir}/results/{args.dataset}/{prompt_type}_unified_su/{args.model}/")

    linguistic_confidence_cache: OrganisedOutputs = load_pickled_results(ling_path, "graded_outputs_0.pkl")
    tp_signal_cache: OrganisedOutputs = load_pickled_results(tp_path, "graded_outputs_0.pkl")
    su_signal_cache: OrganisedOutputs = load_pickled_results(su_path, "graded_outputs_0.pkl")

    save_dir = f"{common_dir}/{prompt_type}_in_domain_calibration/{args.dataset}/{args.model}/"

    df = pd.DataFrame({
        "accuracy": [0.0 if (a is None or (isinstance(a, float) and np.isnan(a))) else a for a in linguistic_confidence_cache.accuracy_scores[0]],
        "original_response": linguistic_confidence_cache.extracted_answers[0],
        "original_lc": linguistic_confidence_cache.extracted_confidences[0],
        "original_tp": tp_signal_cache.extracted_confidences[0],
        "original_su": su_signal_cache.extracted_confidences[0],
    })

    df.dropna(inplace=True)

    # df = df.iloc[:100]  # limit to 100 samples for faster debugging; remove or adjust as needed

    # apply numerical post hoc calibration
    calibrated_lc = in_domain_numerical_post_hoc_calibration(
        np.array(df["original_lc"]),
        np.array(df["accuracy"]),
        method=signal_calibration_method
    )

    calibrated_tp = in_domain_numerical_post_hoc_calibration(
        np.array(df["original_tp"]),
        np.array(df["accuracy"]),
        method=signal_calibration_method
    )

    calibrated_su = in_domain_numerical_post_hoc_calibration(
        np.array(df["original_su"]),
        np.array(df["accuracy"]),
        method=signal_calibration_method
    )

    df["calibrated_lc"] = calibrated_lc
    df["calibrated_tp"] = calibrated_tp
    df["calibrated_su"] = calibrated_su

    df.dropna(inplace=True)
    
    # print ECE and FD for original and calibrated signals
    def _print_signal_metrics(label: str, conf_col: str) -> None:
        metric_df = df[["accuracy", "original_response", conf_col]].dropna().copy()
        if metric_df.empty:
            print(f"[{label}] no valid rows for metric computation")
            return

        organised_output = OrganisedOutputs(
            accuracy_scores=[metric_df["accuracy"].tolist()],
            extracted_confidences=[metric_df[conf_col].tolist()],
            extracted_answers=[metric_df["original_response"].tolist()],
        )
        ece_val = generalised_ece({}, organised_output)[0]
        fd_val = faithfulness_divergence({}, organised_output)[0]
        print(f"[{label}] n={len(metric_df)} | ECE={ece_val:.6f} | FD={fd_val:.6f}")

    _print_signal_metrics("original_lc", "original_lc")
    _print_signal_metrics("calibrated_lc", "calibrated_lc")
    _print_signal_metrics("original_tp", "original_tp")
    _print_signal_metrics("calibrated_tp", "calibrated_tp")
    _print_signal_metrics("original_su", "original_su")
    _print_signal_metrics("calibrated_su", "calibrated_su")

    # reliability diagrams (soft-binned via sampling, 10 bins) before vs after
    def _draw_reliability_ax(
        ax, ax_density, confidences, accuracies, title,
        n_bins=10, num_samples=1000, n_bootstrap=2000,
    ):
        """
        Reliability diagram using Monte Carlo sampling from each BetaDistribution.

        For each bin m the soft probability P(S in Im | X=xn) is estimated by
        sampling from the distribution, then bin accuracy (rm) and mean
        confidence (gm) are computed as weighted averages.  A 90% bootstrap CI
        is drawn as a shaded band.  ax_density receives a histogram of the
        predicted confidence means (mu of each Beta).
        """
        confs, ys = [], []
        for conf, acc in zip(confidences, accuracies):
            if conf is None:
                continue
            try:
                acc_f = float(acc)   # convert first so lists stay in sync on failure
                confs.append(conf)
                ys.append(acc_f)
            except Exception:
                continue

        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7, label="Perfect Calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.tick_params(labelbottom=False)

        ax_density.set_xlim(0, 1)
        ax_density.set_xlabel("Mean Predicted Confidence")
        ax_density.set_ylabel("Density", fontsize=7)
        ax_density.tick_params(axis="y", labelsize=7)

        if not confs:
            ax.legend(fontsize=8)
            return

        from scipy.stats import gaussian_kde
        y    = np.array(ys)
        mus  = np.array([c.mu for c in confs])

        # Bandwidth selection via LOO-CV with a minimum floor for visual smoothness
        h       = max(_select_bandwidth(mus, y), 0.1)
        c_grid  = np.linspace(0.01, 0.99, 200)
        r_hat   = _nw_smooth(c_grid, mus, y, h)

        # 90% bootstrap CI: resample instances, refit NW curve each time
        boot_curves = np.empty((n_bootstrap, len(c_grid)))
        for k in range(n_bootstrap):
            idx = np.random.randint(0, len(y), size=len(y))
            boot_curves[k] = _nw_smooth(c_grid, mus[idx], y[idx], h)
        lower = np.percentile(boot_curves, 5,  axis=0)
        upper = np.percentile(boot_curves, 95, axis=0)

        ax.fill_between(c_grid, lower, upper,
                        color="#2ca02c", alpha=0.25, linewidth=0, label="90% CI")
        ax.plot(c_grid, r_hat, color="black", lw=1.5, label="Reliability Curve")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

        # density strip: KDE of predicted confidence means
        kde     = gaussian_kde(mus, bw_method="scott")
        x_kde   = np.linspace(0.0, 1.0, 300)
        density = kde(x_kde)
        ax_density.plot(x_kde, density, color="#1f77b4", lw=1.5)
        ax_density.fill_between(x_kde, density, alpha=0.25, color="#1f77b4", linewidth=0)
        ax_density.grid(True, linestyle=":", alpha=0.4)

    def _save_reliability_diagram(signal_name, orig_col, cal_col):
        pair_df = df[["accuracy", orig_col, cal_col]].dropna().copy()
        accs    = pair_df["accuracy"].tolist()

        fig = plt.figure(figsize=(10, 5.5))
        gs  = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=0.08, wspace=0.3)
        ax_rel_l  = fig.add_subplot(gs[0, 0])
        ax_den_l  = fig.add_subplot(gs[1, 0], sharex=ax_rel_l)
        ax_rel_r  = fig.add_subplot(gs[0, 1])
        ax_den_r  = fig.add_subplot(gs[1, 1], sharex=ax_rel_r)

        _draw_reliability_ax(ax_rel_l, ax_den_l, pair_df[orig_col].tolist(), accs, f"Before  ({orig_col})")
        _draw_reliability_ax(ax_rel_r, ax_den_r, pair_df[cal_col].tolist(),  accs, f"After  ({cal_col})")
        ax_rel_r.set_ylabel("")

        fig.suptitle(f"Reliability Diagram — {signal_name}  (n={len(pair_df)})", fontsize=12)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"reliability_{signal_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved reliability diagram → {path}")

    _save_reliability_diagram("lc", "original_lc", "calibrated_lc")
    _save_reliability_diagram("tp", "original_tp", "calibrated_tp")
    _save_reliability_diagram("su", "original_su", "calibrated_su")
    
    

    os.makedirs(save_dir, exist_ok=True)

    # check if hedging words have been cached
    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        cache_file = os.path.join(save_dir, f"{conf_method}_hedging_words.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                hedging_words = pickle.load(f)
            print(f"Loaded cached hedging words for {conf_method}: {hedging_words[:10]}")
        else:
            obtain_hedging_words_with_topk = partial(obtain_hedging_words, top_k=top_k)
            with Pool(cpu_count() - 1) as pool:
                hedging_words = list(
                    tqdm(
                        pool.imap(
                            obtain_hedging_words_with_topk,
                            df[conf_method],
                            chunksize=16,
                        ),
                        total=len(df),
                        desc=f"Finding top-{top_k} hedging words for {conf_method}",
                    )
                )
            with open(cache_file, "wb") as f:
                pickle.dump(hedging_words, f)
            print(f"Saved hedging words to {cache_file}")

        df[f"{conf_method}_hedging_words"] = hedging_words

    if args.breakpoint == "hedge":
        print("Hedging words obtained. Breaking here for debugging.")
        sys.exit()

    rewrite_cfg = {
        "rewrite_model": {
            "backend": "vllm",
            "name": "openai/gpt-oss-20b",
            "max_model_len": 5000,
            "temperature": 1.0,
            "max_tokens": 512,
            "reasoning_effort": "low",
        }
    }

    # rewrite outputs based on calibrated signals
    rewrite_targets = []
    beta_targets = []
    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        rewrite_targets.append({
            "conf_method": conf_method,
            "out_col": f"{conf_method}_rewritten_response",
            "cache_file": os.path.join(save_dir, f"{conf_method}_rewrites.pkl"),
            "label": conf_method,
        })
        beta_targets.append({
            "conf_method": conf_method,
            "out_col": f"{conf_method}_beta_guided_response",
            "cache_file": os.path.join(save_dir, f"{conf_method}_beta_guided_rewrites.pkl"),
            "label": f"{conf_method} (beta-guided)",
        })

    cached_rewrites = {}
    pending_rewrite_jobs = []
    batched_rewrite_prompts = []

    for target in rewrite_targets:
        cache_file = target["cache_file"]
        out_col = target["out_col"]
        label = target["label"]

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                rewritten_answers = pickle.load(f)
            if len(rewritten_answers) == len(df):
                cached_rewrites[out_col] = rewritten_answers
                print(f"Loaded cached rewrites for {label}: {len(rewritten_answers)}")
                continue
            print(
                f"Cache length mismatch for rewrites {label} "
                f"({len(rewritten_answers)} != {len(df)}). Recomputing."
            )

        prompts = []
        conf_method = target["conf_method"]
        for _, row in df.iterrows():
            hedges = row[f"{conf_method}_hedging_words"] or []
            prompts.append(
                REWRITE_PROMPT.format(
                    response=row["original_response"],
                    hedges=format_hedges_for_prompt(hedges),
                )
            )
        print(prompts[:2])
        start_idx = len(batched_rewrite_prompts)
        batched_rewrite_prompts.extend(prompts)
        end_idx = len(batched_rewrite_prompts)
        pending_rewrite_jobs.append({
            "out_col": out_col,
            "cache_file": cache_file,
            "label": label,
            "start": start_idx,
            "end": end_idx,
        })

    if pending_rewrite_jobs:
        print(
            f"Generating rewrites for {len(pending_rewrite_jobs)} uncached targets "
            f"in one batched call ({len(batched_rewrite_prompts)} prompts)."
        )
        _update_max_model_len(rewrite_cfg["rewrite_model"], batched_rewrite_prompts)
        model_manager: ModelManager = ModelManager(master_cfg=rewrite_cfg, model_config_type="rewrite_model")
        prompt_collection = PromptCollection(
            system_prompt="You are a linguistic expert.",
            context_texts=batched_rewrite_prompts,
        )
        outputs = model_manager.run_generation(prompt_collection)
        output_texts = outputs[0].output_texts if outputs else []
        if len(output_texts) < len(batched_rewrite_prompts):
            output_texts = output_texts + [""] * (len(batched_rewrite_prompts) - len(output_texts))
        else:
            output_texts = output_texts[:len(batched_rewrite_prompts)]

        cleaned_texts = [
            text.strip().split("assistantfinal")[-1].strip()
            for text in output_texts
        ]

        for job in pending_rewrite_jobs:
            rewritten_answers = cleaned_texts[job["start"]:job["end"]]
            cached_rewrites[job["out_col"]] = rewritten_answers
            with open(job["cache_file"], "wb") as f:
                pickle.dump(rewritten_answers, f)
            print(f"Saved rewrites to {job['cache_file']}")

        del model_manager

    for target in rewrite_targets:
        df[target["out_col"]] = cached_rewrites[target["out_col"]]

    cached_beta_rewrites = {}
    pending_beta_jobs = []
    batched_beta_prompts = []

    for target in beta_targets:
        cache_file = target["cache_file"]
        out_col = target["out_col"]
        label = target["label"]

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                beta_guided_answers = pickle.load(f)
            if len(beta_guided_answers) == len(df):
                cached_beta_rewrites[out_col] = beta_guided_answers
                print(f"Loaded cached beta-guided rewrites for {label}: {len(beta_guided_answers)}")
                continue
            print(
                f"Cache length mismatch for beta-guided rewrites {label} "
                f"({len(beta_guided_answers)} != {len(df)}). Recomputing."
            )

        conf_method = target["conf_method"]
        beta_prompts = []
        for _, row in df.iterrows():
            conf = row[conf_method]
            alpha = float(conf.alpha_param) if conf is not None else 1.0
            beta = float(conf.beta_param) if conf is not None else 1.0
            beta_prompts.append(
                BETA_GUIDED_PROMPT.format(
                    response=row["original_response"],
                    alpha=alpha,
                    beta=beta,
                )
            )

        start_idx = len(batched_beta_prompts)
        batched_beta_prompts.extend(beta_prompts)
        end_idx = len(batched_beta_prompts)
        pending_beta_jobs.append({
            "out_col": out_col,
            "cache_file": cache_file,
            "label": label,
            "start": start_idx,
            "end": end_idx,
        })

    if pending_beta_jobs:
        print(
            f"Generating beta-guided rewrites for {len(pending_beta_jobs)} uncached targets "
            f"in one batched call ({len(batched_beta_prompts)} prompts)."
        )
        _update_max_model_len(rewrite_cfg["rewrite_model"], batched_beta_prompts)
        model_manager: ModelManager = ModelManager(master_cfg=rewrite_cfg, model_config_type="rewrite_model")
        beta_prompt_collection = PromptCollection(
            system_prompt="You are a linguistic expert.",
            context_texts=batched_beta_prompts,
        )
        beta_outputs = model_manager.run_generation(beta_prompt_collection)
        beta_output_texts = beta_outputs[0].output_texts if beta_outputs else []
        if len(beta_output_texts) < len(batched_beta_prompts):
            beta_output_texts = beta_output_texts + [""] * (len(batched_beta_prompts) - len(beta_output_texts))
        else:
            beta_output_texts = beta_output_texts[:len(batched_beta_prompts)]

        cleaned_beta_texts = [
            text.strip().split("assistantfinal")[-1].strip()
            for text in beta_output_texts
        ]

        for job in pending_beta_jobs:
            beta_guided_answers = cleaned_beta_texts[job["start"]:job["end"]]
            cached_beta_rewrites[job["out_col"]] = beta_guided_answers
            with open(job["cache_file"], "wb") as f:
                pickle.dump(beta_guided_answers, f)
            print(f"Saved beta-guided rewrites to {job['cache_file']}")

        del model_manager

    for target in beta_targets:
        df[target["out_col"]] = cached_beta_rewrites[target["out_col"]]
    
    # evaluate rewritten outputs
    eval_cfg = {
        "lc_eval_0": {
            "backend": "vllm",
            "name": "qwen/Qwen3-8B",
            "max_model_len": 10240,
            "temperature": 1.0,
            "max_tokens": 256,
            "repeat": 3,
            "reasoning_effort": None,
        },
        "lc_eval_1": {
            "backend": "vllm",
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "max_model_len": 10240,
            "temperature": 1.0,
            "max_tokens": 256,
            "repeat": 3,
            "reasoning_effort": "low",
        },
        "lc_eval_2": {
            "backend": "vllm",
            "name": "mistralai/Mistral-7B-Instruct-v0.3",
            "max_model_len": 10240,
            "temperature": 1.0,
            "max_tokens": 256,
            "repeat": 3,
            "reasoning_effort": "low",
        },
    }

    evaluator_keys = ["lc_eval_0", "lc_eval_1", "lc_eval_2"]

    confidence_targets = []
    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        confidence_targets.append({
            "out_col": f"{conf_method}_rewritten_lc",
            "response_col": f"{conf_method}_rewritten_response",
            "cache_file": os.path.join(save_dir, f"{conf_method}_rewritten_confidence.pkl"),
            "label": conf_method,
            "target_mean_col": conf_method,
        })
        confidence_targets.append({
            "out_col": f"{conf_method}_beta_guided_lc",
            "response_col": f"{conf_method}_beta_guided_response",
            "cache_file": os.path.join(save_dir, f"{conf_method}_beta_guided_confidence.pkl"),
            "label": f"{conf_method} (beta-guided)",
            "target_mean_col": conf_method,
        })

    cached_confidences = {}
    pending_jobs = []
    batched_responses = []
    batched_target_means = []

    for target in confidence_targets:
        cache_file = target["cache_file"]
        out_col = target["out_col"]
        label = target["label"]

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                confidences = pickle.load(f)
            if len(confidences) == len(df):
                cached_confidences[out_col] = confidences
                print(f"Loaded cached confidences for {label}: {len(confidences)}")
                continue
            print(
                f"Cache length mismatch for {label} ({len(confidences)} != {len(df)}). Recomputing."
            )

        responses = df[target["response_col"]].tolist()
        target_means = df[target["target_mean_col"]].tolist()
        start_idx = len(batched_responses)
        batched_responses.extend(responses)
        batched_target_means.extend(target_means)
        end_idx = len(batched_responses)
        pending_jobs.append({
            "out_col": out_col,
            "cache_file": cache_file,
            "label": label,
            "start": start_idx,
            "end": end_idx,
        })

    if pending_jobs:
        print(
            f"Estimating linguistic confidence for {len(pending_jobs)} uncached targets "
            f"in one batched call ({len(batched_responses)} prompts)."
        )
        eval_prompts = build_linguistic_evaluator_prompts(
            batched_responses, batched_target_means
        )
        for evaluator in evaluator_keys:
            _update_max_model_len(eval_cfg[evaluator], eval_prompts)
        batched_confidences = estimate_linguistic_confidence(
            batched_responses,
            batched_target_means,
            evaluators_cfg=eval_cfg,
            evaluator_keys=evaluator_keys,
        )

        for job in pending_jobs:
            confidences = batched_confidences[job["start"]:job["end"]]
            cached_confidences[job["out_col"]] = confidences
            with open(job["cache_file"], "wb") as f:
                pickle.dump(confidences, f)
            print(f"Saved confidences to {job['cache_file']}")

    for target in confidence_targets:
        df[target["out_col"]] = cached_confidences[target["out_col"]]

    df.dropna(inplace=True)
    
    if args.breakpoint == "eval":
        print("Evaluation completed. Breaking here for debugging.")
        sys.exit()


    print("Computing metrics...")
    from multiprocessing.pool import ThreadPool

    accuracy_list = df["accuracy"].tolist()

    def _compute_metrics(args_tuple):
        label, conf_list, answer_list = args_tuple
        organised_output = OrganisedOutputs(
            accuracy_scores=[accuracy_list],
            extracted_confidences=[conf_list],
            extracted_answers=[answer_list],
        )
        return [
            {"metric": f"{label}_generalised_ECE", "value": generalised_ece({}, organised_output)[0]},
            {"metric": f"{label}_faithfulness_divergence", "value": faithfulness_divergence({}, organised_output)[0]},
            {"metric": f"{label}_ece_mean", "value": dECE_point_mass({}, organised_output)[0]},
            {"metric": f"{label}_dAUROC", "value": dAUROC({}, organised_output)[0]},
            {"metric": f"{label}_auroc_mean", "value": AUROC_point_mass({}, organised_output)[0]},
        ]

    original_answers = df["original_response"].tolist()

    metric_jobs = [
        ("original_lc", df["original_lc"].tolist(), original_answers),
        ("original_tp", df["original_tp"].tolist(), original_answers),
        ("original_su", df["original_su"].tolist(), original_answers),
        ("calibrated_lc", df["calibrated_lc"].tolist(), original_answers),
        ("calibrated_tp", df["calibrated_tp"].tolist(), original_answers),
        ("calibrated_su", df["calibrated_su"].tolist(), original_answers),
    ]

    for conf_method in ["calibrated_lc", "calibrated_tp", "calibrated_su"]:
        metric_jobs.append(
            (
                f"{conf_method}_rewritten_lc",
                df[f"{conf_method}_rewritten_lc"].tolist(),
                df[f"{conf_method}_rewritten_response"].tolist(),
            )
        )
        metric_jobs.append(
            (
                f"{conf_method}_beta_guided_lc",
                df[f"{conf_method}_beta_guided_lc"].tolist(),
                df[f"{conf_method}_beta_guided_response"].tolist(),
            )
        )

    metrics_rows = []
    with ThreadPool(min(8, len(metric_jobs))) as pool:
        for rows in tqdm(
            pool.imap(_compute_metrics, metric_jobs),
            total=len(metric_jobs),
            desc="Computing metrics",
        ):
            metrics_rows.extend(rows)

    metrics_df = pd.DataFrame(metrics_rows)

    details_csv = os.path.join(save_dir, "calibration_details.csv")
    details_pkl = os.path.join(save_dir, "calibration_details.pkl")
    metrics_csv = os.path.join(save_dir, "calibration_performance.csv")

    df.to_csv(details_csv, index=False)
    df.to_pickle(details_pkl)
    metrics_df.to_csv(metrics_csv, index=False)
    print(metrics_df)
    
            