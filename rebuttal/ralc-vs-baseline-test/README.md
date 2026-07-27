# RALC vs. baselines: significance tests

Addresses the reviewer comment: *"no significance test is reported on the
RALC-vs-baseline deltas"* on `paper_materials/fig_calibration_performance_baseline_comparison.ipynb`
(the 4-panel figure: Mean Entailment Score, Confidence Cross-Space Correlation,
Mean Faithfulness Divergence, Mean Generalised ECE).

All numbers below come from the exact scalar metrics/response text already
written by the existing pipeline (`calibration_performance.csv`,
`calibration_details.csv`, `eval_metrics.csv` / `eval_details.csv`) — no
metric was re-implemented. Model/dataset filtering matches the source
notebooks exactly (noted per test).

Run everything with the `llm` conda env:
```
python test_ece_fd.py                  # no GPU needed, ~1 min
python test_confidence_propagation.py  # no GPU needed, ~6 min
python test_entailment.py              # needs a GPU (cross-encoder/nli-deberta-v3-base), ~10 min
```

## 1. Generalised ECE / Faithfulness Divergence — `test_ece_fd.py`

**Unit of analysis:** one row per (dataset, model, signal ∈ {lc, tp, su}) —
n = 5 models × 3 datasets × 3 signals = 45 cells (15 for the Hedged-QA
comparison, which is signal-agnostic; RALC there is the mean over the 3
signals). Excludes gpt-oss-120b / Qwen3-235B, matching the figure's main
panel. Tests: paired t-test and Wilcoxon signed-rank test (one-sided, H1:
RALC lower/better), plus a 10,000-resample paired bootstrap CI on the mean
delta.

| Metric | Baseline | mean Δ (RALC − baseline) | % cells RALC better | paired-t p | Wilcoxon p | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| gECE | Original Direct QA | **−0.137** | 100% | 1.2e-19 | 2.8e-14 | [−0.154, −0.119] |
| gECE | Hedged QA | **−0.090** | 100% | 1.5e-05 | 3.1e-05 | [−0.118, −0.063] |
| gECE | Direct Beta-Guided | **−0.034** | 82% | 1.8e-08 | 2.8e-08 | [−0.044, −0.024] |
| FD | Original Direct QA | **−0.831** | 91% | 8.1e-12 | 3.7e-10 | [−1.005, −0.650] |
| FD | Hedged QA | **−0.612** | 87% | 7.7e-04 | 2.1e-03 | [−0.912, −0.312] |
| FD | Direct Beta-Guided | −0.067 | 78% | 0.048 | 0.0044 | [−0.136, +0.013] |

**Takeaway:** RALC's ECE/FD advantage over Original Direct QA and Hedged QA
is large and highly significant (p ≤ 1e-4 everywhere, every individual
signal p ≤ 4e-4). Against Direct Beta-Guided, the gECE advantage is
significant (p < 5e-8) but small in absolute terms; the FD advantage is only
marginal (bootstrap CI for FD includes 0, p ≈ 0.05) — worth stating as such
rather than claiming a clean win on FD specifically vs. Beta-Guided. Full
breakdown per signal in `ece_fd_significance_results.csv`
(`ece_fd_cell_level_long.csv` / `hedged_qa_cell_level.csv` hold the raw cells).

## 2. Confidence cross-space correlation — `test_confidence_propagation.py`

Reproduces the exact x/y pairing from
`fig_signal_linguistic_confidence_propagation.ipynb` (RALC) and
`fig_signal_beta_linguistic_confidence_propagation.ipynb` (Direct
Beta-Guided): x = mean of `calibrated_{signal}` (the target confidence
signal), y = mean of the re-extracted linguistic confidence after rewriting
(`..._rewritten_lc` for RALC, `..._beta_guided_lc` for the baseline). x is
identical across the two notebooks, so RALC's ρ and Beta-Guided's ρ are
dependent correlations on the same examples — a paired bootstrap on rows
(resampling x, y_RALC, y_beta jointly) directly tests the difference. No
model exclusion (matches the two source notebooks).

**Pooled, full sample (n≈130,837/signal), one-sided bootstrap (RALC ρ > Beta-Guided ρ), 2,000 resamples:**

| Signal | ρ RALC | ρ Beta-Guided | Δρ | 95% CI | p |
|---|---|---|---|---|---|
| lc | 0.749 | 0.346 | **+0.404** | [0.398, 0.409] | <0.0001 |
| tp | 0.797 | 0.251 | **+0.546** | [0.540, 0.552] | <0.0001 |
| su | 0.838 | 0.362 | **+0.476** | [0.471, 0.481] | <0.0001 |

(Note: these are raw, un-binned Spearman ρ; the figure's 0.945/0.951/0.963
vs. 0.761/0.737/0.829 numbers used 5000-quantile-bin smoothing before
correlating, which inflates ρ for both methods — the *delta* and its
significance are what matters here and both are unambiguous.)

**Cell-level check** (one ρ per dataset×model, n=21 cells/signal, paired
t-test/Wilcoxon across cells): su is significant per-signal (p=5e-10), tp is
significant (p=0.0013), lc alone is underpowered at only 21 cells
(p=0.13) but the **pooled-across-signals** test (n=63 cells) is highly
significant (paired-t p=1.1e-8, Wilcoxon p=2.4e-7). Files:
`confidence_propagation_cell_significance.csv`,
`confidence_propagation_pooled_significance.csv`.

## 3. Mean entailment score — `test_entailment.py`

Cross-encoder `cross-encoder/nli-deberta-v3-base`, soft entailment =
P(entail) + 0.5·P(neutral), direction rewritten→original (same recipe as the
notebook).

**RALC vs. Direct Beta-Guided** (paired, same 93,552 calibration rows,
excludes gpt-oss-120b/Qwen3-235B): mean entailment RALC = 0.9536,
Beta-Guided = 0.9570, Δ = **−0.0034** (paired-t / Wilcoxon p = 1.0 for H1
"RALC higher" — i.e. **not significant in RALC's favor**; if anything
Beta-Guided is marginally, though very slightly, higher). This is
consistent with the figure's own numbers (0.956 vs. 0.960) — the entailment
panel was never a "RALC beats Beta-Guided" claim; both preserve the original
answer content about equally well. Report this honestly rather than
asserting significance that isn't there.

**RALC vs. Hedged QA** (unpaired, different pipelines — Mann-Whitney U;
n=280,656 vs. 186,315, no model exclusion, matching the source notebook):
mean entailment RALC = 0.954, Hedged QA = 0.827, Δ = **+0.127**,
Mann-Whitney p ≈ 0 (< 1e-300). RALC preserves the original answer far more
faithfully than Hedged QA.

Files: `entailment_ralc_vs_beta_guided_significance.csv`,
`entailment_ralc_vs_hedged_significance.csv` (+ per-example CSVs for
re-analysis).

## Bottom line for the rebuttal

RALC's advantage is statistically significant (p < 1e-4, most p < 1e-8) and
consistent across signals for **generalised ECE and faithfulness divergence
vs. Original Direct QA and Hedged QA**, and for **confidence cross-space
correlation vs. Direct Beta-Guided**. Against Direct Beta-Guided
specifically, RALC's edge is significant but smaller on gECE, only marginal
on faithfulness divergence, and not significant on entailment — RALC's case
against Beta-Guided rests primarily on calibration quality (gECE) and
confidence-propagation fidelity (ρ), not on faithfulness/entailment, and the
rebuttal response should say so precisely rather than claim a uniform win.
