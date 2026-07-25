Now i need to run in-domain calibration for openai/gpt-oss-20b, meta-llama/Llama-3.1-8B-Instruct, qwen/Qwen3-8B, mistralai/Mistral-7B-Instruct-v0.3: from original sampling to in-domain calibration (with 100% lexicon). 

you may refer to the run script: 
`/home/ivan/lm-confidence-evaluation-harness/scripts/local_direct_truthful_qa.sh` for original sampling
`/home/ivan/lm-confidence-evaluation-harness/scripts/in_domain_truthful.sh` for in domain calibration

note that we do not need to run beta guided method (no need to sample or calibrate)

The code for calibration is in /home/ivan/lm-confidence-evaluation-harness/calibration
the code for confidence sampling is in /home/ivan/lm-confidence-evaluation-harness/lm_conf 

I need to have 5 independent sampling-calibration runs with 5 different seeds. we only need to run on Truthful QA with direct QA prompt with sematic uncertainty (su), linguistic confidence (lc) and token probability (tp). 

We need to compute the generalised ECE and faithfulness divergence pre and post calibration for each model for each signal. Please cache each round in a similar manner like before. 

If you need to edit code or task yaml files, copy them to /home/ivan/lm-confidence-evaluation-harness/rebuttal/re-runs to do the edits and do not edit the original files.

please use appropriate number of agents to optimise the use of GPUs as many people may race to use them, so try to schedule GPU time together so our usage and runs won't be interrupted. You can use all 4 GPUs (0-3)

finally, tabulate the results of each runs ECE and Faithfulness divergence pre and post hoc (in-domain) and save as csv in the same directory. 