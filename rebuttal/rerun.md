Now i need to run in-domain calibration for openai/gpt-oss-20b, meta-llama/Llama-3.1-8B-Instruct, qwen/Qwen3-8B, mistralai/Mistral-7B-Instruct-v0.3: from original sampling to in-domain calibration (with 100% lexicon). 

We only need to ru on Truthful QA.

you may refer to the run script: 
`scripts/local_direct_truthful_qa.sh` for original sampling
`scripts/in_domain_truthful.sh` for in domain calibration

note that we do not need to run direct beta guided method (no need to sample or calibrate).

the code for QA sampling is in `lm_conf/`, please focus on the distributional settings and Truthful QA
The code for calibration is in `calibration/`

Note that the LLM grader should stay as usual: gpt-oss-20b
The LLM ensemble for linguistic confidence should also remain the same, made up with 3 models: openai/gpt-oss-20b, meta-llama/Llama-3.1-8B-Instruct, qwen/Qwen3-8B, each with 3 repeat, resulting in 9 scores. 

I need to have 5 independent sampling-calibration runs with 5 different seeds. we only need to run on Truthful QA with direct QA prompt with sematic uncertainty (su), linguistic confidence (lc) and token probability (tp). 

We need to compute the generalised ECE and faithfulness divergence pre and post calibration for each model for each signal. Please cache each round in a similar manner like before. 

If you need to edit code or task yaml files, copy them to `rebuttal/re-runs/` to do the edits and do not edit the original files.

please use appropriate number of agents to optimise the use of GPUs as many people may race to use them, so try to schedule GPU time together so our usage and runs won't be interrupted. 

Please note that if caching is not available in /hdd, please cache and save results in `rebuttal/re-runs/results/` with appropriate naming so we can analyse each run later (or potentially reproduce results). 

finally, tabulate the results of each run (generalised ECE and Faithfulness divergence) pre and post hoc (in-domain) and save as CSVs in the same directory. 

Another agent has previously written its code in `rebuttal/re-runs/`, please check if there is any error and edit accordingly. 