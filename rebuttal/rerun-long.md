Now i need to run in-domain calibration for openai/gpt-oss-20b, meta-llama/Llama-3.1-8B-Instruct, qwen/Qwen3-8B, mistralai/Mistral-7B-Instruct-v0.3: from original sampling to in-domain calibration (with 100% lexicon and the same 30/70 train-test split). 

Instead of using the existing benchmark, I want to run on a new long form generation QA benchmark: `sentence-transformers/eli5` from huggingface. 
However, the original benchmark is too large, so only run on the first 800 questions. 

you may refer to the run script (reference only, you should create new ones for the new benchmark):
`scripts/local_direct_truthful_qa.sh` for original sampling
`scripts/in_domain_truthful.sh` for in domain calibration

note that we do not need to run direct beta guided method (no need to sample or calibrate).

the code for QA sampling is in `lm_conf/`, please focus on the distributional settings and Truthful QA
The code for calibration is in `calibration/`

Since this is a new benchmark and no task files have been added before in `lm_conf/`, please add appropriate ones accordingly to run and complete the experiment. 

Note that the LLM grader should stay as usual: gpt-oss-20b
The LLM ensemble for linguistic confidence should also remain the same, made up with 3 models:  Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct, Qwen3-8B, each with 3 repeat, resulting in 9 scores. 

we only need to run on this new benchmark with direct QA prompt with sematic uncertainty (su), linguistic confidence (lc) and token probability (tp). 

Please note that the new direct QA prompt template should not contain "answer with at most one sentence" because we want the elicit long form responses from LLMs. You should encourage the LLM to respond with complete explanation. The rewrite in the calibration should also not limit the rewritten sentence(s) to be of a single sentence only. So please change the prompt templates accordingly. Since we are dealing with long form QAs, please note the max token parameter is set appropriately high to accommodate the form. 

We need to compute the generalised ECE and faithfulness divergence pre and post calibration for each model for each signal. Please cache each round in a similar manner like before. 

If you need to edit code or task yaml files, copy them to `rebuttal/re-runs-long/` to do the edits and do not edit the original files.

please use appropriate number of agents to optimise the use of GPUs as many people may race to use them, so try to schedule GPU time together so our usage and runs won't be interrupted. Use all available GPUs in parallel. 

Please note that if caching is not available in `/hdd/ivny/re-runs-long/results/`, please cache and save results in `rebuttal/re-runs-long/results/` with appropriate naming so we can analyse each run later (or potentially reproduce results). 

finally, tabulate the results of each run (generalised ECE and Faithfulness divergence) pre and post hoc (in-domain) and save as CSVs in the same directory. 

You should run and loop until all code is done and results are saved. 