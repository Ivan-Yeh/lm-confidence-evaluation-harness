Now i need to re-estimate the confidence distribution of the in-domain results. The following experiment is to answer the following question:

Can you re-evaluate the rewritten responses with (a) an LLM judge disjoint from the ensemble and absent from the pipeline, and ideally (b) a small human (or agent) study at the statement level?

And my goal is that the original confidence pre and post calibration estimates closely match those of another model and agents. 

For the LLAMA model results in `/hdd/ivny/direct_qa_in_domain_calibration/truthful_qa`, there is `calibration_details.csv`. Each row contains the model's response to a question also with calibrated rewritten responses. But randomly sample 50 questions to perform the following experiments for the interest of time and cost. 

For the following responses (columns):
    - original_response 
    - calibrated_lc_rewritten_response 
    - calibrated_tp_rewritten_response 
    - calibrated_su_rewritten_response 

use Together API to call a model (`MiniMaxAI/MiniMax-M3`) to evaluate the linguistic confidence (I will supply the API key later upon request). do not hard code the api key in any of the code or files in this project. Only write the key to the env. since there is a rate limit, ensure that error is properly handled when the rate limit is reached and retry after waiting. 
For each response, use the following prompt template and perform inference 9 times. You should also add a hint to the evaluator prompt template showing the original confidence scores/distribution of each statement for the evaluator to reference. 

```
LINGUISTIC_EVALUATOR_PROMPT = """
Please provide only a confidence score between 0 and 100, based solely on the degree of confidence expressed in the tone and linguistic cues of the following sentence, without using any external or prior knowledge to assess the knowledge conveyed by the sentence. 
Please pay attention to the hedging language used and the overall assertiveness of the statement. 
If the sentence abstains from answering by pointing out the insufficiency of information with a firm tone, the score should be towards 100. 
If the sentence contains random guesses or abstention, the score should be towards 0. If the sentence is stated with strong certainty or no hedging, the score should be towards 100. 
If the sentence does not contain any hedging language or is a succinct, decisive short answer, the score should be towards 100, too.

To align with human perception, here are some human-annotated linguistic cues with perceived confidence profiles (mean and standard deviation) for your reference: 
{human_annotated_cues}

Here is the sentence:
{sentence}

Confidence Score: [Return only a number between 0 and 100 without any additional text or explanation]
```

human_annotated_cues is a placeholder for some hedging words and their human annotated confidence profiles from the following:
You should randomly sample 15 from low to high confidence to insert into the template with seed = 5.

```
human_annotated_cues = pd.read_csv(os.path.join("/home/ivan/lm-confidence-evaluation-harness/linguistic_confidence_lexicon", "hedging_word_aggregated.csv"))[["hedging_word", "mean", "std"]]
human_annotated_cues["mean"] *= 100.0
human_annotated_cues["std"] *= 100.0
human_annotated_cues = human_annotated_cues.sort_values("mean").round(2).to_dict(orient="records")
_HUMAN_CUE_MEANS = np.array([row["mean"] for row in human_annotated_cues], dtype=float)
```

Additionally, spin up 10 claude agents to perform the same task, each agent needs to have diverse education, occupation and gender background to evaluate each sentence. And then aggregate their scores using the same manner and fit a beta distribution. Regardless of the agent's background, their judgement should solely based on the tone of the target sentence and not the information conveyed. The prompt should also provide the original scores (or range of scores) for the agents to reference. 

After getting all the API and agent responses, extract the numbers using regex and normalise them to [0-1], cache the results and fit a beta distribution over the 9 scores for each response. Therefore, each of the original and calibrated rewritten responses will have a fitted Beta distribution representing its confidence. 

Finally, for each model, compute the generalised ECE, Faithfulness Divergence for each column (original, calibrated rewritten confidence). 

So the final detailed table for each model will have the following columns:
- original response (from the existing CSV)
- correctness (from the existing CSV)
- calibrated_lc_rewritten_response (from the existing CSV)
- calibrated_tp_rewritten_response (from the existing CSV)
- calibrated_su_rewritten_response (from the existing CSV)
- original_response_lc (api model estimated confidence Beta distribution)
- original_response_lc_new_model (api model estimated confidence Beta distribution)
- original_response_lc_annotators (agent estimated confidence Beta distribution)
- calibrated_lc_rewritten_response_lc (from the existing CSV)
- calibrated_tp_rewritten_response_lc (from the existing CSV)
- calibrated_su_rewritten_response_lc (from the existing CSV)
- calibrated_lc_rewritten_response_lc_new_model (api model estimated confidence Beta distribution)
- calibrated_tp_rewritten_response_lc_new_model (api model estimated confidence Beta distribution)
- calibrated_su_rewritten_response_lc_new_model (api model estimated confidence Beta distribution)
- calibrated_lc_rewritten_response_lc_annotators (agent estimated confidence Beta distribution)
- calibrated_tp_rewritten_response_lc_annotators (agent estimated confidence Beta distribution)
- calibrated_su_rewritten_response_lc_annotators (agent estimated confidence Beta distribution)

You should also produce a summary table for genrealised ECE and faithfulness Divergence. The metric calculation is in `lm-confidence-evaluation-harness/lm_conf/post_processing/metrics.py`. 

store the code, results and cache in `lm-confidence-evaluation-harness/rebuttal/disjoint_llm_eval_truthful_qa`. do not change the original code, copy and paste new code into the dir. 