Now i need to re-estimate the confidence distribution of the in-domain results.

For each model in `/hdd/ivny/direct_qa_in_domain_calibration/truthful_qa` (except for google gemma), there is `calibration_details.csv`. Each row contains the model's response to a question also with calibrated rewritten responses. 

For the following responses (columns):
    - original_response
    - calibrated_lc_rewritten_response 
    - calibrated_tp_rewritten_response 
    - calibrated_su_rewritten_response 

use Together API to call a model to evaluate the linguistic confidence (I will supply the API key later).
For each response, use the following prompt template and perform inference 9 times. 

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


After getting all the API responses, extract the numbers using regex and normalise them to [0-1], cache the results and fit a beta distribution over the 9 scores for each response. Therefore, each of the original and calibrated rewritten responses will have a fitted Beta distribution representing its confidence. 

Finally, for each model, compute the generalised ECE, Faithfulness Divergence for each column (original, calibrated rewritten confidence). 

So the final detailed table for each model will have the following columns:
- original response (from the existing CSV)
- correctness (from the existing CSV)
- calibrated_lc_rewritten_response (from the existing CSV)
- calibrated_tp_rewritten_response (from the existing CSV)
- calibrated_su_rewritten_response (from the existing CSV)
- original_response_lc (estimated confidence Beta distribution)
- calibrated_lc_rewritten_response_lc (estimated confidence Beta distribution of calibrated_lc_rewritten_respons)
- calibrated_tp_rewritten_response_lc (estimated confidence Beta distribution of calibrated_tp_rewritten_response)
- calibrated_su_rewritten_response_lc (estimated confidence Beta distribution of calibrated_su_rewritten_response)

You should also produce a summary table for genrealised ECE and faithfulness Divergence. The metric calculation is in `/home/ivan/lm-confidence-evaluation-harness/lm_conf/post_processing/metrics.py`. 

store the code, results and cache in `/home/ivan/lm-confidence-evaluation-harness/rebuttal/disjoint_llm_eval_truthful_qa`