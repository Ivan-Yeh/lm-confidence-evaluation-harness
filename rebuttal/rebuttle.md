You are a computer scientist investigating confidfence calibration in the context of LLMs.

We need to compare the existing metrics (Faithfulness Divergence, generalised ECE and AUROC) from our RLAC method vs Wang et el's method from https://arxiv.org/abs/2410.04315.

Since we have all the metircs of RALC for Truthful QA in `/hdd/ivny/direct_qa_in_domain_calibration/truthful_qa` already. We need to run additional experiments using Wang's method. and comapare the metrics and show that our method is better (larger ECE and Faithfulness Divergence reduction).

Their method uses discrete optimal transport based on an existing lexicon. In our setting, we'll replicate their setting using the following setup and steps:

1. the lexicon we will use is `/mnt/ssd_1/ivn/lm-confidence-evaluation-harness/linguistic_confidence_lexicon/hedging_word_scores.csv` which contains a list of hedging expressions and their corresponding Beta distribution parameters. 
2. You will need to use regex to find the presence of the existing hedging words in each sentence (i.e. the hedging expression in the sentence also exists in the lexicon) and assign the statement with a confidence distribution that is the Beta distribution of that hedging expression in the lexicon. This will be the original, uncalibrated confidnece. 
3.  And then you will perform Wang's method of discrete optimal transport to calibrate each sentence by suggesting alternative word exchange (you do not need to rewrite the sentence) from the lexicon based on their confidnece profile. The new hedging expression's confidence distribution in the lexicon then becomes the calibrated confidnece.
4. You will then compute all metrics (FD, generlaised ECE and AUROC) of from uncalibreated and calibrated confidence. And report them in the table for each model. you shuold also compare the metrics and confidence profiles with the RALC's results in `/hdd/ivny/direct_qa_in_domain_calibration/truthful_qa` 
5. For regex matching, you will need to report missing count (no hedging expressions in the lexicon is present in the target sentence). This sentence is disregarded and excluded from the metrics computation.
6. Finally, you will need to look at the results and write down the justification for why RALC is superior in terms of generalisability. (may be focus on the mismatch count of regex matching, RALC does not use regex, it assess the sentence holistically rather than looking for specific words.)