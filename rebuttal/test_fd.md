For each model and dataset in `/hdd/ivny/direct_qa_in_domain_calibration` there is a file named `calibration_details.csv`

In easch CSV, you need to compare the calibrated signal confidence profile of each entry (lc, su, tp) with their rewritten responses' lc.

Perform test on kappa, the concentration, of both spaces (calibrated signal vs corresponding rewritten lc) and see if they correlate with each other. 

Save the code and results (in csv) in /home/ivan/lm-confidence-evaluation-harness/rebuttal/fd_test/.

I want to see a positive correlation. 