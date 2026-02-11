#!/usr/bin/env bash

SUBSET=300

python -m single_pass_experiments.sampling --subset $SUBSET --dataset mmlu --estimation_method dist_linguistic_confidence
python -m single_pass_experiments.sampling --subset $SUBSET --dataset mmlu --estimation_method dist_lnll

python -m single_pass_experiments.sampling --subset $SUBSET --dataset trivia_qa --estimation_method dist_linguistic_confidence
python -m single_pass_experiments.sampling --subset $SUBSET --dataset trivia_qa --estimation_method dist_lnll

python -m single_pass_experiments.sampling --subset $SUBSET --dataset squadv2 --estimation_method dist_linguistic_confidence
python -m single_pass_experiments.sampling --subset $SUBSET --dataset squadv2 --estimation_method dist_lnll