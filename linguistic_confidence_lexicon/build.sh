#!/usr/bin/env bash

GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -m linguistic_confidence_lexicon.build_lexicon
# CUDA_VISIBLE_DEVICES=$GPU python -m linguistic_confidence_lexicon.build_confidence
# python -m linguistic_confidence_lexicon.build_distributions