#!/bin/bash

cd ./ReproMultiConAD

python -m preprocessing_text.preprocess_individual_english \
  --dataset "${1:-all}" \
  --data-root ./jsonl
