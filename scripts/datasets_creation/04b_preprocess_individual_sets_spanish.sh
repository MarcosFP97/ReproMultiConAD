#!/bin/bash

cd ./ReproMultiConAD

python -m preprocessing_text.preprocess_individual_spanish \
  --dataset "${1:-all}" \
  --data-root ./jsonl
