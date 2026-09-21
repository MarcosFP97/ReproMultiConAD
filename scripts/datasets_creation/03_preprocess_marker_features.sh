#!/bin/bash

SCRIPT="../../Preprocessing_text/preprocess_language_features.py"

languages=(en en en en en spa spa spa spa spa)
modes=(none pause rep ref all none pause rep ref all)

i="${SLURM_ARRAY_TASK_ID:-0}"

python "$SCRIPT" \
  --language "${languages[$i]}" \
  --mode "${modes[$i]}" \
  --data-root ./jsonl
