#!/bin/bash

cd ./ReproMultiConAD
mkdir -p ./jsonl/results_cha_collection

datasets=(Baycrest Delaware Ivanova Kempler Lu PerLA Pitt VAS WLS)
languages=(english english spanish english english spanish english english english)

i="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$i]}"
language="${languages[$i]}"

python -m extracting_data.cha_collection \
  --input-dir "./$dataset" \
  --language "$language" \
  --output-dir ./jsonl/results_cha_collection \
  --output-name "${dataset}.jsonl"
