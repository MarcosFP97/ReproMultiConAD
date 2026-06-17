#!/bin/bash
#SBATCH --job-name=balancedBERT_ES_real_synth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/BERT/%x_%j.log

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

PYTHON_SCRIPT="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/BERT_balanced.py"
TRAIN_DATASET="${TRAIN_DATASET:-ivanova}"

for task in binary multiclass; do
  for real_percentage in 20 40 60 80 100; do
    python "$PYTHON_SCRIPT" \
      --mode synthetic \
      --train-dataset "$TRAIN_DATASET" \
      --task "$task" \
      --train-source real \
      --real-percentage "$real_percentage"
  done

  for synthetic_source in mistral gemini; do
    python "$PYTHON_SCRIPT" \
      --mode synthetic \
      --train-dataset "$TRAIN_DATASET" \
      --task "$task" \
      --train-source synthetic \
      --synthetic-percentage 100 \
      --synthetic-source "$synthetic_source"

    for real_percentage in 20 40 60 80; do
      synthetic_percentage=$((100 - real_percentage))

      python "$PYTHON_SCRIPT" \
        --mode synthetic \
        --train-dataset "$TRAIN_DATASET" \
        --task "$task" \
        --train-source augmented \
        --real-percentage "$real_percentage" \
        --synthetic-percentage "$synthetic_percentage" \
        --synthetic-source "$synthetic_source"
    done
  done
done
