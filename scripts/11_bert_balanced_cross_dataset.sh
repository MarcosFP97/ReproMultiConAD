#!/bin/bash
#SBATCH --job-name=balancedBERT_cross_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/BERT/%x_%j.log

set -euo pipefail

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

TRAIN_DATASET="${TRAIN_DATASET:-pitt}"
TEST_DATASET="${TEST_DATASET:-taukadial}"
TASK="${TASK:-binary}"

python /mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/BERT_balanced.py \
  --mode cross \
  --train-dataset "$TRAIN_DATASET" \
  --test-dataset "$TEST_DATASET" \
  --task "$TASK"
