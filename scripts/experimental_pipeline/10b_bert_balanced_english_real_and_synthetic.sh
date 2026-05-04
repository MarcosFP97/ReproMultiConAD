#!/bin/bash
#SBATCH --job-name=balancedBERT_EN_real_synth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/BERT/%x_%j.log

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

PYTHON_SCRIPT="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/BERT_balanced.py"
TRAIN_DATASET="${TRAIN_DATASET:-pitt}"
SYNTHETIC_SOURCE="${SYNTHETIC_SOURCE:-mistral}"

for task in binary multiclass; do
  python "$PYTHON_SCRIPT" \
    --mode synthetic \
    --train-dataset "$TRAIN_DATASET" \
    --task "$task" \
    --train-source real \
    --real-percentage 100

  python "$PYTHON_SCRIPT" \
    --mode synthetic \
    --train-dataset "$TRAIN_DATASET" \
    --task "$task" \
    --train-source synthetic \
    --synthetic-percentage 100 \
    --synthetic-source "$SYNTHETIC_SOURCE"
done
