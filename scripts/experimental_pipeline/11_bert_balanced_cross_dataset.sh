#!/bin/bash
#SBATCH --job-name=balancedBERT_cross_task_EN
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

python -u "$PYTHON_SCRIPT" \
  --mode cross_task \
  --train-dataset pitt \
  --task binary \
  --binary-task disease_status \
  --cross-test-datasets wls taukadial
