#!/bin/bash
#SBATCH --job-name=BERT_classification_EN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/BERT/%x_%j.log

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/BERT_classification.py"

python -u "$SCRIPT_PATH" --language en --task binary
python -u "$SCRIPT_PATH" --language en --task multiclass
