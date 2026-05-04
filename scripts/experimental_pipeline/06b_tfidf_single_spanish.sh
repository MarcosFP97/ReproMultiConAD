#!/bin/bash
#SBATCH --job-name=TFIDF_single_SPA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/TFIDF/%x_%j.log

set -euo pipefail

source ~/.bashrc
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/TF_IDF_single_classifier.py"

echo "=== Dataset: ivanova | unbalanced ==="
python "$SCRIPT_PATH" --dataset ivanova

echo "=== Dataset: ivanova | balanced ==="
python "$SCRIPT_PATH" --dataset ivanova --balanced
