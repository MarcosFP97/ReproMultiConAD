#!/bin/bash
#SBATCH --job-name=TFIDF_single_EN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/TFIDF/%x_%j.log

set -euo pipefail

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/TF_IDF_single_classifier.py"

for dataset in pitt delaware lu taukadial vas wls; do
    echo "=== Dataset: $dataset | binary | unbalanced ==="
    python "$SCRIPT_PATH" --dataset "$dataset" --task binary

    echo "=== Dataset: $dataset | binary | balanced ==="
    python "$SCRIPT_PATH" --dataset "$dataset" --task binary --balanced

    echo "=== Dataset: $dataset | multiclass | unbalanced ==="
    python "$SCRIPT_PATH" --dataset "$dataset" --task multiclass

    echo "=== Dataset: $dataset | multiclass | balanced ==="
    python "$SCRIPT_PATH" --dataset "$dataset" --task multiclass --balanced
done
