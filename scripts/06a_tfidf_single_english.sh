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
conda init bash
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/MultiConAD/Experiments/TF_IDF_single_classifier.py"

for dataset in pitt delaware lu taukadial vas wls; do
    echo "=== Dataset: $dataset | unbalanced ==="
    python "$SCRIPT_PATH" --dataset "$dataset"

    echo "=== Dataset: $dataset | balanced ==="
    python "$SCRIPT_PATH" --dataset "$dataset" --balanced
done
