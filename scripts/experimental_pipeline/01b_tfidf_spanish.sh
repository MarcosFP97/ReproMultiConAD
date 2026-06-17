#!/bin/bash
#SBATCH --job-name=TFIDF_SPA_binary_multiclass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/TFIDF/%x_%j.log

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/TF_IDF_classifier.py"

python "$SCRIPT_PATH" --test_language spa --task binary
python "$SCRIPT_PATH" --test_language spa --task multiclass
