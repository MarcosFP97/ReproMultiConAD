#!/bin/bash
#SBATCH --job-name=E5_EN_binary_multiclass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/E5/%x_%j.log

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_PATH="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Experiments/e5_larg_classifier.py"

python "$SCRIPT_PATH" --test_language en --task binary
python "$SCRIPT_PATH" --test_language en --task multiclass
