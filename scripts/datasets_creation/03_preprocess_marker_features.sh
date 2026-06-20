#!/bin/bash
#SBATCH --job-name=preprocess_CHAT_markers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --output=/mnt/beegfs/groups/irgroup/sara_tfg/logs/%x_%A_%a.log
#SBATCH --array=0-9

source ~/.bashrc
conda activate sara_tfg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT="/mnt/beegfs/groups/irgroup/sara_tfg/ConvoCognition/Preprocessing_text/preprocess_language_features.py"

languages=(en en en en en spa spa spa spa spa)
modes=(none pause rep ref all none pause rep ref all)

i="${SLURM_ARRAY_TASK_ID:-0}"

python "$SCRIPT" \
  --language "${languages[$i]}" \
  --mode "${modes[$i]}" \
  --data-root /mnt/beegfs/groups/irgroup/sara_tfg/jsonl
